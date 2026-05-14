"""Pass 0: Detect distinct primary cancers in a patient's documents.

Before extracting NAACCR data items, we must determine how many distinct
primary cancers exist.  Each primary gets its own NAACCR abstract, so
this pass runs first and outputs :class:`TumorCandidate` objects that
drive all subsequent extraction passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
import logging

from onc_registry_pipeline.llm.client import VLLMClient
from onc_registry_pipeline.llm.structured_output import SchemaBuilder

logger = logging.getLogger(__name__)

_METASTASIS_PATTERNS = re.compile(
    r"\b(?:metasta(?:sis|tic|ses|sized?)|mets?\s+(?:from|to)|"
    r"secondary\s+(?:deposit|malignan)|spread\s+(?:from|to))\b",
    re.IGNORECASE,
)

# Body-site synonyms for deduplication matching
_SITE_SYNONYMS: dict[str, str] = {
    "breast": "breast",
    "lung": "lung",
    "colon": "colon",
    "rectum": "colorectal",
    "colorectal": "colorectal",
    "prostate": "prostate",
    "kidney": "kidney",
    "renal": "kidney",
    "liver": "liver",
    "hepatic": "liver",
    "pancreas": "pancreas",
    "pancreatic": "pancreas",
    "bladder": "bladder",
    "thyroid": "thyroid",
    "ovary": "ovary",
    "ovarian": "ovary",
    "uterus": "uterus",
    "uterine": "uterus",
    "endometrium": "uterus",
    "endometrial": "uterus",
    "cervix": "cervix",
    "cervical": "cervix",
    "stomach": "stomach",
    "gastric": "stomach",
    "esophagus": "esophagus",
    "esophageal": "esophagus",
    "skin": "skin",
    "melanoma": "skin",
    "brain": "brain",
    "cerebral": "brain",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TumorCandidate:
    """A distinct primary cancer detected in a patient's documents."""

    tumor_index: int              # 0-based index
    cancer_type: str              # free text (e.g. "invasive ductal carcinoma")
    primary_site_hint: str        # body location hint (e.g. "left breast")
    approximate_date: str         # approximate diagnosis date (YYYY or YYYY-MM)
    evidence: str                 # supporting text quote
    histology: str = ""           # free text histology / morphology description
    laterality: str = "unknown"
    diagnosis_key: str = ""
    relevant_chunk_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tumor detector
# ---------------------------------------------------------------------------

class TumorDetector:
    """Detects distinct primary cancers in a patient's document set.

    Usage::

        detector = TumorDetector(llm_client, schema_builder)
        candidates = await detector.detect(chunks)
    """

    def __init__(
        self,
        llm_client: VLLMClient,
        schema_builder: SchemaBuilder,
        llm_log: Any = None,
    ) -> None:
        self._llm = llm_client
        self._schema_builder = schema_builder
        self._llm_log = llm_log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect(self, chunks: list[Any]) -> list[TumorCandidate]:
        """Detect primary cancers from *chunks*.

        Steps
        -----
        1. Call the LLM once for every chronological patient chunk.
        2. Merge all chunk-level diagnosis candidates.
        3. Deduplicate by primary site + histology + laterality + diagnosis date.
        4. Assign relevant chunk IDs.
        5. If nothing found, return a single "unknown" candidate.
        """
        if not chunks:
            logger.warning("No chunks provided for tumor detection.")
            return [self._unknown_candidate()]

        raw_tumors: list[dict] = []
        for chunk in chunks:
            raw_tumors.extend(await self._detect_in_chunk(chunk))

        if not raw_tumors:
            logger.info("Tumor detection found no candidates in any chunk.")
            return [self._unknown_candidate()]

        candidates = self._deduplicate(raw_tumors, chunks)

        if not candidates:
            return [self._unknown_candidate()]

        logger.info(
            "Tumor detection found %d distinct primary cancer(s).", len(candidates)
        )
        return candidates

    # ------------------------------------------------------------------
    # Chunk-level detection
    # ------------------------------------------------------------------

    async def _detect_in_chunk(self, chunk: Any) -> list[dict]:
        """Detect primary cancer diagnoses mentioned in one chunk."""
        text = getattr(chunk, "text", "").strip()
        chunk_id = getattr(chunk, "chunk_id", "unknown")
        if not text:
            return []

        system_prompt, user_prompt = self._build_detection_prompt(chunk)
        llm_response = await self._llm.extract(system_prompt, user_prompt)

        if self._llm_log is not None:
            self._llm_log.log(
                call_type="tumor_detection",
                pass_number=getattr(chunk, "chunk_index", 0),
                chunk_id=chunk_id,
                chunk_type=getattr(chunk, "chunk_type", "sequential"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=llm_response.raw_content,
                reasoning=llm_response.reasoning,
                final_output=llm_response.final_content,
                parsed=llm_response.parsed,
            )

        if llm_response.parsed.get("_error"):
            logger.error(
                "LLM error during tumor detection for chunk %s: %s",
                chunk_id,
                llm_response.parsed.get("_message", "unknown"),
            )
            return []

        raw_tumors = llm_response.parsed.get("tumors", [])
        if not isinstance(raw_tumors, list):
            logger.warning(
                "Unexpected tumors field type for chunk %s: %s",
                chunk_id,
                type(raw_tumors),
            )
            return []

        normalized: list[dict] = []
        for tumor in raw_tumors:
            if not isinstance(tumor, dict):
                continue
            tumor["_source_chunk_id"] = chunk_id
            normalized.append(tumor)
        return normalized

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_detection_prompt(self, chunk: Any) -> tuple[str, str]:
        """Build system and user prompts for one detection chunk."""
        system_prompt = (
            "You are an expert cancer registrar. Analyze the following clinical "
            "text chunk and identify ALL distinct primary cancer diagnoses "
            "mentioned in this chunk.\n\n"
            "IMPORTANT RULES:\n"
            "1. List only PRIMARY cancers, not metastatic sites.\n"
            "2. A metastasis (e.g., \"liver metastasis from colon cancer\") is "
            "NOT a separate primary.\n"
            "3. A recurrence of the same cancer is NOT a separate primary.\n"
            "4. Different lateralities of paired organs ARE separate primaries "
            "(e.g., left breast and right breast).\n"
            "5. If the same primary cancer is mentioned multiple times in this "
            "chunk, return one entry with the earliest diagnosis date stated.\n"
            "6. For each cancer found, provide: cancer type, histology, body "
            "site, laterality, approximate diagnosis date, and a quote from "
            "the text.\n"
            "7. If no primary cancer diagnosis is mentioned in this chunk, "
            "return an empty tumors array.\n\n"
            "Respond with a JSON object containing a \"tumors\" array. "
            "Each tumor should have: \"cancer_type\" (string), "
            "\"histology\" (string; use cancer_type if no separate histology "
            "wording is available), "
            "\"primary_site\" (string), \"laterality\" (one of: left, right, "
            "bilateral, not_applicable, unknown), "
            "\"approximate_diagnosis_date\" (YYYY-MM or YYYY), "
            "and \"evidence\" (short quote, max 300 chars)."
        )

        chunk_id = getattr(chunk, "chunk_id", "unknown")
        chunk_type = getattr(chunk, "chunk_type", "sequential")
        first_date = getattr(chunk, "first_date", "") or getattr(
            chunk, "document_date", "unknown"
        )
        last_date = getattr(chunk, "last_date", "") or getattr(
            chunk, "document_date", "unknown"
        )

        user_prompt = (
            f"Clinical text chunk {chunk_id} "
            f"(type: {chunk_type}, dates: {first_date} to {last_date}):\n\n"
            f"{getattr(chunk, 'text', '')}"
        )

        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _build_schema(self) -> dict:
        """JSON schema for the tumor detection response."""
        return self._schema_builder.build_simple_schema(
            {
                "tumors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cancer_type": {
                                "type": "string",
                                "description": "Cancer type / histology description",
                            },
                            "primary_site": {
                                "type": "string",
                                "description": "Body site of primary cancer",
                            },
                            "histology": {
                                "type": "string",
                                "description": "Histology / morphology description",
                            },
                            "laterality": {
                                "type": "string",
                                "enum": [
                                    "left",
                                    "right",
                                    "bilateral",
                                    "not_applicable",
                                    "unknown",
                                ],
                            },
                            "approximate_diagnosis_date": {
                                "type": "string",
                                "description": "Approximate date YYYY-MM or YYYY",
                            },
                            "evidence": {
                                "type": "string",
                                "maxLength": 300,
                                "description": "Quote from text supporting this finding",
                            },
                        },
                        "required": [
                            "cancer_type",
                            "histology",
                            "primary_site",
                            "approximate_diagnosis_date",
                            "evidence",
                        ],
                    },
                }
            }
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(
        self, raw_tumors: list[dict], chunks: list[Any]
    ) -> list[TumorCandidate]:
        """Deduplicate raw tumor dicts into :class:`TumorCandidate` objects.

        Rules
        -----
        - Same normalized site + histology + laterality + diagnosis date ->
          merge (keep the candidate with the best detail and evidence).
        - If the cancer_type or evidence contains clear metastasis language,
          remove it.
        """
        # First pass: filter out metastatic entries
        filtered: list[dict] = []
        for tumor in raw_tumors:
            cancer_type = tumor.get("cancer_type", "")
            evidence = tumor.get("evidence", "")
            combined = f"{cancer_type} {evidence}"
            if _METASTASIS_PATTERNS.search(combined):
                logger.debug(
                    "Filtered metastatic entry: %s", cancer_type
                )
                continue
            filtered.append(tumor)

        if not filtered:
            return []

        # Second pass: group by the requested diagnosis identity key.
        groups: dict[tuple[str, str, str, str], list[dict]] = {}
        for tumor in filtered:
            site = self._normalize_site(tumor.get("primary_site", ""))
            histology = self._normalize_histology(
                tumor.get("histology") or tumor.get("cancer_type", "")
            )
            laterality = self._normalize_laterality(tumor.get("laterality", "unknown"))
            diagnosis_date = self._normalize_diagnosis_date(
                tumor.get("approximate_diagnosis_date", "")
            )
            key = (site, histology, laterality, diagnosis_date)
            groups.setdefault(key, []).append(tumor)

        # Third pass: pick best from each group and build candidates
        candidates_by_key: list[tuple[tuple[str, str, str, str], TumorCandidate]] = []
        for key, group in groups.items():
            # Prefer candidates with detailed histology/cancer type and evidence.
            best = max(
                group,
                key=lambda t: (
                    len(t.get("histology") or ""),
                    len(t.get("cancer_type") or ""),
                    len(t.get("evidence") or ""),
                ),
            )
            primary_site = best.get("primary_site", "unknown")
            histology = best.get("histology") or best.get("cancer_type", "unknown")
            laterality = self._normalize_laterality(best.get("laterality", "unknown"))
            if laterality and laterality not in ("not_applicable", "unknown"):
                site_hint = f"{laterality} {primary_site}"
            else:
                site_hint = primary_site
            relevant_chunk_ids = self._chunk_ids_for_group(group, best, chunks)

            candidate = TumorCandidate(
                tumor_index=0,
                cancer_type=best.get("cancer_type", "unknown"),
                primary_site_hint=site_hint,
                approximate_date=best.get("approximate_diagnosis_date", ""),
                evidence=best.get("evidence", "")[:300],
                histology=histology,
                laterality=laterality,
                diagnosis_key="|".join(key),
                relevant_chunk_ids=relevant_chunk_ids,
            )
            candidates_by_key.append((key, candidate))

        candidates = []
        for idx, (_key, candidate) in enumerate(sorted(candidates_by_key)):
            candidate.tumor_index = idx
            candidates.append(candidate)
        return candidates

    # ------------------------------------------------------------------
    # Chunk-ID assignment
    # ------------------------------------------------------------------

    def _chunk_ids_for_group(
        self,
        group: list[dict],
        best: dict,
        chunks: list[Any],
    ) -> list[str]:
        """Combine source chunk IDs from detection with keyword matches."""
        source_ids = [
            str(t.get("_source_chunk_id", "")).strip()
            for t in group
            if str(t.get("_source_chunk_id", "")).strip()
        ]
        keyword_ids = self._assign_chunk_ids_from_tumor(best, chunks)
        seen: set[str] = set()
        ordered: list[str] = []
        for chunk_id in source_ids + keyword_ids:
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                ordered.append(chunk_id)
        return ordered

    def _assign_chunk_ids_from_tumor(
        self, tumor: dict, chunks: list[Any]
    ) -> list[str]:
        """Find chunks that mention this tumor's site or cancer type."""
        cancer_type = tumor.get("cancer_type", "").lower()
        primary_site = tumor.get("primary_site", "").lower()

        # Build search keywords from cancer type and site
        keywords = self._extract_keywords(cancer_type, primary_site)
        if not keywords:
            return []

        matched_ids: list[str] = []
        for chunk in chunks:
            chunk_text = getattr(chunk, "text", "").lower()
            chunk_id = getattr(chunk, "chunk_id", "unknown")
            if not chunk_text:
                continue
            # A chunk matches if at least one keyword appears
            if any(kw in chunk_text for kw in keywords):
                matched_ids.append(chunk_id)

        return matched_ids

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_site(site: str) -> str:
        """Map a body-site string to a canonical form for dedup comparison."""
        lower = site.lower().strip()
        # Try each word in the site string against the synonym map
        for word in re.split(r"\s+", lower):
            word_clean = word.strip(",.;:()")
            if word_clean in _SITE_SYNONYMS:
                return _SITE_SYNONYMS[word_clean]
        # Fall back to the full lowered string
        return lower

    @staticmethod
    def _normalize_histology(histology: str) -> str:
        """Normalize free-text histology for diagnosis-key comparison."""
        normalized = re.sub(r"[^a-z0-9]+", " ", histology.lower()).strip()
        return re.sub(r"\s+", " ", normalized) or "unknown"

    @staticmethod
    def _normalize_laterality(laterality: str) -> str:
        """Normalize laterality values emitted by the LLM."""
        value = laterality.lower().strip().replace(" ", "_")
        valid = {"left", "right", "bilateral", "not_applicable", "unknown"}
        return value if value in valid else "unknown"

    @staticmethod
    def _normalize_diagnosis_date(date_str: str) -> str:
        """Normalize dates to YYYY-MM-DD, YYYY-MM, YYYY, or unknown."""
        date_str = date_str.strip()
        match = re.match(r"(\d{4})(?:[-/]?(\d{2}))?(?:[-/]?(\d{2}))?", date_str)
        if match:
            parts = [part for part in match.groups() if part]
            return "-".join(parts)
        return "unknown"

    @staticmethod
    def _extract_keywords(cancer_type: str, primary_site: str) -> list[str]:
        """Build a list of non-trivial keywords for chunk matching."""
        stop_words = {
            "of", "the", "and", "in", "with", "a", "an", "or", "is",
            "was", "for", "to", "at", "by", "on", "not", "no", "from",
            "type", "grade", "stage", "cancer", "carcinoma", "tumor",
            "tumour", "malignant", "malignancy", "primary", "diagnosis",
            "diagnosed", "unknown",
        }
        raw_words: list[str] = []
        for text in (cancer_type, primary_site):
            for word in re.split(r"\s+", text):
                cleaned = word.strip(",.;:()")
                if cleaned and cleaned not in stop_words and len(cleaned) > 2:
                    raw_words.append(cleaned)

        # Deduplicate while preserving order
        seen: set[str] = set()
        keywords: list[str] = []
        for w in raw_words:
            if w not in seen:
                seen.add(w)
                keywords.append(w)
        return keywords

    @staticmethod
    def _unknown_candidate() -> TumorCandidate:
        """Return a placeholder candidate when no tumors can be identified."""
        return TumorCandidate(
            tumor_index=0,
            cancer_type="unknown",
            primary_site_hint="unknown",
            approximate_date="",
            evidence="No cancer diagnosis could be identified from available documents.",
            relevant_chunk_ids=[],
        )
