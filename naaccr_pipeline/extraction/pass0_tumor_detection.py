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

from naaccr_pipeline.llm.client import VLLMClient
from naaccr_pipeline.llm.structured_output import SchemaBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_PATHOLOGY_CHARS = 2000
_MAX_OTHER_CHARS = 1000
_MAX_OTHER_CHUNKS = 3
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

    def __init__(self, llm_client: VLLMClient, schema_builder: SchemaBuilder) -> None:
        self._llm = llm_client
        self._schema_builder = schema_builder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect(self, chunks: list[Any]) -> list[TumorCandidate]:
        """Detect primary cancers from *chunks*.

        Steps
        -----
        1. Select representative chunks (all pathology + first N others).
        2. Build condensed text excerpts.
        3. Call LLM to identify distinct primaries.
        4. Deduplicate results.
        5. Assign relevant chunk IDs.
        6. If nothing found, return a single "unknown" candidate.
        """
        if not chunks:
            logger.warning("No chunks provided for tumor detection.")
            return [self._unknown_candidate()]

        # 1-2. Build condensed texts
        condensed = self._build_condensed_texts(chunks)
        if not condensed:
            logger.warning("No usable text in chunks.")
            return [self._unknown_candidate()]

        # 3. Call LLM
        system_prompt, user_prompt = self._build_detection_prompt(condensed)
        schema = self._build_schema()

        response = await self._llm.extract(system_prompt, user_prompt, schema)

        if response.get("_error"):
            logger.error(
                "LLM error during tumor detection: %s",
                response.get("_message", "unknown"),
            )
            return [self._unknown_candidate()]

        raw_tumors = response.get("tumors", [])
        if not isinstance(raw_tumors, list):
            logger.warning("Unexpected tumors field type: %s", type(raw_tumors))
            return [self._unknown_candidate()]

        if not raw_tumors:
            logger.info("LLM found no tumors; returning unknown candidate.")
            return [self._unknown_candidate()]

        # 4-5. Deduplicate and assign chunk IDs
        candidates = self._deduplicate(raw_tumors, chunks)

        if not candidates:
            return [self._unknown_candidate()]

        logger.info(
            "Tumor detection found %d distinct primary cancer(s).", len(candidates)
        )
        return candidates

    # ------------------------------------------------------------------
    # Condensed text preparation
    # ------------------------------------------------------------------

    def _build_condensed_texts(
        self, chunks: list[Any]
    ) -> list[tuple[str, str, str]]:
        """Select representative chunks and truncate to useful lengths.

        Returns a list of ``(chunk_type, text_excerpt, chunk_id)`` tuples.
        """
        pathology_chunks: list[Any] = []
        other_chunks: list[Any] = []

        for chunk in chunks:
            ctype = getattr(chunk, "chunk_type", "other")
            if ctype == "pathology":
                pathology_chunks.append(chunk)
            else:
                other_chunks.append(chunk)

        # Sort others by document_date ascending (oldest first) for context
        other_chunks.sort(
            key=lambda c: getattr(c, "document_date", "") or ""
        )

        selected: list[tuple[str, str, str]] = []

        # All pathology chunks
        for chunk in pathology_chunks:
            text = getattr(chunk, "text", "")
            chunk_id = getattr(chunk, "chunk_id", "unknown")
            ctype = getattr(chunk, "chunk_type", "pathology")
            truncated = text[:_MAX_PATHOLOGY_CHARS].strip()
            if truncated:
                selected.append((ctype, truncated, chunk_id))

        # First N other chunks
        for chunk in other_chunks[:_MAX_OTHER_CHUNKS]:
            text = getattr(chunk, "text", "")
            chunk_id = getattr(chunk, "chunk_id", "unknown")
            ctype = getattr(chunk, "chunk_type", "other")
            truncated = text[:_MAX_OTHER_CHARS].strip()
            if truncated:
                selected.append((ctype, truncated, chunk_id))

        return selected

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_detection_prompt(
        self, condensed_texts: list[tuple[str, str, str]]
    ) -> tuple[str, str]:
        """Build system and user prompts for tumor detection.

        Parameters
        ----------
        condensed_texts:
            List of ``(chunk_type, text_excerpt, chunk_id)`` tuples.

        Returns
        -------
        tuple[str, str]
            ``(system_prompt, user_prompt)``
        """
        system_prompt = (
            "You are an expert cancer registrar. Analyze the following clinical "
            "documents and identify ALL distinct primary cancer diagnoses for "
            "this patient.\n\n"
            "IMPORTANT RULES:\n"
            "1. List only PRIMARY cancers, not metastatic sites.\n"
            "2. A metastasis (e.g., \"liver metastasis from colon cancer\") is "
            "NOT a separate primary.\n"
            "3. A recurrence of the same cancer is NOT a separate primary.\n"
            "4. Different lateralities of paired organs ARE separate primaries "
            "(e.g., left breast and right breast).\n"
            "5. For each cancer found, provide: cancer type, body site, "
            "approximate diagnosis date, and a quote from the text.\n"
            "6. If only one cancer is found, that is fine.\n\n"
            "Respond with JSON."
        )

        text_sections: list[str] = []
        for idx, (chunk_type, text_excerpt, _chunk_id) in enumerate(
            condensed_texts, start=1
        ):
            label = chunk_type.replace("_", " ").title()
            text_sections.append(
                f"--- Document {idx} ({label}) ---\n{text_excerpt}"
            )

        user_prompt = (
            "Clinical documents for patient:\n\n"
            + "\n\n".join(text_sections)
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
        - Same normalised site + same year -> merge (keep the one with more
          detail in ``cancer_type``).
        - Same normalised site + different laterality -> keep separate.
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

        # Second pass: group by (normalised_site, laterality, year)
        groups: dict[tuple[str, str, str], list[dict]] = {}
        for tumor in filtered:
            site = self._normalize_site(tumor.get("primary_site", ""))
            laterality = tumor.get("laterality", "unknown")
            year = self._extract_year(
                tumor.get("approximate_diagnosis_date", "")
            )
            key = (site, laterality, year)
            groups.setdefault(key, []).append(tumor)

        # Third pass: pick best from each group and build candidates
        candidates: list[TumorCandidate] = []
        for idx, ((_site, _lat, _year), group) in enumerate(groups.items()):
            # Pick the entry with the longest cancer_type (most detail)
            best = max(group, key=lambda t: len(t.get("cancer_type", "")))
            primary_site = best.get("primary_site", "unknown")
            laterality = best.get("laterality", "")
            if laterality and laterality not in ("not_applicable", "unknown"):
                site_hint = f"{laterality} {primary_site}"
            else:
                site_hint = primary_site

            candidate = TumorCandidate(
                tumor_index=idx,
                cancer_type=best.get("cancer_type", "unknown"),
                primary_site_hint=site_hint,
                approximate_date=best.get("approximate_diagnosis_date", ""),
                evidence=best.get("evidence", "")[:300],
                relevant_chunk_ids=self._assign_chunk_ids_from_tumor(
                    best, chunks
                ),
            )
            candidates.append(candidate)

        return candidates

    # ------------------------------------------------------------------
    # Chunk-ID assignment
    # ------------------------------------------------------------------

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
    def _extract_year(date_str: str) -> str:
        """Pull the 4-digit year from a date string like 'YYYY-MM' or 'YYYY'."""
        date_str = date_str.strip()
        match = re.match(r"(\d{4})", date_str)
        if match:
            return match.group(1)
        return ""

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
