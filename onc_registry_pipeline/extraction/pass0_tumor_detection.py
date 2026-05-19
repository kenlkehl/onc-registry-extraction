"""Pass 0: Detect distinct primary cancers in a patient's documents.

Before extracting NAACCR data items, we must determine how many distinct
primary cancers exist.  Each primary gets its own NAACCR abstract, so
this pass runs first and outputs :class:`TumorCandidate` objects that
drive all subsequent extraction passes.

Approach
--------
For each patient we walk chronologically through document chunks and ask
the LLM to maintain a running list of distinct primary cancers,
incrementally refining/merging entries as new chunks arrive.  After the
last chunk, a single coding-pass call assigns ICD-O-3 topography and
morphology codes to every diagnosis with both reference tables in the
prompt.

This replaces the prior per-chunk-then-rules-based-dedup approach,
which failed to merge diagnoses described with different wording,
laterality precision, or date precision across chunks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from onc_registry_pipeline.icd_o3_reference import (
    MORPHOLOGY,
    TOPOGRAPHY_EXACT,
    TOPOGRAPHY_PREFIX,
)
from onc_registry_pipeline.llm.client import VLLMClient
from onc_registry_pipeline.llm.structured_output import SchemaBuilder

logger = logging.getLogger(__name__)


_VALID_LATERALITY: set[str] = {
    "left",
    "right",
    "bilateral",
    "not_applicable",
    "unknown",
}

# Topography: Cxx.x (or Cxxx without the decimal — we normalize to Cxx.x).
_TOPOGRAPHY_RE = re.compile(r"^C(\d{2})\.?(\d)$", re.IGNORECASE)

# Morphology: 4-digit code, optionally followed by /behavior; we strip the
# behavior suffix because the rest of the pipeline stores morphology and
# behavior separately.
_MORPHOLOGY_RE = re.compile(r"^(\d{4})(?:/\d)?$")


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
    diagnosis_key: str = ""       # retained for serialization compatibility
    relevant_chunk_ids: list[str] = field(default_factory=list)
    primary_site_code: str = ""   # ICD-O-3 topography (e.g. "C50.4")
    histology_code: str = ""      # ICD-O-3 morphology, 4-digit (e.g. "8500")


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
        schema_builder: SchemaBuilder | None = None,
        llm_log: Any = None,
    ) -> None:
        self._llm = llm_client
        # schema_builder is unused; kept on the signature for backward
        # compatibility with existing call sites.
        self._schema_builder = schema_builder
        self._llm_log = llm_log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect(self, chunks: list[Any]) -> list[TumorCandidate]:
        """Detect primary cancers from *chunks*.

        Walks chunks in order, asking the LLM to update a running list of
        distinct primary diagnoses after each chunk.  After the last chunk
        runs a single coding-pass call to assign ICD-O-3 codes.
        """
        if not chunks:
            logger.warning("No chunks provided for tumor detection.")
            return [self._unknown_candidate()]

        running: list[dict] = []
        contributing_chunks: list[str] = []

        for chunk in chunks:
            new_running = await self._update_running_list(chunk, running)
            if new_running != running:
                chunk_id = str(getattr(chunk, "chunk_id", "")).strip()
                if chunk_id and chunk_id not in contributing_chunks:
                    contributing_chunks.append(chunk_id)
            running = new_running

        if not running:
            logger.info("Tumor detection produced an empty running list.")
            return [self._unknown_candidate()]

        coded = await self._assign_codes(running)

        candidates: list[TumorCandidate] = []
        for idx, dx in enumerate(running):
            laterality = self._normalize_laterality(dx.get("laterality", "unknown"))
            primary_site = str(dx.get("primary_site", "") or "unknown").strip()
            site_hint = primary_site or "unknown"
            if (
                laterality
                and laterality not in ("not_applicable", "unknown")
                and not primary_site.lower().startswith(laterality)
            ):
                site_hint = f"{laterality} {primary_site}".strip()

            code_entry = coded[idx] if idx < len(coded) else {}
            candidate = TumorCandidate(
                tumor_index=idx,
                cancer_type=str(dx.get("cancer_type") or "unknown").strip(),
                primary_site_hint=site_hint,
                approximate_date=str(dx.get("approximate_diagnosis_date", "")).strip(),
                evidence=str(dx.get("evidence", ""))[:300],
                histology=str(
                    dx.get("histology") or dx.get("cancer_type", "unknown")
                ).strip(),
                laterality=laterality,
                relevant_chunk_ids=list(contributing_chunks),
                primary_site_code=code_entry.get("primary_site_code", ""),
                histology_code=code_entry.get("histology_code", ""),
            )
            candidates.append(candidate)

        logger.info(
            "Tumor detection found %d distinct primary cancer(s).", len(candidates)
        )
        return candidates

    # ------------------------------------------------------------------
    # Running-list update (one call per chunk)
    # ------------------------------------------------------------------

    async def _update_running_list(
        self, chunk: Any, running: list[dict]
    ) -> list[dict]:
        """Ask the LLM to update *running* using the new chunk."""
        text = getattr(chunk, "text", "").strip()
        chunk_id = getattr(chunk, "chunk_id", "unknown")
        if not text:
            return running

        system_prompt = self._build_running_list_system_prompt()
        user_prompt = self._build_running_list_user_prompt(chunk, running)

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
            return running

        diagnoses_raw = llm_response.parsed.get("diagnoses")
        if not isinstance(diagnoses_raw, list):
            logger.warning(
                "Unexpected diagnoses field type for chunk %s: %s",
                chunk_id,
                type(diagnoses_raw),
            )
            return running

        cleaned: list[dict] = []
        for entry in diagnoses_raw:
            if not isinstance(entry, dict):
                continue
            cleaned.append(
                {
                    "cancer_type": str(entry.get("cancer_type", "")).strip(),
                    "primary_site": str(entry.get("primary_site", "")).strip(),
                    "histology": str(entry.get("histology", "")).strip(),
                    "laterality": self._normalize_laterality(
                        entry.get("laterality", "unknown")
                    ),
                    "approximate_diagnosis_date": str(
                        entry.get("approximate_diagnosis_date", "")
                    ).strip(),
                    "evidence": str(entry.get("evidence", ""))[:300],
                }
            )

        if len(cleaned) < len(running):
            logger.info(
                "Chunk %s: running list shrank from %d to %d (LLM merged entries).",
                chunk_id,
                len(running),
                len(cleaned),
            )

        return cleaned

    def _build_running_list_system_prompt(self) -> str:
        return (
            "You are an expert cancer registrar. You maintain a running list "
            "of the distinct PRIMARY cancers diagnosed for one patient as you "
            "see new chunks of clinical text from that patient's chart.\n\n"
            "Each turn you receive:\n"
            "  * the current running list of diagnoses (may be empty)\n"
            "  * a new chunk of clinical text\n\n"
            "Return the UPDATED full list of distinct primary cancers, "
            "incorporating any information from the new chunk.\n\n"
            "RULES:\n"
            "1. List only PRIMARY cancers. A metastasis (e.g. \"liver "
            "metastasis from colon cancer\") is NOT a separate primary; the "
            "underlying primary is.\n"
            "2. A recurrence of a previously listed cancer is NOT a separate "
            "primary.\n"
            "3. Different lateralities of paired organs (e.g. left breast and "
            "right breast) ARE separate primaries.\n"
            "4. Same cancer described with different wording across notes "
            "is ONE primary -- merge such entries and prefer the more "
            "specific description.  Examples that must be merged:\n"
            "   - \"acute lymphoblastic leukemia\" and \"acute lymphocytic "
            "leukemia\" (synonyms)\n"
            "   - \"adenocarcinoma\" and \"mucinous adenocarcinoma\" at the "
            "same site and around the same date (granularity)\n"
            "   - \"left kidney mass\" and \"kidney cancer\" at the same date "
            "(laterality precision difference)\n"
            "5. Date precision differences alone do NOT separate primaries. "
            "If the same site/histology is mentioned with date \"2018\" in one "
            "chunk and \"July 2018\" in another, that is ONE primary; keep the "
            "most precise documented date.\n"
            "6. Add a new entry only when the new chunk reveals a primary not "
            "yet listed.\n"
            "7. Refine an existing entry when the new chunk provides a more "
            "precise date, site, histology, or laterality than the current "
            "entry.\n"
            "8. Do not remove entries unless you now recognize one as a "
            "duplicate of another or as something other than a primary cancer.\n\n"
            "Respond with a JSON object: {\"diagnoses\": [{...}, ...]}. "
            "Each diagnosis has fields: "
            "\"cancer_type\" (string), "
            "\"primary_site\" (string), "
            "\"histology\" (string; use cancer_type if no separate "
            "histology wording is available), "
            "\"laterality\" (one of: left, right, bilateral, not_applicable, "
            "unknown), "
            "\"approximate_diagnosis_date\" (YYYY-MM-DD, YYYY-MM, or YYYY), "
            "\"evidence\" (short quote from the chart, max 300 chars)."
        )

    def _build_running_list_user_prompt(
        self, chunk: Any, running: list[dict]
    ) -> str:
        chunk_id = getattr(chunk, "chunk_id", "unknown")
        chunk_type = getattr(chunk, "chunk_type", "sequential")
        first_date = getattr(chunk, "first_date", "") or getattr(
            chunk, "document_date", "unknown"
        )
        last_date = getattr(chunk, "last_date", "") or getattr(
            chunk, "document_date", "unknown"
        )
        current_json = json.dumps(running, indent=2)
        return (
            f"Current running list of distinct primary cancers:\n"
            f"{current_json}\n\n"
            f"New clinical text chunk {chunk_id} "
            f"(type: {chunk_type}, dates: {first_date} to {last_date}):\n\n"
            f"{getattr(chunk, 'text', '')}"
        )

    # ------------------------------------------------------------------
    # Coding pass (one call per patient)
    # ------------------------------------------------------------------

    async def _assign_codes(self, diagnoses: list[dict]) -> list[dict]:
        """Assign ICD-O-3 topography and morphology codes to each diagnosis.

        Returns a list parallel to *diagnoses* with dicts of the form
        ``{"primary_site_code": "...", "histology_code": "..."}``.
        Empty strings indicate the LLM did not return a valid code; the
        downstream extractor will then re-extract those items.
        """
        if not diagnoses:
            return []

        system_prompt = self._build_coding_system_prompt()
        user_prompt = self._build_coding_user_prompt(diagnoses)

        llm_response = await self._llm.extract(system_prompt, user_prompt)

        if self._llm_log is not None:
            self._llm_log.log(
                call_type="diagnosis_coding",
                pass_number=0,
                chunk_id="diagnosis_coding",
                chunk_type="diagnosis_coding",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=llm_response.raw_content,
                reasoning=llm_response.reasoning,
                final_output=llm_response.final_content,
                parsed=llm_response.parsed,
            )

        empty = [
            {"primary_site_code": "", "histology_code": ""} for _ in diagnoses
        ]

        if llm_response.parsed.get("_error"):
            logger.error(
                "LLM error during diagnosis coding pass: %s",
                llm_response.parsed.get("_message", "unknown"),
            )
            return empty

        coded_raw = llm_response.parsed.get("coded")
        if not isinstance(coded_raw, list):
            logger.warning(
                "Coding pass returned unexpected 'coded' field type: %s",
                type(coded_raw),
            )
            return empty

        results: list[dict] = []
        for idx, dx in enumerate(diagnoses):
            entry = coded_raw[idx] if idx < len(coded_raw) else {}
            if not isinstance(entry, dict):
                entry = {}
            raw_site = str(entry.get("primary_site_code", "")).strip()
            raw_hist = str(entry.get("histology_code", "")).strip()
            site_code = self._normalize_topography(raw_site)
            hist_code = self._normalize_morphology(raw_hist)
            if raw_site and not site_code:
                logger.warning(
                    "Coding pass returned invalid topography %r for "
                    "diagnosis %d (%s).",
                    raw_site,
                    idx,
                    dx.get("cancer_type", ""),
                )
            if raw_hist and not hist_code:
                logger.warning(
                    "Coding pass returned invalid morphology %r for "
                    "diagnosis %d (%s).",
                    raw_hist,
                    idx,
                    dx.get("cancer_type", ""),
                )
            results.append(
                {"primary_site_code": site_code, "histology_code": hist_code}
            )
        return results

    def _build_coding_system_prompt(self) -> str:
        topo_table = self._format_code_table(
            "ICD-O-3 TOPOGRAPHY codes (primary site)",
            sorted(TOPOGRAPHY_EXACT.items()),
        )
        topo_prefix_table = self._format_code_table(
            "ICD-O-3 TOPOGRAPHY 3-character categories "
            "(use the corresponding Cxx.9 \"NOS\" code when the chart "
            "specifies only the broad category)",
            sorted(TOPOGRAPHY_PREFIX.items()),
        )
        morph_table = self._format_code_table(
            "ICD-O-3 MORPHOLOGY codes (histology)",
            sorted(MORPHOLOGY.items()),
        )
        return (
            "You are an expert cancer registrar. For each diagnosis in the "
            "provided list, assign:\n"
            "  * an ICD-O-3 TOPOGRAPHY (primary site) code in the form "
            "\"Cxx.x\" (e.g. \"C50.4\", \"C34.1\", \"C18.7\")\n"
            "  * an ICD-O-3 MORPHOLOGY (histology) code in 4-digit form "
            "(e.g. \"8500\", \"8140\")\n\n"
            "PREFER codes from the reference tables below when they fit. "
            "These tables cover the most common adult oncology codes but are "
            "NOT exhaustive -- for diagnoses not represented in the tables, "
            "assign a valid ICD-O-3 code in the same format.\n\n"
            "CRITICAL RULES:\n"
            "  - Choose topography by the ANATOMIC SITE of the primary, not "
            "    by the metastatic site.\n"
            "  - Choose the most SPECIFIC topography consistent with the "
            "    documented subsite (e.g. \"sigmoid colon\" -> C18.7, not "
            "    C18.9).  If the chart names only the organ with no subsite "
            "    (e.g. \"colon cancer\"), use the Cxx.9 \"NOS\" code.\n"
            "  - Choose morphology by HISTOLOGY TYPE, not by anatomic site. "
            "    Adenocarcinoma NOS is 8140 whether it is in lung, colon, or "
            "    kidney.  Do NOT use 8500 \"infiltrating duct carcinoma\" "
            "    for non-breast primaries.\n"
            "  - For hematologic malignancies, use the modern ICD-O-3 codes: "
            "    DLBCL is 9680, follicular lymphoma NOS is 9690, CLL/SLL is "
            "    9823, B lymphoblastic leukemia/lymphoma (ALL) is 9811, AML "
            "    NOS is 9861.\n\n"
            f"{topo_table}\n\n"
            f"{topo_prefix_table}\n\n"
            f"{morph_table}\n\n"
            "Respond with a JSON object: "
            "{\"coded\": [{\"primary_site_code\": \"...\", "
            "\"histology_code\": \"...\"}, ...]} "
            "where the array order MATCHES the input diagnosis list (one "
            "entry per input diagnosis, same order)."
        )

    @staticmethod
    def _format_code_table(title: str, entries: list[tuple[str, str]]) -> str:
        lines = [f"{title}:"]
        for code, desc in entries:
            lines.append(f"  {code}  {desc}")
        return "\n".join(lines)

    def _build_coding_user_prompt(self, diagnoses: list[dict]) -> str:
        rows: list[str] = []
        for idx, dx in enumerate(diagnoses):
            rows.append(
                f"[{idx}] cancer_type={dx.get('cancer_type', '')!s} | "
                f"primary_site={dx.get('primary_site', '')!s} | "
                f"histology={dx.get('histology', '')!s} | "
                f"laterality={dx.get('laterality', '')!s} | "
                f"date={dx.get('approximate_diagnosis_date', '')!s} | "
                f"evidence={dx.get('evidence', '')!s}"
            )
        return "Diagnoses to code:\n" + "\n".join(rows)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_topography(value: str) -> str:
        text = value.strip().upper().replace(" ", "")
        match = _TOPOGRAPHY_RE.match(text)
        if not match:
            return ""
        return f"C{match.group(1)}.{match.group(2)}"

    @staticmethod
    def _normalize_morphology(value: str) -> str:
        match = _MORPHOLOGY_RE.match(value.strip())
        return match.group(1) if match else ""

    @staticmethod
    def _normalize_laterality(laterality: str) -> str:
        value = str(laterality).lower().strip().replace(" ", "_")
        return value if value in _VALID_LATERALITY else "unknown"

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
