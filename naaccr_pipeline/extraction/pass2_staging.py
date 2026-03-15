"""Pass 2: Staging and Prognostic Factors (site-specific).

This pass uses results from Pass 1 (primary site and histology) to determine
which cancer schema applies, then extracts the core staging items plus
site-specific data items (SSDIs) for that schema.

Chunk priority favours pathology reports (which contain definitive staging),
then radiology (imaging staging), operative notes, consults, and so on.
"""

from __future__ import annotations

from typing import Any
import logging

from naaccr_pipeline.config import PipelineConfig
from naaccr_pipeline.dictionary.loader import NAACCRDictionary, NAACCRDataItem
from naaccr_pipeline.dictionary.code_resolver import CodeResolver
from naaccr_pipeline.dictionary.schema_registry import SchemaRegistry
from naaccr_pipeline.extraction.base import BaseExtractionPass, ExtractionResult
from naaccr_pipeline.extraction.prompts.staging import (
    PASS2_SYSTEM_PROMPT,
    PASS2_USER_TEMPLATE,
)
from naaccr_pipeline.llm.client import VLLMClient
from naaccr_pipeline.llm.structured_output import SchemaBuilder

logger = logging.getLogger(__name__)

# NAACCR item numbers for primary site, histology, and schema discriminator
# (extracted in Pass 1).
_ITEM_PRIMARY_SITE = 400
_ITEM_HISTOLOGY = 522
_ITEM_SCHEMA_DISCRIMINATOR_1 = 3926


class Pass2Staging(BaseExtractionPass):
    """Extract staging and prognostic factor data items.

    The set of target items is determined dynamically based on the primary
    site and histology obtained from Pass 1.  Core staging items (TNM,
    Summary Stage, EOD, etc.) are always included; site-specific items
    (biomarkers, Gleason, receptor status, etc.) are added when the
    schema is recognized.
    """

    PASS_NUMBER: int = 2
    CHUNK_PRIORITY: list[str] = [
        "pathology",
        "radiology",
        "operative",
        "consult",
        "discharge_summary",
        "progress_note",
        "lab",
        "mixed",
    ]

    def __init__(
        self,
        config: PipelineConfig,
        dictionary: NAACCRDictionary,
        code_resolver: CodeResolver,
        llm_client: VLLMClient,
        schema_builder: SchemaBuilder,
        schema_registry: SchemaRegistry,
        llm_log=None,
    ) -> None:
        super().__init__(config, dictionary, code_resolver, llm_client, schema_builder, llm_log=llm_log)
        self._schema_reg = schema_registry
        # Cache the resolved schema so build_prompt can use it without
        # re-computing from prior_results each time.
        self._resolved_schema: str = "generic"
        self._resolved_primary_site: str = ""
        self._resolved_histology: str = ""

    # ------------------------------------------------------------------
    # Abstract interface implementations
    # ------------------------------------------------------------------

    def get_target_items(
        self, prior_results: dict[int, ExtractionResult]
    ) -> list[NAACCRDataItem]:
        """Dynamically determine items based on Pass 1 results.

        Steps
        -----
        1. Extract primary site from ``prior_results[400]``.
        2. Extract histology from ``prior_results[522]``.
        3. Optionally extract schema discriminator from ``prior_results[3926]``.
        4. Determine schema via :class:`SchemaRegistry`.
        5. Get core staging items + schema-specific SSDIs.
        6. Filter to items that exist in the dictionary and are active
           (not retired).
        7. Cache the schema for use by :meth:`build_prompt`.
        """
        # 1. Get primary site
        primary_site = self._extract_prior_value(prior_results, _ITEM_PRIMARY_SITE)
        self._resolved_primary_site = primary_site

        # 2. Get histology
        histology = self._extract_prior_value(prior_results, _ITEM_HISTOLOGY)
        self._resolved_histology = histology

        # 3. Get schema discriminator (optional)
        schema_disc = self._extract_prior_value(
            prior_results, _ITEM_SCHEMA_DISCRIMINATOR_1
        )

        # 4. Determine schema
        if primary_site:
            self._resolved_schema = self._schema_reg.get_schema_for_site_histology(
                primary_site, histology, schema_disc or None
            )
        else:
            self._resolved_schema = "generic"
            logger.warning(
                "Pass 2: No primary site in prior results; using generic schema."
            )

        logger.info(
            "Pass 2: site=%s, histology=%s -> schema=%s",
            primary_site or "(unknown)",
            histology or "(unknown)",
            self._resolved_schema,
        )

        # 5. Get all staging item numbers for this schema
        item_numbers = self._schema_reg.get_all_staging_items(self._resolved_schema)

        # 6. Filter to items that exist in the dictionary and are active
        target_items: list[NAACCRDataItem] = []
        missing_count = 0
        for item_num in item_numbers:
            item = self._dict.get_item(item_num)
            if item is None:
                missing_count += 1
                continue
            # Skip retired items
            if item.year_retired:
                continue
            target_items.append(item)

        if missing_count > 0:
            logger.debug(
                "Pass 2: %d item numbers not found in dictionary (may be "
                "derived-only or from a different NAACCR version).",
                missing_count,
            )

        logger.info(
            "Pass 2: %d target items for schema '%s' (%d core + %d site-specific).",
            len(target_items),
            self._resolved_schema,
            len(self._schema_reg.CORE_STAGING_ITEMS),
            len(self._schema_reg.get_required_ssdis(self._resolved_schema)),
        )

        return target_items

    def build_prompt(
        self,
        chunk: Any,
        target_items: list[NAACCRDataItem],
        prior_results: dict[int, ExtractionResult],
    ) -> tuple[str, str]:
        """Build staging prompt, parameterized by primary site and histology.

        The system prompt includes site-specific extraction context
        (e.g. breast receptor details, prostate Gleason instructions)
        so the LLM knows exactly what to look for.

        The user prompt includes the chunk text and a code reference
        listing valid codes for each target item.

        Returns
        -------
        tuple[str, str]
            ``(system_prompt, user_prompt)``
        """
        # -- Build site-specific context for the system prompt --
        site_context = self._schema_reg.get_site_context(self._resolved_schema)
        site_desc = self._schema_reg.get_primary_site_description(
            self._resolved_schema
        )

        system_prompt = PASS2_SYSTEM_PROMPT.format(
            primary_site_desc=site_desc,
            primary_site=self._resolved_primary_site or "unknown",
            histology=self._resolved_histology or "unknown",
            site_context=site_context,
        )

        # -- Build code reference for each target item --
        code_ref_parts: list[str] = []
        for item in target_items:
            codes_prompt = self._resolver.get_valid_codes_prompt(item.item_number)
            if codes_prompt:
                code_ref_parts.append(
                    f"- {item.name} (#{item.item_number}): {codes_prompt}"
                )
            else:
                # No enumerated codes -- describe the expected format
                fmt_hint = self._format_hint(item)
                code_ref_parts.append(
                    f"- {item.name} (#{item.item_number}): {fmt_hint}"
                )

        code_reference = "\n".join(code_ref_parts) if code_ref_parts else "(none)"

        # -- Chunk metadata --
        chunk_type = getattr(chunk, "chunk_type", "unknown")
        chunk_date = getattr(chunk, "document_date", "unknown")
        chunk_text = getattr(chunk, "text", "")

        user_prompt = PASS2_USER_TEMPLATE.format(
            note_type=chunk_type.replace("_", " ").title(),
            note_date=chunk_date,
            chunk_text=chunk_text,
            code_reference=code_reference,
        )

        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prior_value(
        prior_results: dict[int, ExtractionResult], item_number: int
    ) -> str:
        """Get the resolved code from prior results for a given item number.

        Returns an empty string if the item was not extracted or has
        zero confidence.
        """
        result = prior_results.get(item_number)
        if result is None:
            return ""
        if result.confidence <= 0.0:
            return ""
        # Prefer resolved_code; fall back to extracted_value
        value = result.resolved_code or result.extracted_value
        return value.strip()

    @staticmethod
    def _format_hint(item: NAACCRDataItem) -> str:
        """Generate a format hint for items without enumerated codes.

        Uses the item's data_type and length to describe expected input.
        """
        data_type = (item.data_type or "").strip().lower()
        length = item.length

        if data_type == "date":
            return "Date format YYYYMMDD (use 99 for unknown components)"

        if data_type == "digits":
            if length <= 2:
                return f"{length}-digit numeric code"
            elif length <= 4:
                return f"Numeric value, up to {length} digits, zero-padded"
            else:
                return f"Numeric value, {length} digits"

        if "size" in item.name.lower():
            return "Size in millimeters (000-999, 990+ for special codes)"

        if "lab value" in item.name.lower() or "lab" in item.name.lower():
            return "Numeric lab value as stated in the report"

        if length <= 3:
            return f"Text value, max {length} characters"

        return f"Free text, max {length} characters"
