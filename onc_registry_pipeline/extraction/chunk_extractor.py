"""Unified per-chunk NAACCR extraction.

Extracts ALL NAACCR items from a single chronological chunk of patient text.
Handles the internal dependency chain: demographics -> schema resolution ->
staging -> treatment -> followup -> narratives.
"""

from __future__ import annotations

from typing import Any
import logging

from onc_registry_pipeline.config import PipelineConfig
from onc_registry_pipeline.dictionary.loader import NAACCRDictionary, NAACCRDataItem
from onc_registry_pipeline.dictionary.code_resolver import CodeResolver
from onc_registry_pipeline.dictionary.schema_registry import SchemaRegistry
from onc_registry_pipeline.extraction.base import (
    ExtractionResult,
    HIGH_CONFIDENCE_THRESHOLD,
    merge_results,
    split_items_into_batches,
)
from onc_registry_pipeline.extraction.prompts.chunk_extraction import (
    DEMOGRAPHICS_SYSTEM_PROMPT,
    STAGING_SYSTEM_PROMPT,
    SURGERY_SYSTEM_PROMPT,
    RADIATION_SYSTEM_PROMPT,
    SYSTEMIC_SYSTEM_PROMPT,
    FOLLOWUP_SYSTEM_PROMPT,
    NARRATIVE_SYSTEM_PROMPT,
    CHUNK_USER_TEMPLATE,
    NARRATIVE_USER_TEMPLATE,
    build_prior_state_block,
    build_prior_narratives_block,
)
from onc_registry_pipeline.llm.client import VLLMClient
from onc_registry_pipeline.llm.structured_output import SchemaBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Item number lists (from the original pass files)
# ---------------------------------------------------------------------------

DEMOGRAPHICS_ITEMS = [
    380, 390, 400, 410, 440, 441, 442, 449, 450, 470, 490, 500, 522, 523,
    150, 160, 161, 190, 220, 230, 240, 252, 254,
]

SURGERY_ITEMS = [
    1200, 1290, 1291, 1292, 1294, 1296, 1310, 1320, 1330, 1340, 1350,
    1640, 3170, 3180, 3190,
]

RADIATION_ITEMS = [
    1210, 1360, 1370, 1380, 1430,
    1501, 1502, 1503, 1504, 1505, 1506, 1507,
    1511, 1512, 1513, 1514, 1515, 1516, 1517,
    1521, 1522, 1523, 1524, 1525, 1526, 1527,
    1531, 1532, 1533, 1550, 1570, 3220,
]

SYSTEMIC_ITEMS = [
    1220, 1230, 1240, 1250, 1285, 1390, 1400, 1410, 1420,
    1632, 1633, 1634, 1639, 3230, 3250, 3270,
]

FOLLOWUP_ITEMS = [1750, 1760, 1770, 1772, 1790, 1910]

TEXT_ITEMS = [
    2520, 2530, 2540, 2550, 2560, 2570, 2580, 2590,
    2600, 2610, 2620, 2630, 2640, 2650, 2660, 2670, 2680,
]

# TNM T/N/M and stage-group items (clinical and pathologic, TNM and AJCC TNM).
# Valid values for these are single-character mains (0-4, IS, X, A) with optional
# prefixes/subdivisions, plus the sentinels 88/99 -- never a multi-digit integer.
# The model occasionally echoes a NAACCR item number (e.g. 1001, 970) as the
# value; those resolve to confidence 0.0 and would otherwise pollute output.
_STAGING_CODE_ITEMS = frozenset({
    880, 890, 900, 910, 940, 950, 960, 970,
    1001, 1002, 1003, 1004, 1011, 1012, 1013, 1014,
})


def _is_item_number_leak(value: str) -> bool:
    """True if *value* looks like a NAACCR item number leaked into a staging field.

    Valid T/N/M/stage-group components are 0-4 (single digit) or letter/prefix
    forms; the only valid multi-digit codes are the sentinels 88 and 99. Any
    other pure integer >= 5 (e.g. 970, 1001-1014) is an item-number leak.
    """
    s = value.strip()
    return s.isdigit() and s not in ("88", "99") and int(s) >= 5


# ---------------------------------------------------------------------------
# ChunkExtractor
# ---------------------------------------------------------------------------

class ChunkExtractor:
    """Extracts all NAACCR items from a single sequential chunk.

    Usage::

        extractor = ChunkExtractor(config, dictionary, code_resolver,
                                    llm_client, schema_builder, schema_registry)
        updated = await extractor.extract(chunk, prior_extraction)
    """

    def __init__(
        self,
        config: PipelineConfig,
        dictionary: NAACCRDictionary,
        code_resolver: CodeResolver,
        llm_client: VLLMClient,
        schema_builder: SchemaBuilder,
        schema_registry: SchemaRegistry,
        manual_context_provider: Any = None,
        tumor: Any = None,
        tumor_context: str = "",
        llm_log: Any = None,
    ) -> None:
        self._config = config
        self._dict = dictionary
        self._resolver = code_resolver
        self._llm = llm_client
        self._schema_builder = schema_builder
        self._schema_reg = schema_registry
        self._manual_context_provider = manual_context_provider
        self._tumor = tumor
        self._tumor_context = tumor_context
        self._registry_context = ""
        self._llm_log = llm_log

    async def extract(
        self,
        chunk: Any,
        prior_extraction: dict[int, ExtractionResult],
    ) -> dict[int, ExtractionResult]:
        """Extract all items from this chunk, returning updated extraction.

        Phases (sequential due to data dependencies):
        1. Demographics + Cancer ID
        2. Resolve schema from site/histology
        3. Staging (schema-dependent)
        4-6. Treatment (surgery, radiation, systemic)
        7. Follow-up coded items
        8. Narrative text summaries
        """
        merged = dict(prior_extraction)
        self._registry_context = self._build_registry_context()

        # Phase 1: Demographics + Cancer ID
        demo_results = await self._extract_group(
            chunk, merged, DEMOGRAPHICS_ITEMS,
            DEMOGRAPHICS_SYSTEM_PROMPT,
        )
        merged = merge_results(merged, demo_results)

        # Phase 2: Resolve schema
        primary_site = self._get_value(merged, 400)
        histology = self._get_value(merged, 522)
        schema = self._schema_reg.get_schema_for_site_histology(
            primary_site or "", histology or "", None
        )
        staging_item_numbers = self._schema_reg.get_all_staging_items(schema)
        site_desc = self._schema_reg.get_primary_site_description(schema)
        site_context = self._schema_reg.get_site_context(schema)
        self._registry_context = self._build_registry_context(
            primary_site=primary_site or "",
            histology=histology or "",
            schema=schema,
            site_desc=site_desc,
            site_context=site_context,
        )

        # Phase 3: Staging
        staging_prompt = STAGING_SYSTEM_PROMPT.format(
            primary_site_desc=site_desc,
            primary_site=primary_site or "unknown",
            histology=histology or "unknown",
            site_context=site_context,
            json_format_instructions="{json_format_instructions}",
        )
        staging_results = await self._extract_group(
            chunk, merged, staging_item_numbers, staging_prompt,
        )
        merged = merge_results(merged, staging_results)

        # Phase 4: Surgery
        surgery_prompt = SURGERY_SYSTEM_PROMPT.format(
            primary_site=primary_site or "unknown",
            json_format_instructions="{json_format_instructions}",
        )
        surgery_results = await self._extract_group(
            chunk, merged, SURGERY_ITEMS, surgery_prompt,
        )
        merged = merge_results(merged, surgery_results)

        # Phase 5: Radiation
        rad_results = await self._extract_group(
            chunk, merged, RADIATION_ITEMS, RADIATION_SYSTEM_PROMPT,
        )
        merged = merge_results(merged, rad_results)

        # Phase 6: Systemic
        sys_results = await self._extract_group(
            chunk, merged, SYSTEMIC_ITEMS, SYSTEMIC_SYSTEM_PROMPT,
        )
        merged = merge_results(merged, sys_results)

        # Phase 7: Follow-up coded items
        followup_results = await self._extract_group(
            chunk, merged, FOLLOWUP_ITEMS, FOLLOWUP_SYSTEM_PROMPT,
        )
        merged = merge_results(merged, followup_results)

        # Phase 8: Narrative text summaries
        narrative_results = await self._update_narratives(chunk, merged)
        merged = merge_results(merged, narrative_results)

        return merged

    # ------------------------------------------------------------------
    # Core extraction for a group of items
    # ------------------------------------------------------------------

    async def _extract_group(
        self,
        chunk: Any,
        prior: dict[int, ExtractionResult],
        item_numbers: list[int],
        system_prompt_template: str,
    ) -> list[ExtractionResult]:
        """Extract a group of items from the chunk.

        Resolves item numbers to dictionary items, filters high-confidence,
        batches by items_per_call, calls LLM, parses, resolves codes.
        """
        # Resolve to dictionary items, filter already-confident
        items: list[NAACCRDataItem] = []
        for num in item_numbers:
            item = self._dict.get_item(num)
            if item is None:
                continue
            if item.year_retired:
                continue
            existing = prior.get(num)
            if existing is not None and existing.confidence >= HIGH_CONFIDENCE_THRESHOLD:
                continue
            items.append(item)

        if not items:
            return []

        # Batch by items_per_call
        batches = split_items_into_batches(items, self._config.items_per_call)

        all_results: list[ExtractionResult] = []
        for batch in batches:
            try:
                results = await self._extract_batch(
                    chunk, prior, batch, system_prompt_template,
                )
                all_results.extend(results)
            except Exception:
                logger.exception(
                    "Error extracting batch from chunk %s",
                    getattr(chunk, "chunk_id", "?"),
                )

        return all_results

    async def _extract_batch(
        self,
        chunk: Any,
        prior: dict[int, ExtractionResult],
        items: list[NAACCRDataItem],
        system_prompt_template: str,
    ) -> list[ExtractionResult]:
        """Extract a batch of items via a single LLM call."""
        # Build JSON format instructions for this batch
        json_instructions = self._schema_builder.build_json_format_instructions(
            items, self._resolver,
        )

        # Build system prompt with format instructions
        system_prompt = system_prompt_template.format(
            json_format_instructions=json_instructions,
        )
        if self._registry_context:
            system_prompt = f"{system_prompt}\n\n{self._registry_context}"

        # Build prior state block
        prior_block = build_prior_state_block(
            prior,
            item_numbers=[i.item_number for i in items],
        )

        # Build user prompt
        first_date = getattr(chunk, "first_date", "") or getattr(chunk, "document_date", "unknown")
        last_date = getattr(chunk, "last_date", "") or getattr(chunk, "document_date", "unknown")

        tumor_context = ""
        if self._tumor_context:
            tumor_context = f"TUMOR FOCUS: {self._tumor_context}"

        user_prompt = CHUNK_USER_TEMPLATE.format(
            first_date=first_date,
            last_date=last_date,
            chunk_text=chunk.text,
            tumor_context=tumor_context,
            prior_state_block=prior_block,
            json_field_descriptions=json_instructions,
        )

        # Call LLM
        llm_response = await self._llm.extract(system_prompt, user_prompt)

        # Log if available
        if self._llm_log is not None:
            self._llm_log.log(
                call_type="coded_item_extraction",
                pass_number=getattr(chunk, "chunk_index", 0),
                chunk_id=getattr(chunk, "chunk_id", "?"),
                chunk_type=getattr(chunk, "chunk_type", "sequential"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=llm_response.raw_content,
                reasoning=llm_response.reasoning,
                final_output=llm_response.final_content,
                parsed=llm_response.parsed,
            )

        if llm_response.parsed.get("_error"):
            logger.warning(
                "LLM error for chunk %s: %s",
                getattr(chunk, "chunk_id", "?"),
                llm_response.parsed.get("_message", "unknown"),
            )
            return []

        return self._parse_response(llm_response.parsed, items, chunk)

    # ------------------------------------------------------------------
    # Narrative text summaries (running update)
    # ------------------------------------------------------------------

    async def _update_narratives(
        self,
        chunk: Any,
        prior: dict[int, ExtractionResult],
    ) -> list[ExtractionResult]:
        """Generate/update narrative text summaries for this chunk."""
        # Resolve text items, skip high-confidence
        text_items: list[NAACCRDataItem] = []
        for num in TEXT_ITEMS:
            item = self._dict.get_item(num)
            if item is None:
                continue
            existing = prior.get(num)
            if existing is not None and existing.confidence >= HIGH_CONFIDENCE_THRESHOLD:
                continue
            text_items.append(item)

        if not text_items:
            return []

        # Build narrative prompt
        json_instructions = self._schema_builder.build_json_format_instructions(
            text_items, self._resolver,
        )

        system_prompt = NARRATIVE_SYSTEM_PROMPT.format(
            json_format_instructions=json_instructions,
        )
        if self._registry_context:
            system_prompt = f"{system_prompt}\n\n{self._registry_context}"

        prior_narratives = build_prior_narratives_block(prior, TEXT_ITEMS)

        first_date = getattr(chunk, "first_date", "") or getattr(chunk, "document_date", "unknown")
        last_date = getattr(chunk, "last_date", "") or getattr(chunk, "document_date", "unknown")

        user_prompt = NARRATIVE_USER_TEMPLATE.format(
            first_date=first_date,
            last_date=last_date,
            chunk_text=chunk.text,
            prior_narratives_block=prior_narratives,
            json_field_descriptions=json_instructions,
        )

        llm_response = await self._llm.extract(system_prompt, user_prompt)

        if self._llm_log is not None:
            self._llm_log.log(
                call_type="narrative_summary",
                pass_number=getattr(chunk, "chunk_index", 0),
                chunk_id=getattr(chunk, "chunk_id", "?"),
                chunk_type=getattr(chunk, "chunk_type", "sequential"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=llm_response.raw_content,
                reasoning=llm_response.reasoning,
                final_output=llm_response.final_content,
                parsed=llm_response.parsed,
            )

        if llm_response.parsed.get("_error"):
            logger.warning(
                "LLM error for narratives: %s",
                llm_response.parsed.get("_message", "unknown"),
            )
            return []

        return self._parse_narrative_response(llm_response.parsed, text_items, chunk)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: dict,
        items: list[NAACCRDataItem],
        chunk: Any,
    ) -> list[ExtractionResult]:
        """Parse LLM JSON response into ExtractionResult objects."""
        results: list[ExtractionResult] = []
        chunk_id = getattr(chunk, "chunk_id", "unknown")
        chunk_type = getattr(chunk, "chunk_type", "sequential")
        chunk_index = getattr(chunk, "chunk_index", 0)

        # Build lookup: xml_id -> item
        xml_id_map: dict[str, NAACCRDataItem] = {}
        for item in items:
            key = item.xml_id if item.xml_id else f"item_{item.item_number}"
            xml_id_map[key] = item

        for field_name, payload in response.items():
            if field_name.startswith("_"):
                continue

            item = xml_id_map.get(field_name)
            if item is None:
                logger.debug("LLM returned unknown field '%s'; skipping.", field_name)
                continue

            if not isinstance(payload, dict):
                continue

            raw_value = str(payload.get("value", "")).strip()
            llm_confidence = float(payload.get("confidence", 0.0))
            evidence = str(payload.get("evidence", "")).strip()

            if not raw_value:
                continue

            # Resolve code
            resolved_code, resolution_confidence = self._resolver.resolve(
                item.item_number, raw_value
            )

            # Drop NAACCR item numbers that leaked into a staging field rather
            # than recording them as bogus T/N/M/stage-group values.
            if item.item_number in _STAGING_CODE_ITEMS and _is_item_number_leak(
                resolved_code
            ):
                logger.debug(
                    "Dropping item-number leak %r for staging item %s (%s).",
                    resolved_code, item.item_number, item.name,
                )
                continue

            if resolution_confidence > 0.0:
                final_confidence = min(llm_confidence, resolution_confidence)
            else:
                final_confidence = llm_confidence * 0.5

            results.append(
                ExtractionResult(
                    item_number=item.item_number,
                    item_name=item.name,
                    extracted_value=raw_value,
                    resolved_code=resolved_code,
                    confidence=round(final_confidence, 4),
                    evidence_text=evidence[:300],
                    source_chunk_id=chunk_id,
                    source_chunk_type=chunk_type,
                    pass_number=chunk_index,
                )
            )

        return results

    def _parse_narrative_response(
        self,
        response: dict,
        text_items: list[NAACCRDataItem],
        chunk: Any,
    ) -> list[ExtractionResult]:
        """Parse narrative text summary response (no code resolution)."""
        results: list[ExtractionResult] = []
        chunk_index = getattr(chunk, "chunk_index", 0)

        xml_id_map: dict[str, NAACCRDataItem] = {}
        for item in text_items:
            key = item.xml_id if item.xml_id else f"item_{item.item_number}"
            xml_id_map[key] = item

        for field_name, payload in response.items():
            if field_name.startswith("_"):
                continue

            item = xml_id_map.get(field_name)
            if item is None:
                continue
            if not isinstance(payload, dict):
                continue

            raw_value = str(payload.get("value", "")).strip()
            confidence = float(payload.get("confidence", 0.0))
            evidence = str(payload.get("evidence", "")).strip()

            if not raw_value:
                continue

            if item.length > 0:
                raw_value = raw_value[:item.length]

            results.append(
                ExtractionResult(
                    item_number=item.item_number,
                    item_name=item.name,
                    extracted_value=raw_value,
                    resolved_code=raw_value,  # no code resolution for text
                    confidence=round(confidence, 4),
                    evidence_text=evidence[:300],
                    source_chunk_id="aggregated",
                    source_chunk_type="aggregated",
                    pass_number=chunk_index,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_value(
        state: dict[int, ExtractionResult], item_number: int
    ) -> str | None:
        """Get the resolved code from extraction state."""
        result = state.get(item_number)
        if result is None or result.confidence <= 0.0:
            return None
        return (result.resolved_code or result.extracted_value).strip() or None

    def _build_registry_context(
        self,
        *,
        primary_site: str = "",
        histology: str = "",
        schema: str = "",
        site_desc: str = "",
        site_context: str = "",
    ) -> str:
        """Build bounded NAACCR/SEER context for this tumor focus."""
        if self._manual_context_provider is None:
            return ""

        cancer_type = getattr(self._tumor, "cancer_type", "")
        tumor_histology = getattr(self._tumor, "histology", "")
        try:
            return self._manual_context_provider.build_context(
                tumor_context=self._tumor_context,
                cancer_type=cancer_type,
                primary_site=primary_site,
                histology=histology or tumor_histology,
                schema=schema,
                site_desc=site_desc,
                site_context=site_context,
            )
        except Exception:
            logger.exception("Could not build SEER/NAACCR manual context")
            return ""
