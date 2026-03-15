"""Base extraction pass framework for the NAACCR v26 pipeline.

Every extraction pass (0 through N) subclasses :class:`BaseExtractionPass`
and implements ``get_target_items`` and ``build_prompt``.  The ``run()``
method orchestrates chunk selection, batching, LLM calls, code resolution,
and result merging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable
import logging

from naaccr_pipeline.config import PipelineConfig
from naaccr_pipeline.dictionary.loader import NAACCRDictionary, NAACCRDataItem
from naaccr_pipeline.dictionary.code_resolver import CodeResolver
from naaccr_pipeline.llm.client import VLLMClient, LLMResponse
from naaccr_pipeline.llm.structured_output import SchemaBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk protocol -- satisfied by any chunker implementation that exposes
# the minimal interface below.  This avoids a hard import on a module that
# may not yet exist.
# ---------------------------------------------------------------------------

@runtime_checkable
class Chunk(Protocol):
    """Minimal interface expected from a document chunk."""

    @property
    def chunk_id(self) -> str: ...

    @property
    def chunk_type(self) -> str: ...

    @property
    def text(self) -> str: ...

    @property
    def document_date(self) -> str: ...


# ---------------------------------------------------------------------------
# Extraction result
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """The outcome of extracting a single NAACCR data item from a chunk."""

    item_number: int
    item_name: str
    extracted_value: str       # raw LLM output
    resolved_code: str         # after code resolution
    confidence: float          # 0.0-1.0
    evidence_text: str         # quoted text supporting the extraction
    source_chunk_id: str       # which chunk this came from
    source_chunk_type: str     # chunk type (pathology, etc.)
    pass_number: int           # which pass produced this


# ---------------------------------------------------------------------------
# Source-type priority (higher number = preferred when confidence ties)
# ---------------------------------------------------------------------------

SOURCE_PRIORITY: dict[str, int] = {
    "pathology": 6,
    "operative": 5,
    "radiology": 4,
    "consult": 3,
    "discharge_summary": 2,
    "progress_note": 1,
}

_HIGH_CONFIDENCE_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Base extraction pass
# ---------------------------------------------------------------------------

class BaseExtractionPass(ABC):
    """Abstract base for all extraction passes."""

    PASS_NUMBER: int = 0
    CHUNK_PRIORITY: list[str] = []  # ordered list of preferred chunk types

    def __init__(
        self,
        config: PipelineConfig,
        dictionary: NAACCRDictionary,
        code_resolver: CodeResolver,
        llm_client: VLLMClient,
        schema_builder: SchemaBuilder,
        llm_log: Any = None,
    ) -> None:
        self._config = config
        self._dict = dictionary
        self._resolver = code_resolver
        self._llm = llm_client
        self._schema_builder = schema_builder
        self._llm_log = llm_log  # Optional LLMLog instance for saving raw outputs

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_target_items(
        self, prior_results: dict[int, ExtractionResult]
    ) -> list[NAACCRDataItem]:
        """Return NAACCR data items this pass extracts.

        *prior_results* (keyed by item_number) may influence which items
        are still needed.
        """

    @abstractmethod
    def build_prompt(
        self,
        chunk: Any,
        target_items: list[NAACCRDataItem],
        prior_results: dict[int, ExtractionResult],
    ) -> tuple[str, str]:
        """Return ``(system_prompt, user_prompt)`` for extracting from *chunk*."""

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def run(
        self,
        chunks: list[Any],
        prior_results: dict[int, ExtractionResult],
    ) -> list[ExtractionResult]:
        """Execute the extraction pass.

        Algorithm
        ---------
        1. Get target items (may depend on prior results).
        2. Filter out items already extracted with high confidence
           (>= 0.9) from *prior_results*.
        3. Prioritise chunks for this pass type.
        4. Split target items into batches based on
           ``model_profile.items_per_call``.
        5. For each prioritised chunk (respecting the budget):
           a. For each item batch call ``_extract_from_chunk``.
        6. Merge with existing results (higher confidence wins).
        7. Return **all** ExtractionResults (merged).
        """
        # 1. Determine target items
        all_targets = self.get_target_items(prior_results)
        if not all_targets:
            logger.info("Pass %d: no target items -- skipping.", self.PASS_NUMBER)
            return list(prior_results.values())

        # 2. Filter items already resolved at high confidence
        remaining_items = [
            item
            for item in all_targets
            if item.item_number not in prior_results
            or prior_results[item.item_number].confidence < _HIGH_CONFIDENCE_THRESHOLD
        ]
        if not remaining_items:
            logger.info(
                "Pass %d: all %d target items already at >= %.1f confidence.",
                self.PASS_NUMBER,
                len(all_targets),
                _HIGH_CONFIDENCE_THRESHOLD,
            )
            return list(prior_results.values())

        logger.info(
            "Pass %d: %d target items (%d remaining after filtering high-confidence).",
            self.PASS_NUMBER,
            len(all_targets),
            len(remaining_items),
        )

        # 3. Prioritise chunks
        ordered_chunks = self._prioritize_chunks(chunks)
        if not ordered_chunks:
            logger.warning("Pass %d: no chunks available.", self.PASS_NUMBER)
            return list(prior_results.values())

        # 4. Batch items
        items_per_call = self._llm.model_profile.items_per_call
        batches = self._split_items_into_batches(remaining_items, items_per_call)

        # 5. Process chunks
        merged = dict(prior_results)
        # Budget: process at most (number-of-chunks) chunks.  In practice the
        # caller may further limit this but by default we try all.
        for chunk in ordered_chunks:
            for batch in batches:
                try:
                    new_results = await self._extract_from_chunk(
                        chunk, batch, merged
                    )
                    merged = self._merge_results(merged, new_results)
                except Exception:
                    logger.exception(
                        "Pass %d: error extracting from chunk %s",
                        self.PASS_NUMBER,
                        getattr(chunk, "chunk_id", "?"),
                    )

        return list(merged.values())

    # ------------------------------------------------------------------
    # Chunk prioritisation
    # ------------------------------------------------------------------

    def _prioritize_chunks(self, chunks: list[Any]) -> list[Any]:
        """Reorder *chunks* based on ``CHUNK_PRIORITY``.

        Chunks whose ``chunk_type`` matches earlier entries in the priority
        list sort first.  Within the same type, newer documents come first
        for follow-up-style passes and oldest-first otherwise.
        """
        priority_map: dict[str, int] = {
            ctype: idx for idx, ctype in enumerate(self.CHUNK_PRIORITY)
        }
        default_priority = len(self.CHUNK_PRIORITY)

        def _sort_key(chunk: Any) -> tuple[int, str]:
            ctype = getattr(chunk, "chunk_type", "other")
            prio = priority_map.get(ctype, default_priority)
            doc_date = getattr(chunk, "document_date", "") or ""
            # For follow-up passes (pass > 1) prefer newest; otherwise oldest.
            if self.PASS_NUMBER > 1:
                # Negate date string for descending sort (newest first).
                # We invert by complementing each character for a stable sort.
                inverted = "".join(
                    chr(0xFFFF - ord(c)) if c.isdigit() else c for c in doc_date
                )
                return (prio, inverted)
            return (prio, doc_date)

        return sorted(chunks, key=_sort_key)

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    @staticmethod
    def _split_items_into_batches(
        items: list[NAACCRDataItem], items_per_call: int
    ) -> list[list[NAACCRDataItem]]:
        """Partition *items* into sub-lists of at most *items_per_call*."""
        if items_per_call <= 0:
            return [items]
        return [
            items[i : i + items_per_call]
            for i in range(0, len(items), items_per_call)
        ]

    # ------------------------------------------------------------------
    # Single-chunk extraction
    # ------------------------------------------------------------------

    async def _extract_from_chunk(
        self,
        chunk: Any,
        items: list[NAACCRDataItem],
        prior_results: dict[int, ExtractionResult],
    ) -> list[ExtractionResult]:
        """Extract *items* from a single *chunk*.

        Builds the prompt, constructs a guided-JSON schema, calls the LLM,
        and parses the response into :class:`ExtractionResult` objects.
        """
        system_prompt, user_prompt = self.build_prompt(chunk, items, prior_results)

        # Build a code-resolver adapter that exposes get_codes(item_number) -> list[str]
        # as expected by SchemaBuilder._value_schema.
        resolver_adapter = _CodeResolverAdapter(self._resolver)
        json_schema = self._schema_builder.build_extraction_schema(
            items, resolver_adapter
        )

        llm_response: LLMResponse = await self._llm.extract(
            system_prompt, user_prompt, json_schema
        )

        # Log the full LLM interaction if a logger is attached.
        if self._llm_log is not None:
            self._llm_log.log(
                pass_number=self.PASS_NUMBER,
                chunk_id=getattr(chunk, "chunk_id", "?"),
                chunk_type=getattr(chunk, "chunk_type", "?"),
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

        return self._parse_llm_response(llm_response.parsed, items, chunk)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_llm_response(
        self,
        response: dict,
        items: list[NAACCRDataItem],
        chunk: Any,
    ) -> list[ExtractionResult]:
        """Convert the LLM JSON response into :class:`ExtractionResult` objects.

        The response keys are ``xml_id`` values (or ``item_<number>``).  Each
        value is ``{"value": ..., "confidence": ..., "evidence": ...}``.
        """
        results: list[ExtractionResult] = []
        chunk_id = getattr(chunk, "chunk_id", "unknown")
        chunk_type = getattr(chunk, "chunk_type", "other")

        # Build a lookup: xml_id -> NAACCRDataItem
        xml_id_map: dict[str, NAACCRDataItem] = {}
        for item in items:
            key = item.xml_id if item.xml_id else f"item_{item.item_number}"
            xml_id_map[key] = item

        for field_name, payload in response.items():
            if field_name.startswith("_"):
                # Skip internal keys like _error, _message.
                continue

            item = xml_id_map.get(field_name)
            if item is None:
                logger.debug(
                    "LLM returned unknown field '%s'; skipping.", field_name
                )
                continue

            if not isinstance(payload, dict):
                logger.debug(
                    "Unexpected payload type for '%s': %s", field_name, type(payload)
                )
                continue

            raw_value = str(payload.get("value", "")).strip()
            llm_confidence = float(payload.get("confidence", 0.0))
            evidence = str(payload.get("evidence", "")).strip()

            # Skip blank / empty extractions
            if not raw_value:
                continue

            # Resolve code
            resolved_code, resolution_confidence = self._resolver.resolve(
                item.item_number, raw_value
            )

            # Final confidence = min(llm_confidence, resolution_confidence)
            # but if resolution_confidence is 0.0 (no match at all), we still
            # keep the llm_confidence scaled down.
            if resolution_confidence > 0.0:
                final_confidence = min(llm_confidence, resolution_confidence)
            else:
                # No code match -- reduce confidence significantly
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
                    pass_number=self.PASS_NUMBER,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Result merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_results(
        existing: dict[int, ExtractionResult],
        new_results: list[ExtractionResult],
    ) -> dict[int, ExtractionResult]:
        """Merge *new_results* into *existing*.  Higher confidence wins.

        On a tie the source chunk type with higher clinical priority wins
        (pathology > operative > radiology > consult > discharge > progress).
        """
        merged = dict(existing)

        for result in new_results:
            item_num = result.item_number
            current = merged.get(item_num)

            if current is None:
                merged[item_num] = result
                continue

            # Compare confidence
            if result.confidence > current.confidence:
                merged[item_num] = result
            elif result.confidence == current.confidence:
                # Tie-break on source priority
                new_prio = SOURCE_PRIORITY.get(result.source_chunk_type, 0)
                cur_prio = SOURCE_PRIORITY.get(current.source_chunk_type, 0)
                if new_prio > cur_prio:
                    merged[item_num] = result

        return merged


# ---------------------------------------------------------------------------
# Adapter so CodeResolver (which has build_constrained_vocab) satisfies
# SchemaBuilder's expectation of get_codes(item_number) -> list[str].
# ---------------------------------------------------------------------------

class _CodeResolverAdapter:
    """Wraps :class:`CodeResolver` to expose ``get_codes`` for SchemaBuilder."""

    def __init__(self, resolver: CodeResolver) -> None:
        self._resolver = resolver

    def get_codes(self, item_number: int) -> list[str]:
        return self._resolver.build_constrained_vocab(item_number)
