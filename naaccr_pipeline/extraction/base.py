"""Core data types and merge logic for the NAACCR v26 pipeline.

Provides :class:`ExtractionResult` (the unit of extracted data) and
:func:`merge_results` (higher-confidence-wins merging used across chunks).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Protocol, runtime_checkable
import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk protocol -- satisfied by any chunker implementation
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
    source_chunk_type: str     # chunk type
    pass_number: int           # chunk index (round number) that produced this

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON checkpointing."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionResult":
        """Deserialize from a plain dict."""
        return cls(**d)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Result merging
# ---------------------------------------------------------------------------

def merge_results(
    existing: dict[int, ExtractionResult],
    new_results: list[ExtractionResult],
) -> dict[int, ExtractionResult]:
    """Merge *new_results* into *existing*.  Higher confidence wins."""
    merged = dict(existing)

    for result in new_results:
        item_num = result.item_number
        current = merged.get(item_num)

        if current is None:
            merged[item_num] = result
            continue

        if result.confidence > current.confidence:
            merged[item_num] = result

    return merged


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------

def split_items_into_batches(
    items: list[Any], items_per_call: int
) -> list[list[Any]]:
    """Partition *items* into sub-lists of at most *items_per_call*."""
    if items_per_call <= 0:
        return [items]
    return [
        items[i : i + items_per_call]
        for i in range(0, len(items), items_per_call)
    ]


# ---------------------------------------------------------------------------
# Serialization helpers for checkpointing
# ---------------------------------------------------------------------------

def serialize_extraction_state(
    state: dict[int, ExtractionResult],
) -> str:
    """Serialize extraction state to JSON string."""
    return json.dumps(
        {str(k): v.to_dict() for k, v in state.items()},
        indent=2,
    )


def deserialize_extraction_state(
    data: str,
) -> dict[int, ExtractionResult]:
    """Deserialize extraction state from JSON string."""
    raw = json.loads(data)
    return {
        int(k): ExtractionResult.from_dict(v)
        for k, v in raw.items()
    }
