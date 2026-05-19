"""Per-item provenance audit trail for extracted NAACCR data."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, fields

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    patient_id: str
    tumor_index: int
    item_number: int
    item_name: str
    extracted_value: str
    resolved_code: str
    confidence: float
    evidence_text: str
    source_chunk_id: str
    source_chunk_type: str
    pass_number: int


class AuditTrail:
    """Collects per-item provenance records and exports to CSV."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record_from_result(
        self,
        result,
        patient_id: str = "",
        tumor_index: int = 0,
    ) -> None:
        """Create an audit entry from an ExtractionResult."""
        self._entries.append(AuditEntry(
            patient_id=patient_id,
            tumor_index=tumor_index,
            item_number=getattr(result, "item_number", 0),
            item_name=getattr(result, "item_name", ""),
            extracted_value=getattr(result, "extracted_value", ""),
            resolved_code=getattr(result, "resolved_code", ""),
            confidence=getattr(result, "confidence", 0.0),
            evidence_text=(getattr(result, "evidence_text", "") or "")[:500],
            source_chunk_id=getattr(result, "source_chunk_id", ""),
            source_chunk_type=getattr(result, "source_chunk_type", ""),
            pass_number=getattr(result, "pass_number", 0),
        ))

    def export_csv(self, path: str) -> None:
        """Write all audit entries to *path*."""
        if not self._entries:
            return
        field_names = [f.name for f in fields(AuditEntry)]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=field_names)
            writer.writeheader()
            for entry in self._entries:
                writer.writerow({
                    f.name: getattr(entry, f.name) for f in fields(AuditEntry)
                })
        logger.info("Audit trail written to %s (%d entries)", path, len(self._entries))
