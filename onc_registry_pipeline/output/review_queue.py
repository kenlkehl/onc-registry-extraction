"""Human review queue for items flagged by confidence scoring."""

from __future__ import annotations

import csv
import logging
from dataclasses import fields

logger = logging.getLogger(__name__)


class ReviewQueue:
    """Collects ReviewItem instances and exports a sorted worklist."""

    def __init__(self) -> None:
        self._items: list = []

    def add_items(self, items: list) -> None:
        """Append flagged review items."""
        self._items.extend(items)

    def export_csv(self, path: str) -> None:
        """Write the review queue to *path*, sorted by priority."""
        if not self._items:
            return
        field_names = [f.name for f in fields(self._items[0])]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=field_names)
            writer.writeheader()
            for item in self._items:
                writer.writerow({
                    f.name: getattr(item, f.name) for f in fields(item)
                })
        logger.info("Review queue written to %s (%d items)", path, len(self._items))
