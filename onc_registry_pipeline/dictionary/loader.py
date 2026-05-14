"""Load and index the NAACCR v26 data dictionary from CSV files."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from onc_registry_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NAACCRDataItem:
    """A single NAACCR data-dictionary item."""

    item_number: int
    name: str
    length: int
    source_of_standard: str
    record_type: str
    section: str
    xml_id: str
    parent_element: str
    year_implemented: str
    version_implemented: str
    year_retired: str
    version_retired: str
    npcr_collect: str
    coc_collect: str
    seer_collect: str
    cccr_collect: str
    description: str
    instructions: str
    allowable_values: str
    data_type: str
    format_spec: str
    alternate_names: list[str] = field(default_factory=list)


@dataclass
class CodeEntry:
    """One valid code for a data item."""

    item_number: int
    item_name: str
    length: int
    code: str
    description: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_valid_item_number(value: str) -> bool:
    """Return True if *value* can be parsed as a non-negative integer."""
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------

class NAACCRDictionary:
    """In-memory index of the NAACCR v26 data dictionary.

    Usage::

        d = NAACCRDictionary(config)
        d.load()
        item = d.get_item(400)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # Primary indexes
        self._items_by_number: dict[int, NAACCRDataItem] = {}
        self._items_by_section: dict[str, list[NAACCRDataItem]] = {}
        self._codes_by_item: dict[int, list[CodeEntry]] = {}
        self._items_by_alt_name: dict[str, NAACCRDataItem] = {}

        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Parse all three CSV files and build look-up indexes."""
        self._load_data_items()
        self._load_code_list()
        self._load_alternate_names()
        self._loaded = True
        logger.info(
            "NAACCR dictionary loaded: %d items, %d code entries",
            len(self._items_by_number),
            sum(len(v) for v in self._codes_by_item.values()),
        )

    # -- DataItems.csv --------------------------------------------------

    def _load_data_items(self) -> None:
        path = Path(self.config.data_items_csv)
        with open(path, newline="", encoding="utf-8-sig") as fh:
            lines = fh.readlines()

        # First line is blank; skip it so the CSV header is the first row.
        if lines and lines[0].strip() == "":
            lines = lines[1:]

        reader = csv.DictReader(lines)
        for row in reader:
            raw_num = row.get("Data Item Number", "").strip()
            if not _is_valid_item_number(raw_num):
                continue

            item_number = int(raw_num)
            item = NAACCRDataItem(
                item_number=item_number,
                name=row.get("Data Item Name", "").strip(),
                length=_safe_int(row.get("Length", "").strip()),
                source_of_standard=row.get("Source of Standard", "").strip(),
                record_type=row.get("Record Type", "").strip(),
                section=row.get("Section Name", "").strip(),
                xml_id=row.get("XML NAACCR ID", "").strip(),
                parent_element=row.get("Parent XML Element", "").strip(),
                year_implemented=row.get("Year Implemented", "").strip(),
                version_implemented=row.get("Version Implemented", "").strip(),
                year_retired=row.get("Year Retired", "").strip(),
                version_retired=row.get("Version Retired", "").strip(),
                npcr_collect=row.get("NPCR Collect", "").strip(),
                coc_collect=row.get("CoC Collect", "").strip(),
                seer_collect=row.get("SEER Collect", "").strip(),
                cccr_collect=row.get("CCCR Collect", "").strip(),
                description=row.get("Description", "").strip(),
                instructions=row.get("Instructions for Coding", "").strip(),
                allowable_values=row.get("Allowable Values", "").strip(),
                data_type=row.get("Data Type", "").strip(),
                format_spec=row.get("Format", "").strip(),
            )
            self._items_by_number[item_number] = item
            self._items_by_section.setdefault(item.section, []).append(item)

    # -- CodeList.csv ---------------------------------------------------

    def _load_code_list(self) -> None:
        path = Path(self.config.code_list_csv)
        with open(path, newline="", encoding="utf-8-sig") as fh:
            lines = fh.readlines()

        if lines and lines[0].strip() == "":
            lines = lines[1:]

        reader = csv.DictReader(lines)
        for row in reader:
            raw_num = row.get("Data Item Number", "").strip()
            if not _is_valid_item_number(raw_num):
                continue

            entry = CodeEntry(
                item_number=int(raw_num),
                item_name=row.get("Data Item Name", "").strip(),
                length=_safe_int(row.get("Length", "").strip()),
                code=row.get("Code", "").strip(),
                description=row.get("Description", "").strip(),
            )
            self._codes_by_item.setdefault(entry.item_number, []).append(entry)

    # -- AlternateNames.csv ---------------------------------------------

    def _load_alternate_names(self) -> None:
        path = Path(self.config.alternate_names_csv)
        with open(path, newline="", encoding="utf-8-sig") as fh:
            lines = fh.readlines()

        if lines and lines[0].strip() == "":
            lines = lines[1:]

        reader = csv.DictReader(lines)
        for row in reader:
            raw_num = row.get("Data Item Number", "").strip()
            if not _is_valid_item_number(raw_num):
                continue

            item_number = int(raw_num)
            alt_name = row.get("Alternate Name", "").strip()
            if not alt_name:
                continue

            item = self._items_by_number.get(item_number)
            if item is not None:
                item.alternate_names.append(alt_name)
                # Index by lower-cased alternate name for fast look-up.
                self._items_by_alt_name[alt_name.lower()] = item

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_item(self, item_number: int) -> Optional[NAACCRDataItem]:
        """Return the data item for *item_number*, or ``None``."""
        return self._items_by_number.get(item_number)

    def get_items_by_section(self, section: str) -> list[NAACCRDataItem]:
        """Return all items belonging to *section*."""
        return list(self._items_by_section.get(section, []))

    def get_codes(self, item_number: int) -> list[CodeEntry]:
        """Return all :class:`CodeEntry` rows for *item_number*."""
        return list(self._codes_by_item.get(item_number, []))

    def get_extractable_items(self, section: str) -> list[NAACCRDataItem]:
        """Return items from *section* that should be extracted from text.

        An item is extractable when:
        * At least one of ``npcr_collect``, ``coc_collect``, or
          ``seer_collect`` starts with ``'R'`` (required).
        * The item has not been retired (``year_retired`` is empty).
        * ``npcr_collect`` is not just ``'D'`` (derived-only items are
          excluded).
        """
        results: list[NAACCRDataItem] = []
        for item in self._items_by_section.get(section, []):
            if item.year_retired:
                continue
            if item.npcr_collect == "D":
                continue
            if (
                item.npcr_collect.startswith("R")
                or item.coc_collect.startswith("R")
                or item.seer_collect.startswith("R")
            ):
                results.append(item)
        return results

    def get_active_items(self) -> list[NAACCRDataItem]:
        """Return all items that have not been retired."""
        return [
            item
            for item in self._items_by_number.values()
            if not item.year_retired
        ]

    def lookup_by_alternate_name(self, name: str) -> Optional[NAACCRDataItem]:
        """Find an item by one of its alternate names (case-insensitive)."""
        return self._items_by_alt_name.get(name.lower())
