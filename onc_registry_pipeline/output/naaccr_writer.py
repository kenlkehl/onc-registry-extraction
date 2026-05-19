"""Write NAACCR v26 output in XML, fixed-width flat file, or CSV format."""

from __future__ import annotations

import csv
import logging
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

from onc_registry_pipeline.dictionary.loader import NAACCRDictionary

logger = logging.getLogger(__name__)

NAACCR_XML_NS = "http://naaccr.org/naaccrxml"


class NAACCRWriter:
    """Produce NAACCR-formatted output files.

    Parameters
    ----------
    dictionary : NAACCRDictionary
        Loaded dictionary used to look up xmlNaaccrId, parentElement, and
        field length for each item number.
    """

    def __init__(self, dictionary: NAACCRDictionary) -> None:
        self._dict = dictionary

    # ------------------------------------------------------------------
    # XML
    # ------------------------------------------------------------------

    def write_xml(
        self,
        value_records: list[dict[int, str]],
        xml_path: str,
        patient_groups: dict[str, list[int]],
    ) -> None:
        """Write NAACCR XML (NaaccrData > Patient > Tumor hierarchy).

        Parameters
        ----------
        value_records : list[dict[int, str]]
            Each dict maps item_number -> resolved value string.
        xml_path : str
            Destination file path.
        patient_groups : dict[str, list[int]]
            patient_id -> list of indices into *value_records*.
        """
        root = Element("NaaccrData", xmlns=NAACCR_XML_NS)

        # Collect NaaccrData-level items from the first record (they are
        # file-level attributes shared across all records).
        if value_records:
            self._add_items(root, value_records[0], parent_filter="NaaccrData")

        for patient_id, record_indices in patient_groups.items():
            patient_el = SubElement(root, "Patient")

            # Patient-level items come from the first tumor record for
            # this patient (they are the same across tumors).
            first_rec = value_records[record_indices[0]]
            self._add_items(patient_el, first_rec, parent_filter="Patient")

            for idx in record_indices:
                rec = value_records[idx]
                tumor_el = SubElement(patient_el, "Tumor")
                self._add_items(tumor_el, rec, parent_filter="Tumor")

        indent(root, space="  ")
        tree = ElementTree(root)
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
        logger.info("NAACCR XML written to %s", xml_path)

    # ------------------------------------------------------------------
    # Flat file
    # ------------------------------------------------------------------

    def write_flat_file(
        self,
        value_records: list[dict[int, str]],
        flat_path: str,
    ) -> None:
        """Write a simplified NAACCR-style flat file.

        Without start-column positions in the data dictionary CSVs a true
        fixed-width record cannot be produced.  This writes one
        fixed-width line per record using items sorted by item number,
        each right-padded to its dictionary-defined length.
        """
        # Collect all item numbers used across records, sorted.
        all_items = sorted({n for rec in value_records for n in rec})

        with open(flat_path, "w", encoding="utf-8") as fh:
            for rec in value_records:
                parts: list[str] = []
                for item_num in all_items:
                    item_def = self._dict.get_item(item_num)
                    length = item_def.length if item_def else 1
                    value = rec.get(item_num, "")
                    # Left-justify, pad/truncate to field length
                    parts.append(value[:length].ljust(length))
                fh.write("".join(parts) + "\n")

        logger.info("NAACCR flat file written to %s", flat_path)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def write_csv(
        self,
        value_records: list[dict[int, str]],
        csv_path: str,
    ) -> None:
        """Write one CSV row per record with item names as column headers."""
        all_items = sorted({n for rec in value_records for n in rec})

        headers: list[str] = []
        for item_num in all_items:
            item_def = self._dict.get_item(item_num)
            name = item_def.name if item_def else f"Item_{item_num}"
            headers.append(f"{name} [{item_num}]")

        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for rec in value_records:
                row = [rec.get(item_num, "") for item_num in all_items]
                writer.writerow(row)

        logger.info("CSV written to %s", csv_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_items(
        self,
        parent_el: Element,
        record: dict[int, str],
        parent_filter: str,
    ) -> None:
        """Add <Item> sub-elements for items matching *parent_filter*."""
        for item_num in sorted(record):
            item_def = self._dict.get_item(item_num)
            if item_def is None:
                continue
            if item_def.parent_element != parent_filter:
                continue
            value = record[item_num]
            if not value:
                continue
            item_el = SubElement(parent_el, "Item")
            item_el.set("naaccrId", item_def.xml_id)
            item_el.set("naaccrNum", str(item_num))
            item_el.text = value
