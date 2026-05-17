"""Convert NAACCR XML output to JSON or flat CSV for downstream use.

Usage::

    # XML -> JSON
    onc-registry-convert output/naaccr_output.xml output/naaccr_output.json

    # XML -> CSV
    onc-registry-convert output/naaccr_output.xml output/naaccr_output.csv

    # With human-readable item names instead of XML IDs
    onc-registry-convert output/naaccr_output.xml output/data.csv --readable-names

    # Include only non-empty fields
    onc-registry-convert output/naaccr_output.xml output/data.json --skip-empty
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from onc_registry_pipeline.paths import default_data_dict_dir, resolve_reference_path

logger = logging.getLogger(__name__)

# Namespace used in NAACCR v26 XML
NAACCR_NS = "http://naaccr.org/naaccrxml"
NS_PREFIX = "{%s}" % NAACCR_NS


# -------------------------------------------------------------------
# Parsing
# -------------------------------------------------------------------

def _get_etree():
    """Return the best available ElementTree implementation."""
    try:
        from lxml import etree
        return etree
    except ImportError:
        import xml.etree.ElementTree as etree
        return etree


def parse_naaccr_xml(xml_path: str) -> list[dict]:
    """Parse a NAACCR XML file into a list of flat record dicts.

    Each record is one tumor.  Patient-level and NaaccrData-level items
    are merged into every tumor record for that patient so each row is
    self-contained.

    Returns a list of dicts where keys are ``naaccrId`` attribute values
    and values are the element text.
    """
    etree = _get_etree()
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Collect root-level (NaaccrData) items
    root_items = _collect_items(root)

    records = []
    for patient_el in root.iter(NS_PREFIX + "Patient"):
        patient_items = _collect_items(patient_el)

        tumors = list(patient_el.iter(NS_PREFIX + "Tumor"))
        if not tumors:
            # Patient with no tumor elements — emit one record with
            # just the root + patient items
            record = {}
            record.update(root_items)
            record.update(patient_items)
            records.append(record)
            continue

        for tumor_el in tumors:
            tumor_items = _collect_items(tumor_el)
            record = {}
            record.update(root_items)
            record.update(patient_items)
            record.update(tumor_items)
            records.append(record)

    return records


def _collect_items(element) -> dict[str, str]:
    """Extract ``<Item naaccrId="...">value</Item>`` children of *element*.

    Only direct children are collected (not descendants deeper than one
    level) to avoid pulling Tumor items when iterating a Patient element.
    """
    items = {}
    for child in element:
        tag = child.tag
        # Strip namespace
        if tag.startswith(NS_PREFIX):
            tag = tag[len(NS_PREFIX):]

        if tag == "Item":
            naaccr_id = child.get("naaccrId", "")
            if naaccr_id:
                items[naaccr_id] = (child.text or "").strip()
    return items


# -------------------------------------------------------------------
# Name mapping
# -------------------------------------------------------------------

def load_name_map(data_dict_dir: Optional[str] = None) -> dict[str, str]:
    """Build a mapping from xmlNaaccrId -> human-readable item name.

    Tries the explicit *data_dict_dir* path first, then a working-directory
    ``NAACCRDataItems`` directory, then the vendored package/repo copy.
    """
    candidates = []
    if data_dict_dir:
        candidates.append(
            resolve_reference_path(data_dict_dir, "NAACCRDataItems") / "DataItems.csv"
        )
    else:
        candidates.append(Path("NAACCRDataItems") / "DataItems.csv")
    candidates.append(default_data_dict_dir() / "DataItems.csv")

    for path in candidates:
        if path.exists():
            return _parse_name_map(path)

    logger.debug("DataItems.csv not found; readable names unavailable.")
    return {}


def _parse_name_map(csv_path: Path) -> dict[str, str]:
    """Parse DataItems.csv and return {xml_id: item_name}."""
    name_map = {}
    with open(csv_path, "r", encoding="utf-8-sig") as fh:
        # First line is blank in NAACCR CSVs
        first_line = fh.readline()
        if first_line.strip():
            fh.seek(0)
        reader = csv.reader(fh)
        header = next(reader)

        # Find column indices
        try:
            idx_num = header.index("Data Item Number")
            idx_name = header.index("Data Item Name")
            idx_xml = header.index("XML NAACCR ID")
        except ValueError:
            logger.warning("Unexpected DataItems.csv header; skipping name map.")
            return {}

        for row in reader:
            if len(row) <= max(idx_num, idx_name, idx_xml):
                continue
            num = row[idx_num].strip()
            if not num.isdigit():
                continue
            xml_id = row[idx_xml].strip()
            item_name = row[idx_name].strip()
            if xml_id:
                name_map[xml_id] = item_name

    return name_map


# -------------------------------------------------------------------
# Writers
# -------------------------------------------------------------------

def write_json(
    records: list[dict],
    output_path: str,
    *,
    skip_empty: bool = False,
    name_map: Optional[dict[str, str]] = None,
) -> None:
    """Write records as a JSON array of objects."""
    out_records = []
    for rec in records:
        out = {}
        for key, value in rec.items():
            if skip_empty and not value:
                continue
            display_key = name_map.get(key, key) if name_map else key
            out[display_key] = value
        out_records.append(out)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(out_records, fh, indent=2, ensure_ascii=False)

    logger.info("Wrote JSON to %s (%d records)", output_path, len(out_records))


def write_csv(
    records: list[dict],
    output_path: str,
    *,
    skip_empty: bool = False,
    name_map: Optional[dict[str, str]] = None,
) -> None:
    """Write records as a flat CSV file."""
    if not records:
        logger.warning("No records to write.")
        return

    # Collect all keys in insertion order across records
    all_keys: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for key in rec:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    # Optionally filter to keys that have at least one non-empty value
    if skip_empty:
        non_empty = set()
        for rec in records:
            for key, value in rec.items():
                if value:
                    non_empty.add(key)
        all_keys = [k for k in all_keys if k in non_empty]

    # Map to display names
    headers = [name_map.get(k, k) if name_map else k for k in all_keys]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for rec in records:
            writer.writerow([rec.get(k, "") for k in all_keys])

    logger.info("Wrote CSV to %s (%d records, %d columns)", output_path, len(records), len(all_keys))


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    """CLI entry point for onc-registry-convert."""
    parser = argparse.ArgumentParser(
        description="Convert NAACCR XML output to JSON or CSV",
    )
    parser.add_argument("input", help="Path to NAACCR XML file")
    parser.add_argument(
        "output",
        help="Output path (.json or .csv; format inferred from extension)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default=None,
        help="Output format (default: inferred from output file extension)",
    )
    parser.add_argument(
        "--readable-names",
        action="store_true",
        help="Use human-readable NAACCR item names instead of XML IDs",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Omit fields with empty values",
    )
    parser.add_argument(
        "--data-dict",
        default=None,
        help="Path to NAACCRDataItems directory (for --readable-names)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine output format
    fmt = args.format
    if fmt is None:
        ext = Path(args.output).suffix.lower()
        if ext == ".json":
            fmt = "json"
        elif ext in (".csv", ".tsv"):
            fmt = "csv"
        else:
            logger.error(
                "Cannot infer format from extension '%s'. "
                "Use --format json or --format csv.",
                ext,
            )
            sys.exit(1)

    # Parse XML
    logger.info("Parsing %s ...", args.input)
    records = parse_naaccr_xml(args.input)
    logger.info("Parsed %d tumor records", len(records))

    if not records:
        logger.warning("No records found in input file.")
        sys.exit(0)

    # Load name map if requested
    name_map = None
    if args.readable_names:
        name_map = load_name_map(args.data_dict)
        if name_map:
            logger.info("Loaded %d readable item names", len(name_map))
        else:
            logger.warning(
                "Could not load item names; using XML IDs. "
                "Pass --data-dict to specify the NAACCRDataItems directory."
            )

    # Write output
    if fmt == "json":
        write_json(records, args.output, skip_empty=args.skip_empty, name_map=name_map)
    else:
        write_csv(records, args.output, skip_empty=args.skip_empty, name_map=name_map)

    print(f"Converted {len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()
