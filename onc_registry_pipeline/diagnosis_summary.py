"""Write a one-row-per-diagnosis high-level summary CSV file."""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from onc_registry_pipeline.extraction.base import ExtractionResult
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorCandidate
from onc_registry_pipeline.icd_o3_reference import (
    MORPHOLOGY as _MORPHOLOGY_REFERENCE,
    TOPOGRAPHY_EXACT as _TOPOGRAPHY_EXACT_REFERENCE,
    TOPOGRAPHY_PREFIX as _TOPOGRAPHY_PREFIX_REFERENCE,
)

logger = logging.getLogger(__name__)

DATE_OF_DIAGNOSIS = 390
PRIMARY_SITE = 400
HISTOLOGY_ICD_O_3 = 522
SUMMARY_STAGE_2018 = 764

AJCC_FIELDS: tuple[tuple[str, int], ...] = (
    ("ajcc_clinical_t", 1001),
    ("ajcc_clinical_n", 1002),
    ("ajcc_clinical_m", 1003),
    ("ajcc_clinical_stage_group", 1004),
    ("ajcc_pathologic_t", 1011),
    ("ajcc_pathologic_n", 1012),
    ("ajcc_pathologic_m", 1013),
    ("ajcc_pathologic_stage_group", 1014),
)

SUMMARY_COLUMNS: list[str] = [
    "patient_id",
    "diagnosis_date",
    "primary_site_code",
    "primary_site_text",
    "histology_code",
    "histology_text",
    "seer_summary_stage_code",
    "seer_summary_stage_text",
]
for _field_name, _item_number in AJCC_FIELDS:
    SUMMARY_COLUMNS.extend([f"{_field_name}_code", f"{_field_name}_text"])

_SITE_CODE_RE = re.compile(r"C\d{2}(?:\.?\d)?", re.IGNORECASE)
_HISTOLOGY_CODE_RE = re.compile(r"\b(\d{4})(?:/[0-3])?\b")
_AJCC_BASIS_PREFIX_RE = re.compile(r"^(?:c|p)(?=(?:T|N|M|0|I|V|X))", re.IGNORECASE)
_BLANK_CODE_VALUES = {"blank"}

_TOPOGRAPHY_PREFIX_DESCRIPTIONS = _TOPOGRAPHY_PREFIX_REFERENCE

_TOPOGRAPHY_EXACT_DESCRIPTIONS = _TOPOGRAPHY_EXACT_REFERENCE

_HISTOLOGY_DESCRIPTIONS = _MORPHOLOGY_REFERENCE


DiagnosisSummaryRecord = tuple[
    str,
    int,
    TumorCandidate | None,
    Mapping[int, ExtractionResult],
]


def write_diagnosis_summary_csv(
    records: Sequence[DiagnosisSummaryRecord],
    csv_path: str | Path,
    dictionary: Any,
) -> None:
    """Write one high-level CSV row for each diagnosis/tumor record."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for patient_id, _tumor_index, tumor, record in records:
            writer.writerow(build_diagnosis_summary_row(patient_id, tumor, record, dictionary))

    logger.info("Diagnosis summary CSV written to %s", path)


def build_diagnosis_summary_row(
    patient_id: str,
    tumor: TumorCandidate | None,
    record: Mapping[int, ExtractionResult],
    dictionary: Any,
) -> dict[str, str]:
    """Build the output row for a single diagnosis/tumor record."""
    primary_site_result = record.get(PRIMARY_SITE)
    histology_result = record.get(HISTOLOGY_ICD_O_3)
    summary_stage_raw = _raw_result_value(record.get(SUMMARY_STAGE_2018))
    summary_stage_code = _public_code_value(summary_stage_raw)

    row: dict[str, str] = {
        "patient_id": patient_id,
        "diagnosis_date": _public_code_value(_raw_result_value(record.get(DATE_OF_DIAGNOSIS))),
        "primary_site_code": _public_code_value(_raw_result_value(primary_site_result)),
        "primary_site_text": _primary_site_text(primary_site_result, tumor, dictionary),
        "histology_code": _public_code_value(_raw_result_value(histology_result)),
        "histology_text": _histology_text(histology_result, tumor, dictionary),
        "seer_summary_stage_code": summary_stage_code,
        "seer_summary_stage_text": _code_description(
            dictionary,
            SUMMARY_STAGE_2018,
            summary_stage_raw,
        ),
    }

    for field_name, item_number in AJCC_FIELDS:
        raw_value = _raw_result_value(record.get(item_number))
        row[f"{field_name}_code"] = _public_code_value(raw_value)
        row[f"{field_name}_text"] = _ajcc_text(dictionary, item_number, raw_value)

    return row


def _raw_result_value(result: ExtractionResult | None) -> str:
    if not isinstance(result, ExtractionResult):
        return ""
    value = result.resolved_code or result.extracted_value
    return str(value).strip()


def _public_code_value(value: str) -> str:
    text = str(value).strip()
    if text.lower() in _BLANK_CODE_VALUES:
        return ""
    return text


def _primary_site_text(
    result: ExtractionResult | None,
    tumor: TumorCandidate | None,
    dictionary: Any,
) -> str:
    raw_value = _raw_result_value(result)
    return (
        _code_description(dictionary, PRIMARY_SITE, raw_value)
        or _topography_description(raw_value)
        or _non_code_extracted_value(result, PRIMARY_SITE)
        or _clean_text(getattr(tumor, "primary_site_hint", "") if tumor else "")
    )


def _histology_text(
    result: ExtractionResult | None,
    tumor: TumorCandidate | None,
    dictionary: Any,
) -> str:
    raw_value = _raw_result_value(result)
    histology_code = _normalize_histology_code(raw_value)
    tumor_histology = getattr(tumor, "histology", "") if tumor else ""
    cancer_type = getattr(tumor, "cancer_type", "") if tumor else ""
    return (
        _code_description(dictionary, HISTOLOGY_ICD_O_3, raw_value)
        or _HISTOLOGY_DESCRIPTIONS.get(histology_code, "")
        or _non_code_extracted_value(result, HISTOLOGY_ICD_O_3)
        or _clean_text(tumor_histology)
        or _clean_text(cancer_type)
    )


def _ajcc_text(dictionary: Any, item_number: int, raw_value: str) -> str:
    description = _code_description(dictionary, item_number, raw_value)
    if description:
        return description

    value = _public_code_value(raw_value)
    if not value:
        return ""

    # AJCC v8+ NAACCR items store the stage category text itself
    # (for example cT1 or pN0).  In a clinical/pathologic column, the
    # c/p basis prefix is redundant, so the display text drops it.
    return _AJCC_BASIS_PREFIX_RE.sub("", value).strip()


def _code_description(dictionary: Any, item_number: int, value: str) -> str:
    if dictionary is None:
        return ""

    candidates = _code_lookup_candidates(value)
    try:
        entries = dictionary.get_codes(item_number)
    except Exception:
        return ""

    for entry in entries:
        code = str(getattr(entry, "code", "")).strip()
        for candidate in candidates:
            if code == candidate or code.lower() == candidate.lower():
                return _clean_text(getattr(entry, "description", ""))

    return ""


def _code_lookup_candidates(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    candidates = [text]
    if text.lower() == "blank":
        candidates.extend(["Blank", "blank", ""])
    return list(dict.fromkeys(candidates))


def _topography_description(value: str) -> str:
    code = _normalize_site_code(value)
    if not code:
        return ""
    return (
        _TOPOGRAPHY_EXACT_DESCRIPTIONS.get(code, "")
        or _TOPOGRAPHY_PREFIX_DESCRIPTIONS.get(code[:3], "")
    )


def _normalize_site_code(value: str) -> str:
    match = _SITE_CODE_RE.search(str(value).strip())
    if not match:
        return ""

    code = match.group(0).upper().replace(" ", "")
    if "." in code:
        prefix, subsite = code.split(".", 1)
        return f"{prefix}.{subsite[:1]}"
    if len(code) == 4:
        return f"{code[:3]}.{code[3]}"
    return code[:3]


def _normalize_histology_code(value: str) -> str:
    match = _HISTOLOGY_CODE_RE.search(str(value).strip())
    return match.group(1) if match else ""


def _non_code_extracted_value(
    result: ExtractionResult | None,
    item_number: int,
) -> str:
    if result is None:
        return ""
    extracted = str(result.extracted_value or "").strip()
    if not extracted:
        return ""
    if item_number == PRIMARY_SITE and _normalize_site_code(extracted):
        return ""
    if item_number == HISTOLOGY_ICD_O_3 and _normalize_histology_code(extracted):
        return ""
    resolved = str(result.resolved_code or "").strip()
    if extracted == resolved:
        return ""
    return _clean_text(extracted)


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip(" *")
