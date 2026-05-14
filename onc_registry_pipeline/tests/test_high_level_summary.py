from __future__ import annotations

import csv

from onc_registry_pipeline.dictionary.loader import CodeEntry
from onc_registry_pipeline.extraction.base import ExtractionResult
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorCandidate
from onc_registry_pipeline.diagnosis_summary import (
    build_diagnosis_summary_row,
    write_diagnosis_summary_csv,
)


class FakeDictionary:
    def __init__(self) -> None:
        self._codes = {
            764: [
                CodeEntry(764, "Summary Stage 2018", 1, "2", "Regional by direct extension only"),
            ],
            1014: [
                CodeEntry(
                    1014,
                    "AJCC TNM Path Stage Group",
                    15,
                    "99",
                    "Unknown, not staged",
                ),
            ],
            1001: [
                CodeEntry(
                    1001,
                    "AJCC TNM Clin T",
                    15,
                    "Blank",
                    "Information not available to code this item.",
                ),
            ],
        }

    def get_codes(self, item_number: int) -> list[CodeEntry]:
        return list(self._codes.get(item_number, []))


def result(item_number: int, value: str, extracted: str | None = None) -> ExtractionResult:
    return ExtractionResult(
        item_number=item_number,
        item_name=f"Item {item_number}",
        extracted_value=extracted if extracted is not None else value,
        resolved_code=value,
        confidence=0.95,
        evidence_text="evidence",
        source_chunk_id="chunk_0",
        source_chunk_type="sequential",
        pass_number=0,
    )


def test_diagnosis_summary_row_uses_codes_and_text_descriptions() -> None:
    tumor = TumorCandidate(
        tumor_index=0,
        cancer_type="ductal carcinoma",
        primary_site_hint="left breast",
        approximate_date="2024-01",
        evidence="left breast ductal carcinoma",
        histology="ductal carcinoma",
    )
    record = {
        390: result(390, "20240115"),
        400: result(400, "C50.4"),
        522: result(522, "8500"),
        764: result(764, "2"),
        1001: result(1001, "cT1"),
        1002: result(1002, "cN0"),
        1003: result(1003, "cM0"),
        1004: result(1004, "cIA"),
        1011: result(1011, "pT2"),
        1012: result(1012, "pN1"),
        1013: result(1013, "pM0"),
        1014: result(1014, "99"),
    }

    row = build_diagnosis_summary_row("patient-1", tumor, record, FakeDictionary())

    assert row["patient_id"] == "patient-1"
    assert row["diagnosis_date"] == "20240115"
    assert row["primary_site_code"] == "C50.4"
    assert row["primary_site_text"] == "Upper-outer quadrant of breast"
    assert row["histology_code"] == "8500"
    assert row["histology_text"] == "Infiltrating duct carcinoma, NOS"
    assert row["seer_summary_stage_code"] == "2"
    assert row["seer_summary_stage_text"] == "Regional by direct extension only"
    assert row["ajcc_clinical_t_code"] == "cT1"
    assert row["ajcc_clinical_t_text"] == "T1"
    assert row["ajcc_pathologic_n_code"] == "pN1"
    assert row["ajcc_pathologic_n_text"] == "N1"
    assert row["ajcc_pathologic_stage_group_code"] == "99"
    assert row["ajcc_pathologic_stage_group_text"] == "Unknown, not staged"


def test_write_diagnosis_summary_csv(tmp_path) -> None:
    record = {
        390: result(390, "20240115"),
        400: result(400, "C509"),
        522: result(522, "9999", extracted="rare tumor"),
    }
    csv_path = tmp_path / "diagnosis_summary.csv"

    write_diagnosis_summary_csv(
        [("patient-1", 0, None, record)],
        csv_path,
        FakeDictionary(),
    )

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    assert rows[0]["patient_id"] == "patient-1"
    assert rows[0]["primary_site_text"] == "Breast, NOS"
    assert rows[0]["histology_text"] == "rare tumor"
    assert rows[0]["ajcc_clinical_t_text"] == ""
