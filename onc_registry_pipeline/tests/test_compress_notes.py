from __future__ import annotations

import pandas as pd

from onc_registry_pipeline.compress_notes import (
    COMPRESSION_SYSTEM_PROMPT,
    build_document_prompt,
    compress_notes_dataframe,
    normalize_summary,
)
from onc_registry_pipeline.llm.client import LLMTextResponse


class FakeTextClient:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = outputs
        self.prompts: list[dict] = []

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMTextResponse:
        self.prompts.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return LLMTextResponse(
            raw_content=str(output),
            final_content=str(output),
            reasoning="",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


def test_compression_prompt_captures_required_clinical_rules() -> None:
    prompt = COMPRESSION_SYSTEM_PROMPT

    assert "three sentences or less" in prompt
    assert "multiple independent primary cancers" in prompt
    assert "Biomarkers are NOT routine laboratory" in prompt
    assert "tumor markers belong" in prompt
    assert "Spell drug names out in full" in prompt
    assert "International Prognostic Index" in prompt
    assert "planned next steps" in prompt


def test_build_document_prompt_includes_metadata_and_text() -> None:
    prompt = build_document_prompt(
        "Clinical document body",
        {
            "document_id": "doc-1",
            "patient_id": "patient-1",
            "date": "2025-01-02",
            "note_type": "progress",
        },
    )

    assert "- document_id: doc-1" in prompt
    assert "- patient_id: patient-1" in prompt
    assert "Clinical document body" in prompt


def test_normalize_summary_removes_common_wrappers() -> None:
    assert normalize_summary("Summary: Patient has colon cancer.") == (
        "Patient has colon cancer."
    )
    assert normalize_summary("```\nClinical content.\n```") == "Clinical content."


async def test_compress_notes_dataframe_processes_individual_rows() -> None:
    client = FakeTextClient(["Summary for document 1.", "Summary for document 2."])
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "date": ["2025-01-01", "2025-02-01"],
            "note_type": ["pathology", "progress"],
            "text": ["Document one text.", "Document two text."],
            "sex": ["F", "F"],
        }
    )

    compressed, audit = await compress_notes_dataframe(
        df,
        client,
        max_concurrent=1,
        max_tokens=256,
    )

    assert compressed["text"].tolist() == [
        "Summary for document 1.",
        "Summary for document 2.",
    ]
    assert compressed["patient_id"].tolist() == ["p1", "p1"]
    assert compressed["date"].tolist() == ["2025-01-01", "2025-02-01"]
    assert compressed["note_type"].tolist() == ["pathology", "progress"]
    assert compressed["sex"].tolist() == ["F", "F"]
    assert len(client.prompts) == 2
    assert "Document one text." in client.prompts[0]["user_prompt"]
    assert "Document two text." not in client.prompts[0]["user_prompt"]
    assert "Document two text." in client.prompts[1]["user_prompt"]
    assert client.prompts[0]["system_prompt"] == COMPRESSION_SYSTEM_PROMPT
    assert client.prompts[0]["max_tokens"] == 256
    assert audit["summary"].tolist() == [
        "Summary for document 1.",
        "Summary for document 2.",
    ]
    assert audit["compression_used_fallback"].tolist() == [False, False]
    assert audit["prompt_tokens"].tolist() == [10, 10]
    assert audit["completion_tokens"].tolist() == [5, 5]


async def test_compress_notes_dataframe_falls_back_to_full_note_on_failure() -> None:
    client = FakeTextClient([RuntimeError("endpoint unavailable")])
    df = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "date": ["2025-01-01"],
            "text": ["Original clinical note text."],
        }
    )

    compressed, audit = await compress_notes_dataframe(
        df,
        client,
        max_concurrent=1,
    )

    assert compressed.loc[0, "text"] == "Original clinical note text."
    assert audit.loc[0, "summary"] == ""
    assert bool(audit.loc[0, "compression_used_fallback"]) is True
    assert "RuntimeError: endpoint unavailable" in audit.loc[0, "error"]


async def test_compress_notes_dataframe_uses_custom_columns() -> None:
    client = FakeTextClient(["Short summary."])
    df = pd.DataFrame(
        {
            "record_id": ["patient-1"],
            "service_date": ["2025-03-04"],
            "note_text": ["Long note."],
            "document_id": ["doc-1"],
            "mrn": ["MRN123"],
        }
    )

    compressed, audit = await compress_notes_dataframe(
        df,
        client,
        patient_id_column="record_id",
        date_column="service_date",
        text_column="note_text",
        document_id_column="document_id",
        max_concurrent=1,
    )

    assert compressed.loc[0, "patient_id"] == "patient-1"
    assert compressed.loc[0, "date"] == "2025-03-04"
    assert compressed.loc[0, "text"] == "Short summary."
    assert compressed.loc[0, "record_id"] == "patient-1"
    assert compressed.loc[0, "service_date"] == "2025-03-04"
    assert compressed.loc[0, "document_id"] == "doc-1"
    assert compressed.loc[0, "mrn"] == "MRN123"
    assert audit.loc[0, "document_id"] == "doc-1"
