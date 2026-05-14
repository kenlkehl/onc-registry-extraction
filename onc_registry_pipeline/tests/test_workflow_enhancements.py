from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from onc_registry_pipeline.extraction.chunk_extractor import ChunkExtractor
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorDetector
from onc_registry_pipeline.llm.client import LLMResponse
from onc_registry_pipeline.manuals.seer import SEERManualContextProvider


@dataclass
class DummyChunk:
    chunk_id: str
    text: str
    chunk_type: str = "sequential"
    chunk_index: int = 0


class CapturingLog:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def log(self, **kwargs) -> None:
        self.entries.append(kwargs)


class FakeLLM:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response

    async def extract(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return self.response


@dataclass
class DummyItem:
    item_number: int
    name: str
    xml_id: str
    length: int = 4000


class FakeDictionary:
    def __init__(self, item: DummyItem) -> None:
        self.item = item

    def get_item(self, item_number: int) -> DummyItem | None:
        if item_number == self.item.item_number:
            return self.item
        return None


class FakeSchemaBuilder:
    def build_json_format_instructions(self, items, resolver) -> str:
        return "Return valid JSON."


def test_tumor_deduplication_uses_site_histology_laterality_and_date() -> None:
    detector = TumorDetector(llm_client=None, schema_builder=None)  # type: ignore[arg-type]
    chunks = [
        DummyChunk("chunk_0", "left breast ductal carcinoma and lobular carcinoma"),
        DummyChunk("chunk_1", "left breast ductal carcinoma"),
    ]
    raw_tumors = [
        {
            "cancer_type": "ductal carcinoma",
            "histology": "ductal carcinoma",
            "primary_site": "breast",
            "laterality": "left",
            "approximate_diagnosis_date": "2024-01",
            "evidence": "left breast ductal carcinoma",
            "_source_chunk_id": "chunk_0",
        },
        {
            "cancer_type": "ductal carcinoma NOS",
            "histology": "ductal carcinoma",
            "primary_site": "breast",
            "laterality": "left",
            "approximate_diagnosis_date": "2024-01",
            "evidence": "left breast ductal carcinoma",
            "_source_chunk_id": "chunk_1",
        },
        {
            "cancer_type": "lobular carcinoma",
            "histology": "lobular carcinoma",
            "primary_site": "breast",
            "laterality": "left",
            "approximate_diagnosis_date": "2024-01",
            "evidence": "left breast lobular carcinoma",
            "_source_chunk_id": "chunk_0",
        },
        {
            "cancer_type": "ductal carcinoma",
            "histology": "ductal carcinoma",
            "primary_site": "breast",
            "laterality": "right",
            "approximate_diagnosis_date": "2024-01",
            "evidence": "right breast ductal carcinoma",
            "_source_chunk_id": "chunk_1",
        },
    ]

    candidates = detector._deduplicate(raw_tumors, chunks)

    assert len(candidates) == 3
    diagnosis_keys = {candidate.diagnosis_key for candidate in candidates}
    assert "breast|ductal carcinoma|left|2024-01" in diagnosis_keys
    assert "breast|lobular carcinoma|left|2024-01" in diagnosis_keys
    assert "breast|ductal carcinoma|right|2024-01" in diagnosis_keys


async def test_tumor_detection_logs_reasoning_and_raw_output() -> None:
    log = CapturingLog()
    response = LLMResponse(
        raw_content="<think>diagnosis reasoning</think>{\"tumors\": []}",
        final_content="{\"tumors\": []}",
        parsed={"tumors": []},
        reasoning="<think>diagnosis reasoning",
    )
    detector = TumorDetector(
        llm_client=FakeLLM(response),  # type: ignore[arg-type]
        schema_builder=None,  # type: ignore[arg-type]
        llm_log=log,
    )

    tumors = await detector._detect_in_chunk(
        DummyChunk("chunk_0", "No primary cancer diagnosis.", chunk_index=2)
    )

    assert tumors == []
    assert len(log.entries) == 1
    entry = log.entries[0]
    assert entry["call_type"] == "tumor_detection"
    assert entry["pass_number"] == 2
    assert entry["raw_output"] == response.raw_content
    assert entry["reasoning"] == response.reasoning
    assert entry["final_output"] == response.final_content
    assert entry["parsed"] == response.parsed


async def test_narrative_summary_logs_reasoning_and_raw_output() -> None:
    log = CapturingLog()
    item = DummyItem(2520, "DX Proc--PE", "dxProcPeText")
    response = LLMResponse(
        raw_content="<think>narrative reasoning</think>{\"dxProcPeText\": {}}",
        final_content="{\"dxProcPeText\": {}}",
        parsed={
            "dxProcPeText": {
                "value": "Physical exam summary.",
                "confidence": 0.85,
                "evidence": "Exam note",
            }
        },
        reasoning="<think>narrative reasoning",
    )
    extractor = ChunkExtractor(
        config=None,  # type: ignore[arg-type]
        dictionary=FakeDictionary(item),  # type: ignore[arg-type]
        code_resolver=None,  # type: ignore[arg-type]
        llm_client=FakeLLM(response),  # type: ignore[arg-type]
        schema_builder=FakeSchemaBuilder(),  # type: ignore[arg-type]
        schema_registry=None,  # type: ignore[arg-type]
        llm_log=log,
    )

    results = await extractor._update_narratives(
        DummyChunk("chunk_1", "Physical exam note.", chunk_index=3),
        {},
    )

    assert len(results) == 1
    assert len(log.entries) == 1
    entry = log.entries[0]
    assert entry["call_type"] == "narrative_summary"
    assert entry["pass_number"] == 3
    assert entry["raw_output"] == response.raw_content
    assert entry["reasoning"] == response.reasoning
    assert entry["final_output"] == response.final_content
    assert entry["parsed"] == response.parsed


def test_seer_manual_context_provider_returns_relevant_local_context() -> None:
    manuals_dir = Path(__file__).resolve().parents[2] / "SEERManuals"
    provider = SEERManualContextProvider(manuals_dir, max_chars=4000)

    context = provider.build_context(
        tumor_context="invasive ductal carcinoma at left breast, diagnosed 2024-01",
        cancer_type="invasive ductal carcinoma",
        primary_site="C50.4",
        histology="8500",
        schema="breast",
        site_desc="breast",
        site_context="Extract ER, PR, HER2, Ki-67, and breast-specific factors.",
    )

    assert "NAACCR cancer-specific context" in context
    assert "SEER/NAACCR manual excerpt" in context
    assert "Breast" in context or "breast" in context
