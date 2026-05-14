from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from onc_registry_pipeline.extraction.chunk_extractor import ChunkExtractor
from onc_registry_pipeline.config import PipelineConfig
from onc_registry_pipeline.extraction.base import ExtractionResult
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorDetector
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorCandidate
from onc_registry_pipeline.extraction.round_orchestrator import (
    RoundOrchestrator,
    TumorWorkUnit,
)
from onc_registry_pipeline.ingest.reader import Document
from onc_registry_pipeline.llm.client import LLMResponse, ModelProfile, VLLMClient
from onc_registry_pipeline.main import (
    _diagnosis_document_window,
    _documents_overlapping_window,
)
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


class FakeHTTPResponse:
    def __init__(
        self,
        data: dict,
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self._data = data

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        return None


class RecordingHTTPClient:
    def __init__(self, responses: list[FakeHTTPResponse]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    async def post(
        self,
        url: str,
        json: dict,  # noqa: A002
        headers: dict[str, str] | None = None,
    ) -> FakeHTTPResponse:
        self.requests.append({"method": "POST", "url": url, "json": json})
        return self.responses.pop(0)

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> FakeHTTPResponse:
        self.requests.append({"method": "GET", "url": url})
        return self.responses.pop(0)


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


def test_diagnosis_document_window_filters_exact_diagnosis_date() -> None:
    docs = [
        Document(0, "2023-07-14", "too early"),
        Document(1, "2023-07-15", "window start"),
        Document(2, "2024-01-15", "diagnosis"),
        Document(3, "2024-07-15", "window end"),
        Document(4, "2024-07-16", "too late"),
    ]

    window = _diagnosis_document_window("2024-01-15")

    assert window is not None
    selected = _documents_overlapping_window(docs, *window)
    assert [doc.text for doc in selected] == [
        "window start",
        "diagnosis",
        "window end",
    ]


def test_diagnosis_document_window_handles_month_precision() -> None:
    docs = [
        Document(0, "2023-06-30", "too early"),
        Document(1, "2023-07-01", "window start"),
        Document(2, "2024-01-31", "diagnosis month"),
        Document(3, "2024-07-31", "window end"),
        Document(4, "2024-08-01", "too late"),
    ]

    window = _diagnosis_document_window("2024-01")

    assert window is not None
    selected = _documents_overlapping_window(docs, *window)
    assert [doc.text for doc in selected] == [
        "window start",
        "diagnosis month",
        "window end",
    ]


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


def test_round_checkpoint_restores_partial_chunk_progress(tmp_path) -> None:
    tumor = TumorCandidate(
        tumor_index=0,
        cancer_type="ductal carcinoma",
        primary_site_hint="left breast",
        approximate_date="2024-01",
        evidence="left breast ductal carcinoma",
    )
    chunks = [DummyChunk("chunk_0", "one"), DummyChunk("chunk_1", "two")]
    result = ExtractionResult(
        item_number=400,
        item_name="Primary Site",
        extracted_value="C50.9",
        resolved_code="C50.9",
        confidence=0.95,
        evidence_text="left breast",
        source_chunk_id="chunk_0",
        source_chunk_type="sequential",
        pass_number=0,
    )
    work_units = [
        TumorWorkUnit(
            patient_id="patient-1",
            tumor_index=0,
            tumor=tumor,
            chunks=chunks,
            current_extraction={400: result},
            completed_chunks={0},
        ),
        TumorWorkUnit(
            patient_id="patient-2",
            tumor_index=0,
            tumor=tumor,
            chunks=chunks,
            completed_chunks=set(),
        ),
    ]
    orchestrator = RoundOrchestrator(
        config=PipelineConfig(),
        dictionary=None,  # type: ignore[arg-type]
        code_resolver=None,  # type: ignore[arg-type]
        llm_client=None,  # type: ignore[arg-type]
        schema_builder=None,  # type: ignore[arg-type]
        schema_registry=None,  # type: ignore[arg-type]
    )
    rounds = orchestrator.prepare_rounds(work_units)

    orchestrator._save_checkpoint(0, work_units, tmp_path, rounds)

    restored_work_units = [
        TumorWorkUnit("patient-1", 0, tumor, chunks),
        TumorWorkUnit("patient-2", 0, tumor, chunks),
    ]
    restored_rounds = orchestrator.prepare_rounds(restored_work_units)
    resume_round = orchestrator._load_checkpoints(
        restored_work_units,
        tmp_path,
        restored_rounds,
    )

    assert resume_round == 0
    assert restored_work_units[0].completed_chunks == {0}
    assert restored_work_units[0].current_extraction[400].resolved_code == "C50.9"
    assert restored_work_units[1].completed_chunks == set()


async def test_completion_rate_limit_retries_with_retry_after(monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    http_client = RecordingHTTPClient(
        [
            FakeHTTPResponse(
                {},
                status_code=429,
                text="rate limited",
                headers={"retry-after": "2"},
            ),
            FakeHTTPResponse(
                {"choices": [{"message": {"content": "{\"ok\": true}"}}]},
            ),
        ]
    )
    client = VLLMClient(max_retries=2)
    client._client = http_client  # type: ignore[assignment]
    client._model_profile = ModelProfile(
        model_name="Llama-3.3-70B",
        context_window=131_072,
        model_size_class="large",
    )

    response = await client.extract("system", "user")

    assert response.parsed == {"ok": True}
    assert sleeps == [2.0]
    assert [request["method"] for request in http_client.requests] == ["POST", "POST"]


async def test_model_discovery_rate_limit_retries_with_backoff(monkeypatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    http_client = RecordingHTTPClient(
        [
            FakeHTTPResponse({}, status_code=429, text="rate limited"),
            FakeHTTPResponse({"data": [{"id": "Llama-3.3-70B"}]}),
        ]
    )
    client = VLLMClient(max_retries=2)
    client._client = http_client  # type: ignore[assignment]

    response = await client._get_models_with_auth_retry()

    assert response == {"data": [{"id": "Llama-3.3-70B"}]}
    assert sleeps == [1.0]
    assert [request["method"] for request in http_client.requests] == ["GET", "GET"]
