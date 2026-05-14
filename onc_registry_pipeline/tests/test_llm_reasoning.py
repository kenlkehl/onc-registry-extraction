from __future__ import annotations

import pytest

from onc_registry_pipeline.llm.client import ModelProfile, VLLMClient
from onc_registry_pipeline.llm.reasoning import (
    VLLMReasoningOutputProcessor,
    extract_server_reasoning,
    infer_reasoning_parser,
    message_content_to_text,
    resolve_reasoning_parser,
)


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("google/gemma-4-31b-it", "gemma4"),
        ("Gemma4-31B", "gemma4"),
        ("Qwen/Qwen3.6-27B", "qwen3"),
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", "qwen3"),
        ("openai/gpt-oss-120b", "openai_gptoss"),
        ("GPTOSS-120B", "openai_gptoss"),
        ("meta-llama/Llama-3.3-70B-Instruct", None),
    ],
)
def test_infer_reasoning_parser_defaults(model_name: str, expected: str | None) -> None:
    assert infer_reasoning_parser(model_name) == expected


def test_resolve_reasoning_parser_handles_overrides() -> None:
    assert resolve_reasoning_parser("Qwen/Qwen3.6-27B", "auto") == "qwen3"
    assert resolve_reasoning_parser("Qwen/Qwen3.6-27B", "none") is None
    assert resolve_reasoning_parser("Qwen/Qwen3.6-27B", "") is None
    assert resolve_reasoning_parser("Qwen/Qwen3.6-27B", None) is None
    assert resolve_reasoning_parser("Llama-3.3-70B", "gpt-oss") == "openai_gptoss"


def test_extract_server_reasoning_supports_vllm_field_names() -> None:
    assert extract_server_reasoning({"reasoning": "parsed"}) == "parsed"
    assert extract_server_reasoning({"reasoning_content": "legacy"}) == "legacy"
    assert extract_server_reasoning({"content": "{}"}) is None


def test_message_content_to_text_handles_text_parts() -> None:
    content = [
        {"type": "text", "text": '{"a": '},
        {"type": "text", "text": "1}"},
    ]
    assert message_content_to_text(content) == '{"a": 1}'


class FakeReasoningParser:
    def extract_reasoning(self, text: str, request) -> tuple[str, str]:  # noqa: ANN001
        assert request is None
        assert text == "<think>work</think>{\"ok\": true}"
        return "work", "{\"ok\": true}"


def test_output_processor_uses_vllm_parser_contract() -> None:
    processor = VLLMReasoningOutputProcessor(
        "qwen3",
        parser=FakeReasoningParser(),
    )

    split = processor.split("<think>work</think>{\"ok\": true}")

    assert split.reasoning == "work"
    assert split.final_content == "{\"ok\": true}"


def test_output_processor_leaves_plain_content_alone() -> None:
    processor = VLLMReasoningOutputProcessor(
        "qwen3",
        parser=FakeReasoningParser(),
    )

    split = processor.split("{\"ok\": true}")

    assert split.reasoning == ""
    assert split.final_content == "{\"ok\": true}"


class FakeHTTPResponse:
    def __init__(self, data: dict, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = text
        self._data = data

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        return None


class FakeHTTPClient:
    def __init__(self, data: dict) -> None:
        self.data = data

    async def post(  # noqa: A002
        self,
        url: str,
        json: dict,
        headers: dict[str, str] | None = None,
    ) -> FakeHTTPResponse:
        return FakeHTTPResponse(self.data)


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
        self.requests.append({"url": url, "json": json, "headers": headers or {}})
        return self.responses.pop(0)


async def test_client_uses_vllm_server_reasoning_field() -> None:
    client = VLLMClient()
    client._client = FakeHTTPClient(  # type: ignore[assignment]
        {
            "choices": [
                {
                    "message": {
                        "reasoning": "server parsed reasoning",
                        "content": "{\"ok\": true}",
                    }
                }
            ]
        }
    )
    client._model_profile = ModelProfile(
        model_name="Qwen/Qwen3.6-27B",
        context_window=131_072,
        model_size_class="medium",
        reasoning_parser="qwen3",
    )

    response = await client._post_completion({}, attempt=1)

    assert response.reasoning == "server parsed reasoning"
    assert response.final_content == "{\"ok\": true}"
    assert response.parsed == {"ok": True}
    assert "server parsed reasoning" in response.raw_content


async def test_client_uses_local_vllm_parser_adapter_when_needed() -> None:
    client = VLLMClient(reasoning_parser="qwen3")
    client._client = FakeHTTPClient(  # type: ignore[assignment]
        {
            "choices": [
                {"message": {"content": "<think>work</think>{\"ok\": true}"}}
            ]
        }
    )
    client._model_profile = ModelProfile(
        model_name="Qwen/Qwen3.6-27B",
        context_window=131_072,
        model_size_class="medium",
        reasoning_parser="qwen3",
    )
    client._reasoning_output_processor = VLLMReasoningOutputProcessor(
        "qwen3",
        parser=FakeReasoningParser(),
    )

    response = await client._post_completion({}, attempt=1)

    assert response.reasoning == "work"
    assert response.final_content == "{\"ok\": true}"
    assert response.parsed == {"ok": True}


async def test_azure_client_refreshes_bearer_token_on_auth_failure(
    tmp_path,
    monkeypatch,
) -> None:
    token_script = tmp_path / "token.sh"
    token_script.write_text("#!/bin/sh\nprintf refreshed-token\n")
    token_script.chmod(0o700)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "expired-token")

    http_client = RecordingHTTPClient(
        [
            FakeHTTPResponse({}, status_code=401, text="expired"),
            FakeHTTPResponse(
                {"output_text": "{\"ok\": true}"},
            ),
        ]
    )
    client = VLLMClient(
        provider="azure-openai",
        base_url="https://example.openai.azure.com/openai/v1",
        model="gpt-4o",
        azure_token_refresh_command=(
            f'export AZURE_OPENAI_API_KEY="$({token_script})"'
        ),
    )
    client._client = http_client  # type: ignore[assignment]
    client._model_profile = ModelProfile(
        model_name="gpt-4o",
        context_window=131_072,
        model_size_class="medium",
        provider="azure-openai",
    )

    response = await client.extract("system", "user")

    assert response.parsed == {"ok": True}
    assert http_client.requests[0]["url"] == (
        "https://example.openai.azure.com/openai/v1/responses"
    )
    assert http_client.requests[0]["json"] == {
        "model": "gpt-4o",
        "instructions": "system",
        "input": "user",
        "max_output_tokens": 4096,
        "store": False,
        "temperature": 0.0,
    }
    assert http_client.requests[0]["headers"] == {
        "Authorization": "Bearer expired-token"
    }
    assert http_client.requests[1]["headers"] == {
        "Authorization": "Bearer refreshed-token"
    }
    assert client._auth_token == "refreshed-token"


def test_azure_base_url_normalizes_resource_endpoint() -> None:
    client = VLLMClient(
        provider="azure-openai",
        base_url="https://example.openai.azure.com",
        model="gpt-4o",
    )

    assert client._completion_url() == (
        "https://example.openai.azure.com/openai/v1/responses"
    )


def test_azure_responses_body_omits_temperature_for_reasoning_models() -> None:
    client = VLLMClient(
        provider="azure-openai",
        base_url="https://example.openai.azure.com/openai/v1",
        model="gpt-5",
    )

    body = client._build_completion_body("gpt-5", "system prompt", "user prompt")

    assert body == {
        "model": "gpt-5",
        "instructions": "system prompt",
        "input": "user prompt",
        "max_output_tokens": 4096,
        "store": False,
    }


async def test_azure_responses_output_array_is_parsed() -> None:
    client = VLLMClient(
        provider="azure-openai",
        base_url="https://example.openai.azure.com/openai/v1",
        model="gpt-5",
    )
    client._client = FakeHTTPClient(  # type: ignore[assignment]
        {
            "output": [
                {"type": "reasoning", "summary": [{"text": "brief summary"}]},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "{\"ok\": true}"}
                    ],
                },
            ]
        }
    )
    client._model_profile = ModelProfile(
        model_name="gpt-5",
        context_window=131_072,
        model_size_class="medium",
        provider="azure-openai",
    )

    response = await client._post_completion({}, attempt=1)

    assert response.reasoning == "brief summary"
    assert response.final_content == "{\"ok\": true}"
    assert response.parsed == {"ok": True}


def test_anthropic_vertex_builds_raw_predict_request() -> None:
    client = VLLMClient(
        provider="anthropic-vertex",
        model="claude-sonnet-4-5@20250929",
        anthropic_vertex_project_id="project-id",
        anthropic_vertex_region="global",
    )

    body = client._build_completion_body("ignored", "system prompt", "user prompt")

    assert client._completion_url() == (
        "https://aiplatform.googleapis.com/v1/projects/project-id/locations/global"
        "/publishers/anthropic/models/claude-sonnet-4-5%4020250929:rawPredict"
    )
    assert body["anthropic_version"] == "vertex-2023-10-16"
    assert body["system"] == "system prompt"
    assert body["messages"] == [{"role": "user", "content": "user prompt"}]
    assert body["stream"] is False


async def test_anthropic_vertex_response_content_is_parsed() -> None:
    client = VLLMClient(
        provider="anthropic-vertex",
        model="claude-sonnet-4-5@20250929",
        anthropic_vertex_project_id="project-id",
        anthropic_vertex_region="global",
    )
    client._client = FakeHTTPClient(  # type: ignore[assignment]
        {
            "content": [
                {"type": "text", "text": "{\"ok\":"},
                {"type": "text", "text": " true}"},
            ]
        }
    )
    client._model_profile = ModelProfile(
        model_name="claude-sonnet-4-5@20250929",
        context_window=200_000,
        model_size_class="medium",
        provider="anthropic-vertex",
    )

    response = await client._post_completion({}, attempt=1)

    assert response.final_content == "{\"ok\": true}"
    assert response.parsed == {"ok": True}
