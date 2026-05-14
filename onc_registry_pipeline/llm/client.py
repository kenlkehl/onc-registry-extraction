"""Fully async LLM client for registry extraction model endpoints.

Supports local vLLM, Azure OpenAI v1 chat completions, and Anthropic Claude
on Vertex AI.  Handles model discovery where available, auth-token refresh,
size classification, and retry logic for robust JSON extraction. JSON output
is requested via the prompt text (no guided_json). On parse failure, retries
with error feedback.
"""

import asyncio
import json
import logging
import os
import re
import shlex
from dataclasses import dataclass
from typing import Literal, Optional
from urllib.parse import quote

import httpx

from onc_registry_pipeline.llm.reasoning import (
    AUTO_REASONING_PARSER,
    VLLMReasoningOutputProcessor,
    extract_server_reasoning,
    message_content_to_text,
    resolve_reasoning_parser,
)

logger = logging.getLogger(__name__)

LLMProvider = Literal["vllm", "azure-openai", "anthropic-vertex"]
AzureAuthMode = Literal["bearer", "api-key"]

_DEFAULT_AZURE_TOKEN_REFRESH_COMMAND = (
    "az account get-access-token "
    "--resource=https://cognitiveservices.azure.com/ "
    "--query accessToken --output tsv"
)
_DEFAULT_VERTEX_TOKEN_REFRESH_COMMAND = (
    "gcloud auth application-default print-access-token"
)
_ANTHROPIC_VERTEX_VERSION = "vertex-2023-10-16"
_AUTH_ERROR_STATUSES = {401, 403}
_EXPORT_COMMAND_RE = re.compile(
    r"^\s*export\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*[\"']?\$\((.*)\)[\"']?\s*$",
    re.DOTALL,
)


@dataclass
class ModelProfile:
    """Describes the discovered model and its operational parameters."""

    model_name: str
    context_window: int
    model_size_class: str  # "small" (<15B), "medium" (15-40B), "large" (40B+)
    reasoning_parser: Optional[str] = None
    provider: LLMProvider = "vllm"


@dataclass
class LLMResponse:
    """Full LLM response preserving both reasoning and final output.

    Attributes
    ----------
    raw_content : str
        Best-effort complete model text.  When vLLM returns reasoning in
        a separate response field, this combines reasoning and final text
        for logging/audit visibility.
    final_content : str
        The text after vLLM reasoning parsing.  This is what gets parsed
        as JSON.
    parsed : dict
        The JSON-parsed result from *final_content*.
    reasoning : str
        The reasoning/thinking portion stripped from the output, or
        empty string if there was no reasoning block.
    """

    raw_content: str
    final_content: str
    parsed: dict
    reasoning: str


# ---------------------------------------------------------------------------
# Size-classification helpers
# ---------------------------------------------------------------------------

# Regex that captures a number immediately before 'b' (case-insensitive),
# e.g. "Qwen2.5-72B-Instruct" -> 72, "Meta-Llama-3.1-8B" -> 8
_PARAM_RE = re.compile(r"(\d+)[bB]")

_SIZE_THRESHOLDS: list[tuple[int, str]] = [
    # (upper_bound_exclusive, class_name)
    (15, "small"),
    (40, "medium"),
]


def _classify_model(name: str) -> str:
    """Return size_class inferred from model name."""
    match = _PARAM_RE.search(name)
    if not match:
        logger.info(
            "Cannot determine parameter count from model name '%s'; "
            "defaulting to medium.",
            name,
        )
        return "medium"

    params_b = int(match.group(1))
    for upper, cls in _SIZE_THRESHOLDS:
        if params_b < upper:
            return cls
    return "large"


# ---------------------------------------------------------------------------
# Custom retry predicate for HTTP 5xx responses
# ---------------------------------------------------------------------------

class _ServerError(Exception):
    """Raised when the endpoint returns an HTTP 5xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:300]}")


class _RateLimited(Exception):
    """Raised on HTTP 429 so tenacity can apply a longer back-off."""

    def __init__(self, retry_after: Optional[float] = None) -> None:
        self.retry_after = retry_after
        super().__init__("Rate limited (429)")


class _AuthenticationError(Exception):
    """Raised when endpoint credentials are missing, expired, or rejected."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Authentication failed (HTTP {status_code}): {body[:300]}")


class _JSONParseError(Exception):
    """Raised when the LLM output cannot be parsed as JSON."""

    def __init__(self, parse_error: str, malformed_output: str) -> None:
        self.parse_error = parse_error
        self.malformed_output = malformed_output
        super().__init__(f"JSON parse error: {parse_error}")


# ---------------------------------------------------------------------------
# VLLMClient
# ---------------------------------------------------------------------------

class VLLMClient:
    """Fully async client for model endpoints used by the pipeline.

    The class name is retained for compatibility with the existing extraction
    code, but the implementation can talk to local vLLM, Azure OpenAI, or
    Anthropic Claude on Vertex AI.

    Usage::

        client = VLLMClient()
        await client.initialize()
        result = await client.extract(system_prompt, user_prompt)
        await client.close()

    Or as an async context manager::

        async with VLLMClient() as client:
            await client.initialize()
            result = await client.extract(system, user)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        provider: LLMProvider = "vllm",
        model: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
        reasoning_parser: Optional[str] = AUTO_REASONING_PARSER,
        azure_api_key_env: str = "AZURE_OPENAI_API_KEY",
        azure_auth_mode: AzureAuthMode = "bearer",
        azure_token_refresh_command: Optional[str] = (
            _DEFAULT_AZURE_TOKEN_REFRESH_COMMAND
        ),
        anthropic_vertex_project_id: Optional[str] = None,
        anthropic_vertex_region: Optional[str] = None,
        anthropic_vertex_token_env: str = "ANTHROPIC_VERTEX_ACCESS_TOKEN",
        anthropic_vertex_token_refresh_command: Optional[str] = (
            _DEFAULT_VERTEX_TOKEN_REFRESH_COMMAND
        ),
    ) -> None:
        self._provider = provider
        self._configured_model = model
        self._base_url = self._normalize_base_url(base_url, provider)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._configured_reasoning_parser = reasoning_parser
        self._azure_api_key_env = azure_api_key_env
        self._azure_auth_mode = azure_auth_mode
        self._azure_token_refresh_command = azure_token_refresh_command
        self._anthropic_vertex_project_id = anthropic_vertex_project_id
        self._anthropic_vertex_region = anthropic_vertex_region
        self._anthropic_vertex_token_env = anthropic_vertex_token_env
        self._anthropic_vertex_token_refresh_command = (
            anthropic_vertex_token_refresh_command
        )
        self._auth_token: Optional[str] = None
        self._auth_refresh_lock = asyncio.Lock()
        self._reasoning_output_processor = VLLMReasoningOutputProcessor(None)
        self._client: Optional[httpx.AsyncClient] = None
        self._model_profile: Optional[ModelProfile] = None

    # -- async context manager support --

    async def __aenter__(self) -> "VLLMClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        await self.close()

    # -- lifecycle --

    async def initialize(self) -> ModelProfile:
        """Discover or configure the model and operational parameters.

        Steps
        -----
        1. Create the ``httpx.AsyncClient``.
        2. Configure auth for cloud providers.
        3. Discover model metadata when the provider exposes ``/models``.
        4. Infer model size from the name (e.g. ``72b`` -> large).
        5. Set the context window.
        """
        self._client = httpx.AsyncClient(timeout=self._timeout)
        await self._ensure_auth_token()

        model_name, context_window = await self._resolve_model_profile()
        size_class = _classify_model(model_name)
        reasoning_parser = None
        if self._provider == "vllm":
            reasoning_parser = resolve_reasoning_parser(
                model_name,
                self._configured_reasoning_parser,
            )
        self._reasoning_output_processor = VLLMReasoningOutputProcessor(
            reasoning_parser,
        )
        logger.info(
            "Model profile: provider=%s, model=%s, size_class=%s, "
            "context_window=%d, reasoning_parser=%s, local_reasoning_parser=%s",
            self._provider,
            model_name,
            size_class,
            context_window,
            reasoning_parser or "none",
            self._reasoning_output_processor.local_parser_available,
        )

        self._model_profile = ModelProfile(
            model_name=model_name,
            context_window=context_window,
            model_size_class=size_class,
            reasoning_parser=reasoning_parser,
            provider=self._provider,
        )
        return self._model_profile

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # -- properties --

    @property
    def model_profile(self) -> ModelProfile:
        """Return the discovered model profile.

        Raises ``RuntimeError`` if ``initialize()`` has not been called.
        """
        if self._model_profile is None:
            raise RuntimeError(
                "Client not initialized. Call initialize() first."
            )
        return self._model_profile

    # -- extraction ------------------------------------------------------

    async def extract(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Send a chat completion request and return the full LLM response.

        JSON output is requested via the prompt text (no guided_json).
        On JSON parse failure, the prompt is re-submitted with the error
        and malformed output appended so the LLM can self-correct.

        Parameters
        ----------
        system_prompt:
            The system-level instruction (should include JSON format
            instructions).
        user_prompt:
            The user-level prompt (clinical text + item list + valid codes).

        Returns
        -------
        LLMResponse
            Contains the best-effort raw output, the final output after
            vLLM reasoning parsing, the parsed JSON, and the reasoning
            text separately.  On unrecoverable failure the ``parsed`` dict
            contains ``{"_error": True, "_message": "..."}``.
        """
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Call initialize() first."
            )

        profile = self.model_profile

        body = self._build_completion_body(
            profile.model_name,
            system_prompt,
            user_prompt,
        )

        last_error: Optional[str] = None
        last_malformed_output: str = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                llm_resp = await self._post_completion(body, attempt)
                return llm_resp
            except _RateLimited as exc:
                wait = exc.retry_after or (2 ** attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d). Waiting %.1fs.",
                    attempt,
                    self._max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
                last_error = str(exc)
            except _ServerError as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Server error %d (attempt %d/%d). Waiting %.1fs.",
                    exc.status_code,
                    attempt,
                    self._max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
                last_error = str(exc)
            except _AuthenticationError as exc:
                logger.warning(
                    "Authentication failure (HTTP %d, attempt %d/%d). "
                    "Refreshing credentials before retry.",
                    exc.status_code,
                    attempt,
                    self._max_retries,
                )
                stale_token = self._auth_token or self._read_auth_token_from_env()
                refreshed = await self._refresh_auth_token(stale_token=stale_token)
                if refreshed:
                    last_error = str(exc)
                    continue
                last_error = str(exc)
                break
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Connection issue (attempt %d/%d): %s. Waiting %.1fs.",
                    attempt,
                    self._max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
                last_error = str(exc)
            except _JSONParseError as exc:
                logger.warning(
                    "JSON parse failure (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    exc.parse_error,
                )
                last_malformed_output = exc.malformed_output
                # Re-submit with the error and malformed output so the
                # LLM can self-correct.
                error_feedback = (
                    f"\n\nYour previous response was not valid JSON. "
                    f"Error: {exc.parse_error}\n"
                    f"Your response was:\n{last_malformed_output[:2000]}\n\n"
                    f"Please respond with ONLY valid JSON. "
                    f"No markdown fences, no commentary."
                )
                body = self._build_completion_body(
                    profile.model_name,
                    system_prompt,
                    user_prompt + error_feedback,
                )
                last_error = f"JSON parse error: {exc.parse_error}"

        logger.error(
            "All %d attempts exhausted. Last error: %s",
            self._max_retries,
            last_error,
        )
        return LLMResponse(
            raw_content="",
            final_content="",
            parsed={"_error": True, "_message": last_error or "unknown error"},
            reasoning="",
        )

    async def extract_batch(
        self,
        requests: list[tuple[str, str]],
    ) -> list[LLMResponse]:
        """Dispatch multiple extract() calls concurrently.

        Parameters
        ----------
        requests:
            Each element is ``(system_prompt, user_prompt)``.

        Returns
        -------
        list[LLMResponse]
            Results in the same order as the input requests.
        """
        tasks = [
            self.extract(sys_p, usr_p)
            for sys_p, usr_p in requests
        ]
        return list(await asyncio.gather(*tasks))

    # -- internal helpers ------------------------------------------------

    async def _post_completion(self, body: dict, attempt: int) -> LLMResponse:
        """POST a model request and return the full LLM response.

        Raises
        ------
        _ServerError
            On HTTP 5xx responses.
        _RateLimited
            On HTTP 429 responses.
        httpx.TimeoutException
            On request timeout.
        httpx.ConnectError
            On connection failure.
        _JSONParseError
            When the final content (after reasoning strip) is not valid JSON.
        """
        assert self._client is not None  # noqa: S101

        resp = await self._post_json(self._completion_url(), body)

        # Handle error status codes before reading JSON body.
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            raise _RateLimited(
                retry_after=float(retry_after) if retry_after else None
            )
        if resp.status_code in _AUTH_ERROR_STATUSES:
            stale_token = self._auth_token or self._read_auth_token_from_env()
            refreshed = await self._refresh_auth_token(stale_token=stale_token)
            if refreshed:
                resp = await self._post_json(self._completion_url(), body)
            if resp.status_code in _AUTH_ERROR_STATUSES:
                raise _AuthenticationError(resp.status_code, resp.text)
        if resp.status_code >= 500:
            raise _ServerError(resp.status_code, resp.text)
        resp.raise_for_status()

        # Parse the API response envelope.
        data = resp.json()
        raw_content, final_content, reasoning = self._parse_completion_data(data)

        # Strip markdown code fences from the final output.
        final_content = _strip_code_fences(final_content)

        try:
            parsed = json.loads(final_content)
            parsed = _coerce_to_dict(parsed)
        except json.JSONDecodeError as exc:
            raise _JSONParseError(str(exc), final_content) from exc

        logger.debug(
            "Attempt %d succeeded. Keys: %s (reasoning: %d chars)",
            attempt,
            list(parsed.keys()),
            len(reasoning),
        )
        return LLMResponse(
            raw_content=raw_content,
            final_content=final_content,
            parsed=parsed,
            reasoning=reasoning,
        )

    async def _resolve_model_profile(self) -> tuple[str, int]:
        if self._provider == "anthropic-vertex":
            model_name = self._require_configured_model("Anthropic Vertex")
            return model_name, _infer_anthropic_vertex_context_window(model_name)

        if self._provider == "azure-openai" and self._configured_model != "auto":
            return self._configured_model, 131_072

        models_data = await self._get_models_with_auth_retry()
        model_entries = models_data.get("data", [])
        if not model_entries:
            raise RuntimeError(
                f"No models found for provider {self._provider}. "
                f"GET {self._base_url}/models returned: {models_data}"
            )

        model_info = model_entries[0]
        model_name = model_info["id"]
        if self._provider == "vllm":
            logger.info("Discovered vLLM model: %s", model_name)
        else:
            logger.info("Discovered model: %s", model_name)

        context_window = (
            model_info.get("max_model_len")
            or model_info.get("context_length")
            or 131_072
        )
        return model_name, int(context_window)

    async def _get_models_with_auth_retry(self) -> dict:
        try:
            resp = await self._get_json(f"{self._base_url}/models")
        except _AuthenticationError:
            stale_token = self._auth_token or self._read_auth_token_from_env()
            refreshed = await self._refresh_auth_token(stale_token=stale_token)
            if not refreshed:
                raise
            resp = await self._get_json(f"{self._base_url}/models")

        if resp.status_code == 429:
            raise _RateLimited()
        if resp.status_code >= 500:
            raise _ServerError(resp.status_code, resp.text)
        resp.raise_for_status()
        return resp.json()

    def _build_completion_body(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        if self._provider == "anthropic-vertex":
            return {
                "anthropic_version": _ANTHROPIC_VERTEX_VERSION,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "stream": False,
            }

        return {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    def _completion_url(self) -> str:
        if self._provider == "anthropic-vertex":
            project_id = self._anthropic_vertex_project_id
            region = self._anthropic_vertex_region
            if not project_id or not region:
                raise RuntimeError(
                    "Anthropic Vertex requires ANTHROPIC_VERTEX_PROJECT_ID "
                    "and CLOUD_ML_REGION (or matching CLI options)."
                )
            model = quote(self._require_configured_model("Anthropic Vertex"), safe="")
            host = _anthropic_vertex_host(region)
            return (
                f"https://{host}/v1/projects/{project_id}/locations/{region}"
                f"/publishers/anthropic/models/{model}:rawPredict"
            )

        return f"{self._base_url}/chat/completions"

    def _parse_completion_data(self, data: dict) -> tuple[str, str, str]:
        if self._provider == "anthropic-vertex":
            return _parse_anthropic_message(data)

        message = data["choices"][0]["message"]
        final_content = message_content_to_text(message.get("content"))

        server_reasoning = extract_server_reasoning(message)
        if server_reasoning is not None:
            reasoning = server_reasoning.strip()
            raw_content = _combine_reasoning_and_content(reasoning, final_content)
        else:
            raw_content = final_content
            split = self._reasoning_output_processor.split(raw_content)
            reasoning = split.reasoning
            final_content = split.final_content

        return raw_content, final_content, reasoning

    async def _post_json(self, url: str, body: dict) -> httpx.Response:
        assert self._client is not None  # noqa: S101
        headers = self._auth_headers()
        if headers:
            return await self._client.post(url, json=body, headers=headers)
        return await self._client.post(url, json=body)

    async def _get_json(self, url: str) -> httpx.Response:
        assert self._client is not None  # noqa: S101
        headers = self._auth_headers()
        if headers:
            resp = await self._client.get(url, headers=headers)
        else:
            resp = await self._client.get(url)
        if resp.status_code in _AUTH_ERROR_STATUSES:
            raise _AuthenticationError(resp.status_code, resp.text)
        return resp

    async def _ensure_auth_token(self) -> None:
        if self._provider == "vllm":
            return

        token = self._read_auth_token_from_env()
        if token:
            self._auth_token = token
            return

        refreshed = await self._refresh_auth_token()
        if not refreshed:
            env_name = (
                self._azure_api_key_env
                if self._provider == "azure-openai"
                else self._anthropic_vertex_token_env
            )
            raise RuntimeError(
                f"{self._provider} requires credentials in ${env_name} "
                "or a configured token refresh command."
            )

    async def _refresh_auth_token(self, stale_token: Optional[str] = None) -> bool:
        command = self._token_refresh_command()
        if not command:
            return False

        async with self._auth_refresh_lock:
            current_token = self._auth_token or self._read_auth_token_from_env()
            if (
                stale_token is not None
                and current_token
                and current_token != stale_token
            ):
                self._auth_token = current_token
                return True

            token = await _run_token_command(command)
            self._auth_token = token
            env_name = (
                self._azure_api_key_env
                if self._provider == "azure-openai"
                else self._anthropic_vertex_token_env
            )
            os.environ[env_name] = token
            logger.info("Refreshed %s credentials into $%s", self._provider, env_name)
            return True

    def _read_auth_token_from_env(self) -> Optional[str]:
        if self._provider == "azure-openai":
            return os.getenv(self._azure_api_key_env)
        if self._provider == "anthropic-vertex":
            return os.getenv(self._anthropic_vertex_token_env)
        return None

    def _token_refresh_command(self) -> Optional[str]:
        if self._provider == "azure-openai" and self._azure_auth_mode == "bearer":
            return self._azure_token_refresh_command
        if self._provider == "anthropic-vertex":
            return self._anthropic_vertex_token_refresh_command
        return None

    def _auth_headers(self) -> dict[str, str]:
        if self._provider == "vllm":
            return {}

        token = self._auth_token or self._read_auth_token_from_env()
        if not token:
            return {}

        if self._provider == "azure-openai" and self._azure_auth_mode == "api-key":
            return {"api-key": token}

        return {"Authorization": f"Bearer {token}"}

    def _require_configured_model(self, provider_label: str) -> str:
        if self._configured_model and self._configured_model != "auto":
            return self._configured_model
        raise RuntimeError(
            f"{provider_label} requires --model or a provider model env var."
        )

    @staticmethod
    def _normalize_base_url(base_url: str, provider: LLMProvider) -> str:
        normalized = (base_url or "").rstrip("/")
        if provider == "azure-openai":
            if not normalized:
                return normalized
            if normalized.endswith("/openai/v1"):
                return normalized
            if "/openai/" in normalized:
                return normalized
            return f"{normalized}/openai/v1"
        return normalized


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?\s*```$",
    re.DOTALL,
)


def _combine_reasoning_and_content(reasoning: str, content: str) -> str:
    """Combine separately parsed response fields for legacy raw-output logs."""
    if reasoning and content:
        return f"{reasoning}\n{content}"
    if reasoning:
        return reasoning
    return content


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) wrapping JSON output."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def _coerce_to_dict(parsed: object) -> dict:
    """Ensure parsed JSON is a dict.

    Some models return a JSON array instead of an object.  Common cases:
    - ``[{...}]`` — single-element list wrapping the real result → unwrap.
    - ``[{...}, {...}]`` — fields split across elements → merge.

    Raises ``json.JSONDecodeError`` for anything else so the retry loop
    can nudge the model toward a proper object.
    """
    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        logger.debug("Unwrapping single-element JSON array to dict.")
        return parsed[0]

    raise json.JSONDecodeError(
        f"Expected JSON object, got {type(parsed).__name__}",
        str(parsed)[:200],
        0,
    )


def _parse_anthropic_message(data: dict) -> tuple[str, str, str]:
    """Return raw text, final text, and reasoning from an Anthropic response."""
    content = data.get("content")
    if not isinstance(content, list):
        final_content = message_content_to_text(content)
        return final_content, final_content, ""

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue

        part_type = part.get("type")
        if part_type == "text":
            text_parts.append(str(part.get("text", "")))
        elif part_type in {"thinking", "redacted_thinking"}:
            text = part.get("thinking") or part.get("text") or part.get("content")
            if text is not None:
                reasoning_parts.append(str(text))
        else:
            text = part.get("text") or part.get("content")
            if text is not None:
                text_parts.append(str(text))

    final_content = "".join(text_parts)
    reasoning = "\n".join(reasoning_parts).strip()
    raw_content = _combine_reasoning_and_content(reasoning, final_content)
    return raw_content, final_content, reasoning


def _anthropic_vertex_host(region: str) -> str:
    """Return the Vertex AI host for global, multi-region, or regional Claude."""
    normalized = region.strip().lower()
    if normalized == "global":
        return "aiplatform.googleapis.com"
    if normalized == "us":
        return "aiplatform.us.rep.googleapis.com"
    if normalized == "eu":
        return "aiplatform.eu.rep.googleapis.com"
    return f"{normalized}-aiplatform.googleapis.com"


def _infer_anthropic_vertex_context_window(model_name: str) -> int:
    """Infer a conservative context window for Claude on Vertex AI."""
    normalized = model_name.lower()
    one_million_context_models = (
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    )
    if any(model in normalized for model in one_million_context_models):
        return 1_000_000
    return 200_000


async def _run_token_command(command: str) -> str:
    """Run a token command and return stdout as the credential value."""
    normalized = _normalize_token_command(command)
    argv = shlex.split(normalized)
    if not argv:
        raise RuntimeError("Token refresh command is empty.")

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Token refresh command failed with exit code {proc.returncode}: "
            f"{message}"
        )

    token = stdout.decode("utf-8", errors="replace").strip()
    if not token:
        raise RuntimeError("Token refresh command produced no output.")
    return token


def _normalize_token_command(command: str) -> str:
    """Accept either a token command or a shell-style export wrapper."""
    match = _EXPORT_COMMAND_RE.match(command)
    if match:
        return match.group(1).strip()
    return command.strip()
