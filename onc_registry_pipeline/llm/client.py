"""Fully async LLM client for a local vLLM server.

Handles model discovery, size classification, and retry logic for
robust JSON extraction. JSON output is requested via the prompt text
(no guided_json). On parse failure, retries with error feedback.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Describes the discovered model and its operational parameters."""

    model_name: str
    context_window: int
    model_size_class: str  # "small" (<15B), "medium" (15-40B), "large" (40B+)


@dataclass
class LLMResponse:
    """Full LLM response preserving both reasoning and final output.

    Attributes
    ----------
    raw_content : str
        The complete text returned by the model, including any
        chain-of-thought / reasoning tokens.
    final_content : str
        The text *after* stripping reasoning tokens (``</think>``,
        ``assistantfinal``, etc.).  This is what gets parsed as JSON.
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
    """Raised when the vLLM server returns an HTTP 5xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:300]}")


class _RateLimited(Exception):
    """Raised on HTTP 429 so tenacity can apply a longer back-off."""

    def __init__(self, retry_after: Optional[float] = None) -> None:
        self.retry_after = retry_after
        super().__init__("Rate limited (429)")


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
    """Fully async client for vLLM's OpenAI-compatible API.

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
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._model_profile: Optional[ModelProfile] = None

    # -- async context manager support --

    async def __aenter__(self) -> "VLLMClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        await self.close()

    # -- lifecycle --

    async def initialize(self) -> ModelProfile:
        """Discover the served model and configure operational parameters.

        Steps
        -----
        1. Create the ``httpx.AsyncClient``.
        2. ``GET /v1/models`` to obtain the model id.
        3. Infer model size from the name (e.g. ``72b`` -> large).
        4. Set the context window (default 131 072 if the server does not
           report ``max_model_len``).
        5. Classify model size for logging.
        """
        self._client = httpx.AsyncClient(timeout=self._timeout)

        # 1. Discover model name -----------------------------------------
        resp = await self._client.get(f"{self._base_url}/models")
        resp.raise_for_status()
        models_data = resp.json()

        model_entries = models_data.get("data", [])
        if not model_entries:
            raise RuntimeError(
                "No models found on vLLM server. "
                f"GET {self._base_url}/models returned: {models_data}"
            )

        model_info = model_entries[0]
        model_name: str = model_info["id"]
        logger.info("Discovered vLLM model: %s", model_name)

        # 2. Context window ----------------------------------------------
        # vLLM may expose max_model_len in the model info object.
        context_window: int = (
            model_info.get("max_model_len")
            or model_info.get("context_length")
            or 131_072
        )

        # 3. Size classification ------------------------------------------
        size_class = _classify_model(model_name)
        logger.info(
            "Model profile: size_class=%s, context_window=%d",
            size_class,
            context_window,
        )

        self._model_profile = ModelProfile(
            model_name=model_name,
            context_window=context_window,
            model_size_class=size_class,
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
            Contains the raw output (with reasoning), the final output
            (after stripping reasoning tokens), the parsed JSON, and the
            reasoning text separately.  On unrecoverable failure the
            ``parsed`` dict contains ``{"_error": True, "_message": "..."}``.
        """
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Call initialize() first."
            )

        profile = self.model_profile

        body: dict = {
            "model": profile.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

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
                body["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + error_feedback},
                ]
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
        """POST to /chat/completions and return full LLM response.

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

        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=body,
        )

        # Handle error status codes before reading JSON body.
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            raise _RateLimited(
                retry_after=float(retry_after) if retry_after else None
            )
        if resp.status_code >= 500:
            raise _ServerError(resp.status_code, resp.text)
        resp.raise_for_status()

        # Parse the API response envelope.
        data = resp.json()
        raw_content: str = data["choices"][0]["message"]["content"]

        # Split reasoning from final output.
        reasoning, final_content = _split_reasoning(raw_content)

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# Patterns that mark the transition from reasoning/thinking to final output.
# Order matters: we try each pattern and take the first match.
_REASONING_END_PATTERNS = [
    # DeepSeek-R1, QwQ, and other models using <think>...</think>
    re.compile(r"</think>\s*", re.IGNORECASE),
    # Some models emit this token to signal final output
    re.compile(r"assistantfinal\s*", re.IGNORECASE),
    # Variant spelling
    re.compile(r"assistant_final\s*", re.IGNORECASE),
    # <output> ... </output> wrappers (some fine-tunes)
    re.compile(r"</reasoning>\s*", re.IGNORECASE),
    # Generic [END_THINKING] marker
    re.compile(r"\[END[_\s]?THINKING\]\s*", re.IGNORECASE),
]


def _split_reasoning(text: str) -> tuple[str, str]:
    """Split raw LLM output into (reasoning, final_output).

    Many reasoning/thinking models emit a chain-of-thought block followed
    by a delimiter token, then the structured answer.  This function
    detects common delimiter patterns and splits accordingly.

    Returns
    -------
    tuple[str, str]
        ``(reasoning, final_output)``.  If no reasoning delimiter is
        found, *reasoning* is empty and *final_output* is the full text.
    """
    for pattern in _REASONING_END_PATTERNS:
        m = pattern.search(text)
        if m:
            reasoning = text[: m.start()].strip()
            final = text[m.end() :].strip()
            # Guard against empty final (model might end with the token)
            if final:
                return reasoning, final
            # If the "final" part is empty, the whole text is probably
            # the answer with a stray token -- fall through to next pattern.

    # No reasoning delimiter found.  Check for <think> opening tag without
    # a closing tag (model may have been cut off).  In that case, try to
    # find JSON after the thinking content.
    think_open = re.search(r"<think>", text, re.IGNORECASE)
    if think_open:
        # Look for the first '{' after the <think> tag as a heuristic
        # for where JSON output begins (in case the closing tag was omitted).
        brace = text.rfind("{")
        if brace > think_open.end():
            reasoning = text[: brace].strip()
            final = text[brace:].strip()
            return reasoning, final

    return "", text.strip()


_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?\s*```$",
    re.DOTALL,
)


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
