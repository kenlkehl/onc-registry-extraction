"""Helpers for vLLM reasoning-output handling.

vLLM owns the model-specific reasoning parsers.  The OpenAI-compatible
server normally applies them and returns parsed reasoning in the response
message.  This module keeps the client-side policy small: choose the parser
name for a model, consume server-parsed reasoning, and optionally use vLLM's
parser classes locally when the package is installed in the client process.
"""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

AUTO_REASONING_PARSER = "auto"
NO_REASONING_PARSER_VALUES = {"", "none", "off", "false", "no", "disabled"}

_PARSER_ALIASES = {
    "gptoss": "openai_gptoss",
    "gpt_oss": "openai_gptoss",
    "openai_gpt_oss": "openai_gptoss",
    "openai_gptoss": "openai_gptoss",
    "qwen_3": "qwen3",
    "gemma_4": "gemma4",
}

_TOKENIZER_VOCABS = {
    "qwen3": {
        "<think>": 1,
        "</think>": 2,
        "<tool_call>": 3,
        "</tool_call>": 4,
    },
    "gemma4": {
        "<|channel>": 1,
        "<channel|>": 2,
        "<|turn>": 3,
        "<|tool_call>": 4,
        "<|tool_response>": 5,
    },
}

_REASONING_MARKERS = {
    "qwen3": ("<think>", "</think>", "<tool_call>"),
    "gemma4": ("<|channel>", "<channel|>"),
    "openai_gptoss": ("<|channel|>analysis", "<|channel|>final"),
}


@dataclass(frozen=True)
class ReasoningSplit:
    """Reasoning/content pair returned by a parser."""

    reasoning: str
    final_content: str


def infer_reasoning_parser(model_name: str) -> Optional[str]:
    """Infer the vLLM reasoning parser name for known model families."""
    compact_name = re.sub(r"[^a-z0-9]", "", model_name.lower())

    if "gptoss" in compact_name:
        return "openai_gptoss"
    if "gemma4" in compact_name:
        return "gemma4"
    if "qwen3" in compact_name:
        # Covers Qwen3, Qwen3.5, and Qwen3.6 style model ids.
        return "qwen3"

    return None


def resolve_reasoning_parser(
    model_name: str,
    configured_parser: Optional[str] = AUTO_REASONING_PARSER,
) -> Optional[str]:
    """Resolve a configured parser setting to a vLLM parser name.

    ``auto`` uses conservative model-name defaults.  Values like ``none`` and
    ``off`` disable reasoning parsing.  Any other value is treated as an
    explicit vLLM parser name after alias normalization.
    """
    if configured_parser is None:
        return None

    raw_parser = configured_parser.strip()
    normalized = _normalize_parser_name(raw_parser)

    if normalized in NO_REASONING_PARSER_VALUES:
        return None
    if normalized == AUTO_REASONING_PARSER:
        return infer_reasoning_parser(model_name)

    return _PARSER_ALIASES.get(normalized, normalized)


def extract_server_reasoning(message: dict[str, Any]) -> Optional[str]:
    """Return reasoning already parsed by vLLM's OpenAI-compatible server."""
    for key in ("reasoning", "reasoning_content"):
        if key not in message:
            continue
        value = message[key]
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)
    return None


def message_content_to_text(content: Any) -> str:
    """Normalize an OpenAI message content payload to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text is None:
                    text = part.get("content")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


class VLLMReasoningOutputProcessor:
    """Split raw message text with vLLM parser classes when available."""

    def __init__(
        self,
        parser_name: Optional[str],
        parser: Optional[Any] = None,
    ) -> None:
        self.parser_name = parser_name
        self._parser = parser if parser is not None else _build_vllm_parser(parser_name)
        self._warned_unparsed_reasoning = False

    @property
    def local_parser_available(self) -> bool:
        """Whether client-side vLLM parser classes are available."""
        return self._parser is not None

    def split(self, text: str) -> ReasoningSplit:
        """Return ``(reasoning, content)`` for raw model text.

        If no local parser is available, the text is treated as final content.
        vLLM server-side reasoning fields are handled before this method is
        called, so this path is primarily for deployments that also install
        vLLM in the pipeline client environment.
        """
        stripped = text.strip()
        if not stripped:
            return ReasoningSplit(reasoning="", final_content="")

        if self._parser is None:
            self._warn_if_reasoning_looks_unparsed(stripped)
            return ReasoningSplit(reasoning="", final_content=stripped)

        if not _contains_reasoning_marker(self.parser_name, stripped):
            return ReasoningSplit(reasoning="", final_content=stripped)

        try:
            reasoning, final_content = self._parser.extract_reasoning(
                stripped,
                request=None,
            )
        except NotImplementedError:
            logger.debug(
                "vLLM parser '%s' does not support text-only non-streaming parsing.",
                self.parser_name,
            )
            self._warn_if_reasoning_looks_unparsed(stripped)
            return ReasoningSplit(reasoning="", final_content=stripped)
        except Exception:
            logger.exception(
                "vLLM parser '%s' failed while splitting reasoning output.",
                self.parser_name,
            )
            return ReasoningSplit(reasoning="", final_content=stripped)

        return ReasoningSplit(
            reasoning=(reasoning or "").strip(),
            final_content=(final_content or "").strip(),
        )

    def _warn_if_reasoning_looks_unparsed(self, text: str) -> None:
        if self._warned_unparsed_reasoning or not self.parser_name:
            return
        if not _contains_reasoning_marker(self.parser_name, text):
            return

        logger.warning(
            "vLLM did not return parsed reasoning and no local text parser is "
            "available for '%s'. Start vLLM with --enable-reasoning "
            "--reasoning-parser %s, or install vLLM in the pipeline "
            "environment for local fallback parsing.",
            self.parser_name,
            self.parser_name,
        )
        self._warned_unparsed_reasoning = True


def _normalize_parser_name(parser_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", parser_name.lower()).strip("_")


def _contains_reasoning_marker(parser_name: Optional[str], text: str) -> bool:
    if parser_name is None:
        return False
    markers = _REASONING_MARKERS.get(parser_name, ())
    return any(marker in text for marker in markers)


def _build_vllm_parser(parser_name: Optional[str]) -> Optional[Any]:
    if not parser_name:
        return None

    if parser_name == "openai_gptoss":
        # vLLM's gpt-oss non-streaming parser uses token ids inside the server.
        # The HTTP client only has text unless the server already parsed it.
        logger.debug(
            "Skipping local parser for '%s'; use vLLM server-side parsing.",
            parser_name,
        )
        return None

    vocab = _TOKENIZER_VOCABS.get(parser_name)
    if vocab is None:
        logger.debug(
            "No text-only tokenizer stub is registered for vLLM parser '%s'.",
            parser_name,
        )
        return None

    try:
        reasoning_module = importlib.import_module("vllm.reasoning")
        manager = getattr(reasoning_module, "ReasoningParserManager")
        parser_cls = manager.get_reasoning_parser(parser_name)
        return parser_cls(_TokenizerStub(vocab))
    except ModuleNotFoundError as exc:
        if exc.name == "vllm":
            logger.debug(
                "vLLM is not installed in this environment; relying on "
                "server-side reasoning parsing.",
            )
        else:
            logger.debug(
                "Could not import vLLM parser '%s': %s",
                parser_name,
                exc,
                exc_info=True,
            )
    except Exception as exc:
        logger.debug(
            "Could not initialize vLLM parser '%s': %s",
            parser_name,
            exc,
            exc_info=True,
        )
    return None


class _TokenizerStub:
    """Minimal tokenizer interface needed by delimiter-style vLLM parsers."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab
