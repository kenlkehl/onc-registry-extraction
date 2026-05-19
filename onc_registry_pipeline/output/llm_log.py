"""JSONL logger for LLM call details (prompts, responses, parsed output)."""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)


class LLMLog:
    """Append-only JSONL log of every LLM call made during a pipeline run."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fh = open(path, "a", encoding="utf-8")
        self._count = 0

    def log(
        self,
        *,
        call_type: str = "",
        pass_number: int = 0,
        chunk_id: str = "",
        chunk_type: str = "",
        system_prompt: str = "",
        user_prompt: str = "",
        raw_output: str = "",
        reasoning: str = "",
        final_output: str = "",
        parsed: object = None,
    ) -> None:
        """Write one log entry."""
        entry = {
            "timestamp": time.time(),
            "call_type": call_type,
            "pass_number": pass_number,
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_output": raw_output,
            "reasoning": reasoning,
            "final_output": final_output,
            "parsed": parsed,
        }
        self._fh.write(json.dumps(entry, default=str) + "\n")
        self._fh.flush()
        self._count += 1

    def close(self) -> None:
        """Flush and close the log file."""
        if self._fh and not self._fh.closed:
            self._fh.close()
            logger.info("LLM log closed: %s (%d entries)", self._path, self._count)
