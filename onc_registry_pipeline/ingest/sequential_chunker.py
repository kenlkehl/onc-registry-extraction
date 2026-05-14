"""Simple sequential chunker: concatenate patient notes chronologically, chunk by token count.

Replaces the document-type-aware ClinicalChunker. No document classification,
no section splitting, no pass-specific priorities. Just chronological text
chunked into token-length segments with overlap.
"""

import re
import uuid
from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)

_DATE_HEADER_RE = re.compile(r"=== Clinical Note dated (.+?) ===")


@dataclass
class SequentialChunk:
    """A token-length segment of a patient's concatenated clinical text."""

    chunk_id: str
    text: str
    chunk_index: int        # 0-based position in patient's chunk list
    first_date: str         # earliest note date in this chunk
    last_date: str          # latest note date in this chunk
    token_estimate: int

    # Satisfy the Chunk protocol from extraction/base.py
    @property
    def chunk_type(self) -> str:
        return "sequential"

    @property
    def document_date(self) -> str:
        return self.last_date


class SequentialChunker:
    """Concatenates patient documents chronologically and chunks by token count.

    Usage::

        chunker = SequentialChunker(chunk_target_tokens=50000)
        chunks = chunker.chunk_documents(patient_set.documents)
    """

    def __init__(
        self,
        chunk_target_tokens: int = 50000,
        chunk_overlap_tokens: int = 500,
    ) -> None:
        self._target = chunk_target_tokens
        self._overlap = chunk_overlap_tokens

    def chunk_documents(self, documents: list[Any]) -> list[SequentialChunk]:
        """Concatenate all documents chronologically and chunk by token count.

        Each document is wrapped with a date header::

            === Clinical Note dated 2023-03-15 ===
            <note text>

        Parameters
        ----------
        documents:
            List of Document objects (must have ``date`` and ``text`` attrs),
            assumed to be sorted by date.

        Returns
        -------
        list[SequentialChunk]
            Chronological chunks with overlap between consecutive chunks.
        """
        if not documents:
            return []

        # Build the full concatenated text with date headers
        blocks: list[str] = []
        all_dates: list[str] = []
        for doc in documents:
            text = doc.text.strip()
            if not text:
                continue
            date_str = str(doc.date) if doc.date else "unknown"
            blocks.append(f"=== Clinical Note dated {date_str} ===\n{text}\n")
            all_dates.append(date_str)

        if not blocks:
            return []

        full_text = "\n".join(blocks)
        total_tokens = self._estimate_tokens(full_text)

        # If everything fits in one chunk, return it directly
        if total_tokens <= self._target:
            first_date = all_dates[0] if all_dates else "unknown"
            last_date = all_dates[-1] if all_dates else "unknown"
            chunk = SequentialChunk(
                chunk_id=str(uuid.uuid4()),
                text=full_text,
                chunk_index=0,
                first_date=first_date,
                last_date=last_date,
                token_estimate=total_tokens,
            )
            logger.info(
                "Patient text fits in 1 chunk (%d tokens)", total_tokens
            )
            return [chunk]

        # Chunk with overlap using character-based approximation
        chunks = self._chunk_with_overlap(full_text, all_dates)
        logger.info(
            "Chunked %d documents (%d tokens) into %d chunks of ~%d tokens",
            len(documents),
            total_tokens,
            len(chunks),
            self._target,
        )
        return chunks

    def _chunk_with_overlap(
        self, full_text: str, all_dates: list[str]
    ) -> list[SequentialChunk]:
        """Split full_text into token-sized chunks with overlap.

        Uses character-based slicing (chars_per_token ≈ 3.5) to avoid
        requiring a tokenizer dependency.
        """
        chars_per_token = 3.5
        target_chars = int(self._target * chars_per_token)
        overlap_chars = int(self._overlap * chars_per_token)
        stride_chars = target_chars - overlap_chars

        chunks: list[SequentialChunk] = []
        start = 0
        chunk_index = 0

        while start < len(full_text):
            end = min(start + target_chars, len(full_text))
            chunk_text = full_text[start:end]

            # Extract dates present in this chunk from headers
            found_dates = _DATE_HEADER_RE.findall(chunk_text)
            if found_dates:
                first_date = found_dates[0]
                last_date = found_dates[-1]
            elif chunks:
                # Chunk falls within a single note with no header visible
                first_date = chunks[-1].last_date
                last_date = first_date
            else:
                first_date = all_dates[0] if all_dates else "unknown"
                last_date = first_date

            token_est = self._estimate_tokens(chunk_text)
            chunk = SequentialChunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                chunk_index=chunk_index,
                first_date=first_date,
                last_date=last_date,
                token_estimate=token_est,
            )
            chunks.append(chunk)
            chunk_index += 1

            if end >= len(full_text):
                break
            start += stride_chars

        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: len(text) / 3.5"""
        return max(1, int(len(text) / 3.5))
