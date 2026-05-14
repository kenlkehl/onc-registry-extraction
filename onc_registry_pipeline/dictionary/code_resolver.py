"""Resolve LLM outputs to valid NAACCR codes."""

from __future__ import annotations

import logging
import re
from typing import Optional

from onc_registry_pipeline.dictionary.loader import NAACCRDictionary, CodeEntry

logger = logging.getLogger(__name__)

# Optional dependency -- degrade gracefully when rapidfuzz is absent.
try:
    from rapidfuzz import fuzz as _fuzz, process as _process

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False

# Pre-compiled pattern for numeric ranges like "001-999", "00-88", etc.
_RANGE_RE = re.compile(r"^(\d+)\s*[-\u2013]\s*(\d+)$")


class CodeResolver:
    """Map free-text / LLM output to a valid NAACCR code value.

    Parameters
    ----------
    dictionary:
        A fully-loaded :class:`NAACCRDictionary` instance.
    """

    def __init__(self, dictionary: NAACCRDictionary) -> None:
        self._dict = dictionary

        # Per-item indexes built once at init time.
        # item_number -> {code_str: CodeEntry}
        self._code_index: dict[int, dict[str, CodeEntry]] = {}
        # item_number -> {lower(code_str): CodeEntry}
        self._code_index_lower: dict[int, dict[str, CodeEntry]] = {}
        # item_number -> {lower(description): CodeEntry}
        self._desc_index: dict[int, dict[str, CodeEntry]] = {}
        # item_number -> list[(lower_desc, CodeEntry)]  for fuzzy search
        self._desc_list: dict[int, list[tuple[str, CodeEntry]]] = {}
        # item_number -> list of (lo, hi, width) parsed from allowable_values
        self._numeric_ranges: dict[int, list[tuple[int, int, int]]] = {}

        self._build_indexes()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_indexes(self) -> None:
        """Walk every item that has codes and build look-up structures."""
        seen_items: set[int] = set()

        for item in self._dict.get_active_items():
            codes = self._dict.get_codes(item.item_number)
            if not codes:
                # Try to extract numeric ranges from allowable_values.
                self._parse_numeric_ranges(item.item_number, item.allowable_values)
                continue

            exact: dict[str, CodeEntry] = {}
            lower: dict[str, CodeEntry] = {}
            desc: dict[str, CodeEntry] = {}
            desc_pairs: list[tuple[str, CodeEntry]] = []

            for ce in codes:
                exact[ce.code] = ce
                lower[ce.code.lower()] = ce
                d = ce.description.lower().strip()
                if d:
                    desc[d] = ce
                    desc_pairs.append((d, ce))

            self._code_index[item.item_number] = exact
            self._code_index_lower[item.item_number] = lower
            self._desc_index[item.item_number] = desc
            self._desc_list[item.item_number] = desc_pairs
            seen_items.add(item.item_number)

            # Also parse ranges from allowable_values (some items have both).
            self._parse_numeric_ranges(item.item_number, item.allowable_values)

    def _parse_numeric_ranges(self, item_number: int, allowable: str) -> None:
        """Extract numeric-range spans (e.g. ``001-999``) from *allowable*."""
        if not allowable:
            return
        ranges: list[tuple[int, int, int]] = []
        # Split on comma, semicolon, or whitespace-then-comma.
        for token in re.split(r"[,;]\s*|\s+", allowable):
            token = token.strip()
            m = _RANGE_RE.match(token)
            if m:
                lo_str, hi_str = m.group(1), m.group(2)
                width = max(len(lo_str), len(hi_str))
                try:
                    ranges.append((int(lo_str), int(hi_str), width))
                except ValueError:
                    continue
        if ranges:
            self._numeric_ranges[item_number] = ranges

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, item_number: int, llm_output: str) -> tuple[str, float]:
        """Resolve *llm_output* to a valid NAACCR code.

        Returns ``(resolved_code, confidence)``.

        Resolution strategy (highest priority first):

        1. Exact match against code values -> 1.0
        2. Case-insensitive match against code values -> 0.95
        3. Exact match against code descriptions -> 0.9
        4. Fuzzy match against descriptions (rapidfuzz, score > 85) ->
           confidence * (score / 100)
        5. Numeric-range check from allowable values -> 0.85
        6. No match -> ``(llm_output, 0.0)``
        """
        text = llm_output.strip()

        # 1. Exact code match
        exact = self._code_index.get(item_number, {})
        if text in exact:
            return (text, 1.0)

        # 2. Case-insensitive code match
        lower_idx = self._code_index_lower.get(item_number, {})
        hit = lower_idx.get(text.lower())
        if hit is not None:
            return (hit.code, 0.95)

        # 3. Exact description match (case-insensitive)
        desc_idx = self._desc_index.get(item_number, {})
        hit = desc_idx.get(text.lower())
        if hit is not None:
            return (hit.code, 0.9)

        # 4. Fuzzy description match
        desc_pairs = self._desc_list.get(item_number, [])
        if _HAS_RAPIDFUZZ and desc_pairs:
            descriptions = [d for d, _ in desc_pairs]
            result = _process.extractOne(
                text.lower(),
                descriptions,
                scorer=_fuzz.WRatio,
                score_cutoff=85,
            )
            if result is not None:
                matched_desc, score, idx = result
                entry = desc_pairs[idx][1]
                confidence = 0.9 * (score / 100.0)
                return (entry.code, round(confidence, 4))

        # 5. Numeric-range check
        ranges = self._numeric_ranges.get(item_number, [])
        if ranges and text.isdigit():
            val = int(text)
            for lo, hi, width in ranges:
                if lo <= val <= hi:
                    # Zero-pad to expected width.
                    padded = text.zfill(width)
                    return (padded, 0.85)

        # 6. No match
        return (text, 0.0)

    def get_valid_codes_prompt(self, item_number: int) -> str:
        """Return a compact code-reference string for inclusion in prompts.

        Example::

            Valid codes: 0=In situ, 1=Localized, ..., 9=Unknown

        Returns an empty string if no codes are defined for the item.
        """
        codes = self._dict.get_codes(item_number)
        if not codes:
            return ""
        parts = [f"{c.code}={c.description}" for c in codes]
        return "Valid codes: " + ", ".join(parts)

    def build_constrained_vocab(self, item_number: int) -> list[str]:
        """Return a list of valid code strings (for JSON-schema enums)."""
        codes = self._dict.get_codes(item_number)
        return [c.code for c in codes]

    def has_codes(self, item_number: int) -> bool:
        """Return ``True`` if *item_number* has at least one defined code."""
        return bool(self._dict.get_codes(item_number))
