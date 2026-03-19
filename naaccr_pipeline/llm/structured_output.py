"""Build prompt-level JSON format instructions from NAACCR data items.

Instead of vLLM's guided_json constrained decoding (which is unreliable
with complex schemas), we describe the expected JSON format in the prompt
text and let the LLM produce free-form JSON, retrying on parse failure.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight protocols
# ---------------------------------------------------------------------------

class CodeResolverLike(Protocol):
    """Minimal interface expected from a code resolver."""

    def get_valid_codes_prompt(self, item_number: int) -> str:
        """Return a human-readable string of valid codes, or ''."""
        ...


# ---------------------------------------------------------------------------
# SchemaBuilder
# ---------------------------------------------------------------------------

class SchemaBuilder:
    """Builds prompt-level JSON format instructions from NAACCR data items.

    Generates text blocks that tell the LLM what JSON structure to produce,
    including field names, valid codes, and format expectations.
    """

    # -- public API -------------------------------------------------------

    def build_json_format_instructions(
        self,
        items: list[Any],
        code_resolver: Any,
    ) -> str:
        r"""Generate prompt text describing the expected JSON output format.

        Parameters
        ----------
        items:
            NAACCR data item objects with ``item_number``, ``name``,
            ``length``, ``xml_id``, ``data_type``, ``allowable_values``.
        code_resolver:
            Object with ``get_valid_codes_prompt(item_number) -> str``.

        Returns
        -------
        str
            A text block to embed in the prompt, e.g.::

                Respond with a JSON object. Each field should be ...
                Expected fields:
                - "primarySite": ICD-O-3 topography code (C##.#)
                ...
        """
        field_lines: list[str] = []
        for item in items:
            field_name = self._field_name(item)
            desc = self._field_description(item, code_resolver)
            field_lines.append(f'- "{field_name}": {desc}')

        fields_block = "\n".join(field_lines)

        return (
            "Respond with a JSON object. For each item, provide an object with:\n"
            '  "value": the extracted value (use valid codes listed below),\n'
            '  "confidence": a float 0.0-1.0 indicating extraction confidence,\n'
            '  "evidence": a short quote (max 200 chars) from the text supporting the value.\n'
            "\n"
            "If information is not found in the text, set value to the appropriate "
            '"unknown" code (e.g. "9", "99", "unknown") and confidence to 0.0.\n'
            "\n"
            f"Expected fields:\n{fields_block}"
        )

    def build_simple_schema(self, fields: dict[str, dict]) -> dict:
        """Build a JSON schema dict for simple extractions (e.g. tumor detection).

        This is still used by TumorDetector which has a simple enough
        schema that guided_json works fine.
        """
        return {
            "type": "object",
            "properties": fields,
            "required": list(fields.keys()),
        }

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _field_name(item: Any) -> str:
        """Derive the JSON field name from a NAACCR data item."""
        xml_id = getattr(item, "xml_id", "") or ""
        xml_id = xml_id.strip()
        if xml_id:
            return xml_id
        item_number = getattr(item, "item_number", 0)
        return f"item_{item_number}"

    @staticmethod
    def _field_description(item: Any, code_resolver: Any) -> str:
        """Build a human-readable description for one field."""
        item_number = getattr(item, "item_number", 0)
        item_name = getattr(item, "name", "") or getattr(item, "item_name", "")
        item_length = getattr(item, "length", 0) or 1
        data_type = (getattr(item, "data_type", "") or "").strip().lower()

        # Try to get valid codes prompt
        codes_text = ""
        if code_resolver is not None:
            try:
                codes_text = code_resolver.get_valid_codes_prompt(item_number)
            except Exception:
                pass

        parts = [f"{item_name} (Item {item_number})"]

        if codes_text:
            parts.append(f"Valid codes: {codes_text}")
        elif data_type == "date":
            parts.append("YYYYMMDD format (use 99 for unknown day/month)")
        elif data_type == "digits":
            allowable = getattr(item, "allowable_values", "") or ""
            parts.append(f"{item_length}-digit number. {allowable}".strip())
        elif item_number == 400:  # Primary Site
            parts.append("ICD-O-3 topography code C##.# (e.g., C50.4)")
        elif item_number == 522:  # Histology
            parts.append("4-digit ICD-O-3 morphology code 8000-9989")
        else:
            parts.append(f"max {item_length} characters")

        return ". ".join(parts)
