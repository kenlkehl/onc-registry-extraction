"""Build JSON schemas for vLLM guided decoding from NAACCR data items.

The schemas produced here are passed to ``VLLMClient.extract()`` as the
``json_schema`` parameter, which vLLM uses for constrained (guided) decoding.
This guarantees the model output conforms to the expected structure -- valid
NAACCR field names, code enumerations, and date/digit patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight protocols so we don't create hard import dependencies on the
# dictionary module (which may not yet exist).  Any object that satisfies
# the duck-typed interface will work.
# ---------------------------------------------------------------------------

class NAACCRDataItemLike(Protocol):
    """Minimal interface expected from a NAACCR data item object."""

    @property
    def item_number(self) -> int: ...

    @property
    def item_name(self) -> str: ...

    @property
    def length(self) -> int: ...

    @property
    def xml_id(self) -> str: ...

    @property
    def data_type(self) -> str: ...


class CodeResolverLike(Protocol):
    """Minimal interface expected from a code resolver."""

    def get_codes(self, item_number: int) -> list[str]:
        """Return the list of valid code strings for an item, or []."""
        ...


# ---------------------------------------------------------------------------
# SchemaBuilder
# ---------------------------------------------------------------------------

class SchemaBuilder:
    """Builds JSON schemas for vLLM guided decoding from NAACCR data items.

    Each data item is represented as a JSON object with three fields:

    * ``value`` -- constrained to valid codes, date patterns, digit
      patterns, or free text depending on the item type.
    * ``confidence`` -- a float in [0, 1].
    * ``evidence`` -- a short string (max 200 chars) citing the source text.
    """

    # -- public API -------------------------------------------------------

    def build_extraction_schema(
        self,
        items: list[Any],
        code_resolver: Any,
    ) -> dict:
        """Generate a JSON schema constraining the LLM output to valid values.

        Parameters
        ----------
        items:
            A list of NAACCR data item objects.  Each must expose at minimum:
            ``item_number``, ``item_name``, ``length``, ``xml_id``, and
            ``data_type``.
        code_resolver:
            An object with a ``get_codes(item_number) -> list[str]`` method
            that returns valid codes for a given NAACCR item number.

        Returns
        -------
        dict
            A complete JSON Schema object (``{"type": "object", ...}``) ready
            to be passed to ``VLLMClient.extract()`` as ``json_schema``.
        """
        properties: dict[str, dict] = {}
        required: list[str] = []

        for item in items:
            field_name = self._field_name(item)
            value_schema = self._value_schema(item, code_resolver)

            properties[field_name] = {
                "type": "object",
                "properties": {
                    "value": value_schema,
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "evidence": {
                        "type": "string",
                        "maxLength": 200,
                    },
                },
                "required": ["value", "confidence", "evidence"],
            }
            required.append(field_name)

        schema: dict = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        return schema

    def build_simple_schema(self, fields: dict[str, dict]) -> dict:
        """Build a schema from a manual field specification dict.

        Useful for non-NAACCR extractions such as the Pass 0 tumor
        detection step.

        Parameters
        ----------
        fields:
            A mapping of field names to JSON Schema type definitions.
            Example::

                {
                    "tumors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cancer_type": {"type": "string"},
                                "primary_site": {"type": "string"},
                            },
                            "required": ["cancer_type", "primary_site"],
                        },
                    }
                }

        Returns
        -------
        dict
            A complete JSON Schema object.
        """
        return {
            "type": "object",
            "properties": fields,
            "required": list(fields.keys()),
        }

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _field_name(item: Any) -> str:
        """Derive the JSON field name from a NAACCR data item.

        Prefers the XML NAACCR ID (e.g. ``primarySite``).  Falls back to
        ``item_<number>`` when the XML ID is absent.
        """
        xml_id = getattr(item, "xml_id", "") or ""
        xml_id = xml_id.strip()
        if xml_id:
            return xml_id
        item_number = getattr(item, "item_number", 0)
        return f"item_{item_number}"

    @staticmethod
    def _value_schema(item: Any, code_resolver: Any) -> dict:
        """Build the ``value`` sub-schema for a single NAACCR data item.

        Resolution order
        ----------------
        1. If the code resolver returns valid codes for this item, constrain
           to an enum of those code strings.
        2. If the data type is ``"date"``, constrain to an 8-digit string
           pattern (YYYYMMDD).
        3. If the data type is ``"digits"`` (or the item name / format
           suggests digits), constrain to a digit-only string of the
           appropriate length.
        4. Otherwise treat as free text with a ``maxLength`` based on the
           item length.
        """
        item_number = getattr(item, "item_number", 0)
        item_length = getattr(item, "length", 0) or 1
        data_type = (getattr(item, "data_type", "") or "").strip().lower()

        # 1. Enum of valid codes -----------------------------------------
        codes: list[str] = []
        if code_resolver is not None:
            try:
                codes = code_resolver.get_codes(item_number)
            except Exception:
                logger.debug(
                    "code_resolver.get_codes(%d) failed; skipping enum.",
                    item_number,
                )

        if codes:
            return {"type": "string", "enum": codes}

        # 2. Date fields -------------------------------------------------
        if data_type == "date":
            return {"type": "string", "pattern": r"^\d{8}$"}

        # 3. Digit-only fields -------------------------------------------
        if data_type == "digits":
            return {
                "type": "string",
                "pattern": rf"^\d{{{item_length}}}$",
            }

        # 4. Free text ---------------------------------------------------
        return {"type": "string", "maxLength": max(item_length, 1)}
