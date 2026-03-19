"""Pass 3: Treatment-1st Course extraction with 3 sub-passes."""

from __future__ import annotations

from typing import Any
import logging

from naaccr_pipeline.config import PipelineConfig
from naaccr_pipeline.dictionary.loader import NAACCRDictionary, NAACCRDataItem
from naaccr_pipeline.dictionary.code_resolver import CodeResolver
from naaccr_pipeline.extraction.base import BaseExtractionPass, ExtractionResult
from naaccr_pipeline.extraction.prompts.treatment import (
    SURGERY_SYSTEM_PROMPT,
    RADIATION_SYSTEM_PROMPT,
    SYSTEMIC_SYSTEM_PROMPT,
    TREATMENT_USER_TEMPLATE,
)
from naaccr_pipeline.llm.client import VLLMClient
from naaccr_pipeline.llm.structured_output import SchemaBuilder

logger = logging.getLogger(__name__)

# Confidence threshold above which an item is considered fully resolved.
_HIGH_CONFIDENCE_THRESHOLD = 0.9


class Pass3Treatment(BaseExtractionPass):
    """Extract first-course treatment data using three sub-passes.

    Treatment extraction is split into three focused sub-passes so that each
    LLM call receives a domain-specific system prompt and a manageable set
    of data items:

    1. **Surgery** -- surgical procedures on the primary site, lymph nodes,
       margins, and related items.
    2. **Radiation** -- beam radiation, brachytherapy, phase-level detail,
       and related items.
    3. **Systemic** -- chemotherapy, hormone therapy, immunotherapy (BRM),
       transplant/endocrine, neoadjuvant, and palliative procedures.

    The ``run()`` method is overridden to execute these three sub-passes in
    sequence and merge their results.
    """

    PASS_NUMBER = 3
    CHUNK_PRIORITY = [
        "operative",
        "discharge_summary",
        "progress_note",
        "consult",
        "pathology",
        "radiology",
        "lab",
        "mixed",
    ]

    # ------------------------------------------------------------------
    # Item groups for sub-passes
    # ------------------------------------------------------------------

    SURGERY_ITEMS = [
        1200,  # RX Date Surgery
        1290,  # RX Summ--Surg Prim Site 03-2022
        1291,  # RX Summ--Surg Prim Site 2023
        1292,  # RX Summ--Scope Reg LN Sur
        1294,  # RX Summ--Surg Oth Reg/Dis
        1296,  # RX Summ--Reg LN Examined
        1310,  # RX Summ--Surgical Approach
        1320,  # RX Summ--Surgical Margins
        1330,  # RX Summ--Reconstruct 1st
        1340,  # Reason for No Surgery
        1350,  # RX Summ--DX/Stg Proc
        1640,  # RX Summ--Surgery Type
        3170,  # RX Date Mst Defn Srg
        3180,  # RX Date Surg Disch
        3190,  # Readm Same Hosp 30 Days
    ]

    RADIATION_ITEMS = [
        1210,  # RX Date Radiation
        1360,  # RX Summ--Radiation
        1370,  # RX Summ--Rad to CNS
        1380,  # RX Summ--Surg/Rad Seq
        1430,  # Reason for No Radiation
        1501, 1502, 1503, 1504, 1505, 1506, 1507,  # Phase I
        1511, 1512, 1513, 1514, 1515, 1516, 1517,  # Phase II
        1521, 1522, 1523, 1524, 1525, 1526, 1527,  # Phase III
        1531,  # Radiation Treatment Discontinued Early
        1532,  # Number of Phases
        1533,  # Total Dose
        1550,  # Rad--Location of RX
        1570,  # Rad--Regional RX Modality
        3220,  # RX Date Rad Ended
    ]

    SYSTEMIC_ITEMS = [
        1220,  # RX Date Chemo
        1230,  # RX Date Hormone
        1240,  # RX Date BRM
        1250,  # RX Date Other
        1285,  # RX Summ--Treatment Status
        1390,  # RX Summ--Chemo
        1400,  # RX Summ--Hormone
        1410,  # RX Summ--BRM
        1420,  # RX Summ--Other
        1632,  # Neoadjuvant Therapy
        1633,  # Neoadjuvant Therapy-Clinical Response
        1634,  # Neoadjuvant Therapy-Treatment Effect
        1639,  # RX Summ--Systemic/Sur Seq
        3230,  # RX Date Systemic
        3250,  # RX Summ--Transplnt/Endocr
        3270,  # RX Summ--Palliative Proc
    ]

    def __init__(
        self,
        config: PipelineConfig,
        dictionary: NAACCRDictionary,
        code_resolver: CodeResolver,
        llm_client: VLLMClient,
        schema_builder: SchemaBuilder,
        primary_site: str = "unknown",
        llm_log=None,
    ) -> None:
        """
        Parameters
        ----------
        config:
            Pipeline configuration.
        dictionary:
            Loaded NAACCR v26 data dictionary.
        code_resolver:
            Code resolver for mapping LLM output to valid NAACCR codes.
        llm_client:
            Async vLLM client.
        schema_builder:
            JSON schema builder for guided decoding.
        primary_site:
            Primary site description for the tumour being abstracted
            (e.g. ``"left breast"``).  Used to specialise the surgery
            system prompt.
        llm_log:
            Optional LLMLog for saving raw LLM outputs.
        """
        super().__init__(config, dictionary, code_resolver, llm_client, schema_builder, llm_log=llm_log)
        self._primary_site = primary_site

    # ------------------------------------------------------------------
    # Target items
    # ------------------------------------------------------------------

    def get_target_items(
        self, prior_results: dict[int, ExtractionResult]
    ) -> list[NAACCRDataItem]:
        """Return all treatment items across the three sub-passes."""
        all_nums = self.SURGERY_ITEMS + self.RADIATION_ITEMS + self.SYSTEMIC_ITEMS
        return [
            item
            for num in all_nums
            if (item := self._dict.get_item(num)) is not None
        ]

    # ------------------------------------------------------------------
    # Prompt construction (base-class interface)
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        chunk: Any,
        target_items: list[NAACCRDataItem],
        prior_results: dict[int, ExtractionResult],
    ) -> tuple[str, str]:
        """Not used directly -- sub-passes call ``_build_subpass_prompt``.

        This is required by the abstract interface but never invoked because
        ``run()`` is overridden.  Returns empty strings as a safe fallback.
        """
        return "", ""

    # ------------------------------------------------------------------
    # Core execution (overrides BaseExtractionPass.run)
    # ------------------------------------------------------------------

    async def run(
        self,
        chunks: list[Any],
        prior_results: dict[int, ExtractionResult],
    ) -> list[ExtractionResult]:
        """Execute three treatment sub-passes and merge results.

        Algorithm
        ---------
        1. Run the surgery sub-pass over prioritised chunks.
        2. Run the radiation sub-pass over prioritised chunks.
        3. Run the systemic sub-pass over prioritised chunks.
        4. Return merged results (higher confidence wins).
        """
        merged = dict(prior_results)

        # Prioritise chunks once for all sub-passes.
        ordered_chunks = self._prioritize_chunks(chunks)
        if not ordered_chunks:
            logger.warning("Pass %d: no chunks available.", self.PASS_NUMBER)
            return list(merged.values())

        # Specialise the surgery prompt with the primary site.
        surgery_prompt = SURGERY_SYSTEM_PROMPT.format(
            primary_site=self._primary_site,
        )

        subpasses = [
            ("surgery", self.SURGERY_ITEMS, surgery_prompt),
            ("radiation", self.RADIATION_ITEMS, RADIATION_SYSTEM_PROMPT),
            ("systemic", self.SYSTEMIC_ITEMS, SYSTEMIC_SYSTEM_PROMPT),
        ]

        for subpass_name, item_numbers, system_prompt in subpasses:
            logger.info(
                "Pass %d/%s: starting sub-pass.", self.PASS_NUMBER, subpass_name
            )
            subpass_results = await self._run_subpass(
                ordered_chunks, merged, item_numbers, system_prompt
            )
            merged = self._merge_results(merged, subpass_results)

        return list(merged.values())

    # ------------------------------------------------------------------
    # Sub-pass execution
    # ------------------------------------------------------------------

    async def _run_subpass(
        self,
        chunks: list[Any],
        prior_results: dict[int, ExtractionResult],
        item_numbers: list[int],
        system_prompt: str,
    ) -> list[ExtractionResult]:
        """Run a single sub-pass (surgery, radiation, or systemic).

        Parameters
        ----------
        chunks:
            Already-prioritised chunks.
        prior_results:
            Current merged results (used for filtering and context).
        item_numbers:
            NAACCR item numbers targeted by this sub-pass.
        system_prompt:
            The domain-specific system prompt for this sub-pass.

        Returns
        -------
        list[ExtractionResult]
            New extraction results from this sub-pass.
        """
        # Resolve item numbers to dictionary items, filtering already
        # high-confidence results.
        items = []
        for num in item_numbers:
            item = self._dict.get_item(num)
            if item is None:
                continue
            existing = prior_results.get(num)
            if existing is not None and existing.confidence >= _HIGH_CONFIDENCE_THRESHOLD:
                continue
            items.append(item)

        if not items:
            logger.info(
                "Pass %d: all items for sub-pass already at high confidence.",
                self.PASS_NUMBER,
            )
            return []

        # Batch items according to model capacity.
        items_per_call = self._llm.model_profile.items_per_call
        batches = self._split_items_into_batches(items, items_per_call)

        all_results: list[ExtractionResult] = []
        for chunk in chunks:
            for batch in batches:
                try:
                    system, user = self._build_subpass_prompt(
                        chunk, batch, system_prompt, prior_results
                    )
                    new_results = await self._extract_with_prompt(
                        chunk, batch, system, user
                    )
                    all_results.extend(new_results)
                except Exception:
                    logger.exception(
                        "Pass %d: error in sub-pass for chunk %s",
                        self.PASS_NUMBER,
                        getattr(chunk, "chunk_id", "?"),
                    )

        return all_results

    # ------------------------------------------------------------------
    # Sub-pass prompt construction
    # ------------------------------------------------------------------

    def _build_subpass_prompt(
        self,
        chunk: Any,
        items: list[NAACCRDataItem],
        system_prompt: str,
        prior_results: dict[int, ExtractionResult],
    ) -> tuple[str, str]:
        """Build system and user prompts for a specific sub-pass.

        Parameters
        ----------
        chunk:
            A document chunk satisfying the :class:`Chunk` protocol.
        items:
            NAACCR data items for this sub-pass batch.
        system_prompt:
            Domain-specific system prompt (surgery / radiation / systemic).
        prior_results:
            Results from earlier passes for context.

        Returns
        -------
        tuple[str, str]
            ``(system_prompt, user_prompt)``
        """
        code_blocks: list[str] = []
        for item in items:
            codes_text = self._resolver.get_valid_codes_prompt(item.item_number)
            if codes_text:
                code_blocks.append(
                    f"- {item.name} (Item {item.item_number}): {codes_text}"
                )
            else:
                if item.data_type == "date":
                    code_blocks.append(
                        f"- {item.name} (Item {item.item_number}): YYYYMMDD format"
                    )
                elif item.data_type == "digits":
                    code_blocks.append(
                        f"- {item.name} (Item {item.item_number}): "
                        f"{item.length}-digit number. {item.allowable_values or ''}"
                    )
                else:
                    code_blocks.append(
                        f"- {item.name} (Item {item.item_number}): "
                        f"max {item.length} characters. {item.allowable_values or ''}"
                    )

        note_type = getattr(chunk, "chunk_type", "unknown")
        note_date = getattr(chunk, "document_date", "unknown")

        user = TREATMENT_USER_TEMPLATE.format(
            chunk_text=chunk.text,
            code_reference="\n".join(code_blocks),
            note_type=note_type,
            note_date=note_date,
        )
        return system_prompt, user

    # ------------------------------------------------------------------
    # LLM call with custom prompt (bypasses build_prompt)
    # ------------------------------------------------------------------

    async def _extract_with_prompt(
        self,
        chunk: Any,
        items: list[NAACCRDataItem],
        system_prompt: str,
        user_prompt: str,
    ) -> list[ExtractionResult]:
        """Call the LLM with explicit prompts and parse the response.

        This mirrors ``_extract_from_chunk`` but accepts pre-built prompts
        instead of calling ``build_prompt``.
        """
        from naaccr_pipeline.extraction.base import _CodeResolverAdapter

        resolver_adapter = _CodeResolverAdapter(self._resolver)
        json_schema = self._schema_builder.build_extraction_schema(
            items, resolver_adapter
        )

        llm_response = await self._llm.extract(system_prompt, user_prompt, json_schema)

        if llm_response.parsed.get("_error"):
            logger.warning(
                "LLM error for chunk %s: %s",
                getattr(chunk, "chunk_id", "?"),
                llm_response.parsed.get("_message", "unknown"),
            )
            return []

        return self._parse_llm_response(llm_response.parsed, items, chunk)
