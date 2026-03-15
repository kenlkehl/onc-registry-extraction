"""Pass 1: Demographics and Cancer Identification extraction."""

from __future__ import annotations

from typing import Any

from naaccr_pipeline.config import PipelineConfig
from naaccr_pipeline.dictionary.loader import NAACCRDictionary, NAACCRDataItem
from naaccr_pipeline.dictionary.code_resolver import CodeResolver
from naaccr_pipeline.extraction.base import BaseExtractionPass, ExtractionResult
from naaccr_pipeline.extraction.prompts.demographics import (
    PASS1_SYSTEM_PROMPT,
    PASS1_USER_TEMPLATE,
)
from naaccr_pipeline.llm.client import VLLMClient
from naaccr_pipeline.llm.structured_output import SchemaBuilder


class Pass1Demographics(BaseExtractionPass):
    """Extract demographics and cancer identification data items.

    This is the first content-extraction pass (Pass 1).  It targets the
    core cancer-identification fields (primary site, histology, behavior,
    date of diagnosis, etc.) together with key demographic items (sex,
    race, marital status, etc.).

    Items that were already populated from structured data columns
    (present in *prior_results* with confidence >= 1.0) are skipped.
    """

    PASS_NUMBER = 1
    CHUNK_PRIORITY = [
        "pathology",
        "discharge_summary",
        "consult",
        "progress_note",
        "operative",
        "radiology",
        "lab",
        "mixed",
    ]

    def __init__(
        self,
        config: PipelineConfig,
        dictionary: NAACCRDictionary,
        code_resolver: CodeResolver,
        llm_client: VLLMClient,
        schema_builder: SchemaBuilder,
        tumor_context: str = "",
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
        tumor_context:
            Optional string describing which tumor to focus on (from
            Pass 0 TumorCandidate).  For example:
            ``"Focus on the invasive ductal carcinoma of the left breast
            diagnosed approximately 2023-03."``
        llm_log:
            Optional LLMLog for saving raw LLM outputs.
        """
        super().__init__(config, dictionary, code_resolver, llm_client, schema_builder, llm_log=llm_log)
        self._tumor_context = tumor_context

    # ------------------------------------------------------------------
    # Target items
    # ------------------------------------------------------------------

    def get_target_items(
        self, prior_results: dict[int, ExtractionResult]
    ) -> list[NAACCRDataItem]:
        """Return Cancer Identification + key Demographic items.

        Cancer Identification items (from section "Cancer Identification"):
            380 - Sequence Number--Central
            390 - Date of Diagnosis
            400 - Primary Site
            410 - Laterality
            440 - Grade
            441 - Grade Path Value
            442 - Ambiguous Terminology DX
            449 - Grade Path System
            450 - Site Coding Sys--Current
            470 - Morph Coding Sys--Current
            490 - Diagnostic Confirmation
            500 - Type of Reporting Source
            522 - Histologic Type ICD-O-3
            523 - Behavior Code ICD-O-3

        Key Demographic items:
            150 - Marital Status at DX
            160 - Race 1
            161 - Race 2
            190 - Spanish/Hispanic Origin
            220 - Sex
            230 - Age at Diagnosis
            240 - Date of Birth
            252 - Birthplace--State
            254 - Birthplace--Country

        Items already populated from structured columns (present in
        *prior_results* with confidence >= 1.0) are skipped.
        """
        cancer_id_numbers = [
            380, 390, 400, 410, 440, 441, 442, 449, 450, 470, 490, 500, 522, 523,
        ]
        demo_numbers = [150, 160, 161, 190, 220, 230, 240, 252, 254]

        all_numbers = cancer_id_numbers + demo_numbers
        items: list[NAACCRDataItem] = []
        for num in all_numbers:
            # Skip if already extracted with high confidence (from structured data)
            if num in prior_results and prior_results[num].confidence >= 1.0:
                continue
            item = self._dict.get_item(num)
            if item is not None:
                items.append(item)
        return items

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        chunk: Any,
        target_items: list[NAACCRDataItem],
        prior_results: dict[int, ExtractionResult],
    ) -> tuple[str, str]:
        """Build system and user prompts for Pass 1.

        Parameters
        ----------
        chunk:
            A document chunk satisfying the :class:`Chunk` protocol.
        target_items:
            NAACCR data items to extract in this call.
        prior_results:
            Results from earlier passes (keyed by item number).

        Returns
        -------
        tuple[str, str]
            ``(system_prompt, user_prompt)``
        """
        code_blocks: list[str] = []
        for item in target_items:
            codes_text = self._resolver.get_valid_codes_prompt(item.item_number)
            if codes_text:
                code_blocks.append(
                    f"- {item.name} (Item {item.item_number}): {codes_text}"
                )
            else:
                # For items without code lists, describe the expected format
                if item.data_type == "date":
                    code_blocks.append(
                        f"- {item.name} (Item {item.item_number}): YYYYMMDD format"
                    )
                elif item.data_type == "digits":
                    code_blocks.append(
                        f"- {item.name} (Item {item.item_number}): "
                        f"{item.length}-digit number. {item.allowable_values or ''}"
                    )
                elif item.item_number == 400:  # Primary Site
                    code_blocks.append(
                        "- Primary Site (Item 400): "
                        "ICD-O-3 topography code C##.# (e.g., C50.4)"
                    )
                elif item.item_number == 522:  # Histology
                    code_blocks.append(
                        "- Histologic Type ICD-O-3 (Item 522): "
                        "4-digit morphology code 8000-9989"
                    )

        tumor_context = ""
        if self._tumor_context:
            tumor_context = f"TUMOR FOCUS: {self._tumor_context}"

        note_type = getattr(chunk, "chunk_type", "unknown")
        note_date = getattr(chunk, "document_date", "unknown")

        user = PASS1_USER_TEMPLATE.format(
            chunk_text=chunk.text,
            code_reference="\n".join(code_blocks),
            note_type=note_type,
            note_date=note_date,
            tumor_context=tumor_context,
        )
        return PASS1_SYSTEM_PROMPT, user
