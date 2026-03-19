"""Round-based parallel extraction across patients/tumors.

Organizes work so Round N = Nth chunk from each patient/tumor, enabling
cross-patient parallelism via batched vLLM inference. Supports checkpointing
for resume after interruption.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from naaccr_pipeline.config import PipelineConfig
from naaccr_pipeline.dictionary.loader import NAACCRDictionary
from naaccr_pipeline.dictionary.code_resolver import CodeResolver
from naaccr_pipeline.dictionary.schema_registry import SchemaRegistry
from naaccr_pipeline.extraction.base import (
    ExtractionResult,
    serialize_extraction_state,
    deserialize_extraction_state,
)
from naaccr_pipeline.extraction.chunk_extractor import ChunkExtractor
from naaccr_pipeline.extraction.pass0_tumor_detection import TumorCandidate
from naaccr_pipeline.llm.client import VLLMClient
from naaccr_pipeline.llm.structured_output import SchemaBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Work unit
# ---------------------------------------------------------------------------

@dataclass
class TumorWorkUnit:
    """Tracks extraction state for a single patient/tumor across rounds."""

    patient_id: str
    tumor_index: int
    tumor: TumorCandidate
    chunks: list[Any]  # list[SequentialChunk]
    current_extraction: dict[int, ExtractionResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Round orchestrator
# ---------------------------------------------------------------------------

class RoundOrchestrator:
    """Manages round-based parallel extraction across patients/tumors."""

    def __init__(
        self,
        config: PipelineConfig,
        dictionary: NAACCRDictionary,
        code_resolver: CodeResolver,
        llm_client: VLLMClient,
        schema_builder: SchemaBuilder,
        schema_registry: SchemaRegistry,
        llm_log: Any = None,
    ) -> None:
        self._config = config
        self._dict = dictionary
        self._resolver = code_resolver
        self._llm = llm_client
        self._schema_builder = schema_builder
        self._schema_reg = schema_registry
        self._llm_log = llm_log

    # ------------------------------------------------------------------
    # Round preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_rounds(
        work_units: list[TumorWorkUnit],
    ) -> list[list[tuple[TumorWorkUnit, int]]]:
        """Organize work into rounds.

        Round N contains the Nth chunk from each work unit that has
        at least N+1 chunks.

        Returns
        -------
        list[list[tuple[TumorWorkUnit, int]]]
            List of rounds, each containing (work_unit, chunk_index) pairs.
        """
        if not work_units:
            return []

        max_chunks = max(len(wu.chunks) for wu in work_units)
        rounds: list[list[tuple[TumorWorkUnit, int]]] = []

        for round_idx in range(max_chunks):
            round_items = []
            for wu in work_units:
                if round_idx < len(wu.chunks):
                    round_items.append((wu, round_idx))
            if round_items:
                rounds.append(round_items)

        return rounds

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    async def run_all_rounds(
        self,
        work_units: list[TumorWorkUnit],
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Process all rounds sequentially, with cross-patient parallelism.

        If checkpoint_dir is provided, saves state after each round and
        resumes from the last completed round on restart.
        """
        rounds = self.prepare_rounds(work_units)
        if not rounds:
            logger.warning("No rounds to process.")
            return

        # Resume from checkpoint if available
        start_round = 0
        if checkpoint_dir is not None:
            start_round = self._load_checkpoints(work_units, checkpoint_dir)
            if start_round > 0:
                logger.info("Resuming from round %d (of %d)", start_round, len(rounds))

        semaphore = asyncio.Semaphore(self._config.max_concurrent_patients)

        for round_idx in range(start_round, len(rounds)):
            round_items = rounds[round_idx]
            round_start = time.time()

            logger.info(
                "Round %d/%d: processing %d work units",
                round_idx + 1,
                len(rounds),
                len(round_items),
            )

            await self._process_round(round_items, semaphore)

            elapsed = time.time() - round_start
            logger.info(
                "Round %d/%d completed in %.1fs",
                round_idx + 1,
                len(rounds),
                elapsed,
            )

            # Checkpoint
            if checkpoint_dir is not None:
                self._save_checkpoint(round_idx, work_units, checkpoint_dir)

    async def _process_round(
        self,
        round_items: list[tuple[TumorWorkUnit, int]],
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Process all items in a round concurrently."""

        async def _process_one(wu: TumorWorkUnit, chunk_idx: int) -> None:
            async with semaphore:
                tumor_context = (
                    f"{wu.tumor.cancer_type} at {wu.tumor.primary_site_hint}, "
                    f"diagnosed approximately {wu.tumor.approximate_date}"
                )
                extractor = ChunkExtractor(
                    config=self._config,
                    dictionary=self._dict,
                    code_resolver=self._resolver,
                    llm_client=self._llm,
                    schema_builder=self._schema_builder,
                    schema_registry=self._schema_reg,
                    tumor_context=tumor_context,
                    llm_log=self._llm_log,
                )
                chunk = wu.chunks[chunk_idx]
                updated = await extractor.extract(chunk, wu.current_extraction)
                wu.current_extraction = updated

        tasks = [_process_one(wu, ci) for wu, ci in round_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                wu, ci = round_items[i]
                logger.error(
                    "Error processing patient %s tumor %d chunk %d: %s",
                    wu.patient_id,
                    wu.tumor_index,
                    ci,
                    result,
                )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        round_idx: int,
        work_units: list[TumorWorkUnit],
        checkpoint_dir: Path,
    ) -> None:
        """Save extraction state after a completed round."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save per-work-unit extraction state
        state_data: list[dict] = []
        for wu in work_units:
            state_data.append({
                "patient_id": wu.patient_id,
                "tumor_index": wu.tumor_index,
                "extraction_state": serialize_extraction_state(wu.current_extraction),
            })

        round_file = checkpoint_dir / f"round_{round_idx:04d}.json"
        round_file.write_text(json.dumps(state_data, indent=2))

        # Update metadata
        metadata = {
            "completed_rounds": round_idx + 1,
            "total_work_units": len(work_units),
        }
        meta_file = checkpoint_dir / "metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2))

        logger.info("Checkpoint saved: %s", round_file)

    def _load_checkpoints(
        self,
        work_units: list[TumorWorkUnit],
        checkpoint_dir: Path,
    ) -> int:
        """Load checkpoints and return the round to resume from.

        Returns 0 if no checkpoints found.
        """
        meta_file = checkpoint_dir / "metadata.json"
        if not meta_file.exists():
            return 0

        metadata = json.loads(meta_file.read_text())
        completed_rounds = metadata.get("completed_rounds", 0)

        if completed_rounds <= 0:
            return 0

        # Load the latest round checkpoint
        latest_round = completed_rounds - 1
        round_file = checkpoint_dir / f"round_{latest_round:04d}.json"
        if not round_file.exists():
            logger.warning(
                "Metadata says %d rounds completed but %s not found",
                completed_rounds,
                round_file,
            )
            return 0

        state_data = json.loads(round_file.read_text())

        # Build lookup for work units
        wu_map: dict[tuple[str, int], TumorWorkUnit] = {
            (wu.patient_id, wu.tumor_index): wu for wu in work_units
        }

        restored = 0
        for entry in state_data:
            key = (entry["patient_id"], entry["tumor_index"])
            wu = wu_map.get(key)
            if wu is not None:
                wu.current_extraction = deserialize_extraction_state(
                    entry["extraction_state"]
                )
                restored += 1

        logger.info(
            "Restored %d work units from round %d checkpoint",
            restored,
            latest_round,
        )
        return completed_rounds
