"""Round-based parallel extraction across detected diagnosis work units.

Organizes work so Round N = Nth chunk from each patient/diagnosis work unit,
enabling cross-patient and cross-diagnosis parallelism via batched vLLM
inference. Supports checkpointing for resume after interruption.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from onc_registry_pipeline.checkpoint import atomic_write_json, read_json
from onc_registry_pipeline.config import PipelineConfig
from onc_registry_pipeline.dictionary.loader import NAACCRDictionary
from onc_registry_pipeline.dictionary.code_resolver import CodeResolver
from onc_registry_pipeline.dictionary.schema_registry import SchemaRegistry
from onc_registry_pipeline.extraction.base import (
    ExtractionResult,
    serialize_extraction_state,
    deserialize_extraction_state,
)
from onc_registry_pipeline.extraction.chunk_extractor import ChunkExtractor
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorCandidate
from onc_registry_pipeline.llm.client import VLLMClient
from onc_registry_pipeline.llm.structured_output import SchemaBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Work unit
# ---------------------------------------------------------------------------

@dataclass
class TumorWorkUnit:
    """Tracks extraction state for one detected diagnosis across rounds."""

    patient_id: str
    tumor_index: int
    tumor: TumorCandidate
    chunks: list[Any]  # list[SequentialChunk]
    current_extraction: dict[int, ExtractionResult] = field(default_factory=dict)
    completed_chunks: set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Round orchestrator
# ---------------------------------------------------------------------------

class RoundOrchestrator:
    """Manages round-based parallel extraction across diagnosis work units."""

    def __init__(
        self,
        config: PipelineConfig,
        dictionary: NAACCRDictionary,
        code_resolver: CodeResolver,
        llm_client: VLLMClient,
        schema_builder: SchemaBuilder,
        schema_registry: SchemaRegistry,
        manual_context_provider: Any = None,
        llm_log: Any = None,
    ) -> None:
        self._config = config
        self._dict = dictionary
        self._resolver = code_resolver
        self._llm = llm_client
        self._schema_builder = schema_builder
        self._schema_reg = schema_registry
        self._manual_context_provider = manual_context_provider
        self._llm_log = llm_log
        self._checkpoint_lock = asyncio.Lock()

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

        start_round = 0
        if checkpoint_dir is not None:
            start_round = self._load_checkpoints(work_units, checkpoint_dir, rounds)
            if start_round > 0:
                logger.info("Resuming from round %d (of %d)", start_round, len(rounds))

        semaphore = asyncio.Semaphore(self._config.max_concurrent_patients)

        for round_idx in range(start_round, len(rounds)):
            round_items = [
                (wu, chunk_idx)
                for wu, chunk_idx in rounds[round_idx]
                if chunk_idx not in wu.completed_chunks
            ]
            if not round_items:
                logger.info(
                    "Round %d/%d already complete; skipping.",
                    round_idx + 1,
                    len(rounds),
                )
                continue

            round_start = time.time()

            logger.info(
                "Round %d/%d: processing %d work units",
                round_idx + 1,
                len(rounds),
                len(round_items),
            )

            await self._process_round(
                round_items,
                semaphore,
                checkpoint_dir=checkpoint_dir,
                work_units=work_units,
                rounds=rounds,
                round_idx=round_idx,
            )

            elapsed = time.time() - round_start
            logger.info(
                "Round %d/%d completed in %.1fs",
                round_idx + 1,
                len(rounds),
                elapsed,
            )

            if checkpoint_dir is not None:
                self._save_checkpoint(round_idx, work_units, checkpoint_dir, rounds)

    async def _process_round(
        self,
        round_items: list[tuple[TumorWorkUnit, int]],
        semaphore: asyncio.Semaphore,
        *,
        checkpoint_dir: Path | None,
        work_units: list[TumorWorkUnit],
        rounds: list[list[tuple[TumorWorkUnit, int]]],
        round_idx: int,
    ) -> None:
        """Process all items in a round concurrently."""

        async def _process_one(wu: TumorWorkUnit, chunk_idx: int) -> None:
            async with semaphore:
                if chunk_idx in wu.completed_chunks:
                    return

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
                    manual_context_provider=self._manual_context_provider,
                    tumor=wu.tumor,
                    tumor_context=tumor_context,
                    llm_log=self._llm_log,
                )
                chunk = wu.chunks[chunk_idx]
                updated = await extractor.extract(chunk, wu.current_extraction)
                wu.current_extraction = updated
                wu.completed_chunks.add(chunk_idx)

                if checkpoint_dir is not None:
                    async with self._checkpoint_lock:
                        self._save_checkpoint(
                            round_idx,
                            work_units,
                            checkpoint_dir,
                            rounds,
                        )

        tasks = [_process_one(wu, ci) for wu, ci in round_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        failures: list[Exception] = []
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
                failures.append(result)

        if failures:
            raise RuntimeError(
                f"{len(failures)} work unit(s) failed in round {round_idx}"
            ) from failures[0]

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        round_idx: int,
        work_units: list[TumorWorkUnit],
        checkpoint_dir: Path,
        rounds: list[list[tuple[TumorWorkUnit, int]]],
    ) -> None:
        """Save extraction state and per-work-unit completion progress."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state_data: list[dict] = []
        for wu in work_units:
            state_data.append({
                "patient_id": wu.patient_id,
                "tumor_index": wu.tumor_index,
                "completed_chunks": sorted(wu.completed_chunks),
                "extraction_state": serialize_extraction_state(wu.current_extraction),
            })

        round_file = checkpoint_dir / f"round_{round_idx:04d}.json"
        atomic_write_json(round_file, state_data)

        metadata = {
            "schema_version": 2,
            "latest_round": round_idx,
            "completed_rounds": self._completed_round_count(rounds),
            "total_work_units": len(work_units),
        }
        meta_file = checkpoint_dir / "metadata.json"
        atomic_write_json(meta_file, metadata)

        logger.info("Checkpoint saved: %s", round_file)

    def _load_checkpoints(
        self,
        work_units: list[TumorWorkUnit],
        checkpoint_dir: Path,
        rounds: list[list[tuple[TumorWorkUnit, int]]],
    ) -> int:
        """Load checkpoints and return the round to resume from.

        Returns 0 if no checkpoints found.
        """
        meta_file = checkpoint_dir / "metadata.json"
        if not meta_file.exists():
            return 0

        try:
            metadata = read_json(meta_file)
        except Exception as exc:
            logger.warning("Could not read checkpoint metadata %s: %s", meta_file, exc)
            return 0

        completed_rounds = metadata.get("completed_rounds", 0)
        latest_round = metadata.get("latest_round")

        if completed_rounds <= 0 and latest_round is None:
            return 0

        if latest_round is None:
            latest_round = completed_rounds - 1
        round_file = checkpoint_dir / f"round_{latest_round:04d}.json"
        if not round_file.exists():
            logger.warning(
                "Checkpoint metadata points to round %d but %s was not found",
                latest_round,
                round_file,
            )
            return 0

        try:
            state_data = read_json(round_file)
        except Exception as exc:
            logger.warning("Could not read checkpoint %s: %s", round_file, exc)
            return 0

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
                if "completed_chunks" in entry:
                    completed_chunks: set[int] = set()
                    for chunk_idx in entry.get("completed_chunks", []):
                        try:
                            parsed_idx = int(chunk_idx)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= parsed_idx < len(wu.chunks):
                            completed_chunks.add(parsed_idx)
                    wu.completed_chunks = completed_chunks
                else:
                    wu.completed_chunks = {
                        chunk_idx
                        for chunk_idx in range(min(completed_rounds, len(wu.chunks)))
                    }
                restored += 1

        logger.info(
            "Restored %d work units from round %d checkpoint",
            restored,
            latest_round,
        )
        return self._first_incomplete_round(rounds)

    @staticmethod
    def _completed_round_count(
        rounds: list[list[tuple[TumorWorkUnit, int]]],
    ) -> int:
        """Return contiguous fully completed rounds from the beginning."""
        count = 0
        for round_items in rounds:
            if all(chunk_idx in wu.completed_chunks for wu, chunk_idx in round_items):
                count += 1
                continue
            break
        return count

    @staticmethod
    def _first_incomplete_round(
        rounds: list[list[tuple[TumorWorkUnit, int]]],
    ) -> int:
        """Return the first round that still has pending work."""
        for round_idx, round_items in enumerate(rounds):
            if any(
                chunk_idx not in wu.completed_chunks
                for wu, chunk_idx in round_items
            ):
                return round_idx
        return len(rounds)
