"""Oncology Registry NAACCR v26 Abstraction Pipeline - Main Orchestrator.

Sequential chunking with round-based parallel extraction.
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from onc_registry_pipeline.config import PipelineConfig
from onc_registry_pipeline.dictionary.loader import NAACCRDictionary
from onc_registry_pipeline.dictionary.code_resolver import CodeResolver
from onc_registry_pipeline.dictionary.schema_registry import SchemaRegistry
from onc_registry_pipeline.ingest.reader import DataReader, PatientDocumentSet
from onc_registry_pipeline.ingest.sequential_chunker import SequentialChunker
from onc_registry_pipeline.llm.client import VLLMClient
from onc_registry_pipeline.llm.structured_output import SchemaBuilder
from onc_registry_pipeline.manuals.seer import SEERManualContextProvider
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorDetector
from onc_registry_pipeline.extraction.base import ExtractionResult
from onc_registry_pipeline.convert import parse_naaccr_xml, write_json
from onc_registry_pipeline.extraction.round_orchestrator import (
    RoundOrchestrator,
    TumorWorkUnit,
)

# Validation imports
try:
    from onc_registry_pipeline.validation.cross_field import CrossFieldValidator
except ImportError:
    CrossFieldValidator = None
try:
    from onc_registry_pipeline.validation.consistency import InternalConsistencyChecker
except ImportError:
    InternalConsistencyChecker = None
try:
    from onc_registry_pipeline.validation.confidence import ConfidenceScorer
except ImportError:
    ConfidenceScorer = None

# Output imports
try:
    from onc_registry_pipeline.output.naaccr_writer import NAACCRWriter
except ImportError:
    NAACCRWriter = None
try:
    from onc_registry_pipeline.output.audit_trail import AuditTrail
except ImportError:
    AuditTrail = None
try:
    from onc_registry_pipeline.output.review_queue import ReviewQueue
except ImportError:
    ReviewQueue = None
try:
    from onc_registry_pipeline.output.llm_log import LLMLog
except ImportError:
    LLMLog = None

logger = logging.getLogger(__name__)


class OncRegistryExtractionPipeline:
    """Main pipeline orchestrator with round-based parallel extraction.

    Usage::

        config = PipelineConfig(...)
        pipeline = OncRegistryExtractionPipeline(config)
        summary = asyncio.run(pipeline.run("input.csv", "output_dir/"))
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dictionary = NAACCRDictionary(config)
        self.code_resolver: Optional[CodeResolver] = None
        self.schema_registry = SchemaRegistry()
        self.schema_builder = SchemaBuilder()
        self.manual_context_provider = SEERManualContextProvider(
            manuals_dir=config.seer_manuals_dir,
            max_chars=config.seer_context_max_chars,
        )
        self.llm_client = VLLMClient(
            base_url=config.vllm_base_url,
            temperature=config.vllm_temperature,
            max_tokens=config.vllm_max_tokens,
            timeout=config.vllm_timeout,
            max_retries=config.max_retries,
            reasoning_parser=config.vllm_reasoning_parser,
        )
        self.chunker = SequentialChunker(
            chunk_target_tokens=config.chunk_target_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
        )
        self.audit = AuditTrail() if AuditTrail else None
        self.review_queue = ReviewQueue() if ReviewQueue else None
        self.llm_log = None

        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load dictionary, discover the vLLM model."""
        logger.info("Loading NAACCR v26 data dictionary...")
        self.dictionary.load()
        self.code_resolver = CodeResolver(self.dictionary)

        logger.info("Discovering vLLM model...")
        model_profile = await self.llm_client.initialize()
        logger.info(
            "Model: %s, Size: %s, Context: %d, Reasoning parser: %s",
            model_profile.model_name,
            model_profile.model_size_class,
            model_profile.context_window,
            model_profile.reasoning_parser or "none",
        )

        # Adjust chunk size if model context is small
        effective_chunk = min(
            self.config.chunk_target_tokens,
            model_profile.context_window // 3,
        )
        if effective_chunk != self.config.chunk_target_tokens:
            logger.info(
                "Adjusting chunk target from %d to %d tokens (model context)",
                self.config.chunk_target_tokens,
                effective_chunk,
            )
            self.chunker = SequentialChunker(
                chunk_target_tokens=effective_chunk,
                chunk_overlap_tokens=self.config.chunk_overlap_tokens,
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, input_path: str, output_dir: str) -> dict:
        """Execute the full pipeline."""
        await self.initialize()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize LLM log
        if LLMLog:
            self.llm_log = LLMLog(str(output_path / "llm_calls.jsonl"))

        # 1. Load data
        logger.info("Loading input data from %s ...", input_path)
        reader = DataReader()
        patient_sets, structured_cols = reader.load(input_path)
        logger.info("Loaded %d patients", len(patient_sets))
        if structured_cols:
            logger.info("Detected structured columns: %s", structured_cols)

        self._start_time = time.time()

        # 2. Chunk each patient + detect tumors + build work units
        logger.info("Chunking patients and detecting tumors...")
        all_work_units: list[TumorWorkUnit] = []
        detector = TumorDetector(
            self.llm_client,
            self.schema_builder,
            llm_log=self.llm_log,
        )

        for patient_set in patient_sets:
            chunks = self.chunker.chunk_documents(patient_set.documents)
            if not chunks:
                logger.warning("Patient %s: no chunks produced", patient_set.patient_id)
                continue

            # Detect tumors
            tumors = await detector.detect(chunks)
            logger.info(
                "Patient %s: %d chunk(s), %d tumor(s)",
                patient_set.patient_id,
                len(chunks),
                len(tumors),
            )

            # Build work units
            for tumor in tumors:
                prior = self._build_structured_prior(patient_set)
                wu = TumorWorkUnit(
                    patient_id=patient_set.patient_id,
                    tumor_index=tumor.tumor_index,
                    tumor=tumor,
                    chunks=chunks,
                    current_extraction=prior,
                )
                all_work_units.append(wu)

        logger.info(
            "Total: %d work units (%d patients × tumors)",
            len(all_work_units),
            len(patient_sets),
        )

        # 3. Round-based extraction
        orchestrator = RoundOrchestrator(
            config=self.config,
            dictionary=self.dictionary,
            code_resolver=self.code_resolver,
            llm_client=self.llm_client,
            schema_builder=self.schema_builder,
            schema_registry=self.schema_registry,
            manual_context_provider=self.manual_context_provider,
            llm_log=self.llm_log,
        )

        await orchestrator.run_all_rounds(
            all_work_units,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        # 4. Validate
        for wu in all_work_units:
            self._validate_record(wu.current_extraction, wu.patient_id, wu.tumor_index)

            # Audit trail
            if self.audit:
                for result in wu.current_extraction.values():
                    if isinstance(result, ExtractionResult):
                        self.audit.record_from_result(
                            result,
                            patient_id=wu.patient_id,
                            tumor_index=wu.tumor_index,
                        )

        # 5. Write outputs
        all_records = [
            (wu.patient_id, wu.tumor_index, wu.current_extraction)
            for wu in all_work_units
        ]
        await self._write_outputs(all_records, output_path)

        # Cleanup
        if self.llm_log:
            self.llm_log.close()
        await self.llm_client.close()

        elapsed = time.time() - self._start_time
        summary = {
            "patients_processed": len(patient_sets),
            "total_records": len(all_records),
            "elapsed_seconds": round(elapsed, 1),
            "patients_per_minute": (
                round(len(patient_sets) / (elapsed / 60), 1) if elapsed > 0 else 0
            ),
        }
        logger.info("Pipeline complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_structured_prior(
        self, patient_set: PatientDocumentSet
    ) -> dict[int, ExtractionResult]:
        """Pre-populate structured data into extraction state."""
        prior: dict[int, ExtractionResult] = {}
        for item_num, value in patient_set.structured_data.items():
            item = self.dictionary.get_item(item_num)
            if item:
                prior[item_num] = ExtractionResult(
                    item_number=item_num,
                    item_name=item.name,
                    extracted_value=str(value),
                    resolved_code=str(value),
                    confidence=1.0,
                    evidence_text="From structured input column",
                    source_chunk_id="structured",
                    source_chunk_type="structured",
                    pass_number=0,
                )
        return prior

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_record(
        self,
        record: dict[int, ExtractionResult],
        patient_id: str,
        tumor_index: int,
    ) -> None:
        """Run cross-field validation, consistency checks, and confidence scoring."""
        if CrossFieldValidator is None:
            return

        validator = CrossFieldValidator()
        violations = validator.validate(record)

        if InternalConsistencyChecker is not None:
            checker = InternalConsistencyChecker()
            violations.extend(checker.check(record))

        record_updated, _fix_descriptions = validator.auto_fix(record, violations)
        record.update(record_updated)

        if ConfidenceScorer is not None:
            scorer = ConfidenceScorer()
            scores = scorer.score_record(record, violations)
            review_items = scorer.flag_for_review(
                record, scores,
                patient_id=patient_id,
                tumor_index=tumor_index,
            )
            if self.review_queue and review_items:
                self.review_queue.add_items(review_items)

    # ------------------------------------------------------------------
    # Output generation
    # ------------------------------------------------------------------

    async def _write_outputs(
        self,
        all_records: list[tuple[str, int, dict[int, ExtractionResult]]],
        output_path: Path,
    ) -> None:
        """Write all output files."""
        if not all_records:
            logger.warning("No records to write")
            return

        value_records: list[dict[int, str]] = []
        patient_groups: dict[str, list[int]] = {}

        for patient_id, _tumor_index, record in all_records:
            value_dict: dict[int, str] = {}
            for item_num, result in record.items():
                if isinstance(result, ExtractionResult):
                    value_dict[item_num] = result.resolved_code or result.extracted_value
            value_dict.setdefault(10, "A")
            value_dict.setdefault(50, "260")

            record_index = len(value_records)
            value_records.append(value_dict)
            patient_groups.setdefault(patient_id, []).append(record_index)

        if NAACCRWriter is not None:
            writer = NAACCRWriter(self.dictionary)
            fmt = self.config.output_format

            if fmt == "naaccr_xml":
                xml_path = str(output_path / "naaccr_output.xml")
                writer.write_xml(value_records, xml_path, patient_groups)
                logger.info("NAACCR XML written to %s", xml_path)
                json_path = str(output_path / "naaccr_output.json")
                json_records = parse_naaccr_xml(xml_path)
                write_json(json_records, json_path, skip_empty=True)
                logger.info("JSON written to %s", json_path)
            elif fmt == "naaccr_flat":
                flat_path = str(output_path / "naaccr_output.dat")
                writer.write_flat_file(value_records, flat_path)
                logger.info("NAACCR flat file written to %s", flat_path)
            elif fmt == "csv":
                csv_path = str(output_path / "naaccr_output.csv")
                writer.write_csv(value_records, csv_path)
                logger.info("CSV written to %s", csv_path)

        if self.audit:
            audit_path = str(output_path / "audit_trail.csv")
            self.audit.export_csv(audit_path)
            logger.info("Audit trail written to %s", audit_path)

        if self.review_queue:
            review_path = str(output_path / "review_queue.csv")
            self.review_queue.export_csv(review_path)
            logger.info("Review queue written to %s", review_path)


# ======================================================================
# CLI entry point
# ======================================================================

def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Oncology Registry NAACCR v26 Abstraction Pipeline",
    )
    parser.add_argument("input", help="Path to input CSV/TSV/Parquet file")
    parser.add_argument("output", help="Path to output directory")
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000/v1",
        help="vLLM server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=16,
        help="Max concurrent diagnosis work units (default: %(default)s)",
    )
    parser.add_argument(
        "--format", choices=["naaccr_xml", "naaccr_flat", "csv"],
        default="naaccr_xml",
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7,
        help="Confidence threshold for human review (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dict", default="NAACCRDataItems",
        help="Path to NAACCR data dictionary directory (default: %(default)s)",
    )
    parser.add_argument(
        "--seer-manuals-dir", default="SEERManuals",
        help="Path to vendored SEER/NAACCR manuals directory (default: %(default)s)",
    )
    parser.add_argument(
        "--seer-context-max-chars", type=int, default=12000,
        help="Max SEER/NAACCR manual context characters per prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="LLM sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16384,
        help="Max tokens per LLM response (default: %(default)s)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max LLM call retries (default: %(default)s)",
    )
    parser.add_argument(
        "--reasoning-parser",
        default="auto",
        help=(
            "vLLM reasoning parser name for client-side fallback parsing. "
            "Use 'auto' for model-name defaults, or 'none' to disable "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50000,
        help="Chunk size in tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--items-per-call", type=int, default=50,
        help="NAACCR items per LLM call (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory for round checkpoints (enables resume)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dict_dir = Path(args.data_dict)
    config = PipelineConfig(
        vllm_base_url=args.vllm_url,
        vllm_temperature=args.temperature,
        vllm_max_tokens=args.max_tokens,
        vllm_reasoning_parser=args.reasoning_parser,
        max_retries=args.max_retries,
        max_concurrent_patients=args.max_concurrent,
        output_format=args.format,
        confidence_threshold=args.confidence_threshold,
        chunk_target_tokens=args.chunk_size,
        items_per_call=args.items_per_call,
        data_items_csv=dict_dir / "DataItems.csv",
        code_list_csv=dict_dir / "CodeList.csv",
        alternate_names_csv=dict_dir / "AlternateNames.csv",
        seer_manuals_dir=Path(args.seer_manuals_dir),
        seer_context_max_chars=args.seer_context_max_chars,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
    )

    pipeline = OncRegistryExtractionPipeline(config)
    summary = asyncio.run(pipeline.run(args.input, args.output))

    print(f"\nPipeline complete.")
    print(f"  Patients processed: {summary['patients_processed']}")
    print(f"  NAACCR records:     {summary['total_records']}")
    print(f"  Time:               {summary['elapsed_seconds']}s")
    print(f"  Rate:               {summary['patients_per_minute']} patients/min")


if __name__ == "__main__":
    main()
