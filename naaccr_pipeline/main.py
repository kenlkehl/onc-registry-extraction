"""NAACCR v26 Cancer Registry Abstraction Pipeline - Main Orchestrator."""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from naaccr_pipeline.config import PipelineConfig
from naaccr_pipeline.dictionary.loader import NAACCRDictionary
from naaccr_pipeline.dictionary.code_resolver import CodeResolver
from naaccr_pipeline.dictionary.schema_registry import SchemaRegistry
from naaccr_pipeline.ingest.reader import DataReader, PatientDocumentSet
from naaccr_pipeline.ingest.chunker import ClinicalChunker
from naaccr_pipeline.llm.client import VLLMClient
from naaccr_pipeline.llm.structured_output import SchemaBuilder
from naaccr_pipeline.extraction.pass0_tumor_detection import TumorDetector, TumorCandidate
from naaccr_pipeline.extraction.pass1_demographics import Pass1Demographics
from naaccr_pipeline.extraction.pass2_staging import Pass2Staging
from naaccr_pipeline.extraction.pass3_treatment import Pass3Treatment
from naaccr_pipeline.extraction.pass4_followup import Pass4Followup
from naaccr_pipeline.extraction.base import ExtractionResult

# Validation imports -- these modules may not exist yet.
try:
    from naaccr_pipeline.validation.cross_field import CrossFieldValidator
except ImportError:
    CrossFieldValidator = None
try:
    from naaccr_pipeline.validation.consistency import InternalConsistencyChecker
except ImportError:
    InternalConsistencyChecker = None
try:
    from naaccr_pipeline.validation.confidence import ConfidenceScorer
except ImportError:
    ConfidenceScorer = None

# Output imports -- audit_trail and naaccr_writer exist; review_queue may not.
try:
    from naaccr_pipeline.output.naaccr_writer import NAACCRWriter
except ImportError:
    NAACCRWriter = None
try:
    from naaccr_pipeline.output.audit_trail import AuditTrail
except ImportError:
    AuditTrail = None
try:
    from naaccr_pipeline.output.review_queue import ReviewQueue
except ImportError:
    ReviewQueue = None
try:
    from naaccr_pipeline.output.llm_log import LLMLog
except ImportError:
    LLMLog = None

logger = logging.getLogger(__name__)


class NAACCRExtractionPipeline:
    """Main pipeline orchestrator with async patient-level parallelism.

    Usage::

        config = PipelineConfig(...)
        pipeline = NAACCRExtractionPipeline(config)
        summary = asyncio.run(pipeline.run("input.csv", "output_dir/"))
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dictionary = NAACCRDictionary(config)
        self.code_resolver: Optional[CodeResolver] = None
        self.schema_registry = SchemaRegistry()
        self.schema_builder = SchemaBuilder()
        self.llm_client = VLLMClient(
            base_url=config.vllm_base_url,
            temperature=config.vllm_temperature,
            max_tokens=config.vllm_max_tokens,
            timeout=config.vllm_timeout,
            max_retries=config.max_retries,
        )
        self.chunker = ClinicalChunker(
            chunk_target_tokens=config.chunk_target_tokens,
            chunk_overlap_tokens=config.chunk_overlap_tokens,
        )
        self.audit = AuditTrail() if AuditTrail else None
        self.review_queue = ReviewQueue() if ReviewQueue else None
        self.llm_log = None  # Initialized in run() once output_dir is known

        # Concurrency control
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._patients_processed = 0
        self._total_patients = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load dictionary, discover the vLLM model, and tune chunker."""
        logger.info("Loading NAACCR v26 data dictionary...")
        self.dictionary.load()
        self.code_resolver = CodeResolver(self.dictionary)

        logger.info("Discovering vLLM model...")
        model_profile = await self.llm_client.initialize()
        logger.info(
            "Model: %s, Size: %s, Context: %d, Items/call: %d",
            model_profile.model_name,
            model_profile.model_size_class,
            model_profile.context_window,
            model_profile.items_per_call,
        )

        # Re-create chunker if the model context window is small enough to
        # warrant a smaller chunk target.
        effective_chunk_tokens = min(
            self.config.chunk_target_tokens,
            model_profile.context_window // 3,  # leave room for prompt + output
        )
        if effective_chunk_tokens != self.config.chunk_target_tokens:
            logger.info(
                "Adjusting chunk target from %d to %d tokens based on model context.",
                self.config.chunk_target_tokens,
                effective_chunk_tokens,
            )
            self.chunker = ClinicalChunker(
                chunk_target_tokens=effective_chunk_tokens,
                chunk_overlap_tokens=self.config.chunk_overlap_tokens,
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, input_path: str, output_dir: str) -> dict:
        """Execute the full pipeline.

        Parameters
        ----------
        input_path : str
            Path to the input CSV / TSV / Parquet file.
        output_dir : str
            Directory where output files will be written.

        Returns
        -------
        dict
            Summary statistics with keys ``patients_processed``,
            ``total_records``, ``elapsed_seconds``, ``patients_per_minute``.
        """
        await self.initialize()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize LLM call log
        if LLMLog:
            self.llm_log = LLMLog(str(output_path / "llm_calls.jsonl"))
            logger.info("LLM call logging enabled -> %s", self.llm_log.path)

        # Load data
        logger.info("Loading input data from %s ...", input_path)
        reader = DataReader()
        patient_sets, structured_cols = reader.load(input_path)
        logger.info("Loaded %d patients", len(patient_sets))
        if structured_cols:
            logger.info("Detected structured columns: %s", structured_cols)

        self._total_patients = len(patient_sets)
        self._patients_processed = 0
        self._start_time = time.time()

        # Process patients with bounded concurrency
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_patients)

        tasks = [
            self._process_patient_with_semaphore(patient_set)
            for patient_set in patient_sets
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful records
        all_records: list[tuple[str, int, dict[int, ExtractionResult]]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Patient processing failed: %s", result)
                continue
            all_records.extend(result)

        # Generate output files
        await self._write_outputs(all_records, output_path)

        # Close LLM log
        if self.llm_log:
            self.llm_log.close()

        # Shut down LLM client
        await self.llm_client.close()

        elapsed = time.time() - self._start_time
        summary = {
            "patients_processed": self._patients_processed,
            "total_records": len(all_records),
            "elapsed_seconds": round(elapsed, 1),
            "patients_per_minute": (
                round(self._patients_processed / (elapsed / 60), 1)
                if elapsed > 0
                else 0
            ),
        }
        logger.info("Pipeline complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Patient-level processing
    # ------------------------------------------------------------------

    async def _process_patient_with_semaphore(
        self,
        patient_set: PatientDocumentSet,
    ) -> list[tuple[str, int, dict[int, ExtractionResult]]]:
        """Acquire semaphore, process one patient, log progress."""
        async with self._semaphore:
            try:
                result = await self.process_patient(patient_set)
                self._patients_processed += 1
                if self._patients_processed % 10 == 0:
                    elapsed = time.time() - self._start_time
                    rate = (
                        self._patients_processed / (elapsed / 60) if elapsed > 0 else 0
                    )
                    logger.info(
                        "Progress: %d/%d (%.1f patients/min)",
                        self._patients_processed,
                        self._total_patients,
                        rate,
                    )
                return result
            except Exception as e:
                logger.error(
                    "Error processing patient %s: %s", patient_set.patient_id, e
                )
                raise

    async def process_patient(
        self,
        patient_set: PatientDocumentSet,
    ) -> list[tuple[str, int, dict[int, ExtractionResult]]]:
        """Process a single patient through all extraction passes.

        Returns a list of ``(patient_id, tumor_index, record)`` tuples --
        one per detected tumor.  Each *record* is a dict mapping NAACCR
        item number to :class:`ExtractionResult`.
        """
        patient_id = patient_set.patient_id

        # Chunk all of the patient's documents
        chunks = self.chunker.chunk_documents(patient_set.documents)
        if not chunks:
            logger.warning("Patient %s: no chunks produced", patient_id)
            return []

        # Pre-populate structured data into the results dict
        prior_results: dict[int, ExtractionResult] = {}
        for item_num, value in patient_set.structured_data.items():
            item = self.dictionary.get_item(item_num)
            if item:
                prior_results[item_num] = ExtractionResult(
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

        # Pass 0: Tumor Detection
        detector = TumorDetector(self.llm_client, self.schema_builder)
        tumors = await detector.detect(chunks)
        logger.info("Patient %s: detected %d tumor(s)", patient_id, len(tumors))

        # Process each detected tumor through Passes 1-4
        results: list[tuple[str, int, dict[int, ExtractionResult]]] = []
        for tumor in tumors:
            record = await self._process_tumor(
                patient_id, tumor, chunks, dict(prior_results)
            )
            results.append((patient_id, tumor.tumor_index, record))

        return results

    # ------------------------------------------------------------------
    # Per-tumor processing (Passes 1-4 + validation)
    # ------------------------------------------------------------------

    async def _process_tumor(
        self,
        patient_id: str,
        tumor: TumorCandidate,
        chunks: list,
        prior_results: dict[int, ExtractionResult],
    ) -> dict[int, ExtractionResult]:
        """Process a single tumor through Passes 1-4 and validation.

        Returns a record dict mapping item_number to ExtractionResult.
        """
        # Build a context string for Pass 1 from the tumor candidate
        tumor_context = (
            f"Focus on: {tumor.cancer_type} at {tumor.primary_site_hint}, "
            f"diagnosed approximately {tumor.approximate_date}"
        )

        # Filter to chunks relevant to this tumor (if the detector
        # assigned specific chunk IDs).  Always include all pathology
        # chunks regardless.
        relevant_chunks = chunks
        if tumor.relevant_chunk_ids:
            relevant_set = set(tumor.relevant_chunk_ids)
            relevant = [c for c in chunks if c.chunk_id in relevant_set]
            pathology_extras = [
                c
                for c in chunks
                if c.chunk_type == "pathology" and c.chunk_id not in relevant_set
            ]
            relevant_chunks = relevant + pathology_extras if relevant else chunks

        # Set patient context on the LLM log so entries are tagged.
        if self.llm_log:
            self.llm_log.set_patient_context(patient_id, tumor.tumor_index)

        # -- Pass 1: Demographics + Cancer Identification --
        pass1 = Pass1Demographics(
            config=self.config,
            dictionary=self.dictionary,
            code_resolver=self.code_resolver,
            llm_client=self.llm_client,
            schema_builder=self.schema_builder,
            tumor_context=tumor_context,
            llm_log=self.llm_log,
        )
        results_1 = await pass1.run(relevant_chunks, prior_results)
        record: dict[int, ExtractionResult] = dict(prior_results)
        for r in results_1:
            record[r.item_number] = r

        # -- Pass 2: Staging (site-specific) --
        pass2 = Pass2Staging(
            config=self.config,
            dictionary=self.dictionary,
            code_resolver=self.code_resolver,
            llm_client=self.llm_client,
            schema_builder=self.schema_builder,
            schema_registry=self.schema_registry,
            llm_log=self.llm_log,
        )
        results_2 = await pass2.run(relevant_chunks, record)
        for r in results_2:
            record[r.item_number] = r

        # -- Pass 3: Treatment --
        # Derive a primary-site description from Pass 1 results for the
        # treatment prompt.  Item 400 = Primary Site.
        primary_site_desc = ""
        result_400 = record.get(400)
        if result_400 is not None:
            primary_site_desc = result_400.resolved_code or result_400.extracted_value

        pass3 = Pass3Treatment(
            config=self.config,
            dictionary=self.dictionary,
            code_resolver=self.code_resolver,
            llm_client=self.llm_client,
            schema_builder=self.schema_builder,
            primary_site=primary_site_desc or "unknown",
            llm_log=self.llm_log,
        )
        results_3 = await pass3.run(relevant_chunks, record)
        for r in results_3:
            record[r.item_number] = r

        # -- Pass 4: Follow-up + Text Summaries --
        pass4 = Pass4Followup(
            config=self.config,
            dictionary=self.dictionary,
            code_resolver=self.code_resolver,
            llm_client=self.llm_client,
            schema_builder=self.schema_builder,
            llm_log=self.llm_log,
        )
        results_4 = await pass4.run(relevant_chunks, record)
        for r in results_4:
            record[r.item_number] = r

        # -- Assign Sequence Number (Item 380) --
        record[380] = ExtractionResult(
            item_number=380,
            item_name="Sequence Number--Central",
            extracted_value=f"{tumor.tumor_index:02d}",
            resolved_code=f"{tumor.tumor_index:02d}",
            confidence=1.0,
            evidence_text="Assigned by pipeline based on tumor detection order",
            source_chunk_id="pipeline",
            source_chunk_type="pipeline",
            pass_number=0,
        )

        # -- Validation --
        self._validate_record(record, patient_id, tumor.tumor_index)

        # -- Audit trail --
        if self.audit:
            for item_num, result in record.items():
                if isinstance(result, ExtractionResult):
                    self.audit.record_from_result(
                        result,
                        patient_id=patient_id,
                        tumor_index=tumor.tumor_index,
                    )

        return record

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_record(
        self,
        record: dict[int, ExtractionResult],
        patient_id: str,
        tumor_index: int,
    ) -> None:
        """Run cross-field validation, consistency checks, and confidence
        scoring if the modules are available."""
        if CrossFieldValidator is None:
            return

        validator = CrossFieldValidator()
        violations = validator.validate(record)

        if InternalConsistencyChecker is not None:
            checker = InternalConsistencyChecker()
            violations.extend(checker.check(record))

        # Attempt auto-fixes for fixable violations
        record_updated, _fix_descriptions = validator.auto_fix(record, violations)
        record.update(record_updated)

        # Confidence scoring and review-queue flagging
        if ConfidenceScorer is not None:
            scorer = ConfidenceScorer()
            scores = scorer.score_record(record, violations)
            review_items = scorer.flag_for_review(
                record,
                scores,
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
        """Write all output files (NAACCR XML/flat/CSV, audit, review queue)."""
        if not all_records:
            logger.warning("No records to write")
            return

        # Convert ExtractionResult records into plain value dicts for the
        # writer (item_number -> value_string).
        value_records: list[dict[int, str]] = []
        patient_groups: dict[str, list[int]] = {}

        for patient_id, _tumor_index, record in all_records:
            value_dict: dict[int, str] = {}
            for item_num, result in record.items():
                if isinstance(result, ExtractionResult):
                    value_dict[item_num] = result.resolved_code or result.extracted_value
            # Standard record-level defaults
            value_dict.setdefault(10, "A")    # Record Type = Abstract
            value_dict.setdefault(50, "260")  # NAACCR Record Version = v26

            record_index = len(value_records)
            value_records.append(value_dict)
            patient_groups.setdefault(patient_id, []).append(record_index)

        # Write NAACCR output files
        if NAACCRWriter is not None:
            writer = NAACCRWriter(self.dictionary)
            fmt = self.config.output_format

            if fmt == "naaccr_xml":
                xml_path = str(output_path / "naaccr_output.xml")
                writer.write_xml(value_records, xml_path, patient_groups)
                logger.info("NAACCR XML written to %s", xml_path)
            elif fmt == "naaccr_flat":
                flat_path = str(output_path / "naaccr_output.dat")
                writer.write_flat_file(value_records, flat_path)
                logger.info("NAACCR flat file written to %s", flat_path)
            elif fmt == "csv":
                csv_path = str(output_path / "naaccr_output.csv")
                writer.write_csv(value_records, csv_path)
                logger.info("CSV written to %s", csv_path)
        else:
            logger.warning("NAACCRWriter not available; skipping output file generation.")

        # Audit trail
        if self.audit:
            audit_path = str(output_path / "audit_trail.csv")
            self.audit.export_csv(audit_path)
            logger.info("Audit trail written to %s", audit_path)
            stats = self.audit.summary_stats()
            logger.info("Audit summary: %s", stats)

        # Review queue
        if self.review_queue:
            review_path = str(output_path / "review_queue.csv")
            self.review_queue.export_csv(review_path)
            logger.info("Review queue written to %s", review_path)
            summary = self.review_queue.summary()
            logger.info("Review queue summary: %s", summary)


# ======================================================================
# CLI entry point
# ======================================================================

def main() -> None:
    """Command-line entry point.

    Usage::

        python -m naaccr_pipeline.main input.csv output_dir/
    """
    parser = argparse.ArgumentParser(
        description="NAACCR v26 Cancer Registry Abstraction Pipeline",
    )
    parser.add_argument("input", help="Path to input CSV/TSV/Parquet file")
    parser.add_argument("output", help="Path to output directory")
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Max concurrent patients (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        choices=["naaccr_xml", "naaccr_flat", "csv"],
        default="naaccr_xml",
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for human review (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dict",
        default="NAACCRDataItems",
        help="Path to NAACCR data dictionary directory (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens per LLM response (default: %(default)s)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max LLM call retries (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Build config from CLI arguments
    dict_dir = Path(args.data_dict)
    config = PipelineConfig(
        vllm_base_url=args.vllm_url,
        vllm_temperature=args.temperature,
        vllm_max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        max_concurrent_patients=args.max_concurrent,
        output_format=args.format,
        confidence_threshold=args.confidence_threshold,
        data_items_csv=dict_dir / "DataItems.csv",
        code_list_csv=dict_dir / "CodeList.csv",
        alternate_names_csv=dict_dir / "AlternateNames.csv",
    )

    # Run the async pipeline
    pipeline = NAACCRExtractionPipeline(config)
    summary = asyncio.run(pipeline.run(args.input, args.output))

    # Print summary to stdout
    print(f"\nPipeline complete.")
    print(f"  Patients processed: {summary['patients_processed']}")
    print(f"  NAACCR records:     {summary['total_records']}")
    print(f"  Time:               {summary['elapsed_seconds']}s")
    print(f"  Rate:               {summary['patients_per_minute']} patients/min")


if __name__ == "__main__":
    main()
