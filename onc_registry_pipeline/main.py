"""Oncology Registry NAACCR v26 Abstraction Pipeline - Main Orchestrator.

Sequential chunking with round-based parallel extraction.
"""

from __future__ import annotations

import asyncio
import argparse
import calendar
import hashlib
import logging
import os
import re
import sys
import time
from dataclasses import asdict, fields
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from onc_registry_pipeline.checkpoint import atomic_write_json, read_json
from onc_registry_pipeline.config import PipelineConfig
from onc_registry_pipeline.dictionary.loader import NAACCRDictionary
from onc_registry_pipeline.dictionary.code_resolver import CodeResolver
from onc_registry_pipeline.dictionary.schema_registry import SchemaRegistry
from onc_registry_pipeline.ingest.reader import DataReader, PatientDocumentSet
from onc_registry_pipeline.ingest.sequential_chunker import SequentialChunker
from onc_registry_pipeline.llm.client import VLLMClient
from onc_registry_pipeline.llm.structured_output import SchemaBuilder
from onc_registry_pipeline.manuals.seer import SEERManualContextProvider
from onc_registry_pipeline.paths import (
    default_data_dict_dir,
    default_seer_manuals_dir,
    resolve_reference_path,
)
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorDetector
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorCandidate
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
try:
    from onc_registry_pipeline.diagnosis_summary import write_diagnosis_summary_csv
except ImportError:
    write_diagnosis_summary_csv = None

logger = logging.getLogger(__name__)

_DIAGNOSIS_DOCUMENT_WINDOW_MONTHS = 6
_TUMOR_CANDIDATE_FIELDS = {field.name for field in fields(TumorCandidate)}


def _add_months(value: date, months: int) -> date:
    """Add calendar months while clamping to the destination month length."""
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _parse_date_interval(value: Any) -> tuple[date, date] | None:
    """Parse exact or partial dates into an inclusive date interval."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return None

    match = re.match(
        r"^(\d{4})(?:[-/]?(\d{1,2}))?(?:[-/]?(\d{1,2}))?$",
        text,
    )
    if match:
        year = int(match.group(1))
        month_text = match.group(2)
        day_text = match.group(3)
        if month_text is None:
            return date(year, 1, 1), date(year, 12, 31)
        month = int(month_text)
        if not 1 <= month <= 12:
            return None
        if day_text is None:
            last_day = calendar.monthrange(year, month)[1]
            return date(year, month, 1), date(year, month, last_day)
        day = int(day_text)
        try:
            parsed = date(year, month, day)
        except ValueError:
            return None
        return parsed, parsed

    parsed_timestamp = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed_timestamp):
        return None
    if isinstance(parsed_timestamp, datetime):
        parsed_date = parsed_timestamp.date()
    else:
        parsed_date = parsed_timestamp.to_pydatetime().date()
    return parsed_date, parsed_date


def _diagnosis_document_window(
    diagnosis_date: str,
    months: int = _DIAGNOSIS_DOCUMENT_WINDOW_MONTHS,
) -> tuple[date, date] | None:
    """Return the inclusive extraction document window around a diagnosis date."""
    diagnosis_interval = _parse_date_interval(diagnosis_date)
    if diagnosis_interval is None:
        return None

    diagnosis_start, diagnosis_end = diagnosis_interval
    return _add_months(diagnosis_start, -months), _add_months(
        diagnosis_end, months
    )


def _documents_overlapping_window(
    documents: list[Any],
    window_start: date,
    window_end: date,
) -> list[Any]:
    """Keep documents whose document date interval overlaps the window."""
    selected: list[Any] = []
    for doc in documents:
        doc_interval = _parse_date_interval(getattr(doc, "date", ""))
        if doc_interval is None:
            continue
        doc_start, doc_end = doc_interval
        if doc_start <= window_end and doc_end >= window_start:
            selected.append(doc)
    return selected


def _patient_tumor_checkpoint_path(checkpoint_dir: Path, patient_id: str) -> Path:
    """Return a stable pass-0 checkpoint path for a patient id."""
    digest = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:20]
    return checkpoint_dir / "pass0" / f"patient_{digest}.json"


def _tumor_candidate_from_dict(data: dict[str, Any]) -> TumorCandidate:
    """Build a TumorCandidate while ignoring unknown future fields."""
    kwargs = {k: v for k, v in data.items() if k in _TUMOR_CANDIDATE_FIELDS}
    return TumorCandidate(**kwargs)


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
            base_url=self._resolve_llm_base_url(config),
            provider=config.llm_provider,
            model=config.llm_model or config.vllm_model,
            temperature=config.vllm_temperature,
            max_tokens=config.vllm_max_tokens,
            timeout=config.vllm_timeout,
            max_retries=config.max_retries,
            reasoning_parser=config.vllm_reasoning_parser,
            azure_api_key_env=config.azure_openai_api_key_env,
            azure_auth_mode=config.azure_openai_auth_mode,
            azure_token_refresh_command=config.azure_openai_token_refresh_command,
            anthropic_vertex_project_id=config.anthropic_vertex_project_id,
            anthropic_vertex_region=config.anthropic_vertex_region,
            anthropic_vertex_token_env=config.anthropic_vertex_token_env,
            anthropic_vertex_token_refresh_command=(
                config.anthropic_vertex_token_refresh_command
            ),
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
        """Load dictionary and initialize the configured LLM endpoint."""
        logger.info("Loading NAACCR v26 data dictionary...")
        self.dictionary.load()
        self.code_resolver = CodeResolver(self.dictionary)

        logger.info("Initializing %s model endpoint...", self.config.llm_provider)
        model_profile = await self.llm_client.initialize()
        logger.info(
            "Provider: %s, Model: %s, Size: %s, Context: %d, Reasoning parser: %s",
            model_profile.provider,
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
        checkpoint_dir = self.config.checkpoint_dir or (output_path / "checkpoints")
        logger.info("Using checkpoint directory: %s", checkpoint_dir)

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

            tumors = self._load_patient_tumor_checkpoint(
                patient_set.patient_id,
                checkpoint_dir,
            )
            if tumors is None:
                tumors = await detector.detect(chunks)
                self._save_patient_tumor_checkpoint(
                    patient_set.patient_id,
                    tumors,
                    checkpoint_dir,
                )
            else:
                logger.info(
                    "Patient %s: loaded %d tumor(s) from pass-0 checkpoint",
                    patient_set.patient_id,
                    len(tumors),
                )

            logger.info(
                "Patient %s: %d chunk(s), %d tumor(s)",
                patient_set.patient_id,
                len(chunks),
                len(tumors),
            )

            # Build work units
            for tumor in tumors:
                prior = self._build_structured_prior(patient_set)
                diagnosis_window = _diagnosis_document_window(tumor.approximate_date)
                if diagnosis_window is None:
                    scoped_chunks = chunks
                    if tumor.approximate_date:
                        logger.warning(
                            "Patient %s tumor %d: could not parse diagnosis date %r; "
                            "using all %d document chunk(s) for extraction",
                            patient_set.patient_id,
                            tumor.tumor_index,
                            tumor.approximate_date,
                            len(scoped_chunks),
                        )
                else:
                    window_start, window_end = diagnosis_window
                    scoped_documents = _documents_overlapping_window(
                        patient_set.documents,
                        window_start,
                        window_end,
                    )
                    scoped_chunks = self.chunker.chunk_documents(scoped_documents)
                    logger.info(
                        "Patient %s tumor %d: extraction scoped to %d/%d "
                        "document(s) from %s through %s (%d chunk(s))",
                        patient_set.patient_id,
                        tumor.tumor_index,
                        len(scoped_documents),
                        len(patient_set.documents),
                        window_start.isoformat(),
                        window_end.isoformat(),
                        len(scoped_chunks),
                    )

                wu = TumorWorkUnit(
                    patient_id=patient_set.patient_id,
                    tumor_index=tumor.tumor_index,
                    tumor=tumor,
                    chunks=scoped_chunks,
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
            checkpoint_dir=checkpoint_dir,
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
            (wu.patient_id, wu.tumor_index, wu.tumor, wu.current_extraction)
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

    @staticmethod
    def _resolve_llm_base_url(config: PipelineConfig) -> str:
        """Resolve the provider-specific base URL from config/env/defaults."""
        if config.llm_base_url:
            return config.llm_base_url
        if config.llm_provider == "azure-openai":
            return config.azure_openai_endpoint or ""
        return config.vllm_base_url

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

    def _load_patient_tumor_checkpoint(
        self,
        patient_id: str,
        checkpoint_dir: Path,
    ) -> list[TumorCandidate] | None:
        """Load cached pass-0 tumor detection results for one patient."""
        path = _patient_tumor_checkpoint_path(checkpoint_dir, patient_id)
        if not path.exists():
            return None

        try:
            payload = read_json(path)
            tumors = payload.get("tumors", [])
            if not isinstance(tumors, list):
                raise ValueError("pass-0 checkpoint tumors field is not a list")
            return [_tumor_candidate_from_dict(tumor) for tumor in tumors]
        except Exception as exc:
            logger.warning(
                "Could not load pass-0 checkpoint for patient %s from %s: %s",
                patient_id,
                path,
                exc,
            )
            return None

    def _save_patient_tumor_checkpoint(
        self,
        patient_id: str,
        tumors: list[TumorCandidate],
        checkpoint_dir: Path,
    ) -> None:
        """Persist pass-0 tumor detection results for one patient."""
        path = _patient_tumor_checkpoint_path(checkpoint_dir, patient_id)
        payload = {
            "schema_version": 1,
            "patient_id": patient_id,
            "tumors": [asdict(tumor) for tumor in tumors],
        }
        atomic_write_json(path, payload)
        logger.info("Pass-0 checkpoint saved: %s", path)

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
        all_records: list[
            tuple[str, int, TumorCandidate, dict[int, ExtractionResult]]
        ],
        output_path: Path,
    ) -> None:
        """Write all output files."""
        if not all_records:
            logger.warning("No records to write")
            return

        value_records: list[dict[int, str]] = []
        patient_groups: dict[str, list[int]] = {}

        for patient_id, _tumor_index, _tumor, record in all_records:
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

        if write_diagnosis_summary_csv is not None:
            summary_path = output_path / "diagnosis_summary.csv"
            write_diagnosis_summary_csv(all_records, summary_path, self.dictionary)
            logger.info("Diagnosis summary CSV written to %s", summary_path)

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
        "--provider",
        choices=["vllm", "azure-openai", "anthropic-vertex"],
        default="vllm",
        help="LLM endpoint provider (default: %(default)s)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "Provider endpoint base URL. For Azure, defaults to "
            "$AZURE_OPENAI_ENDPOINT. vLLM still supports --vllm-url."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model/deployment id. Defaults to provider env vars "
            "($LLM_MODEL, $AZURE_OPENAI_MODEL, $ANTHROPIC_VERTEX_MODEL) "
            "or auto when supported."
        ),
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000/v1",
        help="vLLM server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--azure-auth-mode",
        choices=["bearer", "api-key"],
        default="bearer",
        help=(
            "Azure OpenAI auth header mode. Use bearer for Entra tokens "
            "from az account get-access-token; use api-key for resource keys "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--azure-api-key-env",
        default="AZURE_OPENAI_API_KEY",
        help="Env var holding the Azure OpenAI token/key (default: %(default)s)",
    )
    parser.add_argument(
        "--azure-token-refresh-command",
        default=None,
        help=(
            "Command that prints a fresh Azure bearer token. Defaults to "
            "az account get-access-token for cognitiveservices. Pass an "
            "empty string to disable refresh."
        ),
    )
    parser.add_argument(
        "--anthropic-vertex-project-id",
        default=None,
        help=(
            "Google Cloud project id for Anthropic Vertex. Defaults to "
            "$ANTHROPIC_VERTEX_PROJECT_ID."
        ),
    )
    parser.add_argument(
        "--anthropic-vertex-region",
        default=None,
        help="Vertex region/multi-region/global. Defaults to $CLOUD_ML_REGION.",
    )
    parser.add_argument(
        "--anthropic-vertex-token-env",
        default="ANTHROPIC_VERTEX_ACCESS_TOKEN",
        help="Env var holding a Vertex bearer token (default: %(default)s)",
    )
    parser.add_argument(
        "--anthropic-vertex-token-refresh-command",
        default=None,
        help=(
            "Command that prints a fresh Vertex bearer token. Defaults to "
            "gcloud auth application-default print-access-token. Pass an "
            "empty string to disable refresh."
        ),
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
        "--data-dict", default=None,
        help=(
            "Path to NAACCR data dictionary directory "
            "(default: vendored NAACCRDataItems)"
        ),
    )
    parser.add_argument(
        "--seer-manuals-dir", default=None,
        help=(
            "Path to vendored SEER/NAACCR manuals directory "
            "(default: vendored SEERManuals)"
        ),
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
        "--max-retries", type=int, default=10,
        help="Max LLM call attempts (default: %(default)s)",
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
        help=(
            "Directory for resumable checkpoints "
            "(default: OUTPUT/checkpoints)"
        ),
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

    dict_dir = (
        resolve_reference_path(args.data_dict, "NAACCRDataItems")
        if args.data_dict
        else default_data_dict_dir()
    )
    seer_manuals_dir = (
        resolve_reference_path(args.seer_manuals_dir, "SEERManuals")
        if args.seer_manuals_dir
        else default_seer_manuals_dir()
    )
    default_config = PipelineConfig()
    provider_model_env = {
        "vllm": os.getenv("VLLM_MODEL"),
        "azure-openai": os.getenv("AZURE_OPENAI_MODEL"),
        "anthropic-vertex": os.getenv("ANTHROPIC_VERTEX_MODEL"),
    }
    llm_model = (
        args.model
        or os.getenv("LLM_MODEL")
        or provider_model_env.get(args.provider)
        or "auto"
    )
    azure_endpoint = args.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    vertex_project_id = (
        args.anthropic_vertex_project_id
        or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
    )
    vertex_region = args.anthropic_vertex_region or os.getenv("CLOUD_ML_REGION")
    azure_token_refresh_command = (
        args.azure_token_refresh_command
        if args.azure_token_refresh_command is not None
        else default_config.azure_openai_token_refresh_command
    )
    vertex_token_refresh_command = (
        args.anthropic_vertex_token_refresh_command
        if args.anthropic_vertex_token_refresh_command is not None
        else default_config.anthropic_vertex_token_refresh_command
    )

    if args.provider == "azure-openai" and not azure_endpoint:
        parser.error(
            "--provider azure-openai requires --endpoint or $AZURE_OPENAI_ENDPOINT"
        )
    if args.provider == "anthropic-vertex":
        if not vertex_project_id:
            parser.error(
                "--provider anthropic-vertex requires "
                "--anthropic-vertex-project-id or $ANTHROPIC_VERTEX_PROJECT_ID"
            )
        if not vertex_region:
            parser.error(
                "--provider anthropic-vertex requires "
                "--anthropic-vertex-region or $CLOUD_ML_REGION"
            )
        if llm_model == "auto":
            parser.error(
                "--provider anthropic-vertex requires --model, $LLM_MODEL, "
                "or $ANTHROPIC_VERTEX_MODEL"
            )

    config = PipelineConfig(
        llm_provider=args.provider,
        llm_model=llm_model,
        llm_base_url=args.endpoint,
        vllm_base_url=args.vllm_url,
        vllm_model=llm_model,
        vllm_temperature=args.temperature,
        vllm_max_tokens=args.max_tokens,
        vllm_reasoning_parser=args.reasoning_parser,
        azure_openai_endpoint=azure_endpoint,
        azure_openai_api_key_env=args.azure_api_key_env,
        azure_openai_auth_mode=args.azure_auth_mode,
        azure_openai_token_refresh_command=azure_token_refresh_command,
        anthropic_vertex_project_id=vertex_project_id,
        anthropic_vertex_region=vertex_region,
        anthropic_vertex_token_env=args.anthropic_vertex_token_env,
        anthropic_vertex_token_refresh_command=vertex_token_refresh_command,
        max_retries=args.max_retries,
        max_concurrent_patients=args.max_concurrent,
        output_format=args.format,
        confidence_threshold=args.confidence_threshold,
        chunk_target_tokens=args.chunk_size,
        items_per_call=args.items_per_call,
        data_items_csv=dict_dir / "DataItems.csv",
        code_list_csv=dict_dir / "CodeList.csv",
        alternate_names_csv=dict_dir / "AlternateNames.csv",
        seer_manuals_dir=seer_manuals_dir,
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
