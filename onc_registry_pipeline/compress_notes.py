"""Compress individual clinical notes before registry extraction."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from onc_registry_pipeline.config import PipelineConfig
from onc_registry_pipeline.checkpoint import atomic_write_json, read_json
from onc_registry_pipeline.llm.client import (
    LLMTextResponse,
    VLLMClient,
)

logger = logging.getLogger(__name__)


COMPRESSION_CHECKPOINT_SCHEMA_VERSION = 1


COMPRESSION_SYSTEM_PROMPT = """\
You are an expert oncology clinical-document summarizer.

TASK:
Summarize one clinical document using only information explicitly stated in
the document and metadata supplied by the user.

OUTPUT FORMAT:
- Output only the summary text. Do not include headings, bullets, labels,
  markdown, preamble, caveats, or JSON.
- Use one paragraph of three sentences or less for the clinical document.
- If the document explicitly describes multiple independent primary cancers,
  output one paragraph per primary cancer diagnosis; each paragraph must still
  be three sentences or less.
- If a concept below is not mentioned in the document, omit that concept from
  the summary rather than writing that it is missing or unknown.
- Never write meta-commentary about the document itself. Do not state what the
  document does or does not contain, mention, address, describe, or discuss.
  Do not write phrases like "The document does not contain...", "No information
  is provided about...", "This report does not mention...", "The note focuses
  on...", or "This is not an oncology document." Just summarize the clinical
  content that is present.
- If the document has no oncology content, simply summarize the clinical
  content that is present in three sentences or less. Do not add a disclaimer
  that the document is unrelated to cancer.

CONTENT TO CAPTURE WHEN PRESENT OR KNOWN:
- Age and sex.
- Cancer type and histology.
- Disease burden at diagnosis, including original stage, TNM, extent of
  metastatic disease, sites of involvement, and explicit disease risk scores
  such as International Prognostic Index, Follicular Lymphoma International
  Prognostic Index, or similar disease-specific scores.
- Current disease burden, including current sites of disease, progression,
  response, recurrence, remission, tumor markers such as carcinoembryonic
  antigen or CA 19-9, and clinically meaningful disease measurements.
- Biomarkers: capture all documented biomarkers without summarizing away
  individual results. Biomarkers include molecular alterations, cytogenetics,
  gene fusions, immunohistochemistry, microsatellite instability or mismatch
  repair status, tumor mutational burden, programmed death-ligand 1, hormone
  receptors, human epidermal growth factor receptor 2, and similar predictive,
  prognostic, or diagnostic findings. Biomarkers are NOT routine laboratory
  values and are NOT tumor markers such as carcinoembryonic antigen, CA 19-9,
  CA-125, alpha-fetoprotein, or prostate-specific antigen; tumor markers belong
  under current disease burden when clinically relevant.
- Current and prior treatments, with dates when present. Include details of
  each systemic therapy and local therapy, including surgery and radiation.
- Current and prior adverse events.
- Current and prior comorbidities.
- Current and prior performance status.
- For clinician notes, planned next steps.

STYLE RULES:
- Spell drug names out in full. Expand common oncology shorthand when the
  expansion is clear, for example fluorouracil for 5-FU, oxaliplatin for oxali,
  bevacizumab for bev, pembrolizumab for pembro, capecitabine for cape, and
  doxorubicin, bleomycin, vinblastine, and dacarbazine for ABVD. If expansion
  is uncertain, preserve the documented name and do not guess.
- Preserve dates of current and prior events in the categories above when
  dates are present; partial dates are acceptable.
- Do not invent facts, normalize uncertain values into certainty, or infer a
  biomarker or treatment from cancer type alone.
- Prefer concise clinical prose over exhaustive narrative, but do not omit
  documented biomarkers or distinct treatment lines solely to save words.
"""


def build_document_prompt(
    document_text: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Build the user prompt for one clinical document."""
    metadata = metadata or {}
    metadata_lines: list[str] = []
    for key, value in metadata.items():
        clean = _clean_scalar(value)
        if clean is not None:
            metadata_lines.append(f"- {key}: {clean}")
    metadata_block = "\n".join(metadata_lines) if metadata_lines else "- none supplied"

    return f"""\
Document metadata:
{metadata_block}

Clinical document:
<document>
{document_text}
</document>
"""


def normalize_summary(text: str) -> str:
    """Clean common wrapper text without changing clinical content."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    lower = cleaned.lower()
    for prefix in ("summary:", "clinical summary:"):
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    return cleaned


def load_notes_table(path: str | Path) -> pd.DataFrame:
    """Load CSV, TSV, or parquet notes into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    raise ValueError(
        f"Unsupported file format '{suffix}'. Expected .csv, .tsv, or .parquet"
    )


async def compress_notes_dataframe(
    notes_df: pd.DataFrame,
    client: Any,
    *,
    patient_id_column: str = "patient_id",
    date_column: str = "date",
    text_column: str = "text",
    note_type_column: str = "note_type",
    document_id_column: Optional[str] = None,
    source_file: Optional[str] = None,
    max_concurrent: int = 4,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    checkpoint_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    output_prefix: str = "compressed_notes",
    progress_every: int = 10,
    progress_interval: float = 30.0,
    flush_every: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compress each row independently.

    Returns a pipeline-compatible compressed notes DataFrame and a separate
    audit DataFrame. Failed compressions fall back to the original note text.
    If checkpoint_dir is provided, each completed row is durably checkpointed
    and skipped on a later run with the same input rows.
    """
    _validate_columns(
        notes_df,
        patient_id_column=patient_id_column,
        date_column=date_column,
        text_column=text_column,
    )

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    output_path = Path(output_dir) if output_dir is not None else None
    progress_every = max(1, progress_every)
    flush_every = max(1, flush_every)
    progress_interval = max(1.0, progress_interval)

    row_items = list(enumerate(notes_df.iterrows()))
    expected_fingerprints = {
        row_ordinal: _row_fingerprint(
            row_index,
            row,
            patient_id_column=patient_id_column,
            date_column=date_column,
            text_column=text_column,
            document_id_column=document_id_column,
        )
        for row_ordinal, (row_index, row) in row_items
    }
    completed: dict[int, tuple[dict, dict]] = {}
    if checkpoint_path is not None:
        completed = _load_compression_checkpoints(
            checkpoint_path,
            expected_fingerprints=expected_fingerprints,
            total_rows=len(notes_df),
        )
        if completed:
            logger.info(
                "Compression resume: loaded %d/%d completed row(s) from %s",
                len(completed),
                len(notes_df),
                checkpoint_path,
            )

    pending_items = [
        (row_ordinal, row_index, row)
        for row_ordinal, (row_index, row) in row_items
        if row_ordinal not in completed
    ]

    if checkpoint_path is not None:
        _save_compression_metadata(
            checkpoint_path,
            total_rows=len(notes_df),
            completed_rows=len(completed),
            source_file=source_file,
            output_prefix=output_prefix,
        )

    if output_path is not None and completed:
        _write_completed_outputs(completed, output_path, output_prefix)

    if not pending_items:
        if output_path is not None:
            _write_completed_outputs(completed, output_path, output_prefix)
        logger.info(
            "Compression already complete: %d/%d row(s)",
            len(completed),
            len(notes_df),
        )
        return _completed_to_dataframes(completed)

    logger.info(
        "Compressing %d pending note row(s) with max_concurrent=%d",
        len(pending_items),
        max(1, max_concurrent),
    )
    if checkpoint_path is not None:
        logger.info("Compression checkpoints: %s", checkpoint_path)
    if output_path is not None:
        logger.info(
            "Partial outputs will be refreshed every %d completed row(s): %s",
            flush_every,
            output_path,
        )

    queue: asyncio.Queue[tuple[int, Any, pd.Series]] = asyncio.Queue()
    for item in pending_items:
        queue.put_nowait(item)

    lock = asyncio.Lock()
    run_started = time.monotonic()
    run_completed = 0
    active = 0

    async def _flush_locked() -> None:
        if output_path is not None:
            _write_completed_outputs(completed, output_path, output_prefix)
        if checkpoint_path is not None:
            _save_compression_metadata(
                checkpoint_path,
                total_rows=len(notes_df),
                completed_rows=len(completed),
                source_file=source_file,
                output_prefix=output_prefix,
            )

    async def _worker() -> None:
        nonlocal active, run_completed
        while True:
            try:
                row_ordinal, row_index, row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            async with lock:
                active += 1
            try:
                compressed_row, audit_row = await _compress_row(
                    row_index,
                    row,
                    client,
                    patient_id_column=patient_id_column,
                    date_column=date_column,
                    text_column=text_column,
                    note_type_column=note_type_column,
                    document_id_column=document_id_column,
                    source_file=source_file,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if checkpoint_path is not None:
                    _save_compression_checkpoint(
                        checkpoint_path,
                        row_ordinal=row_ordinal,
                        row_fingerprint=expected_fingerprints[row_ordinal],
                        compressed_row=compressed_row,
                        audit_row=audit_row,
                    )

                async with lock:
                    completed[row_ordinal] = (compressed_row, audit_row)
                    run_completed += 1
                    active -= 1
                    completed_count = len(completed)
                    should_report = (
                        run_completed % progress_every == 0
                        or completed_count == len(notes_df)
                    )
                    should_flush = (
                        output_path is not None
                        and (
                            run_completed % flush_every == 0
                            or completed_count == len(notes_df)
                        )
                    )
                    if should_flush:
                        await _flush_locked()
                    if should_report:
                        errors, fallbacks = _compression_error_counts(completed)
                        elapsed = time.monotonic() - run_started
                        logger.info(
                            "Compression progress: %d/%d complete "
                            "(%d new this run, %d active, %d errors, "
                            "%d fallback(s), %.1fs elapsed)",
                            completed_count,
                            len(notes_df),
                            run_completed,
                            active,
                            errors,
                            fallbacks,
                            elapsed,
                        )
            finally:
                queue.task_done()
                async with lock:
                    if active > 0 and row_ordinal not in completed:
                        active -= 1

    stop_heartbeat = asyncio.Event()

    async def _heartbeat() -> None:
        while not stop_heartbeat.is_set():
            try:
                await asyncio.wait_for(
                    stop_heartbeat.wait(),
                    timeout=progress_interval,
                )
            except asyncio.TimeoutError:
                pass
            if stop_heartbeat.is_set():
                return
            async with lock:
                errors, fallbacks = _compression_error_counts(completed)
                logger.info(
                    "Compression still running: %d/%d complete "
                    "(%d active, %d queued, %d errors, %d fallback(s))",
                    len(completed),
                    len(notes_df),
                    active,
                    queue.qsize(),
                    errors,
                    fallbacks,
                )

    worker_count = min(max(1, max_concurrent), len(pending_items))
    workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]
    heartbeat = asyncio.create_task(_heartbeat())
    try:
        worker_results = await asyncio.gather(*workers, return_exceptions=True)
    finally:
        stop_heartbeat.set()
        await heartbeat
    failures = [result for result in worker_results if isinstance(result, Exception)]
    if failures:
        raise RuntimeError(
            f"{len(failures)} compression worker(s) failed"
        ) from failures[0]

    if output_path is not None:
        _write_completed_outputs(completed, output_path, output_prefix)
    if checkpoint_path is not None:
        _save_compression_metadata(
            checkpoint_path,
            total_rows=len(notes_df),
            completed_rows=len(completed),
            source_file=source_file,
            output_prefix=output_prefix,
        )

    return _completed_to_dataframes(completed)


async def compress_notes_file(
    notes_path: str | Path,
    client: Any,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and compress one notes file."""
    notes_path = Path(notes_path)
    notes_df = load_notes_table(notes_path)
    return await compress_notes_dataframe(
        notes_df,
        client,
        source_file=str(notes_path),
        **kwargs,
    )


def write_outputs(
    compressed_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "compressed_notes",
) -> tuple[Path, Path]:
    """Write compressed pipeline input CSV and JSONL audit records."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    jsonl_path = output_dir / f"{prefix}.jsonl"

    tmp_csv = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    compressed_df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)

    tmp_jsonl = jsonl_path.with_name(f".{jsonl_path.name}.tmp.{os.getpid()}")
    with tmp_jsonl.open("w", encoding="utf-8") as fh:
        for record in audit_df.to_dict(orient="records"):
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    tmp_jsonl.replace(jsonl_path)

    return csv_path, jsonl_path


def default_compression_checkpoint_dir(
    output_dir: str | Path,
    prefix: str = "compressed_notes",
) -> Path:
    """Return the default per-row checkpoint directory for a compression run."""
    return Path(output_dir) / f".{prefix}_checkpoints"


def _compression_checkpoint_row_path(checkpoint_dir: Path, row_ordinal: int) -> Path:
    return checkpoint_dir / "rows" / f"row_{row_ordinal:08d}.json"


def _row_fingerprint(
    row_index: Any,
    row: pd.Series,
    *,
    patient_id_column: str,
    date_column: str,
    text_column: str,
    document_id_column: Optional[str],
) -> str:
    text_value = _clean_scalar(row.get(text_column))
    text = "" if text_value is None else str(text_value)
    payload = {
        "row_index": str(row_index),
        "patient_id": _string_or_empty(row.get(patient_id_column)),
        "date": _string_or_empty(row.get(date_column)),
        "document_id": (
            _string_or_empty(row.get(document_id_column))
            if document_id_column
            else ""
        ),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _save_compression_checkpoint(
    checkpoint_dir: Path,
    *,
    row_ordinal: int,
    row_fingerprint: str,
    compressed_row: dict,
    audit_row: dict,
) -> None:
    payload = {
        "schema_version": COMPRESSION_CHECKPOINT_SCHEMA_VERSION,
        "row_ordinal": row_ordinal,
        "row_fingerprint": row_fingerprint,
        "compressed_row": compressed_row,
        "audit_row": audit_row,
    }
    atomic_write_json(
        _compression_checkpoint_row_path(checkpoint_dir, row_ordinal),
        payload,
    )


def _load_compression_checkpoints(
    checkpoint_dir: Path,
    *,
    expected_fingerprints: dict[int, str],
    total_rows: int,
) -> dict[int, tuple[dict, dict]]:
    rows_dir = checkpoint_dir / "rows"
    if not rows_dir.exists():
        return {}

    completed: dict[int, tuple[dict, dict]] = {}
    for path in sorted(rows_dir.glob("row_*.json")):
        try:
            payload = read_json(path)
            row_ordinal = int(payload["row_ordinal"])
            if row_ordinal < 0 or row_ordinal >= total_rows:
                continue
            expected = expected_fingerprints.get(row_ordinal)
            if payload.get("row_fingerprint") != expected:
                logger.warning(
                    "Ignoring stale compression checkpoint with mismatched row "
                    "fingerprint: %s",
                    path,
                )
                continue
            compressed_row = payload["compressed_row"]
            audit_row = payload["audit_row"]
            if not isinstance(compressed_row, dict) or not isinstance(audit_row, dict):
                raise ValueError("checkpoint row payload is malformed")
            completed[row_ordinal] = (compressed_row, audit_row)
        except Exception as exc:
            logger.warning("Could not load compression checkpoint %s: %s", path, exc)
    return completed


def _save_compression_metadata(
    checkpoint_dir: Path,
    *,
    total_rows: int,
    completed_rows: int,
    source_file: Optional[str],
    output_prefix: str,
) -> None:
    payload = {
        "schema_version": COMPRESSION_CHECKPOINT_SCHEMA_VERSION,
        "source_file": source_file,
        "output_prefix": output_prefix,
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(checkpoint_dir / "metadata.json", payload)


def _completed_to_dataframes(
    completed: dict[int, tuple[dict, dict]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    compressed_rows = [completed[idx][0] for idx in sorted(completed)]
    audit_rows = [completed[idx][1] for idx in sorted(completed)]
    return pd.DataFrame(compressed_rows), pd.DataFrame(audit_rows)


def _write_completed_outputs(
    completed: dict[int, tuple[dict, dict]],
    output_dir: str | Path,
    prefix: str,
) -> tuple[Path, Path]:
    compressed_df, audit_df = _completed_to_dataframes(completed)
    return write_outputs(compressed_df, audit_df, output_dir, prefix)


def _compression_error_counts(
    completed: dict[int, tuple[dict, dict]],
) -> tuple[int, int]:
    errors = 0
    fallbacks = 0
    for _compressed_row, audit_row in completed.values():
        if audit_row.get("error"):
            errors += 1
        if audit_row.get("compression_used_fallback"):
            fallbacks += 1
    return errors, fallbacks


async def _compress_row(
    row_index: Any,
    row: pd.Series,
    client: Any,
    *,
    patient_id_column: str,
    date_column: str,
    text_column: str,
    note_type_column: str,
    document_id_column: Optional[str],
    source_file: Optional[str],
    max_tokens: int,
    temperature: float,
) -> tuple[dict, dict]:
    text_value = _clean_scalar(row.get(text_column))
    text = "" if text_value is None else str(text_value)
    patient_id = _clean_scalar(row.get(patient_id_column))
    note_date = _clean_scalar(row.get(date_column))
    note_type = _clean_scalar(row.get(note_type_column))
    document_id = (
        _clean_scalar(row.get(document_id_column))
        if document_id_column
        else None
    )
    if document_id is None:
        prefix = source_file or "notes"
        document_id = f"{prefix}:row{row_index}"

    summary = ""
    error: Optional[str] = None
    used_fallback = False
    output_text = ""
    usage: dict[str, int] | None = None

    if not text.strip():
        error = "empty_text"
    else:
        metadata = {
            "document_id": document_id,
            "patient_id": patient_id,
            "date": note_date,
            "note_type": note_type,
        }
        try:
            response: LLMTextResponse = await client.generate_text(
                COMPRESSION_SYSTEM_PROMPT,
                build_document_prompt(text, metadata),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            summary = normalize_summary(response.final_content)
            usage = response.usage
            if not summary:
                raise ValueError("empty_summary")
            output_text = summary
        except Exception as exc:  # pragma: no cover - API failures are mocked
            logger.exception("Failed to compress document %s", document_id)
            error = f"{type(exc).__name__}: {exc}"
            output_text = text
            used_fallback = True

    compressed_row = _build_compressed_row(
        row,
        patient_id_column=patient_id_column,
        date_column=date_column,
        text_column=text_column,
        output_text=output_text,
    )
    audit_row: dict[str, Any] = {
        "source_file": source_file,
        "source_row_index": row_index,
        "document_id": str(document_id),
        "patient_id": None if patient_id is None else str(patient_id),
        "date": None if note_date is None else str(note_date),
        "note_type": None if note_type is None else str(note_type),
        "text_chars": len(text),
        "summary": summary,
        "error": error,
        "compression_used_fallback": used_fallback,
    }
    if usage:
        audit_row.update(usage)
        audit_row["prompt_tokens"] = usage.get("prompt_tokens") or usage.get(
            "input_tokens"
        )
        audit_row["completion_tokens"] = usage.get("completion_tokens") or usage.get(
            "output_tokens"
        )

    return compressed_row, audit_row


def _build_compressed_row(
    row: pd.Series,
    *,
    patient_id_column: str,
    date_column: str,
    text_column: str,
    output_text: str,
) -> dict:
    patient_id = _clean_scalar(row.get(patient_id_column))
    note_date = _clean_scalar(row.get(date_column))
    out: dict[str, Any] = {
        "patient_id": "" if patient_id is None else str(patient_id),
        "date": "" if note_date is None else str(note_date),
        "text": output_text,
    }

    reserved = {"patient_id", "date", "text", text_column}
    for col in row.index:
        if col in reserved:
            continue
        value = _clean_scalar(row.get(col))
        out[str(col)] = "" if value is None else value
    return out


def _validate_columns(
    df: pd.DataFrame,
    *,
    patient_id_column: str,
    date_column: str,
    text_column: str,
) -> None:
    required = {
        "patient id": patient_id_column,
        "date": date_column,
        "text": text_column,
    }
    missing = [label for label, col in required.items() if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) for {', '.join(missing)}. "
            f"Available columns: {list(df.columns)}"
        )


def _clean_scalar(value: Any) -> Optional[Any]:
    """Return None for pandas-style missing or blank scalar values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    if text.strip() == "":
        return None
    return value


def _string_or_empty(value: Any) -> str:
    cleaned = _clean_scalar(value)
    return "" if cleaned is None else str(cleaned)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress individual clinical notes before registry extraction.",
    )
    parser.add_argument("input", help="Path to input CSV/TSV/Parquet notes file")
    parser.add_argument("output", help="Directory for compressed output files")
    parser.add_argument(
        "--provider",
        choices=["vllm", "azure-openai", "anthropic-vertex"],
        default="vllm",
        help="LLM endpoint provider (default: %(default)s)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Provider endpoint base URL. For Azure, defaults to $AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model/deployment id. Defaults to $LLM_MODEL or provider-specific "
            "model env vars when present."
        ),
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--azure-auth-mode",
        choices=["bearer", "api-key"],
        default="bearer",
        help="Azure OpenAI auth mode (default: %(default)s)",
    )
    parser.add_argument(
        "--azure-api-key-env",
        default="AZURE_OPENAI_API_KEY",
        help="Env var holding the Azure OpenAI token/key (default: %(default)s)",
    )
    parser.add_argument("--azure-token-refresh-command", default=None)
    parser.add_argument("--anthropic-vertex-project-id", default=None)
    parser.add_argument("--anthropic-vertex-region", default=None)
    parser.add_argument(
        "--anthropic-vertex-token-env",
        default="ANTHROPIC_VERTEX_ACCESS_TOKEN",
    )
    parser.add_argument("--anthropic-vertex-token-refresh-command", default=None)
    parser.add_argument(
        "--reasoning-parser",
        default="auto",
        help="vLLM reasoning parser name, auto, or none (default: %(default)s)",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--compression-max-tokens",
        type=int,
        default=1024,
        help="Max output tokens for each note summary (default: %(default)s)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent note compression calls (default: %(default)s)",
    )
    parser.add_argument("--patient-id-column", default="patient_id")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--note-type-column", default="note_type")
    parser.add_argument("--document-id-column", default=None)
    parser.add_argument("--output-prefix", default="compressed_notes")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Directory for resumable per-row compression checkpoints "
            "(default: OUTPUT/.OUTPUT_PREFIX_checkpoints)"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Log progress after this many newly completed rows (default: %(default)s)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Heartbeat log interval in seconds while rows are running (default: %(default)s)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Refresh partial CSV/JSONL outputs after this many rows (default: %(default)s)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _build_client_from_args(args: argparse.Namespace) -> VLLMClient:
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
        raise ValueError(
            "--provider azure-openai requires --endpoint or $AZURE_OPENAI_ENDPOINT"
        )
    if args.provider == "anthropic-vertex":
        if not vertex_project_id:
            raise ValueError(
                "--provider anthropic-vertex requires "
                "--anthropic-vertex-project-id or $ANTHROPIC_VERTEX_PROJECT_ID"
            )
        if not vertex_region:
            raise ValueError(
                "--provider anthropic-vertex requires "
                "--anthropic-vertex-region or $CLOUD_ML_REGION"
            )
        if llm_model == "auto":
            raise ValueError(
                "--provider anthropic-vertex requires --model, $LLM_MODEL, "
                "or $ANTHROPIC_VERTEX_MODEL"
            )

    base_url = azure_endpoint if args.provider == "azure-openai" else args.vllm_url
    if args.endpoint and args.provider != "azure-openai":
        base_url = args.endpoint

    return VLLMClient(
        base_url=base_url or "",
        provider=args.provider,
        model=llm_model,
        temperature=args.temperature,
        max_tokens=args.compression_max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        reasoning_parser=args.reasoning_parser,
        azure_api_key_env=args.azure_api_key_env,
        azure_auth_mode=args.azure_auth_mode,
        azure_token_refresh_command=azure_token_refresh_command,
        anthropic_vertex_project_id=vertex_project_id,
        anthropic_vertex_region=vertex_region,
        anthropic_vertex_token_env=args.anthropic_vertex_token_env,
        anthropic_vertex_token_refresh_command=vertex_token_refresh_command,
    )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    notes_df = load_notes_table(args.input)
    logger.info("Loaded %d note row(s) from %s", len(notes_df), args.input)
    output_dir = Path(args.output)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else default_compression_checkpoint_dir(output_dir, args.output_prefix)
    )
    logger.info("Using compression checkpoint directory: %s", checkpoint_dir)

    client = _build_client_from_args(args)
    await client.initialize()
    try:
        compressed_df, audit_df = await compress_notes_dataframe(
            notes_df,
            client,
            patient_id_column=args.patient_id_column,
            date_column=args.date_column,
            text_column=args.text_column,
            note_type_column=args.note_type_column,
            document_id_column=args.document_id_column,
            source_file=args.input,
            max_concurrent=args.max_concurrent,
            max_tokens=args.compression_max_tokens,
            temperature=args.temperature,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            output_prefix=args.output_prefix,
            progress_every=args.progress_every,
            progress_interval=args.progress_interval,
            flush_every=args.flush_every,
        )
    finally:
        await client.close()

    csv_path, jsonl_path = write_outputs(
        compressed_df,
        audit_df,
        output_dir,
        args.output_prefix,
    )
    errors = int(audit_df["error"].notna().sum()) if "error" in audit_df else 0
    fallbacks = (
        int(audit_df["compression_used_fallback"].sum())
        if "compression_used_fallback" in audit_df
        else 0
    )
    return {
        "documents_processed": len(compressed_df),
        "errors": errors,
        "fallbacks": fallbacks,
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        summary = asyncio.run(_run(args))
    except ValueError as exc:
        parser.error(str(exc))

    print("Note compression complete.")
    print(f"  Documents processed: {summary['documents_processed']}")
    print(f"  Errors:              {summary['errors']}")
    print(f"  Fallbacks:           {summary['fallbacks']}")
    print(f"  CSV:                 {summary['csv_path']}")
    print(f"  JSONL:               {summary['jsonl_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
