# NAACCR v26 Cancer Registry Abstraction Pipeline

## What this project does

Automated extraction of NAACCR v26 cancer registry data from patient EHR documents using a local LLM served via vLLM. Produces registry-grade output (NAACCR XML, flat file, or CSV) with audit trail and human review queue.

## Architecture

```
Input (CSV/Parquet) -> Ingest -> Concatenate & Chunk (50K tokens) -> Tumor Detect ->
  Round 0 (all patients' chunk 0) -> Round 1 (chunk 1) -> ... -> Validate -> Output
                                          |
                                     vLLM server
```

**Input format**: Multiple rows per patient. Required columns: `patient_id`, `date`, `text`. Optional structured columns (name, DOB, sex, etc.) are auto-detected and pre-populated.

**Processing**: Round-based parallelism. All patients' Nth chunk processed in parallel via `asyncio.Semaphore`. Within each chunk, items extracted in domain groups (demographics -> staging -> treatment -> followup) due to data dependency (staging items depend on primary site/histology). No `guided_json` -- JSON is requested in prompt text, with retry on parse failure.

**Checkpointing**: With `--checkpoint-dir`, state is saved after each round for resume on interruption.

## Key modules

| Module | Role |
|--------|------|
| `config.py` | Pydantic `PipelineConfig` with all settings |
| `dictionary/loader.py` | Parses the 3 CSVs in `NAACCRDataItems/`. NOTE: first line of each CSV is blank - must skip before header. Uses `utf-8-sig` encoding. Filters to rows with valid integer `Data Item Number`. |
| `dictionary/code_resolver.py` | 6-tier resolution: exact code -> case-insensitive -> description match -> fuzzy (rapidfuzz) -> numeric range -> fail. Used to constrain LLM outputs. |
| `dictionary/schema_registry.py` | Maps ICD-O-3 site+histology to ~15 cancer schemas. Each schema has 10-30 site-specific data items (SSDIs). Curated dict, not dynamically parsed. |
| `ingest/reader.py` | Loads CSV/Parquet, groups by `patient_id`, auto-detects structured demographic columns (maps column names to NAACCR item numbers). |
| `ingest/sequential_chunker.py` | Concatenates all patient notes chronologically with date headers, chunks by token count (default 50K) with overlap. No document classification. |
| `llm/client.py` | Fully async `httpx.AsyncClient`. Discovers model name + size at startup via `/v1/models`. No guided_json -- requests JSON in prompt, retries with error feedback on parse failure. |
| `llm/structured_output.py` | Builds prompt-level JSON format instructions from NAACCR data items. Describes expected fields, valid codes, and format expectations in prompt text. |
| `extraction/base.py` | `ExtractionResult` dataclass, `merge_results()` (higher confidence wins), `split_items_into_batches()`, serialization helpers for checkpointing. |
| `extraction/pass0_tumor_detection.py` | Detects multiple primaries per patient. Sends chunk excerpts to LLM, deduplicates by site+year+laterality. If one cancer found, minimal overhead. |
| `extraction/chunk_extractor.py` | Unified per-chunk extractor. Extracts ALL NAACCR items from one chunk in domain groups: demographics -> resolve schema -> staging -> surgery -> radiation -> systemic -> followup -> narratives. |
| `extraction/round_orchestrator.py` | Round-based parallel processing. Round N = Nth chunk from each patient/tumor. Manages concurrency, checkpointing, and resume. |
| `extraction/prompts/chunk_extraction.py` | All prompt templates. Domain-specific system prompts + unified user template with prior-extraction-state context for running updates. |
| `validation/cross_field.py` | 11 NAACCR interfield edits (site/sex, site/histology, site/laterality, TNM consistency, treatment dates, etc.). Some are auto-fixable. |
| `validation/consistency.py` | 7 cross-pass checks (surgery done -> path size populated, distant stage -> mets site identified, etc.). |
| `validation/confidence.py` | Scores items, flags for review at 4 priority levels: CRITICAL (key vars <0.9), HIGH (<0.7), MEDIUM (violation), LOW (<0.5). |
| `output/naaccr_writer.py` | XML (nested NaaccrData>Patient>Tumor using `parentElement` from dictionary), flat file, CSV. |
| `output/audit_trail.py` | Per-item provenance: which chunk, which round, what evidence, what confidence. |
| `output/review_queue.py` | Sorted worklist for human abstractors. |

## Data dictionary files

Located in `NAACCRDataItems/`:
- `DataItems.csv` - 771 active items, 35 columns
- `CodeList.csv` - 4,372 valid code entries for 512 items
- `AlternateNames.csv` - 412 historical name mappings

## Extraction flow

```
Tumor Detection -> list[TumorCandidate]
For each chunk (round-based, all patients in parallel):
  For each tumor:
    Demographics + Cancer ID (23 items) -> resolves primary site + histology
    Schema resolution -> determines site-specific staging items
    Staging (75-85 items, batched by items_per_call)
    Treatment: surgery (15) + radiation (25) + systemic (16)
    Follow-up coded items (6)
    Narrative text summaries (17 running-update text fields)
  -> Merge with prior extraction (higher confidence wins)
After all chunks: Validate -> Score confidence -> Flag for review
```

## Running

```bash
uv run naaccr-pipeline input.parquet output/ --vllm-url http://localhost:8000/v1
```

Key options:
- `--chunk-size 50000` — tokens per chunk (default 50K)
- `--items-per-call 50` — NAACCR items per LLM call (default 50)
- `--max-tokens 16384` — max output tokens per LLM call
- `--checkpoint-dir ./checkpoints` — enable round checkpointing for resume
- `--max-concurrent 16` — max parallel patient extractions

Requires a running vLLM server. No external network access needed at runtime.

## Converting output

`naaccr_pipeline/convert.py` converts NAACCR XML to JSON or flat CSV. Standalone script -- no vLLM needed. Parses XML, flattens the Patient>Tumor hierarchy into one row per tumor (patient-level items merged into each row). Supports `--readable-names` (maps xmlNaaccrId -> item name via DataItems.csv) and `--skip-empty`.

```bash
uv run naaccr-convert output/naaccr_output.xml output/data.json --skip-empty
uv run naaccr-convert output/naaccr_output.xml output/data.csv --readable-names --skip-empty
```

## Testing without a vLLM server

The non-LLM components (dictionary loading, code resolution, validation, output writing, XML conversion) can be tested standalone. See the verification script in the README.
