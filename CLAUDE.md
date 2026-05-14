# Oncology Registry NAACCR v26 Abstraction Pipeline

## What this project does

Automated extraction of NAACCR v26 cancer registry data from patient EHR documents using a configured LLM endpoint: local vLLM, Azure OpenAI v1, or Anthropic Claude on Vertex AI. Produces registry-grade output (NAACCR XML, flat file, or CSV) with audit trail and human review queue.

## Architecture

```
Input (CSV/Parquet) -> Ingest -> Concatenate & Chunk (50K tokens) -> Tumor Detect ->
  Diagnosis work units -> Round 0 (all diagnoses' chunk 0) -> Round 1 -> ... -> Validate -> Output
                                          |
                                     model endpoint
```

**Input format**: Multiple rows per patient. Required columns: `patient_id`, `date`, `text`. Optional structured columns (name, DOB, sex, etc.) are auto-detected and pre-populated.

**Processing**: Pass 0 scans every patient chunk for primary cancer diagnoses, merges candidates, and deduplicates by site + histology + laterality + diagnosis date. Each diagnosis becomes an independent work unit. Round-based parallelism then processes all work units' Nth chunk in parallel via `asyncio.Semaphore`. Within each chunk, items are extracted in domain groups (demographics -> staging -> treatment -> followup) due to data dependency (staging items depend on primary site/histology). No `guided_json` -- JSON is requested in prompt text, with retry on parse failure.

**Registry references**: Runtime prompts include NAACCR item definitions/valid codes plus bounded, cancer-type-specific excerpts from vendored SEER/NAACCR manuals in `SEERManuals/`. The retrieval layer is local-only and uses `SEERManuals/manifest.json` + `SEERManuals/text/`.

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
| `llm/client.py` | Fully async `httpx.AsyncClient`. Supports vLLM, Azure OpenAI v1, and Anthropic Vertex endpoints. Discovers model metadata where available, refreshes cloud bearer tokens on auth failures, and retries with prompt-level JSON error feedback. |
| `llm/structured_output.py` | Builds prompt-level JSON format instructions from NAACCR data items. Describes expected fields, valid codes, and format expectations in prompt text. |
| `manuals/seer.py` | Retrieves bounded, cancer-type-specific excerpts from vendored SEER/NAACCR manuals for prompt context. |
| `extraction/base.py` | `ExtractionResult` dataclass, `merge_results()` (higher confidence wins), `split_items_into_batches()`, serialization helpers for checkpointing. |
| `extraction/pass0_tumor_detection.py` | Detects multiple primaries per patient by sending every chunk to the LLM, then deduplicates by site+histology+laterality+diagnosis date. |
| `extraction/chunk_extractor.py` | Unified per-chunk extractor. Extracts ALL NAACCR items from one chunk in domain groups: demographics -> resolve schema -> staging -> surgery -> radiation -> systemic -> followup -> narratives. |
| `extraction/round_orchestrator.py` | Round-based parallel processing. Round N = Nth chunk from each diagnosis work unit. Manages concurrency, checkpointing, and resume. |
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

## SEER/NAACCR manuals

Located in `SEERManuals/`:
- Current SEER Program Coding and Staging Manual 2026 core PDFs/XLSX
- Appendix C site-specific coding, surgery, and treatment-effect modules
- Solid Tumor Rules, hematopoietic instructions, EOD, Summary Stage, SSDI, and Grade manuals
- `manifest.json` and `text/` PDF text extractions used by `manuals/seer.py`

## Extraction flow

```
Tumor Detection over every chunk -> merge/deduplicate -> list[TumorCandidate]
For each detected diagnosis:
  Create an independent TumorWorkUnit with its own extraction state
For each round (all diagnosis work units in parallel):
  For each work unit's Nth chunk:
    Demographics + Cancer ID (23 items) -> resolves primary site + histology
    Registry context retrieval -> NAACCR schema context + relevant local SEER excerpts
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
uv run onc-registry-pipeline input.parquet output/ --vllm-url http://localhost:8000/v1
uv run onc-registry-pipeline input.parquet output/ --provider azure-openai --model <azure-model-or-deployment>
uv run onc-registry-pipeline input.parquet output/ --provider anthropic-vertex --model claude-sonnet-4-5@20250929
```

Key options:
- `--chunk-size 50000` — tokens per chunk (default 50K)
- `--items-per-call 50` — NAACCR items per LLM call (default 50)
- `--max-tokens 16384` — max output tokens per LLM call
- `--checkpoint-dir ./checkpoints` — enable round checkpointing for resume
- `--max-concurrent 16` — max parallel diagnosis work-unit extractions
- `--seer-manuals-dir SEERManuals` — local vendored manuals used for prompt context
- `--seer-context-max-chars 12000` — cap for retrieved manual excerpts per prompt
- `--provider {vllm,azure-openai,anthropic-vertex}` — model endpoint provider
- `--model MODEL` — required for Anthropic Vertex and usually required for Azure
- Azure env: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`; bearer tokens refresh via `az account get-access-token --resource=https://cognitiveservices.azure.com/ --query accessToken --output tsv` on 401/403
- Anthropic Vertex env: `CLOUD_ML_REGION`, `ANTHROPIC_VERTEX_PROJECT_ID`, optional `ANTHROPIC_VERTEX_ACCESS_TOKEN`; tokens refresh via `gcloud auth application-default print-access-token` on 401/403

Runtime network access depends on the selected provider: local vLLM stays local, while Azure OpenAI and Anthropic Vertex send prompts to those configured cloud endpoints.

## Converting output

`onc_registry_pipeline/convert.py` converts NAACCR XML to JSON or flat CSV. Standalone script -- no vLLM needed. Parses XML, flattens the Patient>Tumor hierarchy into one row per tumor (patient-level items merged into each row). Supports `--readable-names` (maps xmlNaaccrId -> item name via DataItems.csv) and `--skip-empty`.

```bash
uv run onc-registry-convert output/naaccr_output.xml output/data.json --skip-empty
uv run onc-registry-convert output/naaccr_output.xml output/data.csv --readable-names --skip-empty
```

## Testing without a vLLM server

The non-LLM components (dictionary loading, code resolution, validation, output writing, XML conversion) can be tested standalone. See the verification script in the README.
