# NAACCR v26 Cancer Registry Abstraction Pipeline

## What this project does

Automated extraction of NAACCR v26 cancer registry data from patient EHR documents using a local LLM served via vLLM. Produces registry-grade output (NAACCR XML, flat file, or CSV) with audit trail and human review queue.

## Architecture

```
Input (CSV/Parquet) -> Ingest -> Chunk -> [Pass 0-4] -> Validate -> Output
                                              |
                                         vLLM server
```

**Input format**: Multiple rows per patient. Required columns: `patient_id`, `date`, `text`. Optional structured columns (name, DOB, sex, etc.) are auto-detected and pre-populated.

**Processing is async**: patients processed in parallel via `asyncio.Semaphore` to keep the GPU busy. Within a patient, passes are sequential (each depends on prior results).

## Key modules

| Module | Role |
|--------|------|
| `config.py` | Pydantic `PipelineConfig` with all settings |
| `dictionary/loader.py` | Parses the 3 CSVs in `NAACCRDataItems/`. NOTE: first line of each CSV is blank - must skip before header. Uses `utf-8-sig` encoding. Filters to rows with valid integer `Data Item Number`. |
| `dictionary/code_resolver.py` | 6-tier resolution: exact code -> case-insensitive -> description match -> fuzzy (rapidfuzz) -> numeric range -> fail. Used to constrain LLM outputs. |
| `dictionary/schema_registry.py` | Maps ICD-O-3 site+histology to ~15 cancer schemas. Each schema has 10-30 site-specific data items (SSDIs). Curated dict, not dynamically parsed. |
| `ingest/reader.py` | Loads CSV/Parquet, groups by `patient_id`, auto-detects structured demographic columns (maps column names to NAACCR item numbers). |
| `ingest/chunker.py` | Classifies documents by type (pathology, radiology, operative, etc.) via regex. Sub-chunks large docs on section boundaries. Never splits mid-diagnosis or mid-staging. |
| `llm/client.py` | Fully async `httpx.AsyncClient`. Discovers model name + size at startup via `/v1/models`. Adjusts items-per-call: small(<15B)=8, medium=15, large(70B+)=25. Uses `extra_body.guided_json` for constrained decoding. |
| `llm/structured_output.py` | Builds JSON schemas from NAACCR data items. Coded items become `enum`, dates become `pattern`, each field wrapped with `confidence` + `evidence` sub-fields. |
| `extraction/base.py` | Abstract `BaseExtractionPass` with the core `run()` loop: prioritize chunks -> batch items -> call LLM -> parse -> resolve codes -> merge (higher confidence wins). |
| `extraction/pass0_tumor_detection.py` | Detects multiple primaries per patient. Sends pathology chunk excerpts to LLM, deduplicates by site+year+laterality. If one cancer found, minimal overhead. |
| `extraction/pass1_demographics.py` | Demographics + Cancer ID (~40 items). **Foundational** - primary site + histology determine everything in Pass 2. |
| `extraction/pass2_staging.py` | Site-specific staging. Uses `SchemaRegistry` to determine which of 342 staging items apply (typically 75-85 per patient). |
| `extraction/pass3_treatment.py` | Overrides `run()` to do 3 sub-passes: surgery, radiation (21 items across 3 phases), systemic therapy. |
| `extraction/pass4_followup.py` | Two-part: coded follow-up items + narrative text summaries (up to 4000 chars each for 17 text fields). |
| `validation/cross_field.py` | 11 NAACCR interfield edits (site/sex, site/histology, site/laterality, TNM consistency, treatment dates, etc.). Some are auto-fixable. |
| `validation/consistency.py` | 7 cross-pass checks (surgery done -> path size populated, distant stage -> mets site identified, etc.). |
| `validation/confidence.py` | Scores items, flags for review at 4 priority levels: CRITICAL (key vars <0.9), HIGH (<0.7), MEDIUM (violation), LOW (<0.5). |
| `output/naaccr_writer.py` | XML (nested NaaccrData>Patient>Tumor using `parentElement` from dictionary), flat file, CSV. |
| `output/audit_trail.py` | Per-item provenance: which chunk, which pass, what evidence, what confidence. |
| `output/review_queue.py` | Sorted worklist for human abstractors. |

## Data dictionary files

Located in `NAACCRDataItems/`:
- `DataItems.csv` - 771 active items, 35 columns
- `CodeList.csv` - 4,372 valid code entries for 512 items
- `AlternateNames.csv` - 412 historical name mappings

## Extraction pass flow

```
Pass 0: Tumor Detection -> list[TumorCandidate]
   For each tumor:
     Pass 1: Demographics + Cancer ID (primary site, histology, diagnosis date)
     Pass 2: Staging (dynamically parameterized by Pass 1 site/histology)
     Pass 3: Treatment (3 sub-passes: surgery, radiation, systemic)
     Pass 4: Follow-up + narrative text summaries
     -> Validate -> Score confidence -> Flag for review
```

## Chunk prioritization by pass

- Pass 1: pathology first, then discharge summaries, consults
- Pass 2: pathology + synoptic first, then imaging, consults
- Pass 3: operative notes first, then discharge summaries, progress notes
- Pass 4: newest documents first (reverse chronological)

## Running

```bash
uv run naaccr-pipeline input.parquet output/ --vllm-url http://localhost:8000/v1
```

Requires a running vLLM server. No external network access needed at runtime.

## Converting output

`naaccr_pipeline/convert.py` converts NAACCR XML to JSON or flat CSV. Standalone script -- no vLLM needed. Parses XML, flattens the Patient>Tumor hierarchy into one row per tumor (patient-level items merged into each row). Supports `--readable-names` (maps xmlNaaccrId -> item name via DataItems.csv) and `--skip-empty`.

```bash
uv run naaccr-convert output/naaccr_output.xml output/data.json --skip-empty
uv run naaccr-convert output/naaccr_output.xml output/data.csv --readable-names --skip-empty
```

## Testing without a vLLM server

The non-LLM components (dictionary loading, code resolution, chunking, validation, output writing, XML conversion) can be tested standalone. See the verification script in the README.
