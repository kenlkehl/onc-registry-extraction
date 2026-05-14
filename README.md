# Oncology Registry NAACCR v26 Abstraction Pipeline

Automated extraction of [NAACCR v26](https://www.naaccr.org/) cancer registry data from EHR clinical text using a local LLM. Produces registry-grade output suitable for state cancer registry submission.

## What it does

Given a dataset of patient clinical documents (pathology reports, operative notes, imaging, progress notes, etc.), this pipeline:

1. **Detects** all distinct primary cancers per patient by scanning every chronological chunk
2. **Creates one independent extraction work unit per diagnosis**, keyed by primary site + histology + laterality + diagnosis date
3. **Extracts** ~580 NAACCR data items per diagnosis/chunk in domain groups:
   - Demographics & cancer identification (ICD-O-3 site/histology codes)
   - Staging & prognostic factors (TNM, biomarkers, site-specific factors)
   - Treatment (surgery, radiation with 3-phase detail, chemotherapy, hormone, immunotherapy)
   - Follow-up & narrative text summaries
4. **Injects cancer-type-specific registry context** from the NAACCR dictionary plus locally vendored SEER/NAACCR manuals
5. **Updates** extraction across chunks (higher confidence wins)
6. **Validates** against NAACCR interfield edit rules (site/sex, site/histology, TNM consistency, etc.)
7. **Scores confidence** and generates a prioritized human review queue
8. **Outputs** NAACCR v26 XML, fixed-width flat file, or CSV with full audit trail

By default, LLM inference runs locally via [vLLM](https://docs.vllm.ai/).
The same pipeline can also target configured Azure OpenAI v1 or Anthropic
Claude on Vertex AI endpoints.

## Quick start

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A running [vLLM](https://docs.vllm.ai/) server with a local LLM loaded, or
  credentials for Azure OpenAI / Anthropic Claude on Vertex AI

### Install

```bash
git clone <this-repo>
cd onc-registry-extraction
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Start vLLM

Start a vLLM server with your preferred model:

```bash
# Example with a large model (recommended for best accuracy)
vllm serve meta-llama/Llama-3.3-70B-Instruct

# Example with a smaller model (faster, lower accuracy)
vllm serve Qwen/Qwen2.5-14B-Instruct
```

For reasoning models, start vLLM with its built-in reasoning parser so the
pipeline receives final JSON separately from reasoning text:

```bash
# Qwen3 / Qwen3.5 / Qwen3.6 variants
vllm serve Qwen/Qwen3.6-27B --enable-reasoning --reasoning-parser qwen3

# Gemma 4 variants
vllm serve google/gemma-4-31b-it --enable-reasoning --reasoning-parser gemma4

# GPT-OSS
vllm serve openai/gpt-oss-120b --enable-reasoning --reasoning-parser openai_gptoss
```

### Optional: use cloud model endpoints

Azure OpenAI v1 endpoints use the OpenAI-compatible chat completions route.
For Entra bearer-token auth, set your endpoint and token as usual:

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/openai/v1"
export AZURE_OPENAI_API_KEY="$(az account get-access-token \
    --resource=https://cognitiveservices.azure.com/ \
    --query accessToken --output tsv)"

uv run onc-registry-pipeline input.parquet output/ \
    --provider azure-openai \
    --model <azure-model-or-deployment>
```

If an Azure inference call receives a 401/403, the pipeline refreshes the
bearer token by running:

```bash
az account get-access-token --resource=https://cognitiveservices.azure.com/ --query accessToken --output tsv
```

For static Azure resource keys, use `--azure-auth-mode api-key` and pass an
empty `--azure-token-refresh-command ""`.

Anthropic Claude on Vertex AI uses `ANTHROPIC_VERTEX_PROJECT_ID` and
`CLOUD_ML_REGION`:

```bash
export CLOUD_ML_REGION=global
export ANTHROPIC_VERTEX_PROJECT_ID=<project-id>

uv run onc-registry-pipeline input.parquet output/ \
    --provider anthropic-vertex \
    --model claude-sonnet-4-5@20250929
```

Vertex auth uses `ANTHROPIC_VERTEX_ACCESS_TOKEN` when present, otherwise it
runs `gcloud auth application-default print-access-token` and refreshes on
401/403.

### Prepare your input data

The input is a CSV or Parquet file with **one row per document**, multiple rows per patient:

| patient_id | date       | text                                    |
|------------|------------|-----------------------------------------|
| P001       | 2023-01-15 | SURGICAL PATHOLOGY REPORT...            |
| P001       | 2023-02-10 | ONCOLOGY CONSULTATION NOTE...           |
| P001       | 2023-03-01 | RADIATION ONCOLOGY TREATMENT SUMMARY... |
| P002       | 2023-04-05 | PATHOLOGY REPORT...                     |

**Required columns:**
- `patient_id` -- unique patient identifier
- `date` -- document date (any parseable date format)
- `text` -- full text of the clinical document

**Optional structured columns** (auto-detected, skip LLM extraction when present):
- `last_name`, `first_name`, `dob`, `sex`, `race`, `ethnicity`
- `address`, `city`, `state`, `zip`, `ssn`, `mrn`

### Run the pipeline

```bash
# Basic usage
uv run onc-registry-pipeline input.csv output/

# With options
uv run onc-registry-pipeline input.parquet output/ \
    --provider vllm \
    --vllm-url http://localhost:8000/v1 \
    --format naaccr_xml \
    --chunk-size 50000 \
    --items-per-call 50 \
    --seer-manuals-dir SEERManuals \
    --max-concurrent 16 \
    --checkpoint-dir ./checkpoints \
    --verbose
```

Or without uv:

```bash
python -m onc_registry_pipeline.main input.csv output/
```

### Output files

The pipeline writes to the output directory:

| File | Description |
|------|-------------|
| `naaccr_output.xml` | NAACCR v26 XML (NaaccrData > Patient > Tumor hierarchy) |
| `naaccr_output.dat` | NAACCR v26 fixed-width flat file |
| `naaccr_output.csv` | Flat CSV with one column per NAACCR item |
| `audit_trail.csv` | Per-item provenance: source chunk, evidence text, confidence, round |
| `review_queue.csv` | Prioritized worklist for human review (CRITICAL/HIGH/MEDIUM/LOW) |
| `llm_calls.jsonl` | Full log of all LLM interactions (prompts, responses, reasoning) |

### Converting output to JSON or CSV

The `onc-registry-convert` tool converts NAACCR XML output to JSON or simple CSV for downstream use (analytics, data science, integration with other systems):

```bash
# XML -> JSON (format inferred from extension)
uv run onc-registry-convert output/naaccr_output.xml output/data.json

# XML -> CSV with human-readable column names
uv run onc-registry-convert output/naaccr_output.xml output/data.csv --readable-names

# Drop empty fields for a cleaner file
uv run onc-registry-convert output/naaccr_output.xml output/data.json --skip-empty

# Combine options
uv run onc-registry-convert output/naaccr_output.xml output/data.csv \
    --readable-names --skip-empty
```

The `--readable-names` flag replaces XML IDs (e.g., `primarySite`) with full NAACCR item names (e.g., `Primary Site`) using the data dictionary. Each tumor is one row/object, with patient-level and root-level items merged in so every record is self-contained.

## CLI reference

```
usage: onc-registry-pipeline [-h]
                       [--provider {vllm,azure-openai,anthropic-vertex}]
                       [--endpoint URL] [--model MODEL] [--vllm-url URL]
                       [--azure-auth-mode {bearer,api-key}]
                       [--azure-api-key-env ENV]
                       [--azure-token-refresh-command CMD]
                       [--anthropic-vertex-project-id PROJECT]
                       [--anthropic-vertex-region REGION]
                       [--anthropic-vertex-token-env ENV]
                       [--anthropic-vertex-token-refresh-command CMD]
                       [--max-concurrent N]
                       [--format {naaccr_xml,naaccr_flat,csv}]
                       [--confidence-threshold FLOAT] [--data-dict DIR]
                       [--temperature FLOAT] [--max-tokens N]
                       [--max-retries N] [--reasoning-parser NAME]
                       [--chunk-size N] [--items-per-call N]
                       [--seer-manuals-dir DIR] [--seer-context-max-chars N]
                       [--checkpoint-dir DIR] [--verbose]
                       input output

positional arguments:
  input                    Path to input CSV/TSV/Parquet file
  output                   Path to output directory

options:
  --provider              LLM endpoint provider (default: vllm)
  --endpoint URL          Provider endpoint base URL (Azure defaults to
                          $AZURE_OPENAI_ENDPOINT)
  --model MODEL           Model/deployment id (or $LLM_MODEL /
                          provider-specific env var)
  --vllm-url URL           vLLM server base URL (default: http://localhost:8000/v1)
  --azure-auth-mode        bearer for Entra tokens, api-key for resource keys
  --azure-api-key-env ENV  Env var holding Azure token/key
  --azure-token-refresh-command CMD
                           Command that prints a fresh Azure bearer token
  --anthropic-vertex-project-id PROJECT
                           Defaults to $ANTHROPIC_VERTEX_PROJECT_ID
  --anthropic-vertex-region REGION
                           Defaults to $CLOUD_ML_REGION
  --anthropic-vertex-token-env ENV
                           Env var holding a Vertex bearer token
  --anthropic-vertex-token-refresh-command CMD
                           Command that prints a fresh Vertex bearer token
  --max-concurrent N       Max concurrent diagnosis work units per round (default: 16)
  --format FORMAT          Output format (default: naaccr_xml)
  --confidence-threshold   Confidence threshold for review flagging (default: 0.7)
  --data-dict DIR          Path to NAACCRDataItems directory (default: NAACCRDataItems)
  --temperature FLOAT      LLM sampling temperature (default: 0.0)
  --max-tokens N           Max tokens per LLM response (default: 16384)
  --max-retries N          Max LLM call retries (default: 3)
  --reasoning-parser NAME  vLLM reasoning parser name: auto, none, qwen3,
                           gemma4, openai_gptoss, etc. (default: auto)
  --chunk-size N           Chunk size in tokens (default: 50000)
  --items-per-call N       NAACCR items per LLM call (default: 50)
  --seer-manuals-dir DIR   Vendored SEER/NAACCR manuals directory (default: SEERManuals)
  --seer-context-max-chars Max registry manual context per prompt (default: 12000)
  --checkpoint-dir DIR     Directory for round checkpoints (enables resume)
  -v, --verbose            Enable debug logging
```

## How it works

### Pipeline architecture

```
Input DataFrame
    |
    v
[Ingest] Group by patient, detect structured columns
    |
    v
[Chunk] Concatenate notes chronologically, chunk by token count (50K default)
    |
    v
[Tumor Detect] Process every chunk, merge candidates, deduplicate diagnoses
    |
    v
[Diagnosis work units]
    One work unit per detected diagnosis:
    primary site + histology + laterality + diagnosis date
    |
    v
[Round-based extraction]
    Round 0: all diagnosis work units' chunk 0 in parallel
    Round 1: all diagnosis work units' chunk 1 in parallel (updating prior results)
    ...
    Per chunk, per diagnosis:
      Demographics + Cancer ID -> resolve site/histology
      Retrieve relevant NAACCR schema context + local SEER manual excerpts
      Staging (schema-specific: breast, prostate, lung, etc.)
      Treatment: surgery, radiation, systemic (3 sub-passes)
      Follow-up coded items
      Narrative text summaries (running update)
    |
    v
[Validate] NAACCR interfield edits, cross-field consistency checks
    |
    v
[Score] Confidence scoring, human review flagging
    |
    v
[Output] NAACCR XML/flat/CSV + audit trail + review queue
```

### Chunking strategy

All patient notes are concatenated chronologically with date headers and chunked into ~50K-token segments with overlap. No document classification or type-based prioritization. Each chunk is processed as a complete unit, extracting all available NAACCR items.

### Diagnosis detection strategy

Pass 0 sends every patient chunk to the LLM to identify primary cancer diagnoses mentioned in that chunk. The chunk-level candidates are merged and deduplicated by normalized primary site, histology, laterality, and diagnosis date. Each resulting diagnosis becomes an independent work unit with its own extraction state and output tumor record.

### Round-based parallelism

- **Round N** = Nth chunk from each diagnosis work unit
- All work units' chunk N processed in parallel via `asyncio.Semaphore`
- Within a chunk, domain extraction is sequential (demographics before staging, since staging items depend on primary site)
- After each round, extraction state is checkpointed (if `--checkpoint-dir` is set)

### Running updates across chunks

For each subsequent chunk, the prior extraction state is included in the prompt:
```
PRIOR EXTRACTION STATE (update only with higher-confidence evidence):
- Primary Site (Item 400): C50.4 (confidence: 0.95)
- Histology (Item 522): 8500 (confidence: 0.90)
...
```
Items are only updated when the new chunk provides stronger evidence (higher confidence wins).

### Site-specific extraction

The 342 staging/prognostic factor items in NAACCR v26 are not all relevant to every cancer. The `SchemaRegistry` maps primary site + histology to one of ~15 cancer schemas (breast, prostate, lung, colon, melanoma, etc.), each with 10-30 site-specific data items. A breast cancer case extracts ER/PR/HER2/Oncotype; a prostate case extracts Gleason/PSA.

The `SEERManualContextProvider` reads `SEERManuals/manifest.json`, scores vendored SEER/NAACCR manuals by the current cancer type, site, histology, and resolved schema, then adds bounded excerpts to the prompt. This keeps the prompt focused on the relevant coding instructions rather than loading full manuals.

### Validation

Implements 11 critical NAACCR interfield edits:
- Site/Sex (prostate requires male, cervix requires female)
- Site/Histology (hepatocellular only for liver, renal cell only for kidney)
- Site/Laterality (paired organs must have laterality)
- TNM stage group consistency
- Treatment dates >= diagnosis date
- Surgery/radiation field internal consistency

Plus 7 cross-field consistency checks (e.g., if surgery performed, pathologic tumor size should exist).

### Confidence and review

Items are flagged at four priority levels:
- **CRITICAL**: Key variables (primary site, histology, sex, age, county) below 0.9 confidence -- NAACCR Gold requires 100% accuracy on these
- **HIGH**: Required items below 0.7 confidence
- **MEDIUM**: Items involved in validation violations
- **LOW**: Any item below 0.5 confidence

### How fields and allowed values reach the LLM

The LLM never sees a raw schema or unconstrained output space. Every prompt includes an explicit list of which NAACCR fields to extract and what values are valid for each one. The flow:

1. **Item selection** (`extraction/chunk_extractor.py`): Hardcoded item-number lists define which NAACCR items to extract per domain (demographics, surgery, radiation, systemic, follow-up, narratives). Staging items are determined dynamically at runtime — `SchemaRegistry` maps the detected primary site + histology to a cancer schema (breast, prostate, lung, etc.) and returns only the relevant staging/prognostic items.

2. **Code lookup** (`dictionary/code_resolver.py`): `CodeResolver.get_valid_codes_prompt()` looks up each item's valid codes from `CodeList.csv` and formats them as a compact reference string, e.g. `"Valid codes: 0=In situ, 1=Localized, 2=Regional by direct extension, ..., 9=Unknown"`.

3. **Field descriptions** (`llm/structured_output.py`): `SchemaBuilder.build_json_format_instructions()` iterates the items for the current batch and builds a text block listing each field with its `xmlNaaccrId` as the JSON key, the item name and number, and the valid codes from step 2. For items without discrete codes, it adds format guidance (e.g. `YYYYMMDD` for dates, `C##.#` for ICD-O-3 sites, digit count for numeric fields). Example output:

   ```
   Expected fields:
   - "primarySite": Primary Site (Item 400). ICD-O-3 topography code C##.# (e.g., C50.4)
   - "behaviorCodeIcdO3": Behavior Code ICD-O-3 (Item 523). Valid codes: 0=Benign, 1=Uncertain..., 3=Malignant
   - "dateOfDiagnosis": Date of Diagnosis (Item 390). YYYYMMDD format (use 99 for unknown day/month)
   ```

4. **Registry reference retrieval** (`manuals/seer.py`): The resolved diagnosis/schema is used to retrieve short relevant excerpts from local SEER/NAACCR manual text under `SEERManuals/text/`.

5. **Prompt injection** (`extraction/chunk_extractor.py` + `extraction/prompts/chunk_extraction.py`): The field descriptions are injected into both the **system prompt** (via `{json_format_instructions}`) and the **user prompt** (via `{json_field_descriptions}`), so the LLM sees the valid codes twice. The domain-specific system prompts also contain hardcoded summaries of the most critical codes (e.g. Summary Stage values, surgical margin codes, radiation modality codes), plus the retrieved registry manual excerpts when available.

6. **Post-hoc validation** (`dictionary/code_resolver.py`): After the LLM responds, `CodeResolver.resolve()` maps each returned value back to a valid NAACCR code using a 6-tier strategy: exact match → case-insensitive → description match → fuzzy match (rapidfuzz, score >85) → numeric range → fail. Resolution confidence is combined with the LLM's self-reported confidence to produce the final score.

### Checkpointing and resume

With `--checkpoint-dir`, the pipeline saves state after each round:
```
checkpoint_dir/
├── metadata.json           # Completed rounds, work unit count
├── round_0000.json         # Extraction state after round 0
├── round_0001.json         # Extraction state after round 1
└── ...
```

On restart with the same checkpoint directory, the pipeline automatically resumes from the last completed round.

## NAACCR data dictionary

The `NAACCRDataItems/` directory contains the NAACCR v26 data dictionary:

- `DataItems.csv` -- 771 active data items with coding instructions
- `CodeList.csv` -- 4,372 valid codes for 512 items
- `AlternateNames.csv` -- 412 historical name mappings

The `SEERManuals/` directory contains current vendored SEER/NAACCR coding, staging, Appendix C, SSDI, Grade, Summary Stage, EOD, Solid Tumor Rules, and hematopoietic manuals. PDF text extractions under `SEERManuals/text/` are used at runtime; no network access is required during extraction.

## License

This project is provided for research and educational purposes. NAACCR data standards are maintained by the [North American Association of Central Cancer Registries](https://www.naaccr.org/).
