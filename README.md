# NAACCR v26 Cancer Registry Abstraction Pipeline

Automated extraction of [NAACCR v26](https://www.naaccr.org/) cancer registry data from EHR clinical text using a local LLM. Produces registry-grade output suitable for state cancer registry submission.

## What it does

Given a dataset of patient clinical documents (pathology reports, operative notes, imaging, progress notes, etc.), this pipeline:

1. **Detects** all distinct primary cancers per patient (handles multiple primaries)
2. **Extracts** ~580 NAACCR data items across 5 sequential passes:
   - Demographics & cancer identification (ICD-O-3 site/histology codes)
   - Staging & prognostic factors (TNM, biomarkers, site-specific factors)
   - Treatment (surgery, radiation with 3-phase detail, chemotherapy, hormone, immunotherapy)
   - Follow-up & narrative text summaries
3. **Validates** against NAACCR interfield edit rules (site/sex, site/histology, TNM consistency, etc.)
4. **Scores confidence** and generates a prioritized human review queue
5. **Outputs** NAACCR v26 XML, fixed-width flat file, or CSV with full audit trail

All LLM inference runs locally via [vLLM](https://docs.vllm.ai/) -- no data leaves the machine.

## Quick start

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A running [vLLM](https://docs.vllm.ai/) server with a local LLM loaded

### Install

```bash
git clone <this-repo>
cd registry_skills
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Start vLLM

Start a vLLM server with your preferred model (the pipeline auto-adapts to model size):

```bash
# Example with Llama 3.3 70B (recommended for best accuracy)
vllm serve meta-llama/Llama-3.3-70B-Instruct

# Example with a smaller model (faster, lower accuracy)
vllm serve Qwen/Qwen2.5-14B-Instruct
```

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
uv run naaccr-pipeline input.csv output/

# With options
uv run naaccr-pipeline input.parquet output/ \
    --vllm-url http://localhost:8000/v1 \
    --format naaccr_xml \
    --max-concurrent 16 \
    --confidence-threshold 0.7 \
    --verbose
```

Or without uv:

```bash
python -m naaccr_pipeline.main input.csv output/
```

### Output files

The pipeline writes to the output directory:

| File | Description |
|------|-------------|
| `naaccr_output.xml` | NAACCR v26 XML (NaaccrData > Patient > Tumor hierarchy) |
| `naaccr_output.dat` | NAACCR v26 fixed-width flat file |
| `naaccr_output.csv` | Flat CSV with one column per NAACCR item |
| `audit_trail.csv` | Per-item provenance: source chunk, evidence text, confidence, pass number |
| `review_queue.csv` | Prioritized worklist for human review (CRITICAL/HIGH/MEDIUM/LOW) |

### Converting output to JSON or CSV

The `naaccr-convert` tool converts NAACCR XML output to JSON or simple CSV for downstream use (analytics, data science, integration with other systems):

```bash
# XML -> JSON (format inferred from extension)
uv run naaccr-convert output/naaccr_output.xml output/data.json

# XML -> CSV with human-readable column names
uv run naaccr-convert output/naaccr_output.xml output/data.csv --readable-names

# Drop empty fields for a cleaner file
uv run naaccr-convert output/naaccr_output.xml output/data.json --skip-empty

# Combine options
uv run naaccr-convert output/naaccr_output.xml output/data.csv \
    --readable-names --skip-empty
```

The `--readable-names` flag replaces XML IDs (e.g., `primarySite`) with full NAACCR item names (e.g., `Primary Site`) using the data dictionary. Each tumor is one row/object, with patient-level and root-level items merged in so every record is self-contained.

## CLI reference

```
usage: naaccr-pipeline [-h] [--vllm-url URL] [--max-concurrent N]
                       [--format {naaccr_xml,naaccr_flat,csv}]
                       [--confidence-threshold FLOAT] [--data-dict DIR]
                       [--temperature FLOAT] [--max-tokens N]
                       [--max-retries N] [--verbose]
                       input output

positional arguments:
  input                    Path to input CSV/TSV/Parquet file
  output                   Path to output directory

options:
  --vllm-url URL           vLLM server base URL (default: http://localhost:8000/v1)
  --max-concurrent N       Max concurrent patients (default: 16)
  --format FORMAT          Output format (default: naaccr_xml)
  --confidence-threshold   Confidence threshold for review flagging (default: 0.7)
  --data-dict DIR          Path to NAACCRDataItems directory (default: NAACCRDataItems)
  --temperature FLOAT      LLM sampling temperature (default: 0.0)
  --max-tokens N           Max tokens per LLM response (default: 4096)
  --max-retries N          Max LLM call retries (default: 3)
  -v, --verbose            Enable debug logging
```

## How it works

### Pipeline architecture

```
Input DataFrame
    |
    v
[Ingest] Group by patient, detect structured columns, classify & chunk documents
    |
    v
[Pass 0] Tumor Detection -- identify distinct primary cancers per patient
    |
    v  (for each detected tumor)
[Pass 1] Demographics + Cancer ID -- primary site, histology, diagnosis date
    |
    v  (site/histology determine what Pass 2 extracts)
[Pass 2] Staging -- TNM, Summary Stage, biomarkers, site-specific factors
    |
    v
[Pass 3] Treatment -- 3 sub-passes: surgery, radiation, systemic therapy
    |
    v
[Pass 4] Follow-up + narrative text summaries
    |
    v
[Validate] NAACCR interfield edits, cross-pass consistency checks
    |
    v
[Score] Confidence scoring, human review flagging
    |
    v
[Output] NAACCR XML/flat/CSV + audit trail + review queue
```

### Model adaptation

The pipeline discovers the model at startup and adapts:

| Model size | Items per LLM call | Prompt style |
|------------|-------------------|--------------|
| Small (<15B) | 8 | Simple, focused |
| Medium (15-40B) | 15 | Moderate detail |
| Large (40B+) | 25 | Full instructions |

### Parallelization

- **Patient-level**: `asyncio.Semaphore` controls concurrent patients (default 16)
- **Chunk-level**: independent chunks within a pass processed via `asyncio.gather()`
- **Sequential within patient**: Pass 0 -> 1 -> 2 -> 3 -> 4 (each depends on prior results)

### Site-specific extraction

The 342 staging/prognostic factor items in NAACCR v26 are not all relevant to every cancer. The `SchemaRegistry` maps primary site + histology to one of ~15 cancer schemas (breast, prostate, lung, colon, melanoma, etc.), each with 10-30 site-specific data items. A breast cancer case extracts ER/PR/HER2/Oncotype; a prostate case extracts Gleason/PSA.

### Validation

Implements 11 critical NAACCR interfield edits:
- Site/Sex (prostate requires male, cervix requires female)
- Site/Histology (hepatocellular only for liver, renal cell only for kidney)
- Site/Laterality (paired organs must have laterality)
- TNM stage group consistency
- Treatment dates >= diagnosis date
- Surgery/radiation field internal consistency

Plus 7 cross-pass consistency checks (e.g., if surgery performed, pathologic tumor size should exist).

### Confidence and review

Items are flagged at four priority levels:
- **CRITICAL**: Key variables (primary site, histology, sex, age, county) below 0.9 confidence -- NAACCR Gold requires 100% accuracy on these
- **HIGH**: Required items below 0.7 confidence
- **MEDIUM**: Items involved in validation violations
- **LOW**: Any item below 0.5 confidence

## NAACCR data dictionary

The `NAACCRDataItems/` directory contains the NAACCR v26 data dictionary:

- `DataItems.csv` -- 771 active data items with coding instructions
- `CodeList.csv` -- 4,372 valid codes for 512 items
- `AlternateNames.csv` -- 412 historical name mappings

## License

This project is provided for research and educational purposes. NAACCR data standards are maintained by the [North American Association of Central Cancer Registries](https://www.naaccr.org/).
