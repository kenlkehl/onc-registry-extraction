# SEER and NAACCR Manuals

Vendored registry abstraction references used by `onc-registry-pipeline`
to build cancer-type-specific prompt context. Sources are official SEER
and NAACCR pages listed in `manifest.json`.

## Contents

- `appendix_c`: 61 file(s)
- `hematopoietic`: 1 file(s)
- `seer_coding_manual`: 7 file(s)
- `solid_tumor_rules`: 1 file(s)
- `source_page`: 7 file(s)
- `ssdi_grade`: 6 file(s)
- `staging`: 4 file(s)

PDF text extractions are stored under `text/` for prompt retrieval.
Regenerate this folder with `uv run python scripts/download_seer_manuals.py`.
