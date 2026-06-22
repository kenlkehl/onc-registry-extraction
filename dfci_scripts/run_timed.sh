#!/usr/bin/env bash
set -uo pipefail
PY=/data1/ken/envs/gptoss3/bin/python
export UV_LINK_MODE=copy
cd /data1/ken/registry_extraction
hms(){ printf '%dh%02dm%02ds' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }
echo "START $(date -Is)"
t0=$(date +%s)
$PY scripts/01_sample_diagnoses.py --n 100 --seed 42 || exit 1
t1=$(date +%s); echo ">>> STAGE1 (sample) = $(hms $((t1-t0)))"
$PY scripts/02_pull_clinical_text.py || exit 1
t2=$(date +%s); echo ">>> STAGE2 (pull text) = $(hms $((t2-t1)))"
$PY scripts/03_run_extraction.py --vllm-url http://localhost:8010/v1 --model auto || exit 1
t3=$(date +%s); echo ">>> STAGE3 (extraction) = $(hms $((t3-t2)))"
$PY scripts/04_score.py || exit 1
t4=$(date +%s); echo ">>> STAGE4 (score) = $(hms $((t4-t3)))"
echo ">>> TOTAL = $(hms $((t4-t0)))"
echo "END $(date -Is)"
