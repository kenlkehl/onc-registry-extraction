#!/usr/bin/env python
"""Stage 3: run onc-registry-extraction against the combined clinical text.

Invokes the pipeline through `uv run --project <repo>` so the pipeline's own
dependencies are used (it is not installed in the data-prep venv). Points it at a
running OpenAI-compatible / vLLM server.

Output dir (default ./data/extraction_output) will contain:
  naaccr_output.csv     (--format csv; one row per tumor, all extracted items)
  diagnosis_summary.csv (always written; high-level per-diagnosis fields)
  audit_trail.csv, review_queue.csv, llm_calls.jsonl
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import common as C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(C.DATA / "clinical_text.parquet"))
    ap.add_argument("--output", default=str(C.DATA / "extraction_output"))
    ap.add_argument("--vllm-url", required=True,
                    help="OpenAI-compatible base URL, e.g. http://localhost:8010/v1")
    ap.add_argument("--model", default="auto",
                    help="model id (default 'auto' -> detected from /models)")
    ap.add_argument("--max-concurrent", type=int, default=16)
    ap.add_argument("--format", default="csv", choices=["csv", "naaccr_xml", "naaccr_flat"])
    ap.add_argument("--repo", default=str(C.REPO))
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args passed verbatim to onc-registry-pipeline")
    args = ap.parse_args()

    if not Path(args.input).exists():
        raise SystemExit(f"Input not found: {args.input} (run stage 2 first)")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoints"

    cmd = [
        "uv", "run", "--project", args.repo, "onc-registry-pipeline",
        args.input, str(out),
        "--provider", "vllm",
        "--vllm-url", args.vllm_url,
        "--model", args.model,
        "--format", args.format,
        "--max-concurrent", str(args.max_concurrent),
        "--checkpoint-dir", str(ckpt),
        *args.extra,
    ]
    print("Running:\n  " + " ".join(cmd) + "\n", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"Pipeline exited with code {proc.returncode}")

    produced = sorted(p.name for p in out.glob("*.csv"))
    print(f"\nDone. Output dir: {out}")
    print(f"CSV files produced: {produced}")
    if not (out / "naaccr_output.csv").exists():
        print("WARNING: naaccr_output.csv not found — stage 4 needs it "
              "(was --format csv used?).", file=sys.stderr)


if __name__ == "__main__":
    main()
