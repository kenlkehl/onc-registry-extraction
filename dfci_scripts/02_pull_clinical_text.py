#!/usr/bin/env python
"""Stage 2: pull ALL clinical text for the sampled patients.

Reads the three derived parquet sources (clinical notes, imaging, path) with a
predicate-pushdown filter on dfci_mrn, harmonizes MRN (int) and date (datetime),
sorts by dfci_mrn then date, and writes ONE pipeline-ready parquet.

The pipeline (onc_registry_pipeline) requires columns: patient_id, date, text.
We also emit `mrn` (-> NAACCR item 2300) so extractions can be joined back to the
ground truth, plus `note_type`/`source` (ignored by the reader).

Output: ./data/clinical_text.parquet
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

import common as C

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", default=str(C.DATA / "sampled_patients.csv"))
    ap.add_argument("--out", default=str(C.DATA / "clinical_text.parquet"))
    args = ap.parse_args()

    pats = pd.read_csv(args.patients)
    mrns = sorted(C.to_int_mrn(pats["dfci_mrn"]).dropna().astype(int).unique())
    mrn_list = [int(m) for m in mrns]
    print(f"[1/4] Pulling text for {len(mrn_list)} patients")

    frames = []
    for path, source in C.TEXT_PARQUETS:
        print(f"[2/4] Reading {source}: {path}")
        part = pd.read_parquet(
            path,
            columns=["dfci_mrn", "date", "text", "note_type"],
            filters=[("dfci_mrn", "in", mrn_list)],
        )
        part["source"] = source
        print(f"      {len(part):,} documents")
        frames.append(part)

    docs = pd.concat(frames, ignore_index=True)
    docs["dfci_mrn"] = C.to_int_mrn(docs["dfci_mrn"])
    docs["date"] = pd.to_datetime(docs["date"], errors="coerce")

    # Drop empty / undated text.
    before = len(docs)
    docs = docs[docs["text"].notna()]
    docs = docs[docs["text"].astype(str).str.strip() != ""]
    docs = docs[docs["date"].notna() & docs["dfci_mrn"].notna()]
    print(f"[3/4] {len(docs):,} usable documents (dropped {before - len(docs):,} empty/undated)")

    # Sort by patient then chronological date.
    docs = docs.sort_values(["dfci_mrn", "date"]).reset_index(drop=True)

    out = pd.DataFrame({
        "patient_id": docs["dfci_mrn"].astype(int),
        "mrn": docs["dfci_mrn"].astype(int),
        "date": docs["date"].dt.strftime("%Y-%m-%d"),   # clean string for robust parsing
        "text": docs["text"].astype(str),
        "note_type": docs["note_type"].astype("string"),
        "source": docs["source"].astype("string"),
    })
    out.to_parquet(args.out, index=False)

    per_pat = docs.groupby("dfci_mrn").size()
    missing = sorted(set(mrn_list) - set(docs["dfci_mrn"].astype(int).unique()))
    print(f"[4/4] Wrote {len(out):,} documents -> {args.out}")
    print(f"      docs/patient: min={per_pat.min()} median={int(per_pat.median())} "
          f"max={per_pat.max()} mean={per_pat.mean():.0f}")
    print(f"      by source: {docs['source'].value_counts().to_dict()}")
    if missing:
        print(f"      WARNING: {len(missing)} sampled patients have ZERO documents: {missing[:20]}"
              + (" ..." if len(missing) > 20 else ""))


if __name__ == "__main__":
    main()
