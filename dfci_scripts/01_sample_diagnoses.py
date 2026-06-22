#!/usr/bin/env python
"""Stage 1: sample N CAREG diagnoses for evaluation.

Filters: diagnosis year in [year-min, year-max], patient in the training split
of split_7-2025.csv, behavior == 3 (malignant primary) by default, and rows not
explicitly flagged DERIVED_BEST_ROW_IND == 'N'.

Outputs (in ./data):
  sampled_diagnoses.csv  -- one row per sampled diagnosis (ground truth)
  sampled_patients.csv   -- unique dfci_mrn of the sampled patients
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

import common as C

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="number of diagnoses to sample")
    ap.add_argument("--seed", type=int, default=42, help="random seed (reproducibility)")
    ap.add_argument("--year-min", type=int, default=2017)
    ap.add_argument("--year-max", type=int, default=2024)
    ap.add_argument("--behavior", default="3",
                    help="BEHAVIOR_CD to keep, comma-separated (default 3=malignant primary; '' = all)")
    ap.add_argument("--out-dir", default=str(C.DATA),
                    help="directory for sampled_diagnoses.csv / sampled_patients.csv (default ./data)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training-split MRNs ------------------------------------------------
    print(f"[1/4] Reading split file: {C.SPLIT_CSV}")
    split = pd.read_csv(C.SPLIT_CSV, dtype=str)
    if "dfci_mrn" not in split.columns:        # tolerate an unnamed index col
        split = split.rename(columns={split.columns[-2]: "dfci_mrn"})
    split["dfci_mrn"] = C.to_int_mrn(split["dfci_mrn"])
    train_mrns = set(split.loc[split["split"] == "train", "dfci_mrn"].dropna().astype(int))
    print(f"      training MRNs: {len(train_mrns):,}")

    # 2. CAREG diagnoses ----------------------------------------------------
    print(f"[2/4] Reading CAREG diagnoses (subset of columns): {C.CAREG_CSV}")
    df = pd.read_csv(C.CAREG_CSV, usecols=C.CAREG_USECOLS, dtype=str)
    df["dfci_mrn"] = C.to_int_mrn(df["DFCI_MRN"])
    df["diagnosis_dt"] = C.parse_careg_date(df["DIAGNOSIS_DT"])
    print(f"      total diagnosis rows: {len(df):,}")

    # 3. Filters ------------------------------------------------------------
    yr = df["diagnosis_dt"].dt.year
    mask = df["diagnosis_dt"].notna() & yr.between(args.year_min, args.year_max)
    mask &= df["dfci_mrn"].isin(train_mrns)
    if args.behavior.strip():
        keep = {b.strip() for b in args.behavior.split(",")}
        mask &= df["BEHAVIOR_CD"].isin(keep)
    # Drop only rows explicitly flagged as NOT the best/deduped row.
    mask &= df["DERIVED_BEST_ROW_IND"].fillna("Y").str.upper().str.strip() != "N"

    pool = df[mask].copy()
    print(f"[3/4] After filters (years {args.year_min}-{args.year_max}, train split, "
          f"behavior={args.behavior or 'all'}): {len(pool):,} candidate diagnoses "
          f"across {pool['dfci_mrn'].nunique():,} patients")

    if len(pool) == 0:
        raise SystemExit("No diagnoses matched the filters; aborting.")

    # 4. Deterministic sample ----------------------------------------------
    pool = pool.sort_values(["dfci_mrn", "diagnosis_dt"]).reset_index(drop=True)
    n = min(args.n, len(pool))
    if n < args.n:
        print(f"      WARNING: only {len(pool)} candidates available; sampling all of them.")
    sample = pool.sample(n=n, random_state=args.seed).sort_values(["dfci_mrn", "diagnosis_dt"])

    out_diag = out_dir / "sampled_diagnoses.csv"
    out_pats = out_dir / "sampled_patients.csv"
    keep_cols = ["dfci_mrn", "diagnosis_dt"] + C.CAREG_USECOLS
    sample[keep_cols].to_csv(out_diag, index=False)
    pd.DataFrame({"dfci_mrn": sorted(sample["dfci_mrn"].dropna().astype(int).unique())}) \
        .to_csv(out_pats, index=False)

    print(f"[4/4] Wrote {len(sample)} diagnoses ({sample['dfci_mrn'].nunique()} patients)")
    print(f"      {out_diag}")
    print(f"      {out_pats}")
    print("\nTop primary sites in sample:")
    print(sample["SITE_DESCR"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
