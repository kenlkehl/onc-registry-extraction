"""Shared helpers for the registry-extraction evaluation harness.

Conventions used throughout:
  * Every MRN column is harmonized to the name ``dfci_mrn`` with an integer
    dtype (pandas nullable ``Int64`` while in-frame; plain python ``int`` in sets).
  * Every date column is converted with ``pd.to_datetime`` before any sorting.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # /data1/ken/registry_extraction
DATA = ROOT / "data"
REPO = Path("/data1/ken/onc-registry-extraction")      # the pipeline under test

CAREG_CSV = "/data1/ken/pan_dfci_2025/structured_data/REQ_KK71_192437_F1_CANCER_DIAGNOSIS_CAREG.csv"
SPLIT_CSV = "/data1/ken/pan_dfci_2025/derived_data/split_7-2025.csv"
TEXT_PARQUETS = [
    ("/data1/ken/pan_dfci_2025/derived_data/all_clinical_notes.parquet", "notes"),
    ("/data1/ken/pan_dfci_2025/derived_data/all_imaging_reports.parquet", "imaging"),
    ("/data1/ken/pan_dfci_2025/derived_data/all_path_reports.parquet", "path"),
]

# CAREG columns pulled for ground truth (kept small to avoid loading 242 cols).
CAREG_USECOLS = [
    "DFCI_MRN", "DIAGNOSIS_DT", "DERIVED_BEST_ROW_IND",
    "SITE_CD", "SITE_DESCR",
    "HISTOLOGY_CD", "HISTOLOGY_DESCR",
    "BEHAVIOR_CD", "BEHAVIOR_DESCR",
    "LATERALITY_CD",
    "GRADE_DIFF_CODE", "GRADE_DIFF_DESC",
    "GENERAL_STAGE_CD", "GENERAL_STAGE_DESCR",
    "BEST_AJCC_STAGE_CD",
    "CLIN_T_CD", "CLIN_N_CD", "CLIN_M_CD",
    "PATH_T_CD", "PATH_N_CD", "PATH_M_CD",
    "SSDI_ER_SUMMARY", "SSDI_PR_SUMMARY", "SSDI_HER2_OVERALL_SUMMARY",
]

# ---------------------------------------------------------------------------
# MRN / date harmonization
# ---------------------------------------------------------------------------
def to_int_mrn(series: pd.Series) -> pd.Series:
    """Coerce an MRN column to nullable integer (handles '12345.0' strings)."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def parse_careg_date(series: pd.Series) -> pd.Series:
    """CAREG diagnosis dates look like '15-Jan-2020' (DD-Mmm-YYYY)."""
    return pd.to_datetime(series, format="%d-%b-%Y", errors="coerce")


# ---------------------------------------------------------------------------
# String / code normalization
# ---------------------------------------------------------------------------
def norm_str(v) -> str:
    """NaN/None-safe string normalization (trimmed)."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def norm_site(v) -> str:
    """ICD-O-3 topography, dot-stripped & upper, e.g. 'C50.4' / 'C504' -> 'C504'."""
    return norm_str(v).upper().replace(".", "").replace(" ", "")


def site3(v) -> str:
    """Organ-level site, e.g. 'C504' -> 'C50'."""
    s = norm_site(v)
    return s[:3] if s.startswith("C") and len(s) >= 3 else s


def histology4(v) -> str:
    """4-digit ICD-O-3 morphology. CAREG stores 5 digits (morph+behavior)."""
    digits = re.sub(r"\D", "", norm_str(v))
    if len(digits) >= 5:            # 85003 -> 8500 (drop trailing behavior digit)
        return digits[:4]
    return digits[:4]


def histology3(v) -> str:
    return histology4(v)[:3]


def behavior_from_histcode(v) -> str:
    """Trailing behavior digit of a 5-digit CAREG HISTOLOGY_CD (85003 -> '3')."""
    digits = re.sub(r"\D", "", norm_str(v))
    return digits[4] if len(digits) >= 5 else ""


_PREFIX_RE = re.compile(r"^[CPYRA]+")


def norm_tnm(v, axis: str) -> str:
    """Normalize an AJCC T/N/M value, dropping clinical/path prefixes & axis letter.

    'C2'->'2', 'CT1MI'->'1MI', 'CN0'(axis N)->'0', 'PIS'(axis T)->'IS', 'CX'->'X'.
    """
    s = norm_str(v).upper().replace(" ", "")
    if not s:
        return ""
    s = _PREFIX_RE.sub("", s)        # strip leading c/p/y/r/a prefixes
    s = re.sub(rf"^{axis}", "", s)   # strip the axis letter if present
    return s


def tnm_main(v_norm: str) -> str:
    """Main category ignoring suffixes: '1MI'->'1', '2A'->'2', 'X'->'X', 'IS'->'IS'."""
    m = re.match(r"(IS|X|\d+)", v_norm)
    return m.group(1) if m else v_norm


def norm_stage_group(v) -> str:
    s = norm_str(v).upper().replace("STAGE", "").replace(" ", "")
    return s


def stage_group_main(v) -> str:
    """Collapse AJCC stage group to 0/1/2/3/4 or 'unknown' (88/99/blank)."""
    s = norm_stage_group(v)
    if s in {"", "88", "99", "NA", "UNK", "UNKNOWN"}:
        return "unknown"
    roman = {"IV": "4", "III": "3", "II": "2", "I": "1", "0": "0"}
    for r, a in roman.items():
        if s.startswith(r):
            return a
    m = re.match(r"(\d)", s)
    return m.group(1) if m else "unknown"


# SEER Summary Stage 2018 / CAREG GENERAL_STAGE_CD share the same numeric scheme.
def collapse_summary_stage(v) -> str:
    d = re.sub(r"\D", "", norm_str(v))[:1]
    return {
        "0": "insitu", "1": "localized",
        "2": "regional", "3": "regional", "4": "regional", "5": "regional",
        "7": "distant",
    }.get(d, "unknown")


def collapse_ajcc_to_stage_axis(v) -> str:
    """Bridge an AJCC stage group to the SEER summary-stage extent axis.

    Approximate crosswalk (defensible for solid tumors; coarse for some
    hematologic/Ann-Arbor-staged cancers): 0->in situ, I->localized,
    II/III->regional, IV->distant, 88/99/blank->unknown. Lets us compare
    CAREG's densely-coded AJCC stage group against the model's densely-coded
    SEER summary stage on a common axis.
    """
    main = stage_group_main(v)  # "0".."4" or "unknown"
    return {
        "0": "insitu", "1": "localized",
        "2": "regional", "3": "regional", "4": "distant",
    }.get(main, "unknown")


def collapse_biomarker(v) -> str:
    """ER/PR/HER2 summary -> positive/negative/unknown (handles '1: ER positive')."""
    s = norm_str(v).lower()
    if not s:
        return "unknown"
    if "pos" in s:
        return "positive"
    if "neg" in s:
        return "negative"
    m = re.match(r"\s*(\d)", s)
    if m:
        return {"1": "positive", "0": "negative"}.get(m.group(1), "unknown")
    return "unknown"


def parse_pred_date(v):
    """Parse the pipeline's date items ('YYYYMMDD', possibly partial)."""
    s = re.sub(r"\D", "", norm_str(v))
    for n, fmt in ((8, "%Y%m%d"), (6, "%Y%m"), (4, "%Y")):
        if len(s) >= n:
            d = pd.to_datetime(s[:n], format=fmt, errors="coerce")
            if pd.notna(d):
                return d
    return pd.NaT


# ---------------------------------------------------------------------------
# NAACCR output helpers
# ---------------------------------------------------------------------------
def find_item_col(df: pd.DataFrame, item_number: int):
    """naaccr_output.csv headers look like 'Estrogen Receptor Summary [3827]'."""
    suffix = f"[{item_number}]"
    for c in df.columns:
        if str(c).strip().endswith(suffix):
            return c
    return None


def item_val(row, df, item_number) -> str:
    col = find_item_col(df, item_number)
    if col is None:
        return ""
    return norm_str(row.get(col, ""))
