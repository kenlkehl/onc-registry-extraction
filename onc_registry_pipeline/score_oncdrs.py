"""Score pipeline output against DFCI OncDRS-formatted cancer registry data.

This tool is **specific to Dana-Farber (DFCI) OncDRS registry exports** -- the
``CANCER_DIAGNOSIS`` (CAREG) table whose one-row-per-diagnosis schema uses column
names such as ``DFCI_MRN``, ``DIAGNOSIS_DT`` (``DD-Mmm-YYYY``), ``SITE_CD``,
``HISTOLOGY_CD``, ``BEHAVIOR_CD``, ``LATERALITY_CD``, ``GRADE_DIFF_CODE``,
``GENERAL_STAGE_CD``, ``BEST_AJCC_STAGE_CD``, ``CLIN_T_CD``/``CLIN_N_CD``/
``CLIN_M_CD`` and the ``SSDI_*`` site-specific items. It is **not** a generic
NAACCR scorer; other registries would need a different column mapping.

It compares this pipeline's per-tumor output (``naaccr_output.csv``, produced with
``--format csv``) against the registry's coded fields:

  1. Predicted tumors are keyed to registry patients by NAACCR item 2300 (Medical
     Record Number). The pipeline only populates item 2300 when the extraction
     input carried an ``mrn`` column equal to ``DFCI_MRN``, so include that column
     when running the pipeline if you intend to score.
  2. For each patient, predicted tumors are greedily matched to that patient's
     registry diagnoses (best ICD-O-3 organ-level site agreement, tie-broken by the
     smallest diagnosis-date difference).
  3. Matched pairs are compared field-by-field after normalization, at an exact and
     a lenient level.

Because DFCI registry data and this pipeline favour different staging systems
(OncDRS codes the AJCC stage group densely but summary stage sparsely; the model
fills SEER summary stage densely and usually abstains on AJCC), every field reports
``accuracy`` (strict; abstention counts as wrong), ``coverage`` (did the model
answer?) and ``accuracy_attempted`` (accuracy when it answered). A
``stage_extent_crosswalk`` metric bridges the registry's AJCC stage group to the
model's SEER summary-stage extent axis so staging skill is not understated.

Usage::

    uv run onc-registry-score-oncdrs \
        output/naaccr_output.csv \
        REQ_..._CANCER_DIAGNOSIS_CAREG.csv \
        scoring/ \
        [--year-min 2017 --year-max 2024 --behaviors 3]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DFCI OncDRS (CAREG CANCER_DIAGNOSIS) column names
# ---------------------------------------------------------------------------
COL_MRN = "DFCI_MRN"
COL_DATE = "DIAGNOSIS_DT"                       # DD-Mmm-YYYY
COL_BEST_ROW = "DERIVED_BEST_ROW_IND"          # 'Y'/'N'/blank
COL_SITE = "SITE_CD"                           # ICD-O-3 topography w/o dot (C504)
COL_SITE_DESCR = "SITE_DESCR"
COL_HIST = "HISTOLOGY_CD"                       # 5-digit: morphology(4) + behavior(1)
COL_HIST_DESCR = "HISTOLOGY_DESCR"
COL_BEHAVIOR = "BEHAVIOR_CD"
COL_LATERALITY = "LATERALITY_CD"
COL_GRADE = "GRADE_DIFF_CODE"
COL_SUMMARY_STAGE = "GENERAL_STAGE_CD"         # SEER-summary-like (often sparse)
COL_AJCC_GROUP = "BEST_AJCC_STAGE_CD"          # AJCC stage group (dense)
COL_CLIN_T = "CLIN_T_CD"
COL_CLIN_N = "CLIN_N_CD"
COL_CLIN_M = "CLIN_M_CD"
COL_ER = "SSDI_ER_SUMMARY"
COL_PR = "SSDI_PR_SUMMARY"
COL_HER2 = "SSDI_HER2_OVERALL_SUMMARY"

# Columns read from the registry export (others ignored).
REGISTRY_USECOLS = [
    COL_MRN, COL_DATE, COL_BEST_ROW, COL_SITE, COL_SITE_DESCR, COL_HIST,
    COL_HIST_DESCR, COL_BEHAVIOR, COL_LATERALITY, COL_GRADE, COL_SUMMARY_STAGE,
    COL_AJCC_GROUP, COL_CLIN_T, COL_CLIN_N, COL_CLIN_M, COL_ER, COL_PR, COL_HER2,
]

# NAACCR item numbers read from naaccr_output.csv.
ITEM_MRN = 2300
ITEM_DX_DATE = 390
ITEM_SITE = 400
ITEM_HIST = 522
ITEM_BEHAVIOR = 523
ITEM_LATERALITY = 410
ITEM_GRADE = 440
ITEM_SUMMARY_STAGE = 764
ITEM_CLIN_T, ITEM_CLIN_N, ITEM_CLIN_M, ITEM_CLIN_GROUP = 1001, 1002, 1003, 1004
ITEM_PATH_T, ITEM_PATH_N, ITEM_PATH_M, ITEM_PATH_GROUP = 1011, 1012, 1013, 1014
ITEM_ER, ITEM_PR, ITEM_HER2 = 3827, 3915, 3855

_TNM_UNK = ("", "x", "88", "99", "unknown", "nan")
_STAGE_UNK = ("", "9", "88", "99", "unknown", "nan")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def norm_str(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def norm_site(v) -> str:
    """'C50.4' / 'C504' -> 'C504'."""
    return norm_str(v).upper().replace(".", "").replace(" ", "")


def site3(v) -> str:
    s = norm_site(v)
    return s[:3] if s.startswith("C") and len(s) >= 3 else s


def histology4(v) -> str:
    """4-digit ICD-O-3 morphology (OncDRS HISTOLOGY_CD is 5 digits incl. behavior)."""
    return re.sub(r"\D", "", norm_str(v))[:4]


def histology3(v) -> str:
    return histology4(v)[:3]


def behavior_from_histcode(v) -> str:
    digits = re.sub(r"\D", "", norm_str(v))
    return digits[4] if len(digits) >= 5 else ""


_PREFIX_RE = re.compile(r"^[CPYRA]+")


def norm_tnm(v, axis: str) -> str:
    """Strip clinical/path prefixes & axis letter: 'CT1MI'->'1MI', 'C2'->'2', 'CX'->'X'."""
    s = norm_str(v).upper().replace(" ", "")
    if not s:
        return ""
    s = _PREFIX_RE.sub("", s)
    s = re.sub(rf"^{axis}", "", s)
    return s


def tnm_main(v_norm: str) -> str:
    m = re.match(r"(IS|X|\d+)", v_norm)
    return m.group(1) if m else v_norm


def norm_stage_group(v) -> str:
    return norm_str(v).upper().replace("STAGE", "").replace(" ", "")


def stage_group_main(v) -> str:
    s = norm_stage_group(v)
    if s in ("", "88", "99", "NA", "UNK", "UNKNOWN"):
        return "unknown"
    for r, a in (("IV", "4"), ("III", "3"), ("II", "2"), ("I", "1"), ("0", "0")):
        if s.startswith(r):
            return a
    m = re.match(r"(\d)", s)
    return m.group(1) if m else "unknown"


def collapse_summary_stage(v) -> str:
    """SEER Summary Stage 2018 / OncDRS GENERAL_STAGE_CD numeric -> extent axis."""
    d = re.sub(r"\D", "", norm_str(v))[:1]
    return {"0": "insitu", "1": "localized", "2": "regional", "3": "regional",
            "4": "regional", "5": "regional", "7": "distant"}.get(d, "unknown")


def collapse_ajcc_to_stage_axis(v) -> str:
    """Approximate AJCC stage group -> SEER extent axis (bridges the two systems)."""
    return {"0": "insitu", "1": "localized", "2": "regional", "3": "regional",
            "4": "distant"}.get(stage_group_main(v), "unknown")


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
    s = re.sub(r"\D", "", norm_str(v))
    for n, fmt in ((8, "%Y%m%d"), (6, "%Y%m"), (4, "%Y")):
        if len(s) >= n:
            d = pd.to_datetime(s[:n], format=fmt, errors="coerce")
            if pd.notna(d):
                return d
    return pd.NaT


def to_int(v):
    digits = re.sub(r"\D", "", norm_str(v))
    return int(digits) if digits else None


def find_item_col(df: pd.DataFrame, item_number: int):
    suffix = f"[{item_number}]"
    for c in df.columns:
        if str(c).strip().endswith(suffix):
            return c
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_predictions(path: str) -> dict:
    """Read naaccr_output.csv into {mrn: [per-tumor dict, ...]}."""
    df = pd.read_csv(path, dtype=str)
    mrn_col = find_item_col(df, ITEM_MRN)
    if mrn_col is None:
        raise SystemExit(
            "Predictions file has no 'Medical Record Number [2300]' column, so it "
            "cannot be joined to OncDRS by DFCI_MRN. Re-run the pipeline with an "
            "'mrn' column (= DFCI_MRN) in the input so item 2300 is populated."
        )

    def val(row, item):
        col = find_item_col(df, item)
        return norm_str(row.get(col, "")) if col is not None else ""

    def tnm_clin_or_path(row, clin_item, path_item, axis):
        """Predicted T/N/M: prefer the clinical value, fall back to pathologic.

        The registry ground truth is the clinical axis (CLIN_T_CD/...), but the
        model frequently records staging only in the pathologic items. Crediting
        the pathologic value when the clinical one is absent mirrors the existing
        stage-group fallback and avoids penalizing correctly-staged-but-pathologic
        extractions.
        """
        c = val(row, clin_item)
        return c if norm_tnm(c, axis) not in _TNM_UNK else val(row, path_item)

    by_mrn: dict[int, list[dict]] = {}
    for _, row in df.iterrows():
        mrn = to_int(row[mrn_col])
        if mrn is None:
            continue
        clin_group = val(row, ITEM_CLIN_GROUP)
        path_group = val(row, ITEM_PATH_GROUP)
        ajcc = path_group if stage_group_main(path_group) != "unknown" else clin_group
        by_mrn.setdefault(mrn, []).append({
            "site": val(row, ITEM_SITE), "hist": val(row, ITEM_HIST),
            "behavior": val(row, ITEM_BEHAVIOR), "lat": val(row, ITEM_LATERALITY),
            "grade": val(row, ITEM_GRADE), "summary": val(row, ITEM_SUMMARY_STAGE),
            "clin_t": tnm_clin_or_path(row, ITEM_CLIN_T, ITEM_PATH_T, "T"),
            "clin_n": tnm_clin_or_path(row, ITEM_CLIN_N, ITEM_PATH_N, "N"),
            "clin_m": tnm_clin_or_path(row, ITEM_CLIN_M, ITEM_PATH_M, "M"),
            "ajcc": ajcc,
            "er": val(row, ITEM_ER), "pr": val(row, ITEM_PR), "her2": val(row, ITEM_HER2),
            "dxdate": parse_pred_date(val(row, ITEM_DX_DATE)),
        })
    return by_mrn


def load_registry(path, year_min=None, year_max=None, behaviors=None,
                  best_rows_only=True) -> pd.DataFrame:
    """Read a DFCI OncDRS CANCER_DIAGNOSIS export and harmonize MRN/date."""
    cols = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in REGISTRY_USECOLS if c in cols]
    missing = set(REGISTRY_USECOLS) - set(usecols)
    if missing:
        logger.warning("OncDRS export missing expected columns (scored as blank): %s",
                       sorted(missing))
    if COL_MRN not in usecols or COL_DATE not in usecols:
        raise SystemExit(f"Registry file must contain {COL_MRN} and {COL_DATE}.")

    df = pd.read_csv(path, usecols=usecols, dtype=str)
    df["dfci_mrn"] = pd.to_numeric(df[COL_MRN], errors="coerce").astype("Int64")
    df["diagnosis_dt"] = pd.to_datetime(df[COL_DATE], format="%d-%b-%Y", errors="coerce")

    mask = pd.Series(True, index=df.index)
    if year_min is not None or year_max is not None:
        yr = df["diagnosis_dt"].dt.year
        lo = year_min if year_min is not None else -10**9
        hi = year_max if year_max is not None else 10**9
        mask &= df["diagnosis_dt"].notna() & yr.between(lo, hi)
    if behaviors and COL_BEHAVIOR in df.columns:
        mask &= df[COL_BEHAVIOR].isin({b.strip() for b in behaviors})
    if best_rows_only and COL_BEST_ROW in df.columns:
        mask &= df[COL_BEST_ROW].fillna("Y").str.upper().str.strip() != "N"

    # Ensure every expected column exists so scoring can read it uniformly.
    for c in REGISTRY_USECOLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Matching + scoring
# ---------------------------------------------------------------------------
def match_patient(truth_rows: pd.DataFrame, preds: list):
    """Greedy one-to-one: truth index -> (pred index, site_match, days)."""
    pairs = []
    for ti, (_, tr) in enumerate(truth_rows.iterrows()):
        ts, td = site3(tr[COL_SITE]), tr["diagnosis_dt"]
        for pi, pr in enumerate(preds):
            site_m = ts != "" and ts == site3(pr["site"])
            if pd.notna(td) and pd.notna(pr["dxdate"]):
                days = abs((td - pr["dxdate"]).days)
            else:
                days = None
            pairs.append((0 if site_m else 1, days if days is not None else 10**9,
                          ti, pi, site_m, days))
    pairs.sort(key=lambda x: (x[0], x[1]))
    used_t, used_p, assign = set(), set(), {}
    for _, _, ti, pi, site_m, days in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi)
        assign[ti] = (pi, site_m, days)
    return assign, len(preds) - len(used_p)


def score_row(tr: pd.Series, pr) -> dict:
    """Normalized GT/pred values + gt-present, pred-present, and correctness flags."""
    g = {
        "site": norm_site(tr[COL_SITE]), "hist": histology4(tr[COL_HIST]),
        "beh": norm_str(tr[COL_BEHAVIOR]), "lat": norm_str(tr[COL_LATERALITY]),
        "grade": norm_str(tr[COL_GRADE]), "ss": norm_str(tr[COL_SUMMARY_STAGE]),
        "ajcc": norm_str(tr[COL_AJCC_GROUP]),
        "t": tr[COL_CLIN_T], "n": tr[COL_CLIN_N], "m": tr[COL_CLIN_M],
        "er": tr[COL_ER], "pr": tr[COL_PR], "her2": tr[COL_HER2],
    }
    is_breast = site3(tr[COL_SITE]) == "C50"
    r = {
        "gt_site": g["site"], "gt_histology": g["hist"], "gt_behavior": g["beh"],
        "gt_laterality": g["lat"], "gt_grade": g["grade"],
        "gt_summary_stage": g["ss"], "gt_ajcc_group": g["ajcc"],
        "gt_clin_t": norm_str(g["t"]), "gt_clin_n": norm_str(g["n"]), "gt_clin_m": norm_str(g["m"]),
        "gt_er": collapse_biomarker(g["er"]), "gt_pr": collapse_biomarker(g["pr"]),
        "gt_her2": collapse_biomarker(g["her2"]),
        "gt_stage_axis": collapse_ajcc_to_stage_axis(g["ajcc"]),
    }

    # gt-present flags (informative ground truth only)
    r["present_site"] = g["site"] != ""
    r["present_hist"] = g["hist"] != ""
    r["present_beh"] = g["beh"] not in ("", "nan")
    r["present_lat"] = g["lat"] not in ("", "9", "nan")
    r["present_grade"] = g["grade"] not in ("", "9", "nan")
    r["present_ss"] = g["ss"] not in ("", "9", "nan")
    r["present_ajcc"] = stage_group_main(g["ajcc"]) != "unknown"
    r["present_stage_axis"] = r["gt_stage_axis"] != "unknown"
    r["present_t"] = norm_tnm(g["t"], "T").lower() not in _TNM_UNK
    r["present_n"] = norm_tnm(g["n"], "N").lower() not in _TNM_UNK
    r["present_m"] = norm_tnm(g["m"], "M").lower() not in _TNM_UNK
    r["present_er"] = is_breast and r["gt_er"] in ("positive", "negative")
    r["present_pr"] = is_breast and r["gt_pr"] in ("positive", "negative")
    r["present_her2"] = is_breast and r["gt_her2"] in ("positive", "negative")
    r["present_date_gt"] = pd.notna(tr["diagnosis_dt"])

    pred_str_keys = ("pred_site", "pred_histology", "pred_behavior", "pred_laterality",
                     "pred_grade", "pred_summary_stage", "pred_ajcc_group", "pred_clin_t",
                     "pred_clin_n", "pred_clin_m", "pred_er", "pred_pr", "pred_her2",
                     "pred_stage_axis")
    ok_keys = ("ok_site_exact", "ok_site3", "ok_hist4", "ok_hist3", "ok_beh", "ok_lat",
               "ok_grade", "ok_ss_exact", "ok_ss_collapse", "ok_stage_axis", "ok_ajcc_exact",
               "ok_ajcc_main", "ok_t_exact", "ok_t_main", "ok_n_exact", "ok_n_main",
               "ok_m_exact", "ok_date_exact", "ok_date30", "ok_date90", "ok_date365",
               "ok_year", "ok_er", "ok_pr", "ok_her2")
    pp_keys = ("pp_site", "pp_hist", "pp_beh", "pp_lat", "pp_grade", "pp_ss", "pp_ajcc",
               "pp_stage_axis", "pp_t", "pp_n", "pp_m", "pp_er", "pp_pr", "pp_her2",
               "present_date_pred")

    if pr is None:
        for k in pred_str_keys:
            r[k] = ""
        for k in ok_keys + pp_keys:
            r[k] = False
        r["present_date"] = False
        return r

    p = {
        "site": norm_site(pr["site"]), "hist": histology4(pr["hist"]),
        "beh": norm_str(pr["behavior"]), "lat": norm_str(pr["lat"]),
        "grade": norm_str(pr["grade"]), "ss": norm_str(pr["summary"]),
        "ajcc": norm_str(pr["ajcc"]),
        "t": pr["clin_t"], "n": pr["clin_n"], "m": pr["clin_m"],
        "er": collapse_biomarker(pr["er"]), "pr": collapse_biomarker(pr["pr"]),
        "her2": collapse_biomarker(pr["her2"]),
    }
    pred_stage_axis = collapse_summary_stage(p["ss"])
    r.update({
        "pred_site": p["site"], "pred_histology": p["hist"], "pred_behavior": p["beh"],
        "pred_laterality": p["lat"], "pred_grade": p["grade"],
        "pred_summary_stage": p["ss"], "pred_ajcc_group": p["ajcc"],
        "pred_clin_t": norm_str(p["t"]), "pred_clin_n": norm_str(p["n"]),
        "pred_clin_m": norm_str(p["m"]), "pred_er": p["er"], "pred_pr": p["pr"],
        "pred_her2": p["her2"], "pred_stage_axis": pred_stage_axis,
    })

    # pred-present (informative answer?) flags
    r["pp_site"] = p["site"] != ""
    r["pp_hist"] = p["hist"] != ""
    r["pp_beh"] = p["beh"] not in ("", "9", "nan")
    r["pp_lat"] = p["lat"].lower() not in ("", "9", "unknown", "nan")
    r["pp_grade"] = p["grade"].lower() not in _STAGE_UNK
    r["pp_ss"] = p["ss"].lower() not in _STAGE_UNK
    r["pp_ajcc"] = stage_group_main(p["ajcc"]) != "unknown"
    r["pp_stage_axis"] = pred_stage_axis != "unknown"
    r["pp_t"] = norm_tnm(p["t"], "T").lower() not in _TNM_UNK
    r["pp_n"] = norm_tnm(p["n"], "N").lower() not in _TNM_UNK
    r["pp_m"] = norm_tnm(p["m"], "M").lower() not in _TNM_UNK
    r["pp_er"] = p["er"] in ("positive", "negative")
    r["pp_pr"] = p["pr"] in ("positive", "negative")
    r["pp_her2"] = p["her2"] in ("positive", "negative")

    r["ok_site_exact"] = g["site"] != "" and g["site"] == p["site"]
    r["ok_site3"] = site3(g["site"]) != "" and site3(g["site"]) == site3(p["site"])
    r["ok_hist4"] = g["hist"] != "" and g["hist"] == p["hist"]
    r["ok_hist3"] = histology3(g["hist"]) != "" and histology3(g["hist"]) == histology3(p["hist"])
    r["ok_beh"] = g["beh"] != "" and g["beh"] == p["beh"]
    r["ok_lat"] = g["lat"] != "" and g["lat"] == p["lat"]
    r["ok_grade"] = g["grade"] != "" and g["grade"] == p["grade"]
    r["ok_ss_exact"] = g["ss"] != "" and re.sub(r"\D", "", g["ss"])[:1] == re.sub(r"\D", "", p["ss"])[:1]
    r["ok_ss_collapse"] = collapse_summary_stage(g["ss"]) != "unknown" \
        and collapse_summary_stage(g["ss"]) == collapse_summary_stage(p["ss"])
    r["ok_stage_axis"] = r["gt_stage_axis"] != "unknown" and r["gt_stage_axis"] == pred_stage_axis
    r["ok_ajcc_exact"] = g["ajcc"] != "" and norm_stage_group(g["ajcc"]) == norm_stage_group(p["ajcc"])
    r["ok_ajcc_main"] = stage_group_main(g["ajcc"]) != "unknown" \
        and stage_group_main(g["ajcc"]) == stage_group_main(p["ajcc"])
    for axis, key in (("T", "t"), ("N", "n"), ("M", "m")):
        gn, pn = norm_tnm(g[key], axis), norm_tnm(p[key], axis)
        r[f"ok_{key}_exact"] = gn != "" and gn == pn
        if axis != "M":
            r[f"ok_{key}_main"] = tnm_main(gn) != "" and tnm_main(gn) == tnm_main(pn)

    td, pdt = tr["diagnosis_dt"], pr["dxdate"]
    r["present_date_pred"] = pd.notna(pdt)
    r["present_date"] = pd.notna(td) and pd.notna(pdt)
    if r["present_date"]:
        delta = abs((td - pdt).days)
        r["ok_date_exact"] = delta == 0
        r["ok_date30"] = delta <= 30
        r["ok_date90"] = delta <= 90
        r["ok_date365"] = delta <= 365
        r["ok_year"] = td.year == pdt.year
    else:
        for k in ("ok_date_exact", "ok_date30", "ok_date90", "ok_date365", "ok_year"):
            r[k] = False

    r["ok_er"] = r["gt_er"] in ("positive", "negative") and r["gt_er"] == p["er"]
    r["ok_pr"] = r["gt_pr"] in ("positive", "negative") and r["gt_pr"] == p["pr"]
    r["ok_her2"] = r["gt_her2"] in ("positive", "negative") and r["gt_her2"] == p["her2"]
    return r


# (label, gt_present_col, pred_present_col, ok_col)
METRICS = [
    ("primary_site_exact", "present_site", "pp_site", "ok_site_exact"),
    ("primary_site_3char", "present_site", "pp_site", "ok_site3"),
    ("histology_4digit", "present_hist", "pp_hist", "ok_hist4"),
    ("histology_3digit", "present_hist", "pp_hist", "ok_hist3"),
    ("behavior", "present_beh", "pp_beh", "ok_beh"),
    ("laterality", "present_lat", "pp_lat", "ok_lat"),
    ("grade", "present_grade", "pp_grade", "ok_grade"),
    ("summary_stage_exact", "present_ss", "pp_ss", "ok_ss_exact"),
    ("summary_stage_collapsed", "present_ss", "pp_ss", "ok_ss_collapse"),
    ("stage_extent_crosswalk", "present_stage_axis", "pp_stage_axis", "ok_stage_axis"),
    ("ajcc_stage_group_exact", "present_ajcc", "pp_ajcc", "ok_ajcc_exact"),
    ("ajcc_stage_group_main", "present_ajcc", "pp_ajcc", "ok_ajcc_main"),
    ("ajcc_clin_T_exact", "present_t", "pp_t", "ok_t_exact"),
    ("ajcc_clin_T_main", "present_t", "pp_t", "ok_t_main"),
    ("ajcc_clin_N_exact", "present_n", "pp_n", "ok_n_exact"),
    ("ajcc_clin_N_main", "present_n", "pp_n", "ok_n_main"),
    ("ajcc_clin_M_exact", "present_m", "pp_m", "ok_m_exact"),
    ("dx_date_exact", "present_date_gt", "present_date_pred", "ok_date_exact"),
    ("dx_date_within30d", "present_date_gt", "present_date_pred", "ok_date30"),
    ("dx_date_within90d", "present_date_gt", "present_date_pred", "ok_date90"),
    ("dx_date_within365d", "present_date_gt", "present_date_pred", "ok_date365"),
    ("dx_year_match", "present_date_gt", "present_date_pred", "ok_year"),
    ("ER_breast", "present_er", "pp_er", "ok_er"),
    ("PR_breast", "present_pr", "pp_pr", "ok_pr"),
    ("HER2_breast", "present_her2", "pp_her2", "ok_her2"),
]


def score(predictions: dict, registry: pd.DataFrame):
    """Return (per_diagnosis DataFrame, metrics DataFrame, summary dict)."""
    rows, n_extra_total = [], 0
    for mrn, tg in registry.groupby("dfci_mrn"):
        plist = predictions.get(int(mrn), [])
        assign, n_extra = (({}, 0) if not plist else match_patient(tg, plist))
        n_extra_total += n_extra
        for ti, (_, tr) in enumerate(tg.iterrows()):
            pi, site_m, days = assign.get(ti, (None, False, None))
            pr = plist[pi] if pi is not None else None
            matched = pr is not None and (site_m or (days is not None and days <= 365))
            base = {
                "dfci_mrn": int(mrn),
                "gt_diagnosis_dt": tr["diagnosis_dt"].date() if pd.notna(tr["diagnosis_dt"]) else "",
                "gt_site_descr": norm_str(tr.get(COL_SITE_DESCR, "")),
                "gt_histology_descr": norm_str(tr.get(COL_HIST_DESCR, "")),
                "n_pred_tumors_for_patient": len(plist),
                "matched": matched, "match_site": site_m, "match_date_delta_days": days,
            }
            base.update(score_row(tr, pr if matched else None))
            rows.append(base)

    scored = pd.DataFrame(rows)
    matched = scored[scored["matched"]] if len(scored) else scored
    recall = float(scored["matched"].mean()) if len(scored) else 0.0

    def rate(num, den):
        return round(float(num) / den, 4) if den else None

    metric_rows = []
    for label, gt_col, pp_col, ok_col in METRICS:
        d = matched[matched[gt_col]] if len(matched) else matched
        n_gt = int(len(d))
        att = d[d[pp_col]] if n_gt else d
        n_att = int(len(att))
        metric_rows.append({
            "metric": label, "n_gt": n_gt,
            "accuracy": rate(d[ok_col].sum(), n_gt) if n_gt else None,
            "coverage": rate(d[pp_col].sum(), n_gt) if n_gt else None,
            "n_attempted": n_att,
            "accuracy_attempted": rate(att[ok_col].sum(), n_att) if n_att else None,
        })
    metrics = pd.DataFrame(metric_rows)
    summary = {
        "n_truth_diagnoses": int(len(scored)),
        "n_patients": int(scored["dfci_mrn"].nunique()) if len(scored) else 0,
        "diagnosis_match_recall": round(recall, 4),
        "n_matched": int(scored["matched"].sum()) if len(scored) else 0,
        "n_extra_predicted_tumors": int(n_extra_total),
        "fields": {r["metric"]: {k: r[k] for k in
                   ("n_gt", "accuracy", "coverage", "n_attempted", "accuracy_attempted")}
                   for r in metric_rows},
    }
    return scored, metrics, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pipeline output against DFCI OncDRS registry data.")
    parser.add_argument("predictions", help="naaccr_output.csv from the pipeline (--format csv)")
    parser.add_argument("registry", help="DFCI OncDRS CANCER_DIAGNOSIS (CAREG) CSV export")
    parser.add_argument("output", help="output directory for scoring results")
    parser.add_argument("--year-min", type=int, default=None,
                        help="keep registry diagnoses with year >= this")
    parser.add_argument("--year-max", type=int, default=None,
                        help="keep registry diagnoses with year <= this")
    parser.add_argument("--behaviors", default=None,
                        help="comma-separated BEHAVIOR_CD values to keep (e.g. '3')")
    parser.add_argument("--keep-non-best-rows", action="store_true",
                        help="do not drop rows flagged DERIVED_BEST_ROW_IND == 'N'")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(args.predictions)
    behaviors = [b for b in args.behaviors.split(",")] if args.behaviors else None
    registry = load_registry(
        args.registry, year_min=args.year_min, year_max=args.year_max,
        behaviors=behaviors, best_rows_only=not args.keep_non_best_rows,
    )
    # Only score registry patients for whom we have predictions.
    registry = registry[registry["dfci_mrn"].isin(predictions.keys())]

    n_pred_tumors = sum(len(v) for v in predictions.values())
    logger.info("Scoring %d registry diagnoses (%d patients) vs %d predicted tumors.",
                len(registry), registry["dfci_mrn"].nunique(), n_pred_tumors)

    scored, metrics, summary = score(predictions, registry)

    per_path = out_dir / "per_diagnosis_scored.csv"
    scored.to_csv(per_path, index=False)
    metrics.to_csv(out_dir / "field_metrics.csv", index=False)
    (out_dir / "field_metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {per_path}")
    print(f"Wrote {out_dir / 'field_metrics.csv'} and field_metrics.json\n")
    print(f"Diagnosis match recall: {summary['diagnosis_match_recall']:.1%} "
          f"({summary['n_matched']}/{summary['n_truth_diagnoses']}); "
          f"extra predicted tumors: {summary['n_extra_predicted_tumors']}\n")
    print("Per-field metrics over matched diagnoses with informative ground truth.")
    print("  accuracy = strict (abstention counts as wrong); coverage = model answered; "
          "accuracy_attempted = accuracy when answered.")
    with pd.option_context("display.max_rows", None):
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
