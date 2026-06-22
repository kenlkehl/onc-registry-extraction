"""Tests for the DFCI OncDRS registry scorer (onc_registry_pipeline.score_oncdrs)."""
from __future__ import annotations

import pandas as pd

from onc_registry_pipeline import score_oncdrs as s


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def test_site_and_histology_normalization() -> None:
    assert s.norm_site("C50.4") == "C504"      # predicted form (with dot)
    assert s.norm_site("C504") == "C504"        # OncDRS form (no dot)
    assert s.site3("C50.4") == "C50"
    assert s.histology4("85003") == "8500"      # OncDRS 5-digit -> 4-digit morphology
    assert s.histology4("8500") == "8500"        # predicted 4-digit unchanged
    assert s.behavior_from_histcode("85003") == "3"


def test_tnm_normalization() -> None:
    assert s.norm_tnm("CT1MI", "T") == "1MI"     # OncDRS clinical T
    assert s.norm_tnm("C2", "T") == "2"
    assert s.norm_tnm("cT1", "T") == "1"          # predicted form
    assert s.norm_tnm("CN0", "N") == "0"
    assert s.norm_tnm("CX", "T") == "X"
    assert s.tnm_main("1MI") == "1"
    assert s.tnm_main("2A") == "2"


def test_stage_crosswalk_and_biomarker() -> None:
    # AJCC stage group -> SEER extent axis
    assert s.collapse_ajcc_to_stage_axis("1A") == "localized"
    assert s.collapse_ajcc_to_stage_axis("3B") == "regional"
    assert s.collapse_ajcc_to_stage_axis("4") == "distant"
    assert s.collapse_ajcc_to_stage_axis("88") == "unknown"
    # model SEER summary stage -> same axis
    assert s.collapse_summary_stage("1") == "localized"
    assert s.collapse_summary_stage("7") == "distant"
    # biomarker collapse handles OncDRS "code: text" form
    assert s.collapse_biomarker("1: ER positive") == "positive"
    assert s.collapse_biomarker("0: HER2 negative; equivocal") == "negative"
    assert s.collapse_biomarker("9: Not documented") == "unknown"


# ---------------------------------------------------------------------------
# End-to-end scoring on a tiny in-memory dataset
# ---------------------------------------------------------------------------
# NOTE: all values below are synthetic/textbook examples (canonical breast ICD-O-3
# codes, an invented MRN and dates) -- no real patient data.
def _write_predictions(path) -> None:
    """One synthetic patient (MRN 990001) with two predicted tumors; NAACCR CSV form."""
    cols = {
        "Medical Record Number [2300]": ["990001", "990001"],
        "Date of Diagnosis [390]": ["20200115", "20240301"],
        "Primary Site [400]": ["C50.4", "C61.9"],
        "Histologic Type ICD-O-3 [522]": ["8500", "8140"],
        "Behavior Code ICD-O-3 [523]": ["3", "3"],
        "Laterality [410]": ["1", "0"],
        "Grade [440]": ["9", "9"],
        "Summary Stage 2018 [764]": ["1", "7"],
        "AJCC TNM Clin Stage Group [1004]": ["99", "99"],
        "AJCC TNM Path Stage Group [1014]": ["99", "99"],
        "AJCC TNM Clin T [1001]": ["88", "88"],
        "AJCC TNM Clin N [1002]": ["88", "88"],
        "AJCC TNM Clin M [1003]": ["88", "88"],
    }
    pd.DataFrame(cols).to_csv(path, index=False)


def _registry_df() -> pd.DataFrame:
    """One synthetic OncDRS diagnosis for MRN 990001 (a textbook breast primary)."""
    df = pd.DataFrame({
        s.COL_MRN: ["990001"],
        s.COL_DATE: ["15-Jan-2020"],
        s.COL_SITE: ["C504"],
        s.COL_SITE_DESCR: ["BREAST, UPPER-OUTER QUADRANT"],
        s.COL_HIST: ["85003"],
        s.COL_HIST_DESCR: ["Infiltrating duct carcinoma, NOS"],
        s.COL_BEHAVIOR: ["3"],
        s.COL_LATERALITY: ["1"],
        s.COL_GRADE: ["9"],
        s.COL_SUMMARY_STAGE: [""],
        s.COL_AJCC_GROUP: ["1A"],
        s.COL_CLIN_T: ["CT1"],
        s.COL_CLIN_N: ["CN0"],
        s.COL_CLIN_M: ["CM0"],
        s.COL_ER: [""],
        s.COL_PR: [""],
        s.COL_HER2: [""],
    })
    df["dfci_mrn"] = pd.to_numeric(df[s.COL_MRN]).astype("Int64")
    df["diagnosis_dt"] = pd.to_datetime(df[s.COL_DATE], format="%d-%b-%Y", errors="coerce")
    return df


def test_matching_picks_correct_tumor(tmp_path) -> None:
    pred_path = tmp_path / "naaccr_output.csv"
    _write_predictions(pred_path)
    preds = s.load_predictions(str(pred_path))
    assert set(preds.keys()) == {990001}
    assert len(preds[990001]) == 2

    scored, metrics, summary = s.score(preds, _registry_df())

    # The breast registry diagnosis matches the breast prediction, NOT the
    # second (prostate) prediction; that second tumor is an unscored extra.
    assert summary["n_matched"] == 1
    assert summary["n_extra_predicted_tumors"] == 1
    row = scored.iloc[0]
    assert row["matched"]
    assert row["ok_site_exact"] and row["ok_hist4"] and row["ok_beh"]
    assert row["ok_date_exact"]

    f = summary["fields"]
    # Site/histology/behavior correct.
    assert f["primary_site_exact"]["accuracy"] == 1.0
    assert f["histology_4digit"]["accuracy"] == 1.0
    # AJCC group: ground truth informative (1A) but model abstained (99) ->
    # strict wrong, but coverage 0 so accuracy_attempted is undefined (None).
    assert f["ajcc_stage_group_main"]["n_gt"] == 1
    assert f["ajcc_stage_group_main"]["accuracy"] == 0.0
    assert f["ajcc_stage_group_main"]["coverage"] == 0.0
    assert f["ajcc_stage_group_main"]["accuracy_attempted"] is None
    # Cross-system crosswalk: AJCC 1A -> localized == model summary stage 1 -> localized.
    assert f["stage_extent_crosswalk"]["accuracy"] == 1.0


def test_clin_t_attempted_vs_strict() -> None:
    """Clinical T matches after normalization when the model actually answers."""
    df = _registry_df()
    preds = {990001: [{
        "site": "C50.4", "hist": "8500", "behavior": "3", "lat": "1", "grade": "9",
        "summary": "1", "clin_t": "cT1", "clin_n": "cN0", "clin_m": "cM0",
        "ajcc": "99", "er": "", "pr": "", "her2": "",
        "dxdate": pd.Timestamp("2020-01-15"),
    }]}
    _, _, summary = s.score(preds, df)
    f = summary["fields"]
    assert f["ajcc_clin_T_main"]["coverage"] == 1.0
    assert f["ajcc_clin_T_main"]["accuracy"] == 1.0          # CT1 vs cT1 -> "1"=="1"
    assert f["ajcc_clin_N_exact"]["accuracy_attempted"] == 1.0
