#!/usr/bin/env python
"""Stage 4: score extractions against CAREG ground truth.

Pipeline predictions come from naaccr_output.csv (one row per tumor, all items),
keyed by NAACCR item 2300 (Medical Record Number) which we seeded with dfci_mrn.

Approach:
  1. For each patient, greedily match predicted tumors to the sampled CAREG
     diagnoses (best 3-char site agreement, tie-broken by smallest |date diff|).
  2. For matched pairs, compare each registry field at an exact and a lenient
     level after normalization.
  3. Emit a per-diagnosis scorecard and aggregate per-field metrics.

Outputs (in ./data/scoring):
  per_diagnosis_scored.csv, field_metrics.csv, field_metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import pandas as pd

import common as C

warnings.filterwarnings("ignore")


def _mrn_int(v):
    digits = re.sub(r"\D", "", C.norm_str(v))
    return int(digits) if digits else None


def load_predictions(path):
    df = pd.read_csv(path, dtype=str)
    mrn_col = C.find_item_col(df, 2300)
    if mrn_col is None:
        raise SystemExit(
            "No 'Medical Record Number [2300]' column in naaccr_output.csv; "
            "cannot join to ground truth. Ensure stage 2 emitted the `mrn` column."
        )
    by_mrn: dict[int, list[dict]] = {}
    for _, row in df.iterrows():
        mrn = _mrn_int(row[mrn_col])
        if mrn is None:
            continue
        clin_group = C.item_val(row, df, 1004)
        path_group = C.item_val(row, df, 1014)
        ajcc = path_group if C.stage_group_main(path_group) != "unknown" else clin_group
        by_mrn.setdefault(mrn, []).append({
            "site": C.item_val(row, df, 400),
            "hist": C.item_val(row, df, 522),
            "behavior": C.item_val(row, df, 523),
            "lat": C.item_val(row, df, 410),
            "grade": C.item_val(row, df, 440),
            "summary": C.item_val(row, df, 764),
            "clin_t": C.item_val(row, df, 1001),
            "clin_n": C.item_val(row, df, 1002),
            "clin_m": C.item_val(row, df, 1003),
            "ajcc": ajcc,
            "er": C.item_val(row, df, 3827),
            "pr": C.item_val(row, df, 3915),
            "her2": C.item_val(row, df, 3855),
            "dxdate": C.parse_pred_date(C.item_val(row, df, 390)),
        })
    return by_mrn


def match_patient(truth_rows, preds):
    """Greedy one-to-one assignment: truth index -> (pred index, site_match, days)."""
    pairs = []
    for ti, (_, tr) in enumerate(truth_rows.iterrows()):
        ts, td = C.site3(tr["SITE_CD"]), tr["diagnosis_dt"]
        for pi, pr in enumerate(preds):
            site_m = ts != "" and ts == C.site3(pr["site"])
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


def score_row(tr, pr):
    """Compute normalized values + correctness flags for one matched/unmatched pair."""
    g = {  # ground-truth normalized
        "site": C.norm_site(tr["SITE_CD"]),
        "hist": C.histology4(tr["HISTOLOGY_CD"]),
        "beh": C.norm_str(tr["BEHAVIOR_CD"]),
        "lat": C.norm_str(tr["LATERALITY_CD"]),
        "grade": C.norm_str(tr["GRADE_DIFF_CODE"]),
        "ss": C.norm_str(tr["GENERAL_STAGE_CD"]),
        "ajcc": C.norm_str(tr["BEST_AJCC_STAGE_CD"]),
        "t": tr["CLIN_T_CD"], "n": tr["CLIN_N_CD"], "m": tr["CLIN_M_CD"],
        "er": tr["SSDI_ER_SUMMARY"], "pr": tr["SSDI_PR_SUMMARY"], "her2": tr["SSDI_HER2_OVERALL_SUMMARY"],
    }
    is_breast = C.site3(tr["SITE_CD"]) == "C50"
    r = {
        "gt_site": g["site"], "gt_histology": g["hist"], "gt_behavior": g["beh"],
        "gt_laterality": g["lat"], "gt_grade": g["grade"],
        "gt_summary_stage": g["ss"], "gt_ajcc_group": g["ajcc"],
        "gt_clin_t": C.norm_str(g["t"]), "gt_clin_n": C.norm_str(g["n"]), "gt_clin_m": C.norm_str(g["m"]),
        "gt_er": C.collapse_biomarker(g["er"]), "gt_pr": C.collapse_biomarker(g["pr"]),
        "gt_her2": C.collapse_biomarker(g["her2"]),
    }

    # gt-present flags (informative ground truth only)
    r["present_site"] = g["site"] != ""
    r["present_hist"] = g["hist"] != ""
    r["present_beh"] = g["beh"] != ""
    r["present_lat"] = g["lat"] not in ("", "9")
    r["present_grade"] = g["grade"] not in ("", "9")
    r["present_ss"] = g["ss"] not in ("", "9")
    r["present_ajcc"] = C.stage_group_main(g["ajcc"]) != "unknown"
    r["present_t"] = C.norm_tnm(g["t"], "T") not in ("", "X", "88", "99")
    r["present_n"] = C.norm_tnm(g["n"], "N") not in ("", "X", "88", "99")
    r["present_m"] = C.norm_tnm(g["m"], "M") not in ("", "X", "88", "99")
    r["present_er"] = is_breast and r["gt_er"] in ("positive", "negative")
    r["present_pr"] = is_breast and r["gt_pr"] in ("positive", "negative")
    r["present_her2"] = is_breast and r["gt_her2"] in ("positive", "negative")
    # Cross-system stage extent: CAREG AJCC group (dense) on the SEER axis.
    r["gt_stage_axis"] = C.collapse_ajcc_to_stage_axis(g["ajcc"])
    r["present_stage_axis"] = r["gt_stage_axis"] != "unknown"
    r["present_date_gt"] = pd.notna(tr["diagnosis_dt"])

    # Names of every pred-present flag, so both branches stay in sync.
    pred_present_keys = (
        "pp_site", "pp_hist", "pp_beh", "pp_lat", "pp_grade", "pp_ss", "pp_ajcc",
        "pp_t", "pp_n", "pp_m", "pp_er", "pp_pr", "pp_her2", "pp_stage_axis",
        "present_date_pred",
    )

    if pr is None:
        for k in ("pred_site", "pred_histology", "pred_behavior", "pred_laterality",
                  "pred_grade", "pred_summary_stage", "pred_ajcc_group",
                  "pred_clin_t", "pred_clin_n", "pred_clin_m", "pred_er", "pred_pr",
                  "pred_her2", "pred_stage_axis"):
            r[k] = ""
        for k in ("ok_site_exact", "ok_site3", "ok_hist4", "ok_hist3", "ok_beh", "ok_lat",
                  "ok_grade", "ok_ss_exact", "ok_ss_collapse", "ok_ajcc_exact", "ok_ajcc_main",
                  "ok_t_exact", "ok_t_main", "ok_n_exact", "ok_n_main", "ok_m_exact",
                  "ok_date_exact", "ok_date30", "ok_date90", "ok_date365", "ok_year",
                  "ok_er", "ok_pr", "ok_her2", "ok_stage_axis"):
            r[k] = False
        for k in pred_present_keys:
            r[k] = False
        r["present_date"] = False
        return r

    p = {
        "site": C.norm_site(pr["site"]), "hist": C.histology4(pr["hist"]),
        "beh": C.norm_str(pr["behavior"]), "lat": C.norm_str(pr["lat"]),
        "grade": C.norm_str(pr["grade"]), "ss": C.norm_str(pr["summary"]),
        "ajcc": C.norm_str(pr["ajcc"]),
        "t": pr["clin_t"], "n": pr["clin_n"], "m": pr["clin_m"],
        "er": C.collapse_biomarker(pr["er"]), "pr": C.collapse_biomarker(pr["pr"]),
        "her2": C.collapse_biomarker(pr["her2"]),
    }
    pred_stage_axis = C.collapse_summary_stage(p["ss"])  # model SEER summary stage
    r.update({
        "pred_site": p["site"], "pred_histology": p["hist"], "pred_behavior": p["beh"],
        "pred_laterality": p["lat"], "pred_grade": p["grade"],
        "pred_summary_stage": p["ss"], "pred_ajcc_group": p["ajcc"],
        "pred_clin_t": C.norm_str(p["t"]), "pred_clin_n": C.norm_str(p["n"]), "pred_clin_m": C.norm_str(p["m"]),
        "pred_er": p["er"], "pred_pr": p["pr"], "pred_her2": p["her2"],
        "pred_stage_axis": pred_stage_axis,
    })

    # pred-present flags: did the model emit an INFORMATIVE value (not blank /
    # unknown / "88"/"99"/"X" abstentions)? Lets us separate coverage from error.
    _UNK = ("", "9", "88", "99", "unknown", "nan")
    r["pp_site"] = p["site"] != ""
    r["pp_hist"] = p["hist"] != ""
    r["pp_beh"] = p["beh"] not in ("", "9", "nan")
    r["pp_lat"] = p["lat"].lower() not in ("", "9", "unknown", "nan")
    r["pp_grade"] = p["grade"].lower() not in _UNK
    r["pp_ss"] = p["ss"].lower() not in _UNK
    r["pp_ajcc"] = C.stage_group_main(p["ajcc"]) != "unknown"
    r["pp_t"] = C.norm_tnm(p["t"], "T").lower() not in ("", "x", "88", "99", "unknown", "nan")
    r["pp_n"] = C.norm_tnm(p["n"], "N").lower() not in ("", "x", "88", "99", "unknown", "nan")
    r["pp_m"] = C.norm_tnm(p["m"], "M").lower() not in ("", "x", "88", "99", "unknown", "nan")
    r["pp_er"] = p["er"] in ("positive", "negative")
    r["pp_pr"] = p["pr"] in ("positive", "negative")
    r["pp_her2"] = p["her2"] in ("positive", "negative")
    r["pp_stage_axis"] = pred_stage_axis != "unknown"

    r["ok_stage_axis"] = r["gt_stage_axis"] != "unknown" and r["gt_stage_axis"] == pred_stage_axis

    r["ok_site_exact"] = g["site"] != "" and g["site"] == p["site"]
    r["ok_site3"] = C.site3(g["site"]) != "" and C.site3(g["site"]) == C.site3(p["site"])
    r["ok_hist4"] = g["hist"] != "" and g["hist"] == p["hist"]
    r["ok_hist3"] = C.histology3(g["hist"]) != "" and C.histology3(g["hist"]) == C.histology3(p["hist"])
    r["ok_beh"] = g["beh"] != "" and g["beh"] == p["beh"]
    r["ok_lat"] = g["lat"] != "" and g["lat"] == p["lat"]
    r["ok_grade"] = g["grade"] != "" and g["grade"] == p["grade"]
    r["ok_ss_exact"] = re.sub(r"\D", "", g["ss"])[:1] == re.sub(r"\D", "", p["ss"])[:1] and g["ss"] != ""
    r["ok_ss_collapse"] = C.collapse_summary_stage(g["ss"]) == C.collapse_summary_stage(p["ss"]) \
        and C.collapse_summary_stage(g["ss"]) != "unknown"
    r["ok_ajcc_exact"] = C.norm_stage_group(g["ajcc"]) == C.norm_stage_group(p["ajcc"]) and g["ajcc"] != ""
    r["ok_ajcc_main"] = C.stage_group_main(g["ajcc"]) == C.stage_group_main(p["ajcc"]) \
        and C.stage_group_main(g["ajcc"]) != "unknown"
    for axis, key in (("T", "t"), ("N", "n"), ("M", "m")):
        gn, pn = C.norm_tnm(g[key], axis), C.norm_tnm(p[key], axis)
        r[f"ok_{key}_exact"] = gn != "" and gn == pn
        if axis != "M":
            r[f"ok_{key}_main"] = C.tnm_main(gn) != "" and C.tnm_main(gn) == C.tnm_main(pn)

    # date
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default=str(C.DATA / "sampled_diagnoses.csv"))
    ap.add_argument("--pred", default=str(C.DATA / "extraction_output" / "naaccr_output.csv"))
    ap.add_argument("--out", default=str(C.DATA / "scoring"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(args.truth, dtype=str)
    truth["dfci_mrn"] = C.to_int_mrn(truth["dfci_mrn"])
    truth["diagnosis_dt"] = pd.to_datetime(truth["diagnosis_dt"], errors="coerce")

    preds = load_predictions(args.pred)
    print(f"Loaded {len(truth)} truth diagnoses; predictions for {len(preds)} patients "
          f"({sum(len(v) for v in preds.values())} tumors).")

    rows, n_extra_total = [], 0
    for mrn, tg in truth.groupby("dfci_mrn"):
        plist = preds.get(int(mrn), [])
        assign, n_extra = (({}, 0) if not plist else match_patient(tg, plist))
        n_extra_total += n_extra
        for ti, (_, tr) in enumerate(tg.iterrows()):
            pi, site_m, days = assign.get(ti, (None, False, None))
            pr = plist[pi] if pi is not None else None
            matched = pr is not None and (site_m or (days is not None and days <= 365))
            base = {
                "dfci_mrn": int(mrn),
                "gt_diagnosis_dt": tr["diagnosis_dt"].date() if pd.notna(tr["diagnosis_dt"]) else "",
                "gt_site_descr": tr["SITE_DESCR"], "gt_histology_descr": tr["HISTOLOGY_DESCR"],
                "n_pred_tumors_for_patient": len(plist),
                "matched": matched,
                "match_site": site_m, "match_date_delta_days": days,
            }
            base.update(score_row(tr, pr if matched else None))
            rows.append(base)

    scored = pd.DataFrame(rows)
    per_path = out_dir / "per_diagnosis_scored.csv"
    scored.to_csv(per_path, index=False)

    # Aggregate metrics over matched diagnoses only.
    #   n_gt        : matched diagnoses where ground truth is informative
    #   accuracy    : correct / n_gt  (model abstention counts as wrong) -- strict
    #   coverage    : fraction of n_gt where the model emitted an informative value
    #   n_attempted : n_gt rows where BOTH gt and pred are informative
    #   acc_attempt : correct / n_attempted  (accuracy when the model actually answered)
    matched = scored[scored["matched"]]
    recall = float(scored["matched"].mean()) if len(scored) else 0.0

    def _rate(num, den):
        return round(float(num) / den, 4) if den else None

    metric_rows = []
    for label, gt_col, pp_col, ok_col in METRICS:
        d = matched[matched[gt_col]]
        n_gt = int(len(d))
        att = d[d[pp_col]]
        n_att = int(len(att))
        metric_rows.append({
            "metric": label,
            "n_gt": n_gt,
            "accuracy": _rate(d[ok_col].sum(), n_gt),
            "coverage": _rate(d[pp_col].sum(), n_gt),
            "n_attempted": n_att,
            "accuracy_attempted": _rate(att[ok_col].sum(), n_att),
        })
    metrics = pd.DataFrame(metric_rows)

    summary = {
        "n_truth_diagnoses": int(len(scored)),
        "n_patients": int(scored["dfci_mrn"].nunique()),
        "diagnosis_match_recall": round(recall, 4),
        "n_matched": int(scored["matched"].sum()),
        "n_extra_predicted_tumors": int(n_extra_total),
        "fields": {r["metric"]: {k: r[k] for k in
                   ("n_gt", "accuracy", "coverage", "n_attempted", "accuracy_attempted")}
                   for r in metric_rows},
    }

    metrics.to_csv(out_dir / "field_metrics.csv", index=False)
    (out_dir / "field_metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {per_path}")
    print(f"Wrote {out_dir / 'field_metrics.csv'} and field_metrics.json\n")
    print(f"Diagnosis match recall: {recall:.1%} "
          f"({summary['n_matched']}/{summary['n_truth_diagnoses']}); "
          f"extra predicted tumors: {n_extra_total}\n")
    print("Per-field metrics over matched diagnoses with informative ground truth.")
    print("  accuracy = strict (abstention counts as wrong); coverage = model answered; "
          "accuracy_attempted = accuracy when answered.")
    with pd.option_context("display.max_rows", None):
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
