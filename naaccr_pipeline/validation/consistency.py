"""Cross-pass consistency checks.

Validates that related fields extracted across different passes are
internally consistent.  These checks catch errors that arise when
different LLM calls produce contradictory information for the same
patient/tumor.
"""

from __future__ import annotations

import logging
from typing import Optional

from naaccr_pipeline.validation.cross_field import (
    EditViolation,
    _is_unknown_or_blank,
    _parse_int,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Item number constants
# ---------------------------------------------------------------------------

_ITEM_SURG_PRIM_SITE = 1290
_ITEM_SURG_PRIM_SITE_2023 = 1291
_ITEM_TUMOR_SIZE_PATH = 754
_ITEM_RADIATION = 1360
_ITEM_RAD_MODALITY = 1570
_ITEM_CHEMO = 1390
_ITEM_CHEMO_TEXT = 2640
_ITEM_SUMMARY_STAGE = 764
_ITEM_METS_DX_BONE = 1112
_ITEM_METS_DX_BRAIN = 1113
_ITEM_METS_DX_DISTANT_LN = 1114
_ITEM_METS_DX_LIVER = 1115
_ITEM_METS_DX_LUNG = 1116
_ITEM_METS_DX_OTHER = 1117
_ITEM_PT = 880
_ITEM_PN = 890
_ITEM_PM = 900
_ITEM_PSTAGE_GROUP = 910
_ITEM_SENTINEL_LN_EXAMINED = 834
_ITEM_SCOPE_LN_SURGERY = 1292
_ITEM_GRADE_PATH = 441


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_value(record: dict, item_number: int) -> str:
    """Safely get resolved_code from record. Returns empty string if missing."""
    result = record.get(item_number)
    if result is None:
        return ""
    return (
        getattr(result, "resolved_code", "")
        or getattr(result, "extracted_value", "")
        or ""
    )


def _surgery_performed(record: dict) -> bool:
    """Determine whether surgery was performed based on RX Summ--Surg Prim Site."""
    raw = _get_value(record, _ITEM_SURG_PRIM_SITE)
    if not raw:
        raw = _get_value(record, _ITEM_SURG_PRIM_SITE_2023)
    if not raw:
        return False
    surg = raw.strip()
    return (
        surg not in ("00", "0000", "0", "98", "99", "")
        and not _is_unknown_or_blank(surg)
    )


# ---------------------------------------------------------------------------
# InternalConsistencyChecker
# ---------------------------------------------------------------------------

class InternalConsistencyChecker:
    """Checks consistency between extraction pass results.

    These edits detect logical contradictions that arise when different
    extraction passes (demographic, staging, treatment, follow-up) produce
    values that disagree with each other.
    """

    def check(self, record: dict) -> list[EditViolation]:
        """Run all cross-pass consistency checks.

        Parameters
        ----------
        record : dict
            Mapping of item_number (int) -> ExtractionResult.

        Returns
        -------
        list[EditViolation]
        """
        violations: list[EditViolation] = []
        violations.extend(self._check_surgery_tumor_size(record))
        violations.extend(self._check_radiation_modality(record))
        violations.extend(self._check_chemo_text(record))
        violations.extend(self._check_distant_stage_mets(record))
        violations.extend(self._check_path_staging_surgery(record))
        violations.extend(self._check_sentinel_ln_scope(record))
        violations.extend(self._check_grade_path_surgery(record))
        return violations

    # ------------------------------------------------------------------
    # 1. Surgery performed => tumor size pathologic should be populated
    # ------------------------------------------------------------------

    def _check_surgery_tumor_size(self, record: dict) -> list[EditViolation]:
        """If surgery performed, pathologic tumor size should be populated."""
        if not _surgery_performed(record):
            return []

        raw_size = _get_value(record, _ITEM_TUMOR_SIZE_PATH)
        if not raw_size or _is_unknown_or_blank(raw_size):
            surg = _get_value(record, _ITEM_SURG_PRIM_SITE) or _get_value(record, _ITEM_SURG_PRIM_SITE_2023)
            return [EditViolation(
                edit_name="Surgery_TumorSizePath_Missing",
                severity="warning",
                item_numbers=[_ITEM_SURG_PRIM_SITE, _ITEM_TUMOR_SIZE_PATH],
                message=(
                    f"Surgery of primary site={surg.strip()} indicates surgery "
                    f"was performed, but Tumor Size Pathologic (item 754) is "
                    f"missing or unknown. Pathologic size is typically measured "
                    f"from the surgical specimen."
                ),
            )]

        return []

    # ------------------------------------------------------------------
    # 2. Radiation given => radiation modality should be populated
    # ------------------------------------------------------------------

    def _check_radiation_modality(self, record: dict) -> list[EditViolation]:
        """If radiation given, radiation modality should be populated."""
        raw_rad = _get_value(record, _ITEM_RADIATION)
        if not raw_rad:
            return []

        rad = raw_rad.strip()
        # 0 = none; 1-6 = radiation given; 7 = refused; 8/9 = unknown
        radiation_given = rad not in ("0", "7", "8", "9", "") and not _is_unknown_or_blank(rad)

        if not radiation_given:
            return []

        raw_mod = _get_value(record, _ITEM_RAD_MODALITY)
        if not raw_mod or _is_unknown_or_blank(raw_mod):
            return [EditViolation(
                edit_name="Radiation_Modality_Missing",
                severity="warning",
                item_numbers=[_ITEM_RADIATION, _ITEM_RAD_MODALITY],
                message=(
                    f"RX Summ--Radiation={rad} indicates radiation was "
                    f"given, but Radiation Treatment Modality (item 1570) is "
                    f"missing or unknown."
                ),
            )]

        return []

    # ------------------------------------------------------------------
    # 3. Chemo given => chemo text ideally populated
    # ------------------------------------------------------------------

    def _check_chemo_text(self, record: dict) -> list[EditViolation]:
        """If chemotherapy given, chemo agent text ideally populated."""
        raw_chemo = _get_value(record, _ITEM_CHEMO)
        if not raw_chemo:
            return []

        chemo = raw_chemo.strip()
        # 00 = none; 01-03 = chemo given; 82/85/86/87 = chemo given;
        # 88 = recommended unknown if given; 99 = unknown
        chemo_given = (
            chemo not in ("00", "0", "82", "85", "86", "87", "88", "99", "")
            and not _is_unknown_or_blank(chemo)
        )

        if not chemo_given:
            return []

        raw_text = _get_value(record, _ITEM_CHEMO_TEXT)
        if not raw_text or _is_unknown_or_blank(raw_text):
            return [EditViolation(
                edit_name="Chemo_Text_Missing",
                severity="warning",
                item_numbers=[_ITEM_CHEMO, _ITEM_CHEMO_TEXT],
                message=(
                    f"RX Summ--Chemo={chemo} indicates chemotherapy was "
                    f"given, but Chemotherapy text (item 2640) is not "
                    f"populated. Ideally, the agent names should be recorded."
                ),
            )]

        return []

    # ------------------------------------------------------------------
    # 4. Summary Stage = 7 (distant) => at least one mets site should be 1
    # ------------------------------------------------------------------

    def _check_distant_stage_mets(self, record: dict) -> list[EditViolation]:
        """If Summary Stage = 7 (distant), at least one mets site should be positive."""
        raw_ss = _get_value(record, _ITEM_SUMMARY_STAGE)
        if not raw_ss:
            return []

        ss = raw_ss.strip()
        if ss != "7":
            return []

        mets_items = [
            _ITEM_METS_DX_BONE,
            _ITEM_METS_DX_BRAIN,
            _ITEM_METS_DX_DISTANT_LN,
            _ITEM_METS_DX_LIVER,
            _ITEM_METS_DX_LUNG,
            _ITEM_METS_DX_OTHER,
        ]

        # Check if any mets site has value "1" (yes, mets present)
        any_mets = False
        any_mets_populated = False
        for item_num in mets_items:
            val = _get_value(record, item_num).strip()
            if val and not _is_unknown_or_blank(val):
                any_mets_populated = True
                if val == "1":
                    any_mets = True
                    break

        # Only flag if mets fields are populated but none are positive,
        # or if no mets fields are populated at all
        if not any_mets:
            return [EditViolation(
                edit_name="DistantStage_No_Mets_Site",
                severity="warning",
                item_numbers=[_ITEM_SUMMARY_STAGE] + mets_items,
                message=(
                    "Summary Stage=7 (distant) but no metastatic site "
                    "(items 1112-1117) is coded as positive. At least one "
                    "metastatic site should be identified for distant-stage "
                    "disease."
                ),
            )]

        return []

    # ------------------------------------------------------------------
    # 5. Pathological staging => surgery should have been performed
    # ------------------------------------------------------------------

    def _check_path_staging_surgery(self, record: dict) -> list[EditViolation]:
        """If pathological staging present, surgery should have been performed."""
        pt = _get_value(record, _ITEM_PT).strip()
        pn = _get_value(record, _ITEM_PN).strip()
        pm = _get_value(record, _ITEM_PM).strip()
        pstage = _get_value(record, _ITEM_PSTAGE_GROUP).strip()

        has_path_staging = any(
            v and not _is_unknown_or_blank(v)
            for v in [pt, pn, pm, pstage]
        )

        if not has_path_staging:
            return []

        if not _surgery_performed(record):
            # Check if we even have a surgery field
            raw_surg = _get_value(record, _ITEM_SURG_PRIM_SITE) or _get_value(record, _ITEM_SURG_PRIM_SITE_2023)
            if raw_surg:
                return [EditViolation(
                    edit_name="PathStaging_No_Surgery",
                    severity="warning",
                    item_numbers=[_ITEM_PT, _ITEM_PN, _ITEM_PM,
                                  _ITEM_PSTAGE_GROUP, _ITEM_SURG_PRIM_SITE],
                    message=(
                        "Pathological staging values are present "
                        f"(pT={pt}, pN={pn}, pM={pm}, pStage={pstage}) "
                        "but no surgery of primary site is indicated. "
                        "Pathological staging typically requires surgical "
                        "resection."
                    ),
                )]

        return []

    # ------------------------------------------------------------------
    # 6. Sentinel LN examined > 0 => scope of LN surgery should indicate sentinel
    # ------------------------------------------------------------------

    def _check_sentinel_ln_scope(self, record: dict) -> list[EditViolation]:
        """If sentinel LN examined > 0, LN surgery scope should indicate sentinel."""
        raw_sln = _get_value(record, _ITEM_SENTINEL_LN_EXAMINED)
        if not raw_sln or _is_unknown_or_blank(raw_sln):
            return []

        sln_count = _parse_int(raw_sln)
        if sln_count is None or sln_count <= 0:
            return []

        raw_scope = _get_value(record, _ITEM_SCOPE_LN_SURGERY)
        if not raw_scope or _is_unknown_or_blank(raw_scope):
            return [EditViolation(
                edit_name="SentinelLN_Scope_Missing",
                severity="warning",
                item_numbers=[_ITEM_SENTINEL_LN_EXAMINED, _ITEM_SCOPE_LN_SURGERY],
                message=(
                    f"Sentinel Nodes Examined={sln_count} (>0) but Scope of "
                    f"Regional LN Surgery (item 1292) is missing or unknown. "
                    f"Scope should indicate sentinel node procedure."
                ),
            )]

        scope = raw_scope.strip()
        # Scope codes that indicate sentinel node: typically 2 (sentinel only)
        # or 3 (sentinel + additional). Code 0 = none, 1 = regional biopsy/sample,
        # 4-7 = various dissections.
        # We flag if scope indicates no LN surgery at all (0)
        if scope == "0":
            return [EditViolation(
                edit_name="SentinelLN_Scope_Mismatch",
                severity="warning",
                item_numbers=[_ITEM_SENTINEL_LN_EXAMINED, _ITEM_SCOPE_LN_SURGERY],
                message=(
                    f"Sentinel Nodes Examined={sln_count} (>0) but Scope of "
                    f"Regional LN Surgery=0 (no LN surgery). These fields "
                    f"are contradictory."
                ),
            )]

        return []

    # ------------------------------------------------------------------
    # 7. Grade pathological populated => surgery should have been performed
    # ------------------------------------------------------------------

    def _check_grade_path_surgery(self, record: dict) -> list[EditViolation]:
        """If pathological grade is populated, surgery should have been performed."""
        raw_grade = _get_value(record, _ITEM_GRADE_PATH)
        if not raw_grade or _is_unknown_or_blank(raw_grade):
            return []

        grade = raw_grade.strip()
        # Grade 9 = unknown/not stated, which doesn't require surgery
        if grade == "9":
            return []

        if not _surgery_performed(record):
            raw_surg = _get_value(record, _ITEM_SURG_PRIM_SITE) or _get_value(record, _ITEM_SURG_PRIM_SITE_2023)
            if raw_surg:
                return [EditViolation(
                    edit_name="GradePath_No_Surgery",
                    severity="warning",
                    item_numbers=[_ITEM_GRADE_PATH, _ITEM_SURG_PRIM_SITE],
                    message=(
                        f"Pathological grade={grade} is populated but no "
                        f"surgery of primary site is indicated. Pathological "
                        f"grade is determined from surgical specimens."
                    ),
                )]

        return []
