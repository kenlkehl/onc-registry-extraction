"""NAACCR cross-field validation edits.

Implements the most critical interfield edit rules from the NAACCR standard
data-editing set.  The full NAACCR edit catalogue has ~2,732 edits; this
module covers the subset most likely to catch LLM extraction errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EditViolation:
    """A single edit-rule violation."""

    edit_name: str                    # e.g., "IF39_Site_Sex"
    severity: str                     # "error" or "warning"
    item_numbers: list[int]           # items involved
    message: str                      # human-readable description
    auto_fixable: bool = False
    suggested_fix: Optional[dict] = None  # {item_number: suggested_value}


# ---------------------------------------------------------------------------
# Site/laterality constants
# ---------------------------------------------------------------------------

# Sites that require male sex (sex code 1)
_MALE_ONLY_SITES: set[str] = {
    "C619",   # prostate
    "C620", "C621", "C629",  # testis
    "C630", "C631", "C632", "C637", "C638", "C639",  # other male genital
}

# Site prefix patterns requiring male sex
_MALE_ONLY_PREFIXES: list[str] = ["C62", "C63"]

# Sites that require female sex (sex code 2)
_FEMALE_ONLY_SITES: set[str] = {
    "C530", "C531", "C538", "C539",  # cervix uteri
    "C540", "C541", "C542", "C543", "C548", "C549",  # corpus uteri
    "C559",  # uterus NOS
    "C569",  # ovary
    "C570", "C571", "C572", "C573", "C574", "C577", "C578", "C579",  # other female genital
    "C580",  # placenta
}

_FEMALE_ONLY_PREFIXES: list[str] = ["C53", "C54", "C55", "C56", "C57", "C58"]

# Paired organ site prefixes (laterality must not be 0/not-paired)
_PAIRED_ORGAN_PREFIXES: list[str] = [
    "C07",   # parotid gland
    "C08",   # submandibular / sublingual glands
    "C09",   # tonsil
    "C30",   # nasal cavity (C30.0 specifically, but we check prefix)
    "C34",   # lung/bronchus
    "C38",   # pleura (C38.4)
    "C40",   # bones of extremities
    "C41",   # bones (C41.3, C41.4 -- pelvis/limb)
    "C44",   # skin
    "C47",   # peripheral nerves
    "C49",   # connective/soft tissue
    "C50",   # breast
    "C56",   # ovary
    "C57",   # fallopian tube (C57.0)
    "C62",   # testis
    "C63",   # epididymis (C63.0)
    "C64",   # kidney
    "C65",   # renal pelvis
    "C66",   # ureter
    "C69",   # eye/orbit
    "C74",   # adrenal gland
    "C75",   # carotid/aortic body (C75.4, C75.5)
]

# More precise paired-organ check for sites that are only partially paired
_PAIRED_ORGAN_EXACT: set[str] = {
    "C300",            # nasal cavity
    "C384",            # pleura
    "C413", "C414",    # pelvis/limb bones
    "C471", "C472",    # peripheral nerves of upper/lower limb
    "C491", "C492",    # soft tissue of upper/lower limb
    "C570",            # fallopian tube
    "C630",            # epididymis
    "C754", "C755",    # carotid body, aortic body
}

_FULLY_PAIRED_PREFIXES: set[str] = {
    "C07", "C08", "C09", "C34", "C40", "C44", "C50",
    "C56", "C62", "C64", "C65", "C66", "C69", "C74",
}

# NAACCR item number constants used throughout
_ITEM_PRIMARY_SITE = 400
_ITEM_HISTOLOGY = 522
_ITEM_BEHAVIOR = 523
_ITEM_SEX = 220
_ITEM_AGE = 230
_ITEM_LATERALITY = 410
_ITEM_DIAG_DATE = 390
_ITEM_SURGERY_DATE = 1200
_ITEM_RADIATION_DATE = 1210
_ITEM_CHEMO_DATE = 1220
_ITEM_HORMONE_DATE = 1230
_ITEM_BRM_DATE = 1240
_ITEM_SURG_PRIM_SITE = 1290
_ITEM_SURG_PRIM_SITE_2023 = 1291
_ITEM_REASON_NO_SURG = 1340
_ITEM_SURGICAL_MARGINS = 1320
_ITEM_RADIATION = 1360
_ITEM_REASON_NO_RAD = 1430
_ITEM_DIAG_CONFIRM = 490
_ITEM_DATE_LAST_CONTACT = 1750
_ITEM_VITAL_STATUS = 1760
_ITEM_CAUSE_OF_DEATH = 1910
_ITEM_CT = 940
_ITEM_CN = 950
_ITEM_CM = 960
_ITEM_CSTAGE_GROUP = 970
_ITEM_PT = 880
_ITEM_PN = 890
_ITEM_PM = 900
_ITEM_PSTAGE_GROUP = 910
_ITEM_GRADE_PATH = 441
_ITEM_SUMMARY_STAGE = 764


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_site(raw: str) -> str:
    """Normalize a primary-site code to 'CXXX' (no dot)."""
    s = raw.strip().upper().replace(".", "")
    if s and not s.startswith("C"):
        s = "C" + s
    return s


def _parse_histology(raw: str) -> Optional[int]:
    """Parse a histology code string to int, or None."""
    s = raw.strip().lstrip("0")
    if not s:
        return None
    m = re.match(r"(\d{4})", raw.strip())
    if m:
        return int(m.group(1))
    return None


def _parse_int(raw: str) -> Optional[int]:
    """Parse an integer value, returning None for blank/unknown."""
    s = raw.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _is_unknown_or_blank(value: str) -> bool:
    """Check if value is blank or a standard NAACCR unknown sentinel."""
    s = value.strip()
    if not s:
        return True
    # Common unknown sentinels
    if s in ("9", "99", "999", "9999", "99999999", "88", "888", "8888",
             "88888888", "00000000"):
        return True
    return False


def _site_matches_prefix(site: str, prefixes: list[str] | set[str]) -> bool:
    """Check if a normalized site code starts with any of the given prefixes."""
    for prefix in prefixes:
        if site.startswith(prefix):
            return True
    return False


def _is_paired_site(site: str) -> bool:
    """Determine whether *site* (normalized, no dot) is a paired organ."""
    if site in _PAIRED_ORGAN_EXACT:
        return True
    if _site_matches_prefix(site, _FULLY_PAIRED_PREFIXES):
        return True
    return False


def _compare_dates(date1: str, date2: str) -> Optional[int]:
    """Compare two YYYYMMDD date strings.

    Returns negative if date1 < date2, zero if equal, positive if date1 > date2.
    Returns None if either date is blank/unparseable.
    """
    d1 = date1.strip()
    d2 = date2.strip()
    if not d1 or not d2 or _is_unknown_or_blank(d1) or _is_unknown_or_blank(d2):
        return None
    # Replace unknown day/month parts with 01 for comparison
    d1 = d1.ljust(8, "0")
    d2 = d2.ljust(8, "0")
    try:
        n1 = int(d1[:8])
        n2 = int(d2[:8])
        return n1 - n2
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CrossFieldValidator
# ---------------------------------------------------------------------------

class CrossFieldValidator:
    """Implements critical NAACCR standard data edits.

    The full NAACCR edit set has 2,732 edits.  This implements the most
    critical interfield edits that are most likely to catch LLM extraction
    errors.
    """

    def validate(self, record: dict) -> list[EditViolation]:
        """Run all edit checks on a record.

        Parameters
        ----------
        record : dict
            Mapping of item_number (int) -> ExtractionResult.

        Returns
        -------
        list[EditViolation]
        """
        violations: list[EditViolation] = []
        violations.extend(self._check_site_sex(record))
        violations.extend(self._check_site_histology(record))
        violations.extend(self._check_site_laterality(record))
        violations.extend(self._check_behavior_histology(record))
        violations.extend(self._check_age_site(record))
        violations.extend(self._check_tnm_consistency(record))
        violations.extend(self._check_treatment_dates(record))
        violations.extend(self._check_diagnostic_confirmation(record))
        violations.extend(self._check_surgery_consistency(record))
        violations.extend(self._check_radiation_consistency(record))
        violations.extend(self._check_vital_status_dates(record))
        return violations

    # ------------------------------------------------------------------
    # Value access helper
    # ------------------------------------------------------------------

    def _get_value(self, record: dict, item_number: int) -> str:
        """Safely get resolved_code from record.  Returns empty string if missing."""
        result = record.get(item_number)
        if result is None:
            return ""
        return (
            getattr(result, "resolved_code", "")
            or getattr(result, "extracted_value", "")
            or ""
        )

    # ------------------------------------------------------------------
    # IF39 / IF177:  Site / Sex consistency
    # ------------------------------------------------------------------

    def _check_site_sex(self, record: dict) -> list[EditViolation]:
        """Site/Sex consistency.

        Male-only sites (prostate, testis, male genital) require Sex=1.
        Female-only sites (cervix, corpus, uterus, ovary, female genital)
        require Sex=2.
        """
        violations: list[EditViolation] = []

        raw_site = self._get_value(record, _ITEM_PRIMARY_SITE)
        raw_sex = self._get_value(record, _ITEM_SEX)

        if not raw_site or not raw_sex:
            return violations

        site = _normalize_site(raw_site)
        sex = raw_sex.strip()

        if not site or sex not in ("1", "2", "3", "4", "5", "6"):
            return violations

        # Male-only check
        is_male_site = (
            site in _MALE_ONLY_SITES
            or _site_matches_prefix(site, _MALE_ONLY_PREFIXES)
        )
        if is_male_site and sex != "1":
            violations.append(EditViolation(
                edit_name="IF39_Site_Sex",
                severity="error",
                item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_SEX],
                message=(
                    f"Site {raw_site} ({site}) is a male-only site but "
                    f"Sex={sex}. Expected Sex=1 (male)."
                ),
                auto_fixable=False,
            ))

        # Female-only check
        is_female_site = (
            site in _FEMALE_ONLY_SITES
            or _site_matches_prefix(site, _FEMALE_ONLY_PREFIXES)
        )
        if is_female_site and sex != "2":
            violations.append(EditViolation(
                edit_name="IF39_Site_Sex",
                severity="error",
                item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_SEX],
                message=(
                    f"Site {raw_site} ({site}) is a female-only site but "
                    f"Sex={sex}. Expected Sex=2 (female)."
                ),
                auto_fixable=False,
            ))

        return violations

    # ------------------------------------------------------------------
    # IF25:  Site / Histology compatibility
    # ------------------------------------------------------------------

    def _check_site_histology(self, record: dict) -> list[EditViolation]:
        """Site/Histology compatibility -- selected critical checks."""
        violations: list[EditViolation] = []

        raw_site = self._get_value(record, _ITEM_PRIMARY_SITE)
        raw_hist = self._get_value(record, _ITEM_HISTOLOGY)

        if not raw_site or not raw_hist:
            return violations

        site = _normalize_site(raw_site)
        hist = _parse_histology(raw_hist)

        if not site or hist is None:
            return violations

        # 8170-8175: hepatocellular carcinoma -- only liver (C220)
        if 8170 <= hist <= 8175:
            if not site.startswith("C22"):
                violations.append(EditViolation(
                    edit_name="IF25_Site_Histology",
                    severity="error",
                    item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_HISTOLOGY],
                    message=(
                        f"Hepatocellular carcinoma (histology {hist}) is "
                        f"only valid for liver (C22.x) but site is {raw_site}."
                    ),
                ))

        # 8312: renal cell carcinoma -- only kidney (C64.x)
        if hist == 8312:
            if not site.startswith("C64"):
                violations.append(EditViolation(
                    edit_name="IF25_Site_Histology",
                    severity="warning",
                    item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_HISTOLOGY],
                    message=(
                        f"Renal cell carcinoma (histology 8312) is typically "
                        f"only for kidney (C64.x) but site is {raw_site}."
                    ),
                ))

        # 8720-8790: melanoma -- should NOT be internal organs
        # Internal organ sites generally start with C15-C26 (digestive),
        # C30-C39 (respiratory/thorax), C48 (retroperitoneum), etc.
        if 8720 <= hist <= 8790:
            internal_prefixes = [
                "C15", "C16", "C17", "C18", "C19", "C20", "C21",
                "C22", "C23", "C24", "C25", "C26",
                "C30", "C31", "C32", "C33", "C34",
                "C37", "C38",
                "C48",
                "C61",  # prostate
                "C64", "C65", "C66", "C67", "C68",  # urinary
            ]
            if _site_matches_prefix(site, internal_prefixes):
                violations.append(EditViolation(
                    edit_name="IF25_Site_Histology",
                    severity="error",
                    item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_HISTOLOGY],
                    message=(
                        f"Melanoma (histology {hist}) is not expected for "
                        f"internal organ site {raw_site}."
                    ),
                ))

        # 9050-9055: mesothelioma -- only pleura (C38.4) / peritoneum (C48.x)
        if 9050 <= hist <= 9055:
            valid_meso = site.startswith("C384") or site.startswith("C48")
            if not valid_meso:
                violations.append(EditViolation(
                    edit_name="IF25_Site_Histology",
                    severity="error",
                    item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_HISTOLOGY],
                    message=(
                        f"Mesothelioma (histology {hist}) is typically only "
                        f"for pleura (C38.4) or peritoneum (C48.x) but site "
                        f"is {raw_site}."
                    ),
                ))

        # 9800-9989: leukemias -- should be C42.x (hematopoietic)
        if 9800 <= hist <= 9989:
            if not site.startswith("C42"):
                violations.append(EditViolation(
                    edit_name="IF25_Site_Histology",
                    severity="warning",
                    item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_HISTOLOGY],
                    message=(
                        f"Leukemia (histology {hist}) should typically have "
                        f"hematopoietic site (C42.x) but site is {raw_site}."
                    ),
                ))

        return violations

    # ------------------------------------------------------------------
    # Site / Laterality
    # ------------------------------------------------------------------

    def _check_site_laterality(self, record: dict) -> list[EditViolation]:
        """Laterality is required for paired organs and should not be 0."""
        violations: list[EditViolation] = []

        raw_site = self._get_value(record, _ITEM_PRIMARY_SITE)
        raw_lat = self._get_value(record, _ITEM_LATERALITY)

        if not raw_site:
            return violations

        site = _normalize_site(raw_site)
        if not _is_paired_site(site):
            return violations

        # For paired sites, laterality should be present and not 0
        lat = raw_lat.strip()

        if not lat or lat == "0":
            violations.append(EditViolation(
                edit_name="Laterality_PairedSite",
                severity="error",
                item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_LATERALITY],
                message=(
                    f"Site {raw_site} ({site}) is a paired organ but "
                    f"Laterality is '{lat or 'blank'}'. Expected 1 (right), "
                    f"2 (left), 3 (bilateral), or 4 (only one side)."
                ),
                auto_fixable=False,
            ))
        elif lat == "9" and not _is_unknown_or_blank(lat):
            # 9 = unknown laterality -- acceptable but flag as warning
            violations.append(EditViolation(
                edit_name="Laterality_PairedSite",
                severity="warning",
                item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_LATERALITY],
                message=(
                    f"Site {raw_site} ({site}) is a paired organ but "
                    f"Laterality=9 (unknown). Review for laterality specificity."
                ),
                auto_fixable=False,
            ))

        return violations

    # ------------------------------------------------------------------
    # Behavior / Histology
    # ------------------------------------------------------------------

    def _check_behavior_histology(self, record: dict) -> list[EditViolation]:
        """Behavior/Histology consistency checks."""
        violations: list[EditViolation] = []

        raw_hist = self._get_value(record, _ITEM_HISTOLOGY)
        raw_behav = self._get_value(record, _ITEM_BEHAVIOR)

        if not raw_hist or not raw_behav:
            return violations

        hist = _parse_histology(raw_hist)
        behav = raw_behav.strip()

        if hist is None or behav not in ("0", "1", "2", "3"):
            return violations

        # In situ (behavior=2) not valid for leukemias (9800+)
        if behav == "2" and hist >= 9800:
            violations.append(EditViolation(
                edit_name="Behavior_Histology",
                severity="error",
                item_numbers=[_ITEM_HISTOLOGY, _ITEM_BEHAVIOR],
                message=(
                    f"In situ (behavior=2) is not valid for leukemia/lymphoma "
                    f"histology {hist}. Leukemias are systemic diseases."
                ),
            ))

        # In situ (behavior=2) not valid for lymphomas (9590-9729)
        if behav == "2" and 9590 <= hist <= 9729:
            violations.append(EditViolation(
                edit_name="Behavior_Histology",
                severity="error",
                item_numbers=[_ITEM_HISTOLOGY, _ITEM_BEHAVIOR],
                message=(
                    f"In situ (behavior=2) is not valid for lymphoma "
                    f"histology {hist}."
                ),
            ))

        # Benign (behavior=0) unusual for carcinomas (8010-8576)
        if behav == "0" and 8010 <= hist <= 8576:
            violations.append(EditViolation(
                edit_name="Behavior_Histology",
                severity="warning",
                item_numbers=[_ITEM_HISTOLOGY, _ITEM_BEHAVIOR],
                message=(
                    f"Benign behavior (behavior=0) is unusual for carcinoma "
                    f"histology {hist}. Verify behavior code."
                ),
            ))

        return violations

    # ------------------------------------------------------------------
    # Age / Site plausibility
    # ------------------------------------------------------------------

    def _check_age_site(self, record: dict) -> list[EditViolation]:
        """Age/Site plausibility warnings."""
        violations: list[EditViolation] = []

        raw_site = self._get_value(record, _ITEM_PRIMARY_SITE)
        raw_age = self._get_value(record, _ITEM_AGE)
        raw_hist = self._get_value(record, _ITEM_HISTOLOGY)

        if not raw_site or not raw_age:
            return violations

        site = _normalize_site(raw_site)
        age = _parse_int(raw_age)
        hist = _parse_histology(raw_hist) if raw_hist else None

        if age is None or age == 999:
            return violations

        # Prostate cancer (C61) very rare under age 20
        if site.startswith("C61") and age < 20:
            violations.append(EditViolation(
                edit_name="Age_Site_Plausibility",
                severity="warning",
                item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_AGE],
                message=(
                    f"Prostate cancer (site {raw_site}) is very rare under "
                    f"age 20 (patient age={age}). Verify site and age."
                ),
            ))

        # Breast cancer (C50) very rare under age 15
        if site.startswith("C50") and age < 15:
            violations.append(EditViolation(
                edit_name="Age_Site_Plausibility",
                severity="warning",
                item_numbers=[_ITEM_PRIMARY_SITE, _ITEM_AGE],
                message=(
                    f"Breast cancer (site {raw_site}) is very rare under "
                    f"age 15 (patient age={age}). Verify site and age."
                ),
            ))

        # Retinoblastoma (9510-9514) rare over age 10
        if hist is not None and 9510 <= hist <= 9514 and age > 10:
            violations.append(EditViolation(
                edit_name="Age_Histology_Plausibility",
                severity="warning",
                item_numbers=[_ITEM_HISTOLOGY, _ITEM_AGE],
                message=(
                    f"Retinoblastoma (histology {hist}) is rare over age 10 "
                    f"(patient age={age}). Verify histology and age."
                ),
            ))

        # Neuroblastoma (9500) rare over age 20
        if hist is not None and hist == 9500 and age > 20:
            violations.append(EditViolation(
                edit_name="Age_Histology_Plausibility",
                severity="warning",
                item_numbers=[_ITEM_HISTOLOGY, _ITEM_AGE],
                message=(
                    f"Neuroblastoma (histology 9500) is rare over age 20 "
                    f"(patient age={age}). Verify histology and age."
                ),
            ))

        # Pediatric cancers in elderly -- Wilms tumor (8960) rare over age 15
        if hist is not None and hist == 8960 and age > 15:
            violations.append(EditViolation(
                edit_name="Age_Histology_Plausibility",
                severity="warning",
                item_numbers=[_ITEM_HISTOLOGY, _ITEM_AGE],
                message=(
                    f"Wilms tumor (histology 8960) is rare over age 15 "
                    f"(patient age={age}). Verify histology and age."
                ),
            ))

        return violations

    # ------------------------------------------------------------------
    # TNM staging internal consistency
    # ------------------------------------------------------------------

    def _check_tnm_consistency(self, record: dict) -> list[EditViolation]:
        """TNM staging internal consistency."""
        violations: list[EditViolation] = []

        ct = self._get_value(record, _ITEM_CT).strip()
        cn = self._get_value(record, _ITEM_CN).strip()
        cm = self._get_value(record, _ITEM_CM).strip()
        cstage = self._get_value(record, _ITEM_CSTAGE_GROUP).strip()

        pt = self._get_value(record, _ITEM_PT).strip()
        pn = self._get_value(record, _ITEM_PN).strip()
        pm = self._get_value(record, _ITEM_PM).strip()
        pstage = self._get_value(record, _ITEM_PSTAGE_GROUP).strip()

        raw_behav = self._get_value(record, _ITEM_BEHAVIOR).strip()

        # If any clinical T/N/M present, cStage Group should be present
        has_clinical_tnm = any(
            v and not _is_unknown_or_blank(v) for v in [ct, cn, cm]
        )
        if has_clinical_tnm and (not cstage or _is_unknown_or_blank(cstage)):
            violations.append(EditViolation(
                edit_name="TNM_Clinical_Consistency",
                severity="warning",
                item_numbers=[_ITEM_CT, _ITEM_CN, _ITEM_CM, _ITEM_CSTAGE_GROUP],
                message=(
                    "Clinical T/N/M values are present but Clinical Stage "
                    "Group is missing or unknown."
                ),
            ))

        # If any pathological T/N/M present, pStage Group should be present
        has_path_tnm = any(
            v and not _is_unknown_or_blank(v) for v in [pt, pn, pm]
        )
        if has_path_tnm and (not pstage or _is_unknown_or_blank(pstage)):
            violations.append(EditViolation(
                edit_name="TNM_Pathological_Consistency",
                severity="warning",
                item_numbers=[_ITEM_PT, _ITEM_PN, _ITEM_PM, _ITEM_PSTAGE_GROUP],
                message=(
                    "Pathological T/N/M values are present but Pathological "
                    "Stage Group is missing or unknown."
                ),
            ))

        # T0 with N+ is unusual
        for t_val, n_val, prefix in [
            (ct, cn, "Clinical"), (pt, pn, "Pathological")
        ]:
            t_stripped = t_val.upper().replace(" ", "")
            n_stripped = n_val.upper().replace(" ", "")
            if t_stripped in ("T0", "0", "C0", "P0"):
                # N+ means any N value that isn't N0, NX, blank, or unknown
                if (n_stripped
                        and n_stripped not in ("N0", "0", "NX", "X", "")
                        and not _is_unknown_or_blank(n_val)):
                    violations.append(EditViolation(
                        edit_name=f"TNM_{prefix}_T0_Npositive",
                        severity="warning",
                        item_numbers=(
                            [_ITEM_CT, _ITEM_CN]
                            if prefix == "Clinical"
                            else [_ITEM_PT, _ITEM_PN]
                        ),
                        message=(
                            f"{prefix} T0 (no primary tumor) with "
                            f"{prefix} N={n_val} (node-positive) is unusual. "
                            f"Verify staging."
                        ),
                    ))

        # Tis (in situ) with behavior 3 (malignant) inconsistency
        for t_val, prefix in [(ct, "Clinical"), (pt, "Pathological")]:
            t_upper = t_val.upper().replace(" ", "")
            if t_upper in ("TIS", "IS"):
                if raw_behav == "3":
                    violations.append(EditViolation(
                        edit_name=f"TNM_{prefix}_Tis_Behavior",
                        severity="error",
                        item_numbers=(
                            [_ITEM_CT, _ITEM_BEHAVIOR]
                            if prefix == "Clinical"
                            else [_ITEM_PT, _ITEM_BEHAVIOR]
                        ),
                        message=(
                            f"{prefix} stage Tis (in situ) is inconsistent "
                            f"with Behavior=3 (malignant). In situ should "
                            f"have Behavior=2."
                        ),
                        auto_fixable=True,
                        suggested_fix={_ITEM_BEHAVIOR: "2"},
                    ))

        return violations

    # ------------------------------------------------------------------
    # Treatment dates >= diagnosis date
    # ------------------------------------------------------------------

    def _check_treatment_dates(self, record: dict) -> list[EditViolation]:
        """Treatment dates must be on or after diagnosis date."""
        violations: list[EditViolation] = []

        diag_date = self._get_value(record, _ITEM_DIAG_DATE)
        if not diag_date or _is_unknown_or_blank(diag_date):
            return violations

        treatment_items = [
            (_ITEM_SURGERY_DATE, "Surgery"),
            (_ITEM_RADIATION_DATE, "Radiation"),
            (_ITEM_CHEMO_DATE, "Chemotherapy"),
            (_ITEM_HORMONE_DATE, "Hormone Therapy"),
            (_ITEM_BRM_DATE, "Immunotherapy/BRM"),
        ]

        for item_num, label in treatment_items:
            tx_date = self._get_value(record, item_num)
            if not tx_date or _is_unknown_or_blank(tx_date):
                continue

            cmp = _compare_dates(tx_date, diag_date)
            if cmp is not None and cmp < 0:
                violations.append(EditViolation(
                    edit_name="Treatment_Date_Before_Diagnosis",
                    severity="error",
                    item_numbers=[item_num, _ITEM_DIAG_DATE],
                    message=(
                        f"{label} date ({tx_date}) is before Date of "
                        f"Diagnosis ({diag_date}). Treatment cannot precede "
                        f"diagnosis."
                    ),
                    auto_fixable=False,
                ))

        return violations

    # ------------------------------------------------------------------
    # Diagnostic confirmation consistency
    # ------------------------------------------------------------------

    def _check_diagnostic_confirmation(self, record: dict) -> list[EditViolation]:
        """Diagnostic confirmation and histology/staging consistency."""
        violations: list[EditViolation] = []

        raw_dc = self._get_value(record, _ITEM_DIAG_CONFIRM)
        raw_hist = self._get_value(record, _ITEM_HISTOLOGY)

        if not raw_dc or _is_unknown_or_blank(raw_dc):
            return violations

        dc = _parse_int(raw_dc)
        if dc is None:
            return violations

        # Codes 1-5 indicate microscopic/pathological confirmation
        # If microscopically confirmed, histology should be present
        if 1 <= dc <= 5:
            if not raw_hist or _is_unknown_or_blank(raw_hist):
                violations.append(EditViolation(
                    edit_name="DiagConfirm_Histology",
                    severity="warning",
                    item_numbers=[_ITEM_DIAG_CONFIRM, _ITEM_HISTOLOGY],
                    message=(
                        f"Diagnostic confirmation={dc} (microscopically "
                        f"confirmed) but histology is missing or unknown. "
                        f"Path-confirmed cases should have a specific histology."
                    ),
                ))

        # Codes 6-9 indicate clinical-only diagnosis
        # Pathological staging should generally be absent
        if 6 <= dc <= 9:
            pt = self._get_value(record, _ITEM_PT).strip()
            pn = self._get_value(record, _ITEM_PN).strip()
            pm = self._get_value(record, _ITEM_PM).strip()
            pstage = self._get_value(record, _ITEM_PSTAGE_GROUP).strip()

            has_path_staging = any(
                v and not _is_unknown_or_blank(v)
                for v in [pt, pn, pm, pstage]
            )
            if has_path_staging:
                violations.append(EditViolation(
                    edit_name="DiagConfirm_PathStaging",
                    severity="warning",
                    item_numbers=[_ITEM_DIAG_CONFIRM, _ITEM_PT, _ITEM_PN,
                                  _ITEM_PM, _ITEM_PSTAGE_GROUP],
                    message=(
                        f"Diagnostic confirmation={dc} (clinical only) but "
                        f"pathological staging values are present. Clinical-"
                        f"only cases should not have pathological staging."
                    ),
                ))

        return violations

    # ------------------------------------------------------------------
    # Surgery field consistency
    # ------------------------------------------------------------------

    def _check_surgery_consistency(self, record: dict) -> list[EditViolation]:
        """Surgery fields internal consistency."""
        violations: list[EditViolation] = []

        raw_surg = self._get_value(record, _ITEM_SURG_PRIM_SITE)
        # Fallback to 2023+ item number
        if not raw_surg:
            raw_surg = self._get_value(record, _ITEM_SURG_PRIM_SITE_2023)

        raw_surg_date = self._get_value(record, _ITEM_SURGERY_DATE)
        raw_no_surg = self._get_value(record, _ITEM_REASON_NO_SURG)
        raw_margins = self._get_value(record, _ITEM_SURGICAL_MARGINS)

        if not raw_surg:
            return violations

        surg = raw_surg.strip()

        # Determine whether surgery was performed:
        # 00 or 0000 = no surgery; anything else = surgery performed
        # Also handle blank or unknown (98, 99) as not-confirmed
        surgery_performed = (
            surg not in ("00", "0000", "0", "98", "99", "")
            and not _is_unknown_or_blank(surg)
        )

        if surgery_performed:
            # Surgery date should be populated
            if not raw_surg_date or _is_unknown_or_blank(raw_surg_date):
                violations.append(EditViolation(
                    edit_name="Surgery_Date_Missing",
                    severity="warning",
                    item_numbers=[_ITEM_SURG_PRIM_SITE, _ITEM_SURGERY_DATE],
                    message=(
                        f"Surgery of primary site={surg} indicates surgery "
                        f"performed, but surgery date is missing."
                    ),
                ))

            # Surgical margins should be populated
            if not raw_margins or _is_unknown_or_blank(raw_margins):
                violations.append(EditViolation(
                    edit_name="Surgery_Margins_Missing",
                    severity="warning",
                    item_numbers=[_ITEM_SURG_PRIM_SITE, _ITEM_SURGICAL_MARGINS],
                    message=(
                        f"Surgery of primary site={surg} indicates surgery "
                        f"performed, but surgical margins status is missing."
                    ),
                ))
        else:
            # No surgery -- Reason for No Surgery should be populated
            # (but only if surg code is definitively 00/0000 not unknown)
            if surg in ("00", "0000", "0"):
                if not raw_no_surg or _is_unknown_or_blank(raw_no_surg):
                    violations.append(EditViolation(
                        edit_name="Reason_No_Surgery_Missing",
                        severity="warning",
                        item_numbers=[_ITEM_SURG_PRIM_SITE, _ITEM_REASON_NO_SURG],
                        message=(
                            "No surgery performed but Reason for No Surgery "
                            "is missing."
                        ),
                    ))

        return violations

    # ------------------------------------------------------------------
    # Radiation field consistency
    # ------------------------------------------------------------------

    def _check_radiation_consistency(self, record: dict) -> list[EditViolation]:
        """Radiation fields internal consistency."""
        violations: list[EditViolation] = []

        raw_rad = self._get_value(record, _ITEM_RADIATION)
        raw_rad_date = self._get_value(record, _ITEM_RADIATION_DATE)
        raw_no_rad = self._get_value(record, _ITEM_REASON_NO_RAD)

        if not raw_rad:
            return violations

        rad = raw_rad.strip()

        # 0 = no radiation; 1-6 = radiation given; 7 = refused; 8 = recommended unknown;
        # 9 = unknown
        radiation_given = rad not in ("0", "7", "8", "9", "") and not _is_unknown_or_blank(rad)

        if radiation_given:
            # Radiation date should be populated
            if not raw_rad_date or _is_unknown_or_blank(raw_rad_date):
                violations.append(EditViolation(
                    edit_name="Radiation_Date_Missing",
                    severity="warning",
                    item_numbers=[_ITEM_RADIATION, _ITEM_RADIATION_DATE],
                    message=(
                        f"Radiation={rad} indicates radiation given, but "
                        f"radiation date is missing."
                    ),
                ))
        else:
            # No radiation -- Reason for No Radiation should be populated
            if rad == "0":
                if not raw_no_rad or _is_unknown_or_blank(raw_no_rad):
                    violations.append(EditViolation(
                        edit_name="Reason_No_Radiation_Missing",
                        severity="warning",
                        item_numbers=[_ITEM_RADIATION, _ITEM_REASON_NO_RAD],
                        message=(
                            "No radiation given but Reason for No Radiation "
                            "is missing."
                        ),
                    ))

        return violations

    # ------------------------------------------------------------------
    # Vital status / date of last contact
    # ------------------------------------------------------------------

    def _check_vital_status_dates(self, record: dict) -> list[EditViolation]:
        """Vital status and date consistency."""
        violations: list[EditViolation] = []

        diag_date = self._get_value(record, _ITEM_DIAG_DATE)
        dlc = self._get_value(record, _ITEM_DATE_LAST_CONTACT)
        vital = self._get_value(record, _ITEM_VITAL_STATUS).strip()
        cod = self._get_value(record, _ITEM_CAUSE_OF_DEATH)

        # Date of Last Contact >= Date of Diagnosis
        if diag_date and dlc and not _is_unknown_or_blank(diag_date) and not _is_unknown_or_blank(dlc):
            cmp = _compare_dates(dlc, diag_date)
            if cmp is not None and cmp < 0:
                violations.append(EditViolation(
                    edit_name="DLC_Before_Diagnosis",
                    severity="error",
                    item_numbers=[_ITEM_DATE_LAST_CONTACT, _ITEM_DIAG_DATE],
                    message=(
                        f"Date of Last Contact ({dlc}) is before Date of "
                        f"Diagnosis ({diag_date})."
                    ),
                ))

        # If dead (vital status=0), Cause of Death should be populated
        if vital == "0":
            if not cod or _is_unknown_or_blank(cod):
                violations.append(EditViolation(
                    edit_name="Dead_No_CauseOfDeath",
                    severity="warning",
                    item_numbers=[_ITEM_VITAL_STATUS, _ITEM_CAUSE_OF_DEATH],
                    message=(
                        "Vital Status=0 (dead) but Cause of Death is "
                        "missing or unknown."
                    ),
                ))

        return violations

    # ------------------------------------------------------------------
    # Auto-fix
    # ------------------------------------------------------------------

    def auto_fix(
        self,
        record: dict,
        violations: list[EditViolation],
    ) -> tuple[dict, list[str]]:
        """Apply auto-fixes for fixable violations.

        Returns
        -------
        tuple[dict, list[str]]
            (modified_record, list_of_fix_descriptions)
        """
        modified = dict(record)
        descriptions: list[str] = []

        for v in violations:
            if not v.auto_fixable or not v.suggested_fix:
                continue

            for item_num, suggested_value in v.suggested_fix.items():
                result = modified.get(item_num)
                if result is None:
                    continue

                old_code = getattr(result, "resolved_code", "")

                # We create a shallow copy-like approach: set the attribute
                # directly since ExtractionResult is a dataclass.
                try:
                    result.resolved_code = suggested_value
                    desc = (
                        f"[{v.edit_name}] Item {item_num}: "
                        f"changed resolved_code from '{old_code}' to "
                        f"'{suggested_value}'"
                    )
                    descriptions.append(desc)
                    logger.info("Auto-fix applied: %s", desc)
                except AttributeError:
                    logger.warning(
                        "Cannot auto-fix item %d: result object is immutable.",
                        item_num,
                    )

        return modified, descriptions
