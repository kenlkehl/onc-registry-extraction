"""Schema Registry: maps cancer site/histology to required site-specific data items.

Each cancer schema defines the site-specific data items (SSDIs) that registrars
must collect beyond the core staging fields (TNM, Summary Stage, EOD, etc.).
The registry uses ICD-O-3 topography codes and histology to determine which
schema applies, then returns the full set of NAACCR item numbers needed for
that cancer type.
"""

from __future__ import annotations

from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)

# Pre-compiled pattern for extracting the C## prefix from topography codes
# Handles both "C50.4" and "C509" formats.
_SITE_PREFIX_RE = re.compile(r"^(C\d{2})", re.IGNORECASE)


class SchemaRegistry:
    """Maps primary site + histology -> cancer schema -> required SSDIs."""

    # ------------------------------------------------------------------
    # Core staging items that apply to ALL cancer types
    # ------------------------------------------------------------------

    CORE_STAGING_ITEMS: list[int] = [
        752,   # Tumor Size Clinical
        754,   # Tumor Size Pathologic
        756,   # Tumor Size Summary
        764,   # Summary Stage 2018
        772,   # EOD Primary Tumor
        774,   # EOD Regional Nodes
        776,   # EOD Mets
        820,   # Regional Nodes Positive
        830,   # Regional Nodes Examined
        832,   # Date of Sentinel Lymph Node Biopsy
        834,   # Sentinel Lymph Nodes Examined
        835,   # Sentinel Lymph Nodes Positive
        880,   # TNM Path T
        890,   # TNM Path N
        900,   # TNM Path M
        910,   # TNM Path Stage Group
        940,   # TNM Clin T
        950,   # TNM Clin N
        960,   # TNM Clin M
        970,   # TNM Clin Stage Group
        1001,  # AJCC TNM Clin T
        1002,  # AJCC TNM Clin N
        1003,  # AJCC TNM Clin M
        1004,  # AJCC TNM Clin Stage Group
        1011,  # AJCC TNM Path T
        1012,  # AJCC TNM Path N
        1013,  # AJCC TNM Path M
        1014,  # AJCC TNM Path Stage Group
        1060,  # TNM Edition Number
        1112,  # Mets at DX-Bone
        1113,  # Mets at DX-Brain
        1114,  # Mets at Dx-Distant LN
        1115,  # Mets at DX-Liver
        1116,  # Mets at DX-Lung
        1117,  # Mets at DX-Other
        1182,  # Lymphovascular Invasion
        3843,  # Grade Clinical
        3844,  # Grade Pathological
    ]

    # ------------------------------------------------------------------
    # Site-specific data items (SSDIs) by schema
    # These are the NAACCR item numbers for site-specific factors,
    # using the correct v26 numbering from DataItems.csv.
    # ------------------------------------------------------------------

    SCHEMA_SSDI_MAP: dict[str, list[int]] = {
        "breast": [
            3827,  # Estrogen Receptor Summary
            3826,  # Estrogen Receptor Percent Positive or Range
            3828,  # Estrogen Receptor Total Allred Score
            3915,  # Progesterone Receptor Summary
            3914,  # Progesterone Receptor Percent Positive or Range
            3916,  # Progesterone Receptor Total Allred Score
            3850,  # HER2 IHC Summary
            3851,  # HER2 ISH Dual Probe Copy Number
            3852,  # HER2 ISH Dual Probe Ratio
            3853,  # HER2 ISH Single Probe Copy Number
            3854,  # HER2 ISH Summary
            3855,  # HER2 Overall Summary
            3863,  # Ki-67
            3894,  # Multigene Signature Method
            3895,  # Multigene Signature Results
            3903,  # Oncotype Dx Recurrence Score-DCIS
            3904,  # Oncotype Dx Recurrence Score-Invasive
            3905,  # Oncotype Dx Risk Level-DCIS
            3906,  # Oncotype Dx Risk Level-Invasive
            3882,  # LN Positive Axillary Level I-II
            3922,  # Response to Neoadjuvant Therapy
        ],
        "prostate": [
            3838,  # Gleason Patterns Clinical
            3839,  # Gleason Patterns Pathological
            3840,  # Gleason Score Clinical
            3841,  # Gleason Score Pathological
            3842,  # Gleason Tertiary Pattern
            3920,  # PSA (Prostatic Specific Antigen) Lab Value
            3897,  # Number of Cores Examined
            3898,  # Number of Cores Positive
            3919,  # EOD Prostate Pathologic Extension
        ],
        "colon_rectum": [
            3823,  # Circumferential Resection Margin (CRM)
            3819,  # CEA Pretreatment Interpretation
            3820,  # CEA Pretreatment Lab Value
            3909,  # Perineural Invasion
            3890,  # Microsatellite Instability (MSI)
            3866,  # KRAS
            3934,  # Tumor Deposits
            3929,  # Separate Tumor Nodules
        ],
        "lung": [
            3938,  # ALK Rearrangement
            3939,  # EGFR Mutational Analysis
            3866,  # KRAS
            3940,  # BRAF Mutational Analysis
            1174,  # PD-L1
            1176,  # Spread Through Air Spaces (STAS)
            3937,  # Visceral and Parietal Pleural Invasion
            3929,  # Separate Tumor Nodules
        ],
        "melanoma_skin": [
            3817,  # Breslow Tumor Thickness
            3893,  # Mitotic Rate Melanoma
            3936,  # Ulceration
        ],
        "kidney_renal_pelvis": [
            3925,  # Sarcomatoid Features
        ],
        "bladder": [
            3922,  # Response to Neoadjuvant Therapy
        ],
        "thyroid": [
            3830,  # Extranodal Extension Clin (non-Head and Neck)
            3833,  # Extranodal Extension Path (non-Head and Neck)
        ],
        "cervix": [
            3836,  # FIGO Stage
            3956,  # p16
        ],
        "ovary": [
            3818,  # CA-125 Pretreatment Interpretation
            3836,  # FIGO Stage
            3921,  # Residual Tumor Volume Post Cytoreduction
            3911,  # Peritoneal Cytology
        ],
        "testis": [
            3807,  # AFP Pre-Orchiectomy Lab Value
            3808,  # AFP Pre-Orchiectomy Range
            3805,  # AFP Post-Orchiectomy Lab Value
            3806,  # AFP Post-Orchiectomy Range
            3848,  # hCG Pre-Orchiectomy Lab Value
            3849,  # hCG Pre-Orchiectomy Range
            3846,  # hCG Post-Orchiectomy Lab Value
            3847,  # hCG Post-Orchiectomy Range
            3868,  # LDH Pre-Orchiectomy Range
            3867,  # LDH Post-Orchiectomy Range
            3923,  # S Category Clinical
            3924,  # S Category Pathological
        ],
        "liver": [
            3809,  # AFP Pretreatment Interpretation
            3810,  # AFP Pretreatment Lab Value
            3835,  # Fibrosis Score
        ],
        "pancreas": [
            3942,  # CA 19-9 PreTX Lab Value
        ],
        "head_neck": [
            3831,  # Extranodal Extension Head and Neck Clinical
            3832,  # Extranodal Extension Head and Neck Pathological
            3956,  # p16
        ],
        "brain_cns": [
            3816,  # Brain Molecular Markers
        ],
    }

    # ------------------------------------------------------------------
    # Maps ICD-O-3 topography code prefixes to schema names.
    # The key is the C## prefix (without the sub-site digit).
    # ------------------------------------------------------------------

    SITE_SCHEMA_MAP: dict[str, str] = {
        # Breast
        "C50": "breast",
        # Prostate
        "C61": "prostate",
        # Colorectal
        "C18": "colon_rectum",
        "C19": "colon_rectum",
        "C20": "colon_rectum",
        "C21": "colon_rectum",
        # Lung and bronchus
        "C34": "lung",
        # Skin -- melanoma only (histology check required)
        "C44": "melanoma_skin",
        # Kidney and renal pelvis
        "C64": "kidney_renal_pelvis",
        "C65": "kidney_renal_pelvis",
        # Bladder
        "C67": "bladder",
        # Thyroid
        "C73": "thyroid",
        # Cervix
        "C53": "cervix",
        # Ovary
        "C56": "ovary",
        # Testis
        "C62": "testis",
        # Liver and intrahepatic bile ducts
        "C22": "liver",
        # Pancreas
        "C25": "pancreas",
        # Head and neck sites
        "C00": "head_neck",  # Lip
        "C01": "head_neck",  # Base of tongue
        "C02": "head_neck",  # Other tongue
        "C03": "head_neck",  # Gum
        "C04": "head_neck",  # Floor of mouth
        "C05": "head_neck",  # Palate
        "C06": "head_neck",  # Other mouth
        "C07": "head_neck",  # Parotid gland
        "C08": "head_neck",  # Other salivary glands
        "C09": "head_neck",  # Tonsil
        "C10": "head_neck",  # Oropharynx
        "C11": "head_neck",  # Nasopharynx
        "C12": "head_neck",  # Pyriform sinus
        "C13": "head_neck",  # Hypopharynx
        "C14": "head_neck",  # Other pharynx
        "C30": "head_neck",  # Nasal cavity
        "C31": "head_neck",  # Accessory sinuses
        "C32": "head_neck",  # Larynx
        # Brain and CNS
        "C70": "brain_cns",
        "C71": "brain_cns",
        "C72": "brain_cns",
    }

    # ------------------------------------------------------------------
    # Melanoma histology range (ICD-O-3): 8720-8790
    # ------------------------------------------------------------------

    _MELANOMA_HIST_LO = 8720
    _MELANOMA_HIST_HI = 8790

    # ------------------------------------------------------------------
    # Site-specific extraction context for LLM prompts
    # ------------------------------------------------------------------

    _SITE_CONTEXT: dict[str, str] = {
        "breast": (
            "For breast cancer, extract:\n"
            "- ER (Estrogen Receptor) status: summary (positive/negative/borderline), "
            "percent positive (exact value if stated, e.g. 95%%), Allred score (0-8).\n"
            "- PR (Progesterone Receptor) status: summary, percent positive, Allred score.\n"
            "- HER2 status: IHC score (0, 1+, 2+, 3+), ISH result (positive/negative, "
            "dual probe copy number and ratio, single probe copy number), overall summary.\n"
            "- Ki-67 proliferation index (percentage if stated).\n"
            "- Multigene signature: method (Oncotype Dx, MammaPrint, Prosigna/PAM50, "
            "EndoPredict, Breast Cancer Index), results/recurrence score, and risk category.\n"
            "- Oncotype Dx recurrence score (0-100) for both DCIS and invasive, with risk level.\n"
            "- Axillary lymph node involvement: number of positive nodes at Level I-II.\n"
            "- Response to neoadjuvant therapy if applicable (pathological complete response, etc.).\n"
            "Receptor percentages should be exact values when available. "
            "If Allred score components are given (proportion score + intensity score), "
            "calculate the total."
        ),
        "prostate": (
            "For prostate cancer, extract:\n"
            "- Gleason patterns: both clinical and pathological. Record the two pattern numbers "
            "(e.g. 3+4, 4+3). The first number is the primary/predominant pattern.\n"
            "- Gleason score: clinical and pathological (sum of the two patterns, e.g. 7).\n"
            "- Gleason tertiary pattern: if a third pattern is mentioned (e.g. tertiary pattern 5).\n"
            "- PSA lab value: the most recent pre-treatment PSA in ng/mL. Record the numeric value.\n"
            "- Number of biopsy cores examined and number positive.\n"
            "- EOD prostate pathologic extension: extent of disease beyond the prostate capsule.\n"
            "IMPORTANT: Distinguish between clinical Gleason (from biopsy, before treatment) "
            "and pathological Gleason (from prostatectomy specimen). "
            "Grade group may be mentioned (Group 1=3+3, Group 2=3+4, Group 3=4+3, "
            "Group 4=4+4/3+5/5+3, Group 5=4+5/5+4/5+5)."
        ),
        "colon_rectum": (
            "For colorectal cancer, extract:\n"
            "- CEA (carcinoembryonic antigen): pretreatment lab value (ng/mL) and interpretation "
            "(normal/abnormal/unknown).\n"
            "- Circumferential resection margin (CRM): distance in mm, or positive/negative.\n"
            "- Microsatellite instability (MSI): MSI-H (high), MSI-L (low), MSS (stable), or "
            "results of mismatch repair (MMR) IHC testing (MLH1, MSH2, MSH6, PMS2 "
            "intact/lost).\n"
            "- KRAS mutation status: mutated/wild type, specific mutation if stated.\n"
            "- Perineural invasion: present/absent/not identified.\n"
            "- Tumor deposits: number of discrete tumor deposits in pericolorectal/perirectal fat.\n"
            "- Separate tumor nodules: satellite nodules in the pericolorectal fat.\n"
            "For rectal cancers, pay particular attention to CRM and tumor deposits "
            "as these are critical prognostic factors."
        ),
        "lung": (
            "For lung cancer, extract:\n"
            "- ALK rearrangement: positive/negative (from FISH, IHC, or NGS).\n"
            "- EGFR mutational analysis: mutated/not mutated, specific mutations if stated "
            "(e.g. exon 19 deletion, L858R, T790M, exon 20 insertion).\n"
            "- KRAS mutation: mutated/wild type, specific mutation (e.g. G12C).\n"
            "- BRAF mutation: V600E or other specific mutations.\n"
            "- PD-L1 expression: tumor proportion score (TPS) percentage if stated, "
            "and positive/negative/not tested.\n"
            "- Spread through air spaces (STAS): present/absent.\n"
            "- Visceral pleural invasion: present/absent, PL0/PL1/PL2/PL3 if specified.\n"
            "- Separate tumor nodules: same lobe, different lobe, or absent.\n"
            "IMPORTANT: Molecular testing results are critical for treatment decisions. "
            "Record the specific mutation when available, not just positive/negative."
        ),
        "melanoma_skin": (
            "For cutaneous melanoma, extract:\n"
            "- Breslow tumor thickness: depth in mm (e.g. 1.2 mm). This is the single most "
            "important prognostic factor.\n"
            "- Mitotic rate: number of mitoses per mm2. Record the exact count.\n"
            "- Ulceration: present or absent. This affects staging.\n"
            "IMPORTANT: Breslow thickness should be recorded to the nearest 0.1 mm. "
            "Clark level may also be mentioned (I-V) but Breslow thickness is the "
            "primary measurement for staging."
        ),
        "kidney_renal_pelvis": (
            "For kidney/renal pelvis cancer, extract:\n"
            "- Sarcomatoid features: percentage of sarcomatoid component if stated, "
            "or present/absent.\n"
            "Sarcomatoid differentiation is an important adverse prognostic factor "
            "in renal cell carcinoma regardless of histologic subtype."
        ),
        "bladder": (
            "For bladder cancer, extract:\n"
            "- Response to neoadjuvant therapy if given (complete response, partial response, etc.).\n"
            "Pay attention to whether the tumor is muscle-invasive (T2+) versus non-muscle-invasive "
            "(Ta, Tis, T1), as this distinction is critical for treatment and prognosis."
        ),
        "thyroid": (
            "For thyroid cancer, extract:\n"
            "- Extranodal extension: clinical and pathological (non-head and neck coding).\n"
            "Note the histologic subtype (papillary, follicular, medullary, anaplastic) "
            "as it affects staging. BRAF V600E mutation may also be mentioned for papillary "
            "thyroid carcinoma."
        ),
        "cervix": (
            "For cervical cancer, extract:\n"
            "- FIGO stage: the clinical staging used for cervix (IA1, IA2, IB1, IB2, IB3, "
            "IIA1, IIA2, IIB, IIIA, IIIB, IIIC1, IIIC2, IVA, IVB).\n"
            "- p16 immunohistochemistry: positive (block positive)/negative/not done. "
            "p16 overexpression is a surrogate for HPV-associated carcinoma.\n"
            "FIGO staging is the primary clinical staging system for cervical cancer."
        ),
        "ovary": (
            "For ovarian cancer, extract:\n"
            "- CA-125 pretreatment interpretation: elevated/normal/unknown. "
            "Record the lab value if available.\n"
            "- FIGO stage (IA, IB, IC, IIA, IIB, IIIA, IIIB, IIIC, IVA, IVB).\n"
            "- Residual tumor volume post cytoreduction: no residual disease, "
            "optimal (<=1cm), suboptimal (>1cm).\n"
            "- Peritoneal cytology: positive/negative/not done.\n"
            "Residual disease after primary debulking is one of the strongest "
            "prognostic factors in advanced ovarian cancer."
        ),
        "testis": (
            "For testicular cancer, extract:\n"
            "- AFP (alpha-fetoprotein): pre- and post-orchiectomy lab values and ranges.\n"
            "- hCG (human chorionic gonadotropin): pre- and post-orchiectomy lab values and ranges.\n"
            "- LDH (lactate dehydrogenase): pre- and post-orchiectomy ranges.\n"
            "- S Category: clinical and pathological (S0, S1, S2, S3, SX).\n"
            "IMPORTANT: Serum tumor markers (AFP, hCG, LDH) must be recorded both "
            "before and after orchiectomy. The S category is derived from these markers "
            "and is part of TNM staging for testicular cancer."
        ),
        "liver": (
            "For liver cancer (hepatocellular carcinoma), extract:\n"
            "- AFP pretreatment interpretation: elevated/normal/unknown, and the lab value.\n"
            "- Fibrosis score (Ishak score): F0-F6 or equivalent staging. "
            "This affects staging for HCC.\n"
            "The fibrosis score of the non-tumorous liver is an important staging factor "
            "for hepatocellular carcinoma."
        ),
        "pancreas": (
            "For pancreatic cancer, extract:\n"
            "- CA 19-9 pretreatment lab value: record the numeric value in U/mL.\n"
            "CA 19-9 is the most commonly used serum biomarker for pancreatic cancer "
            "and is used for monitoring treatment response."
        ),
        "head_neck": (
            "For head and neck cancer, extract:\n"
            "- Extranodal extension (ENE): clinical and pathological. This is coded "
            "specifically for head and neck sites.\n"
            "- p16 status: positive/negative, especially important for oropharyngeal cancers "
            "as HPV-associated (p16+) oropharyngeal carcinomas have a distinct staging system.\n"
            "IMPORTANT: HPV/p16 status changes the staging system for oropharyngeal cancers."
        ),
        "brain_cns": (
            "For brain and CNS tumors, extract:\n"
            "- Brain molecular markers: IDH1/IDH2 mutation status, 1p/19q codeletion, "
            "MGMT methylation status, ATRX loss, H3K27M mutation.\n"
            "Molecular markers are now integrated into the WHO classification of CNS tumors "
            "and are essential for accurate diagnosis and grading."
        ),
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_schema_for_site_histology(
        self,
        primary_site: str,
        histology: str,
        schema_discriminator: Optional[str] = None,
    ) -> str:
        """Determine schema from ICD-O-3 topography + morphology.

        Parameters
        ----------
        primary_site:
            ICD-O-3 topography code, e.g. ``"C50.4"`` or ``"C509"``.
        histology:
            ICD-O-3 morphology code (4 digits), e.g. ``"8500"``.
        schema_discriminator:
            Optional schema discriminator value for ambiguous cases.

        Returns
        -------
        str
            Schema name (e.g. ``"breast"``, ``"prostate"``).
            Returns ``"generic"`` when no specific schema matches.

        Logic
        -----
        1. Normalize site to C## prefix (uppercase).
        2. Look up in SITE_SCHEMA_MAP.
        3. Special case: C44 (skin) maps to ``melanoma_skin`` ONLY if
           histology is in the melanoma range (8720-8790). Other skin
           histologies fall through to ``"generic"``.
        4. Default to ``"generic"`` if no match.
        """
        # 1. Normalize to C## prefix
        prefix = self._normalize_site_prefix(primary_site)
        if not prefix:
            logger.debug(
                "Could not parse site prefix from '%s'; defaulting to generic.",
                primary_site,
            )
            return "generic"

        # 2. Look up in the map
        schema = self.SITE_SCHEMA_MAP.get(prefix)
        if schema is None:
            logger.debug(
                "No schema mapping for site prefix '%s'; defaulting to generic.",
                prefix,
            )
            return "generic"

        # 3. Special case: skin (C44) requires melanoma histology
        if prefix == "C44":
            hist_num = self._parse_histology(histology)
            if hist_num is None or not (
                self._MELANOMA_HIST_LO <= hist_num <= self._MELANOMA_HIST_HI
            ):
                logger.debug(
                    "C44 with non-melanoma histology '%s'; defaulting to generic.",
                    histology,
                )
                return "generic"

        return schema

    def get_required_ssdis(self, schema: str) -> list[int]:
        """Return SSDI item numbers for this schema.

        Returns an empty list for unknown schemas (including ``"generic"``).
        """
        return list(self.SCHEMA_SSDI_MAP.get(schema, []))

    def get_all_staging_items(self, schema: str) -> list[int]:
        """Return core staging items + schema-specific SSDIs.

        The result is deduplicated (some SSDIs may also appear in CORE)
        and returned in a stable order (core items first, then SSDIs).
        """
        core = list(self.CORE_STAGING_ITEMS)
        ssdis = self.get_required_ssdis(schema)

        # Deduplicate while preserving order
        seen: set[int] = set(core)
        combined = list(core)
        for item_num in ssdis:
            if item_num not in seen:
                seen.add(item_num)
                combined.append(item_num)

        return combined

    def get_site_context(self, schema: str) -> str:
        """Return site-specific extraction instructions for LLM prompts.

        Returns a detailed text block describing what site-specific data
        items to look for and how to code them.  Returns a generic
        instruction string when no site-specific context is defined.
        """
        context = self._SITE_CONTEXT.get(schema)
        if context:
            return context

        return (
            "Extract all available staging information including TNM stage "
            "(clinical and pathological), Summary Stage 2018, EOD fields, "
            "tumor size, regional lymph node status, and any biomarkers or "
            "prognostic factors mentioned in the clinical text."
        )

    def get_primary_site_description(self, schema: str) -> str:
        """Return a human-readable description of the cancer site for prompts."""
        descriptions: dict[str, str] = {
            "breast": "breast",
            "prostate": "prostate",
            "colon_rectum": "colorectal",
            "lung": "lung/bronchus",
            "melanoma_skin": "cutaneous melanoma",
            "kidney_renal_pelvis": "kidney/renal pelvis",
            "bladder": "urinary bladder",
            "thyroid": "thyroid",
            "cervix": "cervix uteri",
            "ovary": "ovary/fallopian tube",
            "testis": "testis",
            "liver": "liver/intrahepatic bile duct",
            "pancreas": "pancreas",
            "head_neck": "head and neck",
            "brain_cns": "brain/central nervous system",
            "generic": "cancer (site not specified)",
        }
        return descriptions.get(schema, "cancer")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_site_prefix(primary_site: str) -> Optional[str]:
        """Extract and uppercase the C## prefix from a topography code.

        Handles formats like ``"C50.4"``, ``"C509"``, ``"c50.4"``,
        ``"C50"``.  Returns ``None`` if the code cannot be parsed.
        """
        if not primary_site:
            return None
        cleaned = primary_site.strip().replace(" ", "")
        m = _SITE_PREFIX_RE.match(cleaned)
        if m:
            return m.group(1).upper()
        return None

    @staticmethod
    def _parse_histology(histology: str) -> Optional[int]:
        """Parse a histology code string to an integer.

        Returns ``None`` if the string is not a valid 4-digit number.
        """
        if not histology:
            return None
        cleaned = histology.strip()
        # Accept 4-digit codes, possibly with a behavior suffix (e.g. "8500/3")
        cleaned = cleaned.split("/")[0].strip()
        try:
            return int(cleaned)
        except (ValueError, TypeError):
            return None
