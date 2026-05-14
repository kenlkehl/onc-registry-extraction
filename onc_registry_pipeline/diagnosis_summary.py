"""Write a one-row-per-diagnosis high-level summary CSV file."""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from onc_registry_pipeline.extraction.base import ExtractionResult
from onc_registry_pipeline.extraction.pass0_tumor_detection import TumorCandidate

logger = logging.getLogger(__name__)

DATE_OF_DIAGNOSIS = 390
PRIMARY_SITE = 400
HISTOLOGY_ICD_O_3 = 522
SUMMARY_STAGE_2018 = 764

AJCC_FIELDS: tuple[tuple[str, int], ...] = (
    ("ajcc_clinical_t", 1001),
    ("ajcc_clinical_n", 1002),
    ("ajcc_clinical_m", 1003),
    ("ajcc_clinical_stage_group", 1004),
    ("ajcc_pathologic_t", 1011),
    ("ajcc_pathologic_n", 1012),
    ("ajcc_pathologic_m", 1013),
    ("ajcc_pathologic_stage_group", 1014),
)

SUMMARY_COLUMNS: list[str] = [
    "patient_id",
    "diagnosis_date",
    "primary_site_code",
    "primary_site_text",
    "histology_code",
    "histology_text",
    "seer_summary_stage_code",
    "seer_summary_stage_text",
]
for _field_name, _item_number in AJCC_FIELDS:
    SUMMARY_COLUMNS.extend([f"{_field_name}_code", f"{_field_name}_text"])

_SITE_CODE_RE = re.compile(r"C\d{2}(?:\.?\d)?", re.IGNORECASE)
_HISTOLOGY_CODE_RE = re.compile(r"\b(\d{4})(?:/[0-3])?\b")
_AJCC_BASIS_PREFIX_RE = re.compile(r"^(?:c|p)(?=(?:T|N|M|0|I|V|X))", re.IGNORECASE)
_BLANK_CODE_VALUES = {"blank"}

_TOPOGRAPHY_PREFIX_DESCRIPTIONS: dict[str, str] = {
    "C00": "Lip",
    "C01": "Base of tongue",
    "C02": "Other and unspecified parts of tongue",
    "C03": "Gum",
    "C04": "Floor of mouth",
    "C05": "Palate",
    "C06": "Other and unspecified parts of mouth",
    "C07": "Parotid gland",
    "C08": "Other and unspecified major salivary glands",
    "C09": "Tonsil",
    "C10": "Oropharynx",
    "C11": "Nasopharynx",
    "C12": "Pyriform sinus",
    "C13": "Hypopharynx",
    "C14": "Other and ill-defined sites in lip, oral cavity and pharynx",
    "C15": "Esophagus",
    "C16": "Stomach",
    "C17": "Small intestine",
    "C18": "Colon",
    "C19": "Rectosigmoid junction",
    "C20": "Rectum",
    "C21": "Anus and anal canal",
    "C22": "Liver and intrahepatic bile ducts",
    "C23": "Gallbladder",
    "C24": "Other and unspecified parts of biliary tract",
    "C25": "Pancreas",
    "C26": "Other and ill-defined digestive organs",
    "C30": "Nasal cavity and middle ear",
    "C31": "Accessory sinuses",
    "C32": "Larynx",
    "C33": "Trachea",
    "C34": "Bronchus and lung",
    "C37": "Thymus",
    "C38": "Heart, mediastinum and pleura",
    "C39": "Other and ill-defined respiratory and intrathoracic organs",
    "C40": "Bones, joints and articular cartilage of limbs",
    "C41": "Bones, joints and articular cartilage of other sites",
    "C42": "Hematopoietic and reticuloendothelial systems",
    "C44": "Skin",
    "C47": "Peripheral nerves and autonomic nervous system",
    "C48": "Retroperitoneum and peritoneum",
    "C49": "Connective, subcutaneous and other soft tissues",
    "C50": "Breast",
    "C51": "Vulva",
    "C52": "Vagina",
    "C53": "Cervix uteri",
    "C54": "Corpus uteri",
    "C55": "Uterus, NOS",
    "C56": "Ovary",
    "C57": "Other and unspecified female genital organs",
    "C58": "Placenta",
    "C60": "Penis",
    "C61": "Prostate gland",
    "C62": "Testis",
    "C63": "Other and unspecified male genital organs",
    "C64": "Kidney",
    "C65": "Renal pelvis",
    "C66": "Ureter",
    "C67": "Bladder",
    "C68": "Other and unspecified urinary organs",
    "C69": "Eye and adnexa",
    "C70": "Meninges",
    "C71": "Brain",
    "C72": "Spinal cord, cranial nerves, and other parts of CNS",
    "C73": "Thyroid gland",
    "C74": "Adrenal gland",
    "C75": "Other endocrine glands and related structures",
    "C76": "Other and ill-defined sites",
    "C77": "Lymph nodes",
    "C80": "Unknown primary site",
}

_TOPOGRAPHY_EXACT_DESCRIPTIONS: dict[str, str] = {
    "C15.0": "Cervical esophagus",
    "C15.1": "Thoracic esophagus",
    "C15.2": "Abdominal esophagus",
    "C15.3": "Upper third of esophagus",
    "C15.4": "Middle third of esophagus",
    "C15.5": "Lower third of esophagus",
    "C15.8": "Overlapping lesion of esophagus",
    "C15.9": "Esophagus, NOS",
    "C16.0": "Cardia, NOS",
    "C16.1": "Fundus of stomach",
    "C16.2": "Body of stomach",
    "C16.3": "Gastric antrum",
    "C16.4": "Pylorus",
    "C16.5": "Lesser curvature of stomach, NOS",
    "C16.6": "Greater curvature of stomach, NOS",
    "C16.8": "Overlapping lesion of stomach",
    "C16.9": "Stomach, NOS",
    "C17.0": "Duodenum",
    "C17.1": "Jejunum",
    "C17.2": "Ileum",
    "C17.3": "Meckel diverticulum",
    "C17.8": "Overlapping lesion of small intestine",
    "C17.9": "Small intestine, NOS",
    "C18.0": "Cecum",
    "C18.1": "Appendix",
    "C18.2": "Ascending colon",
    "C18.3": "Hepatic flexure of colon",
    "C18.4": "Transverse colon",
    "C18.5": "Splenic flexure of colon",
    "C18.6": "Descending colon",
    "C18.7": "Sigmoid colon",
    "C18.8": "Overlapping lesion of colon",
    "C18.9": "Colon, NOS",
    "C19.9": "Rectosigmoid junction",
    "C20.9": "Rectum, NOS",
    "C21.0": "Anus, NOS",
    "C21.1": "Anal canal",
    "C21.2": "Cloacogenic zone",
    "C21.8": "Overlapping lesion of rectum, anus and anal canal",
    "C22.0": "Liver",
    "C22.1": "Intrahepatic bile duct",
    "C22.9": "Liver, NOS",
    "C25.0": "Head of pancreas",
    "C25.1": "Body of pancreas",
    "C25.2": "Tail of pancreas",
    "C25.3": "Pancreatic duct",
    "C25.4": "Islets of Langerhans",
    "C25.7": "Other specified parts of pancreas",
    "C25.8": "Overlapping lesion of pancreas",
    "C25.9": "Pancreas, NOS",
    "C34.0": "Main bronchus",
    "C34.1": "Upper lobe, lung",
    "C34.2": "Middle lobe, lung",
    "C34.3": "Lower lobe, lung",
    "C34.8": "Overlapping lesion of lung",
    "C34.9": "Lung, NOS",
    "C44.0": "Skin of lip, NOS",
    "C44.1": "Eyelid",
    "C44.2": "External ear",
    "C44.3": "Skin of other and unspecified parts of face",
    "C44.4": "Skin of scalp and neck",
    "C44.5": "Skin of trunk",
    "C44.6": "Skin of upper limb and shoulder",
    "C44.7": "Skin of lower limb and hip",
    "C44.8": "Overlapping lesion of skin",
    "C44.9": "Skin, NOS",
    "C50.0": "Nipple",
    "C50.1": "Central portion of breast",
    "C50.2": "Upper-inner quadrant of breast",
    "C50.3": "Lower-inner quadrant of breast",
    "C50.4": "Upper-outer quadrant of breast",
    "C50.5": "Lower-outer quadrant of breast",
    "C50.6": "Axillary tail of breast",
    "C50.8": "Overlapping lesion of breast",
    "C50.9": "Breast, NOS",
    "C53.0": "Endocervix",
    "C53.1": "Exocervix",
    "C53.8": "Overlapping lesion of cervix uteri",
    "C53.9": "Cervix uteri",
    "C54.0": "Isthmus uteri",
    "C54.1": "Endometrium",
    "C54.2": "Myometrium",
    "C54.3": "Fundus uteri",
    "C54.8": "Overlapping lesion of corpus uteri",
    "C54.9": "Corpus uteri",
    "C56.9": "Ovary",
    "C61.9": "Prostate gland",
    "C62.0": "Undescended testis",
    "C62.1": "Descended testis",
    "C62.9": "Testis, NOS",
    "C64.9": "Kidney, NOS",
    "C65.9": "Renal pelvis",
    "C67.0": "Trigone of bladder",
    "C67.1": "Dome of bladder",
    "C67.2": "Lateral wall of bladder",
    "C67.3": "Anterior wall of bladder",
    "C67.4": "Posterior wall of bladder",
    "C67.5": "Bladder neck",
    "C67.6": "Ureteric orifice",
    "C67.7": "Urachus",
    "C67.8": "Overlapping lesion of bladder",
    "C67.9": "Bladder, NOS",
    "C70.0": "Cerebral meninges",
    "C70.1": "Spinal meninges",
    "C70.9": "Meninges, NOS",
    "C71.0": "Cerebrum",
    "C71.1": "Frontal lobe",
    "C71.2": "Temporal lobe",
    "C71.3": "Parietal lobe",
    "C71.4": "Occipital lobe",
    "C71.5": "Ventricle, NOS",
    "C71.6": "Cerebellum, NOS",
    "C71.7": "Brain stem",
    "C71.8": "Overlapping lesion of brain",
    "C71.9": "Brain, NOS",
    "C73.9": "Thyroid gland",
}

_HISTOLOGY_DESCRIPTIONS: dict[str, str] = {
    "8000": "Neoplasm, malignant",
    "8001": "Tumor cells, malignant",
    "8010": "Carcinoma, NOS",
    "8020": "Carcinoma, undifferentiated, NOS",
    "8041": "Small cell carcinoma, NOS",
    "8046": "Non-small cell carcinoma",
    "8050": "Papillary carcinoma, NOS",
    "8070": "Squamous cell carcinoma, NOS",
    "8071": "Squamous cell carcinoma, keratinizing, NOS",
    "8072": "Squamous cell carcinoma, large cell, nonkeratinizing, NOS",
    "8077": "Squamous intraepithelial neoplasia, grade III",
    "8120": "Transitional cell carcinoma, NOS",
    "8130": "Papillary transitional cell carcinoma",
    "8140": "Adenocarcinoma, NOS",
    "8210": "Adenocarcinoma in adenomatous polyp",
    "8230": "Solid carcinoma, NOS",
    "8240": "Carcinoid tumor, NOS",
    "8246": "Neuroendocrine carcinoma, NOS",
    "8250": "Bronchiolo-alveolar adenocarcinoma, NOS",
    "8255": "Adenocarcinoma with mixed subtypes",
    "8260": "Papillary adenocarcinoma, NOS",
    "8310": "Clear cell adenocarcinoma, NOS",
    "8312": "Renal cell carcinoma, NOS",
    "8323": "Mixed cell adenocarcinoma",
    "8380": "Endometrioid adenocarcinoma, NOS",
    "8441": "Serous cystadenocarcinoma, NOS",
    "8460": "Papillary serous cystadenocarcinoma",
    "8480": "Mucinous adenocarcinoma",
    "8490": "Signet ring cell carcinoma",
    "8500": "Infiltrating duct carcinoma, NOS",
    "8502": "Secretory carcinoma of breast",
    "8520": "Lobular carcinoma, NOS",
    "8522": "Infiltrating duct and lobular carcinoma",
    "8550": "Acinar cell carcinoma",
    "8720": "Malignant melanoma, NOS",
    "8770": "Mixed epithelioid and spindle cell melanoma",
    "8800": "Sarcoma, NOS",
    "8890": "Leiomyosarcoma, NOS",
    "8936": "Gastrointestinal stromal tumor",
    "9061": "Seminoma, NOS",
    "9070": "Embryonal carcinoma, NOS",
    "9080": "Teratoma, malignant, NOS",
    "9100": "Choriocarcinoma, NOS",
    "9380": "Glioma, malignant",
    "9400": "Astrocytoma, NOS",
    "9440": "Glioblastoma, NOS",
    "9470": "Medulloblastoma, NOS",
    "9530": "Meningioma, NOS",
    "9590": "Malignant lymphoma, NOS",
    "9670": "Small lymphocytic lymphoma, NOS",
    "9680": "Diffuse large B-cell lymphoma, NOS",
    "9732": "Plasma cell myeloma",
    "9823": "B-cell chronic lymphocytic leukemia/small lymphocytic lymphoma",
    "9861": "Acute myeloid leukemia, NOS",
}


DiagnosisSummaryRecord = tuple[
    str,
    int,
    TumorCandidate | None,
    Mapping[int, ExtractionResult],
]


def write_diagnosis_summary_csv(
    records: Sequence[DiagnosisSummaryRecord],
    csv_path: str | Path,
    dictionary: Any,
) -> None:
    """Write one high-level CSV row for each diagnosis/tumor record."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for patient_id, _tumor_index, tumor, record in records:
            writer.writerow(build_diagnosis_summary_row(patient_id, tumor, record, dictionary))

    logger.info("Diagnosis summary CSV written to %s", path)


def build_diagnosis_summary_row(
    patient_id: str,
    tumor: TumorCandidate | None,
    record: Mapping[int, ExtractionResult],
    dictionary: Any,
) -> dict[str, str]:
    """Build the output row for a single diagnosis/tumor record."""
    primary_site_result = record.get(PRIMARY_SITE)
    histology_result = record.get(HISTOLOGY_ICD_O_3)
    summary_stage_raw = _raw_result_value(record.get(SUMMARY_STAGE_2018))
    summary_stage_code = _public_code_value(summary_stage_raw)

    row: dict[str, str] = {
        "patient_id": patient_id,
        "diagnosis_date": _public_code_value(_raw_result_value(record.get(DATE_OF_DIAGNOSIS))),
        "primary_site_code": _public_code_value(_raw_result_value(primary_site_result)),
        "primary_site_text": _primary_site_text(primary_site_result, tumor, dictionary),
        "histology_code": _public_code_value(_raw_result_value(histology_result)),
        "histology_text": _histology_text(histology_result, tumor, dictionary),
        "seer_summary_stage_code": summary_stage_code,
        "seer_summary_stage_text": _code_description(
            dictionary,
            SUMMARY_STAGE_2018,
            summary_stage_raw,
        ),
    }

    for field_name, item_number in AJCC_FIELDS:
        raw_value = _raw_result_value(record.get(item_number))
        row[f"{field_name}_code"] = _public_code_value(raw_value)
        row[f"{field_name}_text"] = _ajcc_text(dictionary, item_number, raw_value)

    return row


def _raw_result_value(result: ExtractionResult | None) -> str:
    if not isinstance(result, ExtractionResult):
        return ""
    value = result.resolved_code or result.extracted_value
    return str(value).strip()


def _public_code_value(value: str) -> str:
    text = str(value).strip()
    if text.lower() in _BLANK_CODE_VALUES:
        return ""
    return text


def _primary_site_text(
    result: ExtractionResult | None,
    tumor: TumorCandidate | None,
    dictionary: Any,
) -> str:
    raw_value = _raw_result_value(result)
    return (
        _code_description(dictionary, PRIMARY_SITE, raw_value)
        or _topography_description(raw_value)
        or _non_code_extracted_value(result, PRIMARY_SITE)
        or _clean_text(getattr(tumor, "primary_site_hint", "") if tumor else "")
    )


def _histology_text(
    result: ExtractionResult | None,
    tumor: TumorCandidate | None,
    dictionary: Any,
) -> str:
    raw_value = _raw_result_value(result)
    histology_code = _normalize_histology_code(raw_value)
    tumor_histology = getattr(tumor, "histology", "") if tumor else ""
    cancer_type = getattr(tumor, "cancer_type", "") if tumor else ""
    return (
        _code_description(dictionary, HISTOLOGY_ICD_O_3, raw_value)
        or _HISTOLOGY_DESCRIPTIONS.get(histology_code, "")
        or _non_code_extracted_value(result, HISTOLOGY_ICD_O_3)
        or _clean_text(tumor_histology)
        or _clean_text(cancer_type)
    )


def _ajcc_text(dictionary: Any, item_number: int, raw_value: str) -> str:
    description = _code_description(dictionary, item_number, raw_value)
    if description:
        return description

    value = _public_code_value(raw_value)
    if not value:
        return ""

    # AJCC v8+ NAACCR items store the stage category text itself
    # (for example cT1 or pN0).  In a clinical/pathologic column, the
    # c/p basis prefix is redundant, so the display text drops it.
    return _AJCC_BASIS_PREFIX_RE.sub("", value).strip()


def _code_description(dictionary: Any, item_number: int, value: str) -> str:
    if dictionary is None:
        return ""

    candidates = _code_lookup_candidates(value)
    try:
        entries = dictionary.get_codes(item_number)
    except Exception:
        return ""

    for entry in entries:
        code = str(getattr(entry, "code", "")).strip()
        for candidate in candidates:
            if code == candidate or code.lower() == candidate.lower():
                return _clean_text(getattr(entry, "description", ""))

    return ""


def _code_lookup_candidates(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    candidates = [text]
    if text.lower() == "blank":
        candidates.extend(["Blank", "blank", ""])
    return list(dict.fromkeys(candidates))


def _topography_description(value: str) -> str:
    code = _normalize_site_code(value)
    if not code:
        return ""
    return (
        _TOPOGRAPHY_EXACT_DESCRIPTIONS.get(code, "")
        or _TOPOGRAPHY_PREFIX_DESCRIPTIONS.get(code[:3], "")
    )


def _normalize_site_code(value: str) -> str:
    match = _SITE_CODE_RE.search(str(value).strip())
    if not match:
        return ""

    code = match.group(0).upper().replace(" ", "")
    if "." in code:
        prefix, subsite = code.split(".", 1)
        return f"{prefix}.{subsite[:1]}"
    if len(code) == 4:
        return f"{code[:3]}.{code[3]}"
    return code[:3]


def _normalize_histology_code(value: str) -> str:
    match = _HISTOLOGY_CODE_RE.search(str(value).strip())
    return match.group(1) if match else ""


def _non_code_extracted_value(
    result: ExtractionResult | None,
    item_number: int,
) -> str:
    if result is None:
        return ""
    extracted = str(result.extracted_value or "").strip()
    if not extracted:
        return ""
    if item_number == PRIMARY_SITE and _normalize_site_code(extracted):
        return ""
    if item_number == HISTOLOGY_ICD_O_3 and _normalize_histology_code(extracted):
        return ""
    resolved = str(result.resolved_code or "").strip()
    if extracted == resolved:
        return ""
    return _clean_text(extracted)


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip(" *")
