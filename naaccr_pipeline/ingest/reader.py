from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single clinical document for a patient."""
    doc_index: int          # position in patient's document list
    date: str               # document date from input 'date' column
    text: str               # document text


@dataclass
class PatientDocumentSet:
    """All documents for a single patient, sorted by date."""
    patient_id: str
    documents: list[Document]
    structured_data: dict[int, str] = field(default_factory=dict)
    # Maps NAACCR item number -> pre-populated value from structured columns


# Mapping of known column name patterns to NAACCR item numbers
STRUCTURED_COLUMN_MAP = {
    # Column name pattern -> NAACCR item number
    'mrn': 2300,
    'medical_record_number': 2300,
    'last_name': 2230,
    'name_last': 2230,
    'first_name': 2240,
    'name_first': 2240,
    'middle_name': 2250,
    'name_middle': 2250,
    'dob': 240,
    'date_of_birth': 240,
    'birth_date': 240,
    'sex': 220,
    'gender': 220,
    'race': 160,
    'race1': 160,
    'ethnicity': 190,
    'hispanic': 190,
    'spanish_hispanic_origin': 190,
    'address': 2330,
    'street': 2330,
    'addr_at_dx': 2330,
    'city': 70,
    'state': 80,
    'zip': 100,
    'zip_code': 100,
    'postal_code': 100,
    'ssn': 2320,
    'social_security': 2320,
    'social_security_number': 2320,
    'marital_status': 150,
    'county': 90,
}


class DataReader:
    def __init__(self):
        pass

    def load(self, path: str) -> tuple[list[PatientDocumentSet], dict[str, int]]:
        """Load CSV or Parquet file, group by patient_id.

        Required columns: patient_id, date, text
        Optional: any structured demographic columns

        Returns:
            - list of PatientDocumentSet (one per patient, documents sorted by date)
            - structured_columns_map: {column_name: naaccr_item_number} for detected structured columns

        Steps:
        1. Load file (detect format from extension)
        2. Validate required columns exist
        3. Drop rows with null text
        4. Detect structured columns by matching column names against STRUCTURED_COLUMN_MAP
        5. Group by patient_id
        6. For each patient group:
           a. Sort documents by date
           b. Create Document objects
           c. Extract structured data from first row (demographics are same across rows)
           d. Create PatientDocumentSet
        """
        filepath = Path(path)
        logger.info("Loading data from %s", filepath)

        # 1. Load file based on extension
        ext = filepath.suffix.lower()
        if ext == '.parquet':
            df = pd.read_parquet(filepath)
        elif ext in ('.csv', '.tsv'):
            sep = '\t' if ext == '.tsv' else ','
            df = pd.read_csv(filepath, sep=sep, dtype=str)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. Expected .csv, .tsv, or .parquet"
            )

        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        # 2. Validate required columns exist
        required = {'patient_id', 'date', 'text'}
        # Case-insensitive column lookup
        col_map = {c.lower().strip(): c for c in df.columns}
        missing = required - set(col_map.keys())
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        # Normalize required column names so downstream code can rely on them
        rename = {}
        for req in required:
            actual = col_map[req]
            if actual != req:
                rename[actual] = req
        if rename:
            df = df.rename(columns=rename)

        # 3. Drop rows with null text
        before_count = len(df)
        df = df.dropna(subset=['text'])
        # Also drop rows where text is empty after stripping whitespace
        df = df[df['text'].str.strip().astype(bool)]
        dropped = before_count - len(df)
        if dropped:
            logger.warning("Dropped %d rows with null/empty text", dropped)

        if len(df) == 0:
            logger.warning("No rows remaining after filtering null text")
            return [], {}

        # 4. Detect structured columns
        structured_columns = self._detect_structured_columns(df)
        if structured_columns:
            logger.info(
                "Detected %d structured columns: %s",
                len(structured_columns),
                {col: item for col, item in structured_columns.items()},
            )

        # 5. Group by patient_id
        patient_sets = []
        grouped = df.groupby('patient_id', sort=True)

        for patient_id, group in grouped:
            # 6a. Sort documents by date
            group_sorted = group.sort_values('date', na_position='last')

            # 6b. Create Document objects
            documents = []
            for idx, (_, row) in enumerate(group_sorted.iterrows()):
                doc = Document(
                    doc_index=idx,
                    date=str(row['date']) if pd.notna(row['date']) else '',
                    text=str(row['text']),
                )
                documents.append(doc)

            # 6c. Extract structured data from first row
            structured_data: dict[int, str] = {}
            first_row = group_sorted.iloc[0]
            for col_name, naaccr_item in structured_columns.items():
                value = first_row.get(col_name)
                if pd.notna(value):
                    str_value = str(value).strip()
                    if str_value:
                        structured_data[naaccr_item] = str_value

            # 6d. Create PatientDocumentSet
            pds = PatientDocumentSet(
                patient_id=str(patient_id),
                documents=documents,
                structured_data=structured_data,
            )
            patient_sets.append(pds)

        logger.info(
            "Created %d patient document sets (%d total documents)",
            len(patient_sets),
            sum(len(ps.documents) for ps in patient_sets),
        )

        return patient_sets, structured_columns

    def _detect_structured_columns(self, df: pd.DataFrame) -> dict[str, int]:
        """Match DataFrame column names against known patterns.
        Returns {column_name: naaccr_item_number}."""
        result = {}
        for col in df.columns:
            normalized = col.lower().strip().replace(' ', '_').replace('-', '_')
            if normalized in STRUCTURED_COLUMN_MAP:
                result[col] = STRUCTURED_COLUMN_MAP[normalized]
        return result
