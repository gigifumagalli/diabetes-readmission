"""Data preparation shared by train.py and predict.py.

The same rules as notebooks/02_modelling.ipynb (sections 1 and 2), where each decision is explained. Training and the
service both import from here, so a patient sent to the API is prepared exactly like the patients the model learned from.
"""

from pathlib import Path

import pandas as pd

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

# Discharged as expired (11, 19, 20, 21) or to hospice (13, 14)
EXCLUDED_DISCHARGE = [11, 13, 14, 19, 20, 21]

# Codes that mean "not available / NULL / not mapped" in IDS_mapping.csv
UNKNOWN_CODES = {
    "admission_type_id": [5, 6, 8],
    "discharge_disposition_id": [18, 25, 26],
    "admission_source_id": [9, 15, 17, 20, 21],
}

# Categories with fewer training patients than this become "Other"
MIN_PATIENTS = 100
RARE_COLS = ["medical_specialty", *UNKNOWN_CODES]  # unpacking a dict gives its keys

# The 29 inputs of the final model (the reduced list of section 2.3)
FEATURES = [
    "race", "age", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "medical_specialty", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "glimepiride", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "insulin",
    "change", "diabetesMed",
    "diag_1_group", "diag_2_group", "diag_3_group",
]

TARGET = "target"


def load_raw(path: Path) -> pd.DataFrame:
    """Read diabetic_data.csv with the dataset's missing-value conventions."""
    return pd.read_csv(
        path,
        na_values=["?"],        # the dataset's only missing-value token
        keep_default_na=False,  # keep "None" (test not measured) as a category
        low_memory=False,
    )


def scope(df_raw: pd.DataFrame) -> pd.DataFrame:
    """First encounter per patient, no death / hospice, known gender; add the 0/1 target."""
    df = df_raw.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE)]
    df = df[df["gender"] != "Unknown/Invalid"].reset_index(drop=True)

    # 1 = readmitted within 30 days; ">30" and "NO" both become 0
    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns="readmitted")


def icd9_group(code: str | float | None) -> str:
    """Map an ICD-9 code to one of the 9 disease groups of Strack et al. (2014)."""
    if pd.isna(code):
        return "Missing"
    if code[0] in "VE":  # supplementary (V) and external-cause (E) codes
        return "Other"
    value = float(code)
    chapter = int(value)  # integer part: "786.05" -> 786
    if 390 <= value < 460 or chapter == 785:
        return "Circulatory"
    if 460 <= value < 520 or chapter == 786:
        return "Respiratory"
    if 520 <= value < 580 or chapter == 787:
        return "Digestive"
    if chapter == 250:
        return "Diabetes"
    if 800 <= value < 1000:
        return "Injury"
    if 710 <= value < 740:
        return "Musculoskeletal"
    if 580 <= value < 630 or chapter == 788:
        return "Genitourinary"
    if 140 <= value < 240:
        return "Neoplasms"
    return "Other"


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed rules: diagnosis groups, missing labels, unknown ID codes. Nothing is learned from the data."""
    data = data.copy()  # leave the caller's DataFrame untouched

    # Empty code beyond number_diagnoses = no such diagnosis -> "Absent" instead of "Missing"
    for position, col in enumerate(DIAG_COLS, start=1):
        absent = data[col].isna() & (data["number_diagnoses"] < position)
        data[f"{col}_group"] = data[col].map(icd9_group).mask(absent, "Absent")

    data[["race", "medical_specialty"]] = data[["race", "medical_specialty"]].fillna("Missing")
    for col, codes in UNKNOWN_CODES.items():
        # Codes become text, so the model cannot treat them as quantities
        data[col] = data[col].astype(str).mask(data[col].isin(codes), "Unknown")
    return data


def fit_kept_categories(train: pd.DataFrame) -> dict[str, list[str]]:
    """Return, per column, the categories with at least MIN_PATIENTS patients in the training data."""
    kept = {}
    for col in RARE_COLS:
        counts = train[col].value_counts()
        kept[col] = counts[counts >= MIN_PATIENTS].index.tolist()
    return kept


def merge_rare(data: pd.DataFrame, kept: dict[str, list[str]]) -> pd.DataFrame:
    """Replace every category outside the kept lists with "Other"."""
    data = data.copy()
    for col, categories in kept.items():
        # .where keeps the value when the condition is true, otherwise writes "Other"
        data[col] = data[col].where(data[col].isin(categories), "Other")
    return data


def to_dicts(data: pd.DataFrame) -> list[dict]:
    """Select the model's inputs and return one dict per patient, the format DictVectorizer expects."""
    return data[FEATURES].to_dict(orient="records")
