"""FastAPI service: 30-day readmission risk for one discharged diabetic patient.

The request holds the raw fields of the hospital record (the original ICD-9 codes, not the groups); the service applies
the same clean() and merge_rare() steps as training before scoring.

Run locally from the repository root: uv run uvicorn diabetes_readmission.predict:app --port 9696
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from diabetes_readmission.features import clean, merge_rare, to_dicts

ROOT = Path(__file__).resolve().parents[2]  # src/diabetes_readmission/predict.py -> repository root
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.bin"))

with MODEL_PATH.open("rb") as f:
    bundle = pickle.load(f)
PIPELINE = bundle["pipeline"]
KEPT_CATEGORIES = bundle["kept_categories"]
MODEL_SHA256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

Drug = Literal["No", "Steady", "Up", "Down"]  # dosage change during the stay
Age = Literal[
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]

# A real test-set patient, shown as the example request in /docs
EXAMPLE = {
    "race": "Caucasian", "age": "[60-70)",
    "admission_type_id": 2, "discharge_disposition_id": 1, "admission_source_id": 4,
    "time_in_hospital": 3, "medical_specialty": None,
    "num_lab_procedures": 21, "num_procedures": 6, "num_medications": 33,
    "number_outpatient": 2, "number_emergency": 1, "number_inpatient": 2, "number_diagnoses": 9,
    "max_glu_serum": "None", "A1Cresult": "None",
    "metformin": "No", "repaglinide": "No", "glimepiride": "No", "glipizide": "No", "glyburide": "No",
    "pioglitazone": "No", "rosiglitazone": "No", "insulin": "Steady",
    "change": "No", "diabetesMed": "Yes",
    "diag_1": "997", "diag_2": "250.42", "diag_3": "403",
}


class Patient(BaseModel):
    """One hospital encounter, with the values and codes of the UCI dataset (see IDS_mapping.csv for the IDs)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})  # unknown fields -> 422

    race: Literal["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"] | None = None
    age: Age
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int = Field(ge=1, le=14)  # days
    medical_specialty: str | None = None  # admitting physician's specialty, null if not recorded
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(ge=0)  # visits in the year before the encounter
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    number_diagnoses: int = Field(ge=1)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"]  # "None" = test not done
    A1Cresult: Literal["None", "Norm", ">7", ">8"]
    metformin: Drug
    repaglinide: Drug
    glimepiride: Drug
    glipizide: Drug
    glyburide: Drug
    pioglitazone: Drug
    rosiglitazone: Drug
    insulin: Drug
    change: Literal["No", "Ch"]  # any diabetes medication changed
    diabetesMed: Literal["Yes", "No"]
    diag_1: str | None = None  # ICD-9 codes, e.g. "250.83"; null if not recorded
    diag_2: str | None = None
    diag_3: str | None = None


app = FastAPI(title="Diabetes 30-day readmission risk")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_sha256": MODEL_SHA256}


@app.post("/predict")
def predict(patient: Patient) -> dict[str, float]:
    # The same preparation as training: fixed rules, then the kept-category lists learned at training time
    data = merge_rare(clean(pd.DataFrame([patient.model_dump()])), KEPT_CATEGORIES)
    probability = float(PIPELINE.predict_proba(to_dicts(data))[0, 1])
    return {"readmission_probability": round(probability, 4)}
