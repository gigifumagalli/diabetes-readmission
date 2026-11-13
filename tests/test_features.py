import pandas as pd
import pytest

from diabetes_readmission.features import clean, icd9_group, merge_rare


@pytest.mark.parametrize("code, group", [
    ("428", "Circulatory"), ("785.5", "Circulatory"),  # 785 is circulatory although outside 390-459
    ("250.83", "Diabetes"), ("486", "Respiratory"), ("V57", "Other"), ("E888", "Other"),
    ("996", "Injury"), (None, "Missing"),
])
def test_icd9_group(code, group):
    assert icd9_group(code) == group


def test_clean_marks_unused_diagnosis_slots_as_absent():
    # One diagnosis recorded: slots 2 and 3 are empty by design, not missing
    patient = pd.DataFrame([{
        "diag_1": "428", "diag_2": None, "diag_3": None, "number_diagnoses": 1,
        "race": None, "medical_specialty": None,
        "admission_type_id": 6, "discharge_disposition_id": 1, "admission_source_id": 7,
    }])
    row = clean(patient).iloc[0]
    assert (row["diag_2_group"], row["diag_3_group"]) == ("Absent", "Absent")
    assert (row["race"], row["admission_type_id"], row["discharge_disposition_id"]) == ("Missing", "Unknown", "1")


def test_merge_rare_sends_unseen_categories_to_other():
    data = pd.DataFrame({"medical_specialty": ["Cardiology", "Dentistry"]})
    merged = merge_rare(data, {"medical_specialty": ["Cardiology"]})
    assert merged["medical_specialty"].tolist() == ["Cardiology", "Other"]
