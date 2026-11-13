from fastapi.testclient import TestClient

from diabetes_readmission.predict import EXAMPLE, app

client = TestClient(app)  # calls the app in-process, no server needed


def test_example_patient_matches_the_notebook():
    # The first test-set patient scored 0.1562 in notebooks/02_modelling.ipynb (4.6)
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    assert response.json() == {"readmission_probability": 0.1562}


def test_unseen_specialty_is_scored_as_other():
    response = client.post("/predict", json={**EXAMPLE, "medical_specialty": "NotASpecialty"})
    assert response.status_code == 200


def test_invalid_input_is_rejected():
    assert client.post("/predict", json={**EXAMPLE, "age": "65"}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "gender": "Male"}).status_code == 422  # not a model input
