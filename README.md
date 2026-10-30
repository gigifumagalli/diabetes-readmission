# Diabetes 30-day readmission

Predict whether a diabetic patient will be readmitted to hospital **within 30 days** of discharge, from the data of the hospital stay. ML Zoomcamp 2026 midterm project.

> Work in progress. Problem framing, EDA, model and service are added as the project grows.

## Problem

Hospitals in the US are penalised for early readmissions, and a readmission within 30 days is a cost for the hospital and a risk for the patient. A model that flags high-risk patients at discharge lets staff target follow-up calls, medication reviews or a later discharge at the patients who need it most.

- **Target:** `readmitted == '<30'` (binary; about 11% of encounters)
- **Metric:** ROC AUC (the classes are imbalanced, and the model ranks patients for follow-up)

## Data

[Diabetes 130-US Hospitals for years 1999–2008](https://archive.ics.uci.edu/dataset/296) (UCI Machine Learning Repository, dataset 296): 101,766 hospital encounters of 71,518 patients across 130 US hospitals, 50 columns.

- Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The raw files are committed in `data/raw/`.
- Source paper: Strack B. et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*, BioMed Research International, [doi:10.1155/2014/781670](https://doi.org/10.1155/2014/781670).

| File | Content |
| --- | --- |
| `data/raw/diabetic_data.csv` | one row per encounter |
| `data/raw/IDS_mapping.csv` | meaning of `admission_type_id`, `discharge_disposition_id`, `admission_source_id` |

To download them again: `uv run python scripts/download_data.py`.

### Scoping rules

Following the source paper:

1. **One encounter per patient**, so the same patient cannot appear in both training and validation data.
2. **Drop encounters that ended in death or hospice** (`discharge_disposition_id` 11, 13, 14, 19, 20, 21): these patients cannot be readmitted.
3. **Group the ICD-9 diagnosis codes** (`diag_1`, `diag_2`, `diag_3`) into about 9 disease categories.

## Project structure

```text
├── data/raw/          raw data (CC BY 4.0)
├── notebooks/         EDA and model selection
├── scripts/           data download
├── pyproject.toml     dependencies (uv)
└── uv.lock            locked versions
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # runtime + dev (notebook) dependencies
```
