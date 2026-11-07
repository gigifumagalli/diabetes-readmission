# Diabetes 30-day readmission

Predict whether a diabetic patient will be readmitted to hospital **within 30 days** of discharge, from the data of the hospital stay. ML Zoomcamp 2026 midterm project.

> Work in progress. Problem framing, EDA, model and service are added as the project grows.

## Problem

Hospitals in the US are penalised for early readmissions (Hospital Readmissions Reduction Program), and a readmission within 30 days is often a sign that the patient left hospital without enough support.

**Use case.** On the day of discharge, the model gives each diabetic patient a probability of being readmitted within 30 days. The hospital's care-transition team (nurses, case managers, pharmacists) cannot call, visit or review the medication of every patient, so it uses the score to rank the day's discharges and spend its limited capacity on the patients most likely to come back.

- **Target:** `readmitted == '<30'` (binary; 9.0% of the encounters after scoping). Readmissions after 30 days count as negative.
- **Metric:** ROC AUC, because the team acts on a ranking and the classes are imbalanced (always predicting "no readmission" is already 91% accurate).
- **Features:** only information available at discharge (demographics, admission data, length of stay, lab results, medications and their changes during the stay, diagnoses).
- **Bar in practice:** hospitals already use simple clinical scores such as the LACE index for this task.

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

Rule 1 uses the lowest `encounter_id` as the first encounter (the data has no dates; the notebook checks that the id order is consistent with time order). The 3 rows with `gender` = `Unknown/Invalid` are also dropped. Result: **69,970 encounters**, one per patient (the paper reports 69,984; the notebook explains the difference).

## Limitations

- **The model predicts risk; it does not estimate the effect of any intervention.** It says who is likely to come back, not whether a follow-up call would prevent it. That is a causal question, the one the source paper addresses.
- **Readmission depends on factors that are not in the data** (support at home, adherence to treatment, socio-economic situation), which limits the achievable performance.
- **The data covers 1999–2008.** Clinical practice and coding have changed since (the US moved from ICD-9 to ICD-10 in 2015), so a real deployment would need retraining on recent data.

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
