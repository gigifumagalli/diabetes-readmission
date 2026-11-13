"""Train the final model and save it for the service.

Reproduces notebooks/02_modelling.ipynb section 4.6: the settings chosen in sections 3-4 are fitted on train + validation
(80% of the patients) and scored once on the held-out test set (20%). Hyperparameters are fixed here; they were tuned
with manual loops on a fixed validation set, as in the course. In production, GridSearchCV (cross-validated search
inside the training data) would be the standard way to tune them.

Run from the repository root: uv run python -m diabetes_readmission.train
"""

import argparse
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost import XGBClassifier

from diabetes_readmission.features import (
    FEATURES, TARGET, clean, fit_kept_categories, load_raw, merge_rare, scope, to_dicts,
)

ROOT = Path(__file__).resolve().parents[2]  # src/diabetes_readmission/train.py -> repository root
DEFAULT_DATA = ROOT / "data" / "raw" / "diabetic_data.csv"
DEFAULT_MODEL = ROOT / "models" / "model.bin"

XGB_PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 30}
SEED = 1  # the same seed as the notebooks, so the test set holds the same patients


def build_pipeline() -> Pipeline:
    """DictVectorizer (one-hot for text, numbers passed through) + the tuned XGBoost."""
    return make_pipeline(DictVectorizer(), XGBClassifier(**XGB_PARAMS, random_state=SEED, n_jobs=-1))


def train(data_path: Path, model_path: Path) -> None:
    df = scope(load_raw(data_path))
    df_full_train, df_test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)

    # The rare-category lists are learned on the training patients only; the test set just receives them
    full_train = clean(df_full_train)
    kept_categories = fit_kept_categories(full_train)
    full_train = merge_rare(full_train, kept_categories)
    test = merge_rare(clean(df_test), kept_categories)

    pipeline = build_pipeline()
    pipeline.fit(to_dicts(full_train), full_train[TARGET])
    test_auc = roc_auc_score(test[TARGET], pipeline.predict_proba(to_dicts(test))[:, 1])

    # The service needs both: the fitted pipeline and the lists merge_rare() applies to each request
    model_path.parent.mkdir(exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump({"pipeline": pipeline, "kept_categories": kept_categories}, f)

    metadata = {
        "model": "DictVectorizer + XGBClassifier",
        "params": XGB_PARAMS,
        "features": FEATURES,
        "train_patients": len(full_train),
        "test_patients": len(test),
        "test_auc": round(test_auc, 4),
        "kept_categories": kept_categories,
    }
    model_path.with_name("metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"trained on {len(full_train)} patients, test AUC {test_auc:.4f} on {len(test)} patients")
    print(f"saved {model_path.relative_to(ROOT)} and metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 30-day readmission model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    train(args.data, args.model)
