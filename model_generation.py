# train_and_save.py
"""
Train/tune a pruned DecisionTree on preprocessed splits and save a single
inference bundle (model + scaler + metadata) for FastAPI.

Prereqs:
1) You have already run your preprocessing script to produce:
   - train_set_processed.csv
   - validation_set_processed.csv
   - test_set_processed.csv
   and you've saved the fitted RobustScaler:
   - robust_scaler.joblib   (dump(scaler, ".../robust_scaler.joblib"))

2) Activate your venv and run:
   python train_and_save.py
"""

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier


def resolve_data_dir() -> Path:
    """Prefer ./FastAPILab/FastAPI_Nicholas, else fall back to script dir."""
    here = Path(__file__).resolve().parent
    candidate = here / "FastAPILab" / "FastAPI_Nicholas"
    if (candidate / "train_set_processed.csv").exists():
        return candidate
    return here


def main():
    data_dir = resolve_data_dir()
    print(f"[INFO] Using data dir: {data_dir}")

    # Files
    TRAIN = data_dir / "train_set_processed.csv"
    VAL = data_dir / "validation_set_processed.csv"
    TEST = data_dir / "test_set_processed.csv"
    SCALER_PATH = data_dir / "robust_scaler.joblib"   # must exist (saved during preprocessing)
    ARTIFACTS_PATH = data_dir / "artifacts_dt_pruned.joblib"
    REPORT_PATH = data_dir / "test_classification_report.json"
    CM_PATH = data_dir / "test_confusion_matrix.json"

    # Sanity checks
    for p in [TRAIN, VAL, TEST]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p} (did you run preprocessing and point paths correctly?)")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing scaler at {SCALER_PATH}\n"
            "Please dump the fitted RobustScaler in your preprocessing step:\n"
            "    from joblib import dump\n"
            "    dump(scaler, 'FastAPILab/FastAPI_Nicholas/robust_scaler.joblib')"
        )

    # Load data
    train_df = pd.read_csv(TRAIN)
    val_df = pd.read_csv(VAL)
    test_df = pd.read_csv(TEST)

    # Split features/targets
    X_train = train_df.drop(columns=["Response"])
    y_train = train_df["Response"]
    X_val = val_df.drop(columns=["Response"])
    y_val = val_df["Response"]
    X_test = test_df.drop(columns=["Response"])
    y_test = test_df["Response"]

    # Ensure consistent feature schema
    if list(X_train.columns) != list(X_val.columns) or list(X_train.columns) != list(X_test.columns):
        raise ValueError("Feature columns differ across splits. Ensure preprocessing produced aligned columns.")

    feature_names = X_train.columns.tolist()

    # Hyperparameter grid (your settings)
    max_depths = [5, 10, 15, 20, None]
    min_leafs = [1, 10, 100, 500]
    min_splits = [2, 10, 100, 500]

    best_model = None
    best_f1 = -1.0
    best_params = {}
    best_times = {}

    print("Tuning Decision Tree with pruning (using validation F1)...\n")
    for max_depth, min_leaf, min_split in product(max_depths, min_leafs, min_splits):
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            min_samples_split=min_split,
            random_state=42
        )

        # Train
        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()
        train_time = end_train - start_train

        # Validate
        start_test = time.time()
        y_val_pred = model.predict(X_val)
        end_test = time.time()
        test_time = end_test - start_test

        f1 = f1_score(y_val, y_val_pred)

        print(
            f"max_depth={max_depth}, min_samples_leaf={min_leaf}, min_samples_split={min_split} "
            f"→ F1={f1:.4f} | Train: {train_time:.4f}s | Test: {test_time:.4f}s"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = {
                "max_depth": max_depth,
                "min_samples_leaf": min_leaf,
                "min_samples_split": min_split
            }
            best_times = {
                "train_time": train_time,
                "test_time": test_time,
                "train_start": start_train,
                "train_end": end_train,
                "test_start": start_test,
                "test_end": end_test
            }

    # Final test evaluation
    y_test_pred = best_model.predict(X_test)
    report_dict = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
    cm = confusion_matrix(y_test, y_test_pred)

    with open(REPORT_PATH, "w") as f:
        json.dump(report_dict, f, indent=2)
    with open(CM_PATH, "w") as f:
        json.dump({"labels": [0, 1], "matrix": cm.tolist()}, f, indent=2)

    print("\n[RESULT] Best params:", best_params)
    print(f"[RESULT] Best Val F1: {best_f1:.4f}")
    print(f"[INFO] Wrote test report → {REPORT_PATH.name}")
    print(f"[INFO] Wrote confusion matrix → {CM_PATH.name}")

    # Load the exact RobustScaler used during preprocessing
    scaler = load(SCALER_PATH)

    # Persist bundle for FastAPI
    # These mirror your original (raw) preprocessing so the API can transform raw inputs:
    columns_to_drop = ['Previously_Insured', 'Driving_License', 'id', 'Age']
    categorical_maps = {
        "Gender": {"Male": 1, "Female": 0},
        "Vehicle_Damage": {"Yes": 1, "No": 0},
        "Vehicle_Age": {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
    }
    columns_to_scale = ['Annual_Premium', 'Vintage']

    # Version info (helps with portability)
    versions = {}
    try:
        import sklearn, imblearn, numpy, pandas
        versions = {
            "python": sys.version,
            "sklearn": sklearn.__version__,
            "imblearn": imblearn.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__
        }
    except Exception:
        pass

    bundle = {
        "model": best_model,
        "best_params": best_params,
        "best_val_f1": float(best_f1),
        "best_times": best_times,
        "feature_names": feature_names,      # exact order the model expects
        "columns_to_drop": columns_to_drop,  # raw-input drops
        "categorical_maps": categorical_maps,
        "columns_to_scale": columns_to_scale,
        "scaler": scaler,
        "versions": versions,
        "created_at": time.time(),
        "model_type": "DecisionTreeClassifier",
    }

    dump(bundle, ARTIFACTS_PATH, compress=3)
    print(f"[INFO] Saved artifacts → {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
