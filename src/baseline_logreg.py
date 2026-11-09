"""Compare logistic regression settings using the existing validation split
Uses the existing prepared dataset and record split. No test evaluation occurs here.
"""
import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from io_utils import CFG, CLASSES
from metrics import format_report, summary

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/processed/beats.npz"
OUTPUT_DIR = ROOT / "results/baseline_logreg_search"
SEED = CFG["reproducibility"]["seed"]
# Fix the candidates and selection rule before running the comparison
C_VALUES = (0.01, 0.1, 1.0, 10.0)
SELECTION_METRIC = "macro_f1_nsv"


def main():
    # 1. Load the dataset produced by Task 1
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Prepared data not found. From the project root, run: "
            "run_prepare.py"
        )

    with np.load(DATA_PATH, allow_pickle=False) as data:
        windows = data["windows"]   # One row = 256 samples describing one beat
        labels = data["labels"]     # 0=N, 1=S, 2=V, 3=F
        records = data["record"]    # Recording that each beat came from
        subsets = data["subset"]    # Existing train / val / test assignments

    # 2. Reuse the existing training and validation assignments
    # No train_test_split: that could mix beats from the same record
    train_rows = np.flatnonzero(subsets == "train")
    val_rows = np.flatnonzero(subsets == "val")
    X_train, y_train = windows[train_rows], labels[train_rows]
    X_val, y_val = windows[val_rows], labels[val_rows]

    train_records = set(records[train_rows].tolist())
    val_records = set(records[val_rows].tolist())
    expected_val = set(CFG["data"]["val"])
    expected_train = set(CFG["data"]["ds1"]) - expected_val
    if train_records != expected_train or val_records != expected_val:
        raise ValueError("Prepared train/validation records do not match configs/prep.yaml")
    if train_records & val_records:
        raise ValueError("A record appears in both training and validation")
    if X_train.shape[1] != 256 or X_val.shape[1] != 256:
        raise ValueError("Expected the shared 256-sample input used by CNN arm B2")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Training or validation inputs contain NaN or infinity")

    print(f"Training input:   {X_train.shape}; labels: {y_train.shape}")
    print(f"Validation input: {X_val.shape}; labels: {y_val.shape}")

    # 3. Compute class weights from TRAINING labels only
    counts = np.bincount(y_train, minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError("All four classes must be represented in training")
    weights = counts.sum() / counts.astype(float)
    weights = np.minimum(weights, 20 * weights.min())
    weights = weights / weights.mean()
    class_weights = {i: float(weight) for i, weight in enumerate(weights)}
    print("Training counts:", dict(zip(CLASSES, counts.tolist())))

    # 4. Train a fresh pipeline for each C using exactly the same data.
    # Smaller C = stronger regularization (more penalty on large weights).
    best_model = None
    best_score = -np.inf
    best_C = None
    best_predictions = None
    best_probabilities = None
    search_results = []

    print("\nComparing C values:", C_VALUES, flush=True)
    print("Selection metric: macro F1 over N/S/V (same as the CNN)", flush=True)
    for C in C_VALUES:
        print(f"\nTraining C={C:g} ...", flush=True)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=2000,
                tol=1e-5,
                class_weight=class_weights,
                random_state=SEED,
            ),
        )

        # Each scaler is fitted on TRAINING beats only, inside this pipeline
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            with threadpool_limits(limits=1):
                model.fit(X_train, y_train)
                probabilities = model.predict_proba(X_val)
        predictions = model.classes_[probabilities.argmax(axis=1)]
        converged = not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )

        # 5. Score each candidate on validation, using the shared metrics
        scores = summary(y_val, predictions)
        search_results.append({
            "C": C,
            "converged": converged,
            "iterations": int(model[-1].n_iter_.max()),
            "accuracy": scores["accuracy"],
            "macro_f1": scores["macro_f1"],
            "macro_f1_nsv": scores["macro_f1_nsv"],
        })
        print(
            f"  accuracy={scores['accuracy']:.4f} | "
            f"macro-F1(N/S/V/F)={scores['macro_f1']:.4f} | "
            f"macro-F1(N/S/V)={scores['macro_f1_nsv']:.4f} | "
            f"converged={converged}",
            flush=True,
        )

        # Exact ties keep the earlier (smaller C) candidate
        if converged and scores[SELECTION_METRIC] > best_score:
            best_score = scores[SELECTION_METRIC]
            best_C = C
            best_model = model
            best_predictions = predictions.copy()
            best_probabilities = probabilities.copy()

    if best_model is None:
        raise RuntimeError("No candidate converged. Resolve fitting on DS1 before testing.")

    # metric definitions: accuracy, macro F1, per-class
    # precision/recall/F1 and a confusion matrix
    print(f"\nSELECTED C={best_C:g} using validation {SELECTION_METRIC}")
    print("\nBEST MODEL: VALIDATION RESULTS\n")
    print(format_report(y_val, best_predictions))

    # 6. Save the selected TRAINING-fitted model and its validation outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, OUTPUT_DIR / "model.joblib")
    with open(OUTPUT_DIR / "validation_search.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(search_results[0]))
        writer.writeheader()
        writer.writerows(search_results)
    selection = {
        "selected_C": best_C,
        "selection_metric": SELECTION_METRIC,
        "selection_score": best_score,
        "candidate_C_values": list(C_VALUES),
        "seed": SEED,
        "class_weight_cap": 20,
        "class_weights": class_weights,
        "train_records": sorted(train_records),
        "validation_records": sorted(val_records),
        "test_evaluated_in_this_run": False,
    }
    with open(OUTPUT_DIR / "selection.json", "w", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2, allow_nan=False)
        stream.write("\n")
    np.savez_compressed(
        OUTPUT_DIR / "val_predictions.npz",
        dataset_row=val_rows,
        y_true=y_val,
        y_pred=best_predictions,
        probabilities=best_probabilities,
        record=records[val_rows],
        classes=np.array(CLASSES),
    )
    print(f"\nSaved selected model, comparison and validation predictions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
