"""Evaluate the selected logistic regression model on DS2 beats.
Add --noise to evaluate the same model with NSTDB electrode-motion noise at 0 dB.
"""
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from baseline_logreg import DATA_PATH, OUTPUT_DIR
from io_utils import CFG, CLASSES
from metrics import format_report, summary


def main(use_noise=False):
    # 1. Load the model that won the validation comparison.
    model_path = OUTPUT_DIR / "model.joblib"
    selection_path = OUTPUT_DIR / "selection.json"
    test_dir = OUTPUT_DIR / ("test_em_0db" if use_noise else "test")
    if test_dir.exists():
        raise FileExistsError(f"Test outputs already exist in {test_dir}")
    if not model_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("Run baseline_logreg.py to select and save a model first")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    model = joblib.load(BytesIO(model_bytes))
    if model[-1].C != selection["selected_C"]:
        raise ValueError("The saved model's C differs from selection.json")
    if not np.array_equal(model.classes_, np.arange(len(CLASSES))):
        raise ValueError("Expected a saved model with classes N, S, V and F")

    # 2. Select the test rows from the existing record split
    if not DATA_PATH.is_file():
        raise FileNotFoundError("Run run_prepare.py to prepare beats.npz first")
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("subset", "record", "windows", "labels")}
    test_rows = np.flatnonzero(data["subset"] == "test")
    records = data["record"][test_rows]
    y_test = data["labels"][test_rows]

    test_records = set(records.tolist())
    development_records = set(selection["train_records"]) | set(selection["validation_records"])
    if test_records != set(CFG["data"]["ds2"]) or test_records & development_records:
        raise ValueError("Test rows must match DS2 and have no records in the development set")
    if not np.isin(y_test, model.classes_).all():
        raise ValueError("Test labels must follow the project's N/S/V/F mapping")

    condition = "clean"
    if use_noise:
        from ecg_noise import noisy_windows

        # Pair the noisy run with the clean predictions from this exact model
        clean_metrics_path = OUTPUT_DIR / "test" / "metrics.json"
        if not clean_metrics_path.is_file():
            raise FileNotFoundError("Run the clean evaluation before adding --noise")
        clean_metrics = json.loads(clean_metrics_path.read_text(encoding="utf-8"))
        if clean_metrics["model_sha256"] != model_sha256:
            raise ValueError("The clean test outputs belong to a different model.joblib")
        X_test, noise_details = noisy_windows(data, test_rows, "em", 0.0, selection["seed"])
        condition = "electrode motion at 0 dB raw-signal SNR"
    else:
        X_test = data["windows"][test_rows]

    # 3. Predict with the fitted pipeline and score against the test labels
    with threadpool_limits(limits=1):
        probabilities = model.predict_proba(X_test)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    scores = summary(y_test, predictions)
    metrics = {
        "model": "logistic regression",
        "selected_C": selection["selected_C"],
        "selection_metric": selection["selection_metric"],
        "validation_selection_score": selection["selection_score"],
        "split": "DS2",
        "condition": condition,
        "test_records": sorted(test_records),
        "test_beats": len(test_rows),
        "classifier_parameters": int(model[-1].coef_.size + model[-1].intercept_.size),
        "model_sha256": model_sha256,
        **scores,
    }
    if use_noise:
        metrics["noise_details"] = noise_details

    # 4. Save the scores and matching prediction rows together
    # The noisy run also saves the waveforms used for inference
    waveform_data = {"windows": X_test} if use_noise else {}
    with TemporaryDirectory(dir=OUTPUT_DIR) as temp_dir:
        staged = Path(temp_dir) / "test"
        staged.mkdir()
        np.savez_compressed(
            staged / "predictions.npz",
            dataset_row=test_rows,
            y_true=y_test,
            y_pred=predictions,
            probabilities=probabilities,
            record=records,
            classes=np.array(CLASSES),
            **waveform_data,
        )
        (staged / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staged.rename(test_dir)

    print(f"\nDS2 TEST | {condition} | C={selection['selected_C']:g}")
    print(f"{len(test_rows):,} beats from {len(test_records)} records\n")
    print(format_report(y_test, predictions))
    print(f"\nSaved test metrics and predictions in {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", action="store_true",
                        help="Add NSTDB electrode-motion noise at 0 dB")
    main(use_noise=parser.parse_args().noise)
