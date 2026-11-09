"""Plot representative ECG classification errors"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from io_utils import CFG, CLASSES

ROOT = Path(__file__).resolve().parent.parent


def representative_indices(mask, scores, record, n):
    """Select near-median scores and prefer distinct records"""
    candidates = np.flatnonzero(mask)
    if not len(candidates):
        return []
    median = np.median(scores[candidates])
    order = candidates[np.lexsort((candidates, np.abs(scores[candidates] - median)))]
    chosen, used_records = [], set()
    for i in order:
        if record[i] not in used_records:
            chosen.append(int(i))
            used_records.add(record[i])
            if len(chosen) == n:
                return chosen
    chosen.extend(int(i) for i in order if i not in chosen)
    return chosen[:n]


def load_predictions(path, d):
    with np.load(path, allow_pickle=False) as archive:
        p = {k: archive[k] for k in archive.files}
    required = {"dataset_row", "y_true", "y_pred", "probabilities", "record", "classes"}
    if not required.issubset(p) or tuple(p["classes"]) != CLASSES:
        raise ValueError(f"Predictions need {sorted(required)} in N/S/V/F order")
    rows = p["dataset_row"]
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(np.unique(rows)) != len(rows):
        raise ValueError("dataset_row must contain unique integer row IDs")
    if np.any(rows < 0) or np.any(rows >= len(d["labels"])):
        raise ValueError("Prediction row IDs are outside the dataset")
    for key, source in (("y_true", "labels"), ("record", "record")):
        if not np.array_equal(p[key], d[source][rows]):
            raise ValueError(f"Prediction {key} does not align with the dataset")
    proba = p["probabilities"]
    if proba.shape != (len(rows), 4) or not np.isfinite(proba).all() or np.any(proba < 0):
        raise ValueError("Expected finite nonnegative class scores with shape (beats, 4)")
    if not np.allclose(proba.sum(1), 1, atol=1e-5):
        raise ValueError("Class probabilities must sum to one")
    if not np.array_equal(proba.argmax(1), p["y_pred"]):
        raise ValueError("Predictions must be argmax of N/S/V/F class probabilities")
    if len(np.unique(d["subset"][rows])) != 1:
        raise ValueError("Analyze one split at a time")
    if "windows" in p and (p["windows"].shape != d["windows"][rows].shape or
                           not np.isfinite(p["windows"]).all()):
        raise ValueError("Noisy inputs must align with prediction rows")
    return p


def analyze(args):
    if args.out.exists():
        raise FileExistsError(f"Use a new output directory: {args.out}")
    with np.load(args.data, allow_pickle=False) as archive:
        d = {k: archive[k] for k in ("windows", "labels", "record", "subset", "rr_prev", "rr_local")}
    p = load_predictions(args.predictions, d)
    rows, y, pred = p["dataset_row"], p["y_true"], p["y_pred"]
    proba = p["probabilities"]
    scores = proba.max(1)
    has_noise = "windows" in p
    x = p["windows"] if has_noise else d["windows"][rows]
    clean, clean_lookup = None, {}
    if args.clean_predictions:
        clean = load_predictions(args.clean_predictions, d)
        clean_lookup = {int(row): i for i, row in enumerate(clean["dataset_row"])}
    groups = [("N_to_S", (y == 0) & (pred == 1)),
              ("S_to_N", (y == 1) & (pred == 0)),
              ("V_incorrect", (y == 2) & (pred != 2)),
              ("F_incorrect", (y == 3) & (pred != 3))]
    if has_noise:
        wrong = y != pred
        induced = np.array([int(row) in clean_lookup and
                            clean["y_pred"][clean_lookup[int(row)]] == y[i]
                            for i, row in enumerate(rows)]) if clean is not None else np.zeros(len(rows), bool)
        groups.insert(0, ("noise_induced", wrong & induced))
        groups.append(("noisy_errors", wrong))
    args.out.mkdir(parents=True)
    fs = CFG["resample"]["fs_target"]
    t = (np.arange(x.shape[1]) - CFG["segmentation"]["store_pre"]) / fs * 1000
    selected, counts, seen = [], {}, set()
    for name, mask in groups:
        counts[name] = int(mask.sum())
        chosen = representative_indices(mask, scores, p["record"], args.examples)
        if not chosen:
            continue
        for i in chosen:
            row = int(rows[i])
            if (name == "noisy_errors") and row in seen:
                continue
            seen.add(row)
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
            ax = axes[0]
            clean_pred = None
            if row in clean_lookup:
                clean_pred = CLASSES[int(clean["y_pred"][clean_lookup[row]])]
            if has_noise:
                clean_label = f"Clean pred. {clean_pred}" if clean_pred else "Clean"
                ax.plot(t, d["windows"][row], color="#9aa7b3", lw=1.4, label=clean_label)
            ax.plot(t, x[i], color="#b54239", lw=1, label="Noisy" if has_noise else "Error")
            ax.axvline(0, color="#576574", linestyle=":", lw=1)
            ax.set(xlabel="Time from beat marker (ms)", ylabel="Normalized amplitude",
                   title=f"Record {p['record'][i]} | Row {row}\n"
                         f"True {CLASSES[y[i]]} → Pred. {CLASSES[pred[i]]}")
            ax.legend(fontsize=8, loc="lower right")
            # Prefer a correct beat of the same class and record.
            control_mask = (y == y[i]) & (pred == y) & (p["record"] == p["record"][i])
            if not control_mask.any():
                control_mask = (y == y[i]) & (pred == y)
            controls = representative_indices(control_mask, scores, p["record"], 1)
            ax = axes[1]
            control_row = None
            if controls:
                j = controls[0]
                control_row = int(rows[j])
                ax.plot(t, x[j], color="#237e73", lw=1.1)
                ax.set_title(f"Record {p['record'][j]} | Row {control_row}\n"
                             f"Correct {CLASSES[y[j]]} control")
            else:
                ax.text(.5, .5, "No correct control",
                        ha="center", va="center", transform=ax.transAxes)
            ax.axvline(0, color="#576574", linestyle=":", lw=1)
            ax.set(xlabel="Time from beat marker (ms)", ylabel="Normalized amplitude")
            # Share the amplitude scale.
            signals = [x[i], d["windows"][row]] + ([x[controls[0]]] if controls else [])
            low, high = min(float(z.min()) for z in signals), max(float(z.max()) for z in signals)
            margin = max(.05, (high - low) * .08)
            for ax in axes:
                ax.set_ylim(low - margin, high + margin)
                ax.grid(alpha=.18)
                ax.spines[["top", "right"]].set_visible(False)
            filename = f"{name}_record{p['record'][i]}_row{row}.png"
            fig.suptitle(f"{args.label} | {name.replace('_', ' ')} | Score {scores[i]:.3f}", fontsize=11)
            fig.savefig(args.out / filename, dpi=160)
            plt.close(fig)
            selected.append({"category": name, "figure": filename, "dataset_row": row,
                             "record": int(p["record"][i]), "true": CLASSES[y[i]],
                             "predicted": CLASSES[pred[i]], "predicted_score": float(scores[i]),
                             "true_class_score": float(proba[i, y[i]]),
                             "rr_prev_seconds": float(d["rr_prev"][row]),
                             "rr_prev_over_local": float(d["rr_prev"][row] / d["rr_local"][row]),
                             "control_dataset_row": control_row, "clean_prediction": clean_pred})
    (args.out / "category_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    if selected:
        with open(args.out / "selected_examples.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    print(json.dumps(counts, indent=2))
    print(f"Saved {len(selected)} plots to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/beats.npz")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--clean-predictions", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be positive")

    if args.predictions is not None or args.out is not None or args.clean_predictions is not None:
        if args.predictions is None or args.out is None:
            parser.error("Provide both --predictions and --out for a custom run")
        if args.label is None:
            args.label = "ECG model"
        analyze(args)
    else:
        baseline_results = ROOT / "results/baseline_logreg_search"
        clean_predictions = baseline_results / "test/predictions.npz"
        noisy_predictions = baseline_results / "test_em_0db/predictions.npz"
        if not clean_predictions.is_file():
            parser.error("Run evaluate_baseline_logreg.py first")

        requested_label = args.label
        runs = (
            ("clean", clean_predictions, None, "Clean DS2"),
            ("noise", noisy_predictions, clean_predictions, "DS2 at 0 dB"),
        )
        for condition, predictions, clean_pair, label in runs:
            if not predictions.is_file():
                continue
            args.predictions = predictions
            args.clean_predictions = clean_pair
            args.label = label if requested_label is None else requested_label
            folder = ROOT / "figures" / f"baseline_logreg_errors_{condition}"
            args.out = folder
            suffix = 2
            while args.out.exists():
                args.out = folder.with_name(f"{folder.name}_{suffix}")
                suffix += 1
            analyze(args)
