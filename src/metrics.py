"""
Classification metrics
"""

from __future__ import annotations
import numpy as np
from io_utils import CLASSES
NSV = (0, 1, 2)          # N, S, V — the classes used for model selection

def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int = len(CLASSES)) -> np.ndarray:
    """Rows are true classes, columns predicted."""
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)
    return cm

def per_class(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall and F1 per class, plus the support."""
    tp = np.diag(cm).astype(np.float64)
    pred = cm.sum(0).astype(np.float64)       # column totals
    true = cm.sum(1).astype(np.float64)       # row totals
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred > 0, tp / pred, 0.0)
        rec = np.where(true > 0, tp / true, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {"precision": prec, "recall": rec, "f1": f1, "support": true.astype(int)}


def macro_f1(cm: np.ndarray, classes: tuple[int, ...] | None = None) -> float:
    """Unweighted mean F1."""
    f1 = per_class(cm)["f1"]
    idx = list(classes) if classes is not None else range(len(f1))
    return float(np.mean([f1[i] for i in idx]))


def accuracy(cm: np.ndarray) -> float:
    return float(np.diag(cm).sum() / max(cm.sum(), 1))


def summary(y_true: np.ndarray, y_pred: np.ndarray):
    """Summary"""
    cm = confusion(y_true, y_pred)
    pc = per_class(cm)
    return {
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(cm),
        "macro_f1_nsv": macro_f1(cm, NSV),
        "per_class": {c: {k: float(pc[k][i]) if k != "support" else int(pc[k][i])
                          for k in pc}
                      for i, c in enumerate(CLASSES)},
        "confusion": cm.tolist(),
    }

def format_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Human-readable block for logs and the report."""
    s = summary(y_true, y_pred)
    out = [f"{'class':<8}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}"]
    for c in CLASSES:
        p = s["per_class"][c]
        out.append(f"{c:<8}{p['precision']:>8.3f}{p['recall']:>8.3f}"
                   f"{p['f1']:>8.3f}{p['support']:>9,}")
    out.append(f"{'':<8}{'':>8}{'':>8}{'':>8}{'':>9}")
    out.append(f"accuracy {s['accuracy']:.4f}   macro-F1 {s['macro_f1']:.4f}   "
               f"macro-F1(N/S/V) {s['macro_f1_nsv']:.4f}")
    out.append("")
    out.append("confusion (rows = true, cols = predicted)")
    cm = np.array(s["confusion"])
    out.append("        " + "".join(f"{c:>9}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        out.append(f"{c:<8}" + "".join(f"{v:>9,}" for v in cm[i]))
    return "\n".join(out)
