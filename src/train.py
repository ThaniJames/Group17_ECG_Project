"""
Training

Every run writes results/runs/<arm>_<seed>/ containing the best checkpoint,
the per-epoch history and a summary with the exact configuration used.
"""

from __future__ import annotations
import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import MCFG, load_beats, make_loaders
from io_utils import CFG, CLASSES
from metrics import NSV, confusion, format_report, macro_f1, summary
from model import build

_ROOT = Path(__file__).resolve().parent.parent
RUNS = _ROOT / "results" / "runs"


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(arm: str, seed: int, device: torch.device | None = None,
              data: dict | None = None, out_root: Path = RUNS,
              max_epochs: int | None = None, quiet: bool = False) -> dict:
    """Train one arm with one seed. Returns the run summary."""
    t = MCFG["training"]
    device = device or pick_device()
    epochs = max_epochs or t["max_epochs"]
    patience = t["early_stopping_patience"]

    set_seed(seed)
    loaders = make_loaders(arm, seed=seed, d=data)
    model = build(arm).to(device)
    weights = loaders["class_weights"].to(device)
    lossf = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out = out_root / f"{arm}_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    best, best_epoch, history = -1.0, -1, []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        sched.step()

        y, p = predict(model, loaders["val"], device)
        cm = confusion(y, p)
        score = macro_f1(cm, NSV)
        history.append({
            "epoch": epoch,
            "train_loss": total / n,
            "val_macro_f1_nsv": score,
            "val_macro_f1": macro_f1(cm),
            "lr": sched.get_last_lr()[0],
        })

        if score > best:
            best, best_epoch = score, epoch
            torch.save({"state_dict": model.state_dict(), "arm": arm,
                        "seed": seed, "epoch": epoch, "val_macro_f1_nsv": score},
                       out / "best.pt")

        if not quiet and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:>3}  loss {total/n:.4f}  "
                  f"val macro-F1(N/S/V) {score:.4f}"
                  f"{'  *' if epoch == best_epoch else ''}")

        if epoch - best_epoch >= patience:
            if not quiet:
                print(f"  early stop at epoch {epoch} "
                      f"(no gain for {patience} epochs)")
            break

    # reload the best checkpoint and report it
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["state_dict"])
    y, p = predict(model, loaders["val"], device)
    val = summary(y, p)

    run = {
        "arm": arm,
        "seed": seed,
        "device": str(device),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "seconds": round(time.perf_counter() - t0, 1),
        "parameters": model.n_params,
        "class_weights": [round(float(w), 4) for w in loaders["class_weights"]],
        "validation": val,
        "config": {
            "arms": MCFG["arms"][arm],
            "architecture": MCFG["architecture"],
            "training": t,
            "augmentation": MCFG["augmentation"],
            "segmentation": CFG["segmentation"],
            "val_records": CFG["data"]["val"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "report.txt").write_text(format_report(y, p))
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one arm, or the full ablation.")
    ap.add_argument("--arm", choices=list(MCFG["arms"]), default="C")
    ap.add_argument("--seed", type=int, default=MCFG["training"]["seeds"][0])
    ap.add_argument("--all", action="store_true", help="every arm x every seed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None, help="override max_epochs")
    args = ap.parse_args()

    device = pick_device(args.device)
    data = load_beats()
    print(f"device {device}   beats {data['windows'].shape}")

    jobs = ([(a, s) for a in MCFG["arms"] for s in MCFG["training"]["seeds"]]
            if args.all else [(args.arm, args.seed)])

    results = []
    for arm, seed in jobs:
        print(f"\n=== arm {arm}  seed {seed} ===")
        r = train_one(arm, seed, device, data, max_epochs=args.epochs)
        v = r["validation"]
        print(f"  best epoch {r['best_epoch']}  "
              f"val macro-F1(N/S/V) {v['macro_f1_nsv']:.4f}  "
              f"S recall {v['per_class']['S']['recall']:.3f}  "
              f"({r['seconds']}s)")
        results.append(r)

    if len(results) > 1:
        print("\n" + "=" * 64)
        print(f"{'arm':5}{'runs':>6}{'macro-F1(N/S/V)':>18}{'S recall':>12}{'F recall':>12}")
        for arm in MCFG["arms"]:
            rs = [r for r in results if r["arm"] == arm]
            if not rs:
                continue
            f = np.array([r["validation"]["macro_f1_nsv"] for r in rs])
            s = np.array([r["validation"]["per_class"]["S"]["recall"] for r in rs])
            fr = np.array([r["validation"]["per_class"]["F"]["recall"] for r in rs])
            print(f"{arm:5}{len(rs):>6}{f.mean():>12.4f} ±{f.std():.3f}"
                  f"{s.mean():>9.3f} ±{s.std():.3f}{fr.mean():>9.3f} ±{fr.std():.3f}")
        print(f"\nruns written to {RUNS}")


if __name__ == "__main__":
    main()
