"""
Independent checks of the MAX78002 port. Run inside WSL with the ai8x venv:
    python max78002/verify_toolchain.py [--izer-samples-per-class 5]

1. ai87ecgnet has the same layers and parameter shapes as src/model.py arm B2.
2. The ai8x ECG loader yields exactly the beats/labels of src/dataset.py, with
   augmentation on train rows only, and never touches DS2.
3. The quantized checkpoint holds 8-bit integer weights and biases.
4. PyTorch 8-bit simulation reproduces train.py's validation confusion matrix.
5. The synthesizer's own simulation (sampleoutput.h) gives the same logits as
   PyTorch for real beats of every class, i.e. the YAML matches the model.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
AI8X = Path(os.environ.get("AI8X", "/root/max78002"))
TRAIN, SYNTH = AI8X / "ai8x-training", AI8X / "ai8x-synthesis"
QCKPT = REPO / "max78002" / "quantization_output" / "ai87-ecg-b2-qat8-q.pth.tar"
YAML = REPO / "max78002" / "model_configuration" / "ecg-b2.yaml"
TRAIN_LOG_CM = [[9108, 115, 14, 13], [96, 57, 81, 0], [164, 99, 973, 2], [10, 0, 11, 0]]

sys.path[:0] = [str(TRAIN), str(REPO / "src")]
os.chdir(TRAIN)

import ai8x  # noqa: E402
from distiller import apputils  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def build_ai8x(eight_bit: bool):
    from models.ai87ecgnet import ai87ecgnet
    q = dict(weight_bits=8, bias_bits=8, quantize_activation=True) if eight_bit else {}
    return ai87ecgnet(num_classes=4, num_channels=1, dimensions=(256, 1), bias=True, **q)


def check_architecture() -> None:
    ai8x.set_device(87, False, False)
    from model import build
    ref = [(n, tuple(p.shape)) for n, p in build("B2").named_parameters()]
    port = [(n, tuple(p.shape)) for n, p in build_ai8x(False).named_parameters()
            if n.endswith((".weight", ".bias")) and ".op." in n]
    check("architecture: parameter shapes match src/model.py B2",
          [s for _, s in ref] == [s for _, s in port],
          f"{len(port)} tensors, {sum(np.prod(s) for _, s in port):,} parameters")


def check_loader() -> None:
    import dataset as src_ds
    import datasets.ecg as ecg

    d = src_ds.load_beats()
    args = types.SimpleNamespace(act_mode_8bit=False)
    train, test = ecg.ecg_get_datasets(("data", args))
    ref_val = src_ds.BeatDataset(d, "val", "B2")

    val_rows = np.asarray(train.valid_indices)
    tr_rows = np.setdiff1d(np.arange(len(train)), val_rows)
    check("loader: no DS2 beats in training or validation",
          not np.isin(train.subset, ["test"]).any() and not np.isin(test.subset, ["test"]).any())
    check("loader: train and validation records are disjoint",
          not set(train.record[tr_rows]) & set(train.record[val_rows]),
          f"val records {sorted(set(train.record[val_rows].tolist()))}")
    check("loader: train/val sizes match src/dataset.py",
          len(tr_rows) == len(src_ds.BeatDataset(d, "train", "B2")) and len(val_rows) == len(ref_val),
          f"train {len(tr_rows):,}, val {len(val_rows):,}")

    same_labels = np.array_equal(test.y, ref_val.labels.numpy())
    same_records = np.array_equal(test.record, ref_val.record)
    xs = torch.stack([test[i][0] for i in range(len(test))]).numpy()[:, 0]
    expect = np.clip(np.round(ref_val.windows / 4.0 * 128), -128, 127) / 128
    check("loader: validation beats/labels/records identical to src/dataset.py",
          same_labels and same_records and np.array_equal(xs, expect.astype(np.float32)),
          f"max |diff| {np.abs(xs - expect).max():.2e}")

    rng_rows = val_rows[:50]
    val_stable = all(torch.equal(train[i][0], train[i][0]) for i in rng_rows)
    tr_varies = any(not torch.equal(train[i][0], train[i][0]) for i in tr_rows[:200])
    check("loader: augmentation on train rows only", val_stable and tr_varies)

    try:
        ecg.ecg_ds2_get_datasets(("data", args), load_train=True)
        check("loader: DS2 refuses to load for training", False)
    except RuntimeError:
        check("loader: DS2 refuses to load for training", True)


def check_quantized_checkpoint() -> None:
    sd = torch.load(QCKPT, map_location="cpu")["state_dict"]

    def int8(v):
        return torch.equal(v, v.round()) and v.min() >= -128 and v.max() <= 127

    weights = [v for k, v in sd.items() if k.endswith("op.weight")]
    # quantize.py stores biases pre-multiplied by 2**7, the scale of a weight x data product
    biases = [v / 128 for k, v in sd.items() if k.endswith("op.bias")]
    bits = {int(v) for k, v in sd.items() if k.endswith(("weight_bits", "bias_bits"))}
    check("quantized checkpoint: weights and biases are int8",
          all(map(int8, weights + biases)) and bits == {8},
          f"{sum(v.numel() for v in weights + biases):,} values, weight/bias bits {bits}")


def load_int8_model():
    ai8x.set_device(87, True, False)
    model = build_ai8x(True)
    model = apputils.load_lean_checkpoint(model, str(QCKPT), model_device="cpu")
    ai8x.initiate_qat(model, {"start_epoch": 0, "weight_bits": 8})
    ai8x.update_model(model)
    return model.eval()


def check_int8_eval(model):
    import datasets.ecg as ecg
    from metrics import NSV, confusion, macro_f1

    _, test = ecg.ecg_get_datasets(("data", types.SimpleNamespace(act_mode_8bit=True)),
                                   load_train=False)
    xs = torch.stack([test[i][0] for i in range(len(test))])
    with torch.no_grad():
        logits = torch.cat([model(xs[i:i + 1024]) for i in range(0, len(xs), 1024)])
    cm = confusion(test.y, logits.argmax(1).numpy())
    check("INT8 simulation reproduces train.py -8 confusion matrix", cm.tolist() == TRAIN_LOG_CM,
          f"macro-F1(N/S/V) {macro_f1(cm, NSV):.4f}")
    return test, xs, logits


def izer_logits(sample: np.ndarray, workdir: Path) -> np.ndarray:
    npy = workdir / "sample.npy"
    np.save(npy, sample.astype(np.int64), allow_pickle=False)
    subprocess.run(
        [sys.executable, "ai8xize.py", "--no-version-check", "--test-dir", str(workdir),
         "--prefix", "v", "--checkpoint-file", str(QCKPT), "--config-file", str(YAML),
         "--sample-input", str(npy), "--device", "MAX78002", "--softmax", "--compact-data",
         "--mexpress", "--overwrite"],
        cwd=SYNTH, check=True, capture_output=True, text=True)
    words = re.findall(r"0x[0-9a-f]{8}", (workdir / "v" / "sampleoutput.h").read_text())
    # layout: address, mask, count, values..., 0 terminator
    n = int(words[2], 16)
    return np.array([int(w, 16) - (1 << 32) if int(w, 16) >= 1 << 31 else int(w, 16)
                     for w in words[3:3 + n]])


def check_synthesis(test, xs, logits, per_class: int) -> None:
    rng = np.random.default_rng(0)
    picks = np.concatenate([rng.choice(np.flatnonzero(test.y == c), per_class, replace=False)
                            for c in range(4)])
    torch_out = logits[picks].numpy()
    izer_out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in picks:
            izer_out.append(izer_logits(xs[i].numpy(), Path(tmp)))
    izer_out = np.array(izer_out, dtype=np.float64)
    # PyTorch wide output and the 32-bit hardware word differ by a constant power-of-two scale
    scale = np.median(izer_out[torch_out != 0] / torch_out[torch_out != 0])
    exact = np.array_equal(izer_out, torch_out * scale)
    same_class = np.array_equal(izer_out.argmax(1), torch_out.argmax(1))
    check("synthesis simulation == PyTorch INT8 logits (bit-exact up to fixed scale)",
          exact, f"{len(picks)} beats, scale {scale:g}")
    check("synthesis simulation predicts the same class as PyTorch",
          same_class, f"classes tested {sorted(set(test.y[picks].tolist()))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--izer-samples-per-class", type=int, default=5)
    a = ap.parse_args()

    check_architecture()
    check_loader()
    check_quantized_checkpoint()
    model = load_int8_model()
    test, xs, logits = check_int8_eval(model)
    check_synthesis(test, xs, logits, a.izer_samples_per_class)

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
