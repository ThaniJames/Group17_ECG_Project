#Dataset and input arms

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from io_utils import CFG, CLASSES

_ROOT = Path(__file__).resolve().parent.parent
with open(_ROOT / "configs" / "model.yaml") as _fh:
    MCFG: dict = yaml.safe_load(_fh)

_NPZ = _ROOT / "data" / "processed" / "beats.npz"

STORE_PRE = CFG["segmentation"]["store_pre"]
PRE, POST = CFG["segmentation"]["pre"], CFG["segmentation"]["post"]
SLICE_LO, SLICE_HI = STORE_PRE - PRE, STORE_PRE + POST


def load_beats(path: str | Path | None = None):
    """Load beats.npz"""
    d = np.load(Path(path or _NPZ), allow_pickle=False)
    return {k: d[k] for k in d.files}


def rr_record_median(rr_local: np.ndarray, record: np.ndarray):
    """Per-record median of rr_local — the patient's own resting rhythm.Computed per record, not globally"""
    out = np.empty_like(rr_local)
    for r in np.unique(record):
        m = record == r
        out[m] = np.median(rr_local[m])
    return out


def rr_channels(d: dict, idx: np.ndarray):
    """The four RR ratios for arm C, clipped and centred"""
    f = MCFG["rr_features"]
    lo, hi = f["clip_min"], f["clip_max"]
    mid, half = (lo + hi) / 2, (hi - lo) / 2

    prev, nxt = d["rr_prev"][idx], d["rr_next"][idx]
    local = d["rr_local"][idx]
    rec = rr_record_median(d["rr_local"], d["record"])[idx]

    ratios = np.stack([prev / local, nxt / local, local / rec, prev / nxt], 1)
    return ((np.clip(ratios, lo, hi) - mid) / half).astype(np.float32)


class BeatDataset(Dataset):
    """One input arm over one split."""
    def __init__(self, d: dict, subset: str, arm: str,
                 augment: bool = False, seed: int | None = None):
        if augment and subset != "train":
            raise ValueError(f"augmentation requested for split '{subset}'")
        a = MCFG["arms"][arm]
        self.arm, self.channels, self.length = arm, a["channels"], a["length"]
        self.slice = a["slice"]
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        idx = np.flatnonzero(d["subset"] == subset)
        self.windows = d["windows"][idx]
        self.labels = torch.from_numpy(d["labels"][idx].astype(np.int64))
        self.record = d["record"][idx]
        self.rr = rr_channels(d, idx) if self.channels > 1 else None
        aug = MCFG["augmentation"]
        self.amp_lo, self.amp_hi = aug["amplitude_scale"]
        self.max_shift = aug["time_shift_samples"] if augment else 0
        self.p_apply = aug["p_apply"]

    def __len__(self) -> int:
        return len(self.labels)

    def _window(self, i: int) -> np.ndarray:
        w = self.windows[i]
        shift = 0
        if self.max_shift and self.rng.random() < self.p_apply:
            shift = int(self.rng.integers(-self.max_shift, self.max_shift + 1))

        if self.slice:
            lo = SLICE_LO + shift
            x = w[lo:lo + self.length]
        else:
            # arm B uses the whole stored window, so a shift has no margin to take from; roll instead. +/-4 of 256 moves only baseline samples
            # at the far edges, well away from the R-peak at index 96.
            x = np.roll(w, shift) if shift else w

        if self.augment and self.rng.random() < self.p_apply:
            x = x * self.rng.uniform(self.amp_lo, self.amp_hi)
        return x.astype(np.float32)

    def __getitem__(self, i: int):
        x = self._window(i)[None, :]                           
        if self.rr is not None:
            broadcast = np.repeat(self.rr[i][:, None], self.length, 1)
            x = np.concatenate([x, broadcast], 0)
        return torch.from_numpy(x), self.labels[i]


def class_weights(ds: BeatDataset, cap: float | None = None):
    """Inverse-frequency weights with the max:min ratio capped, mean 1."""
    if cap is None:
        cap = MCFG["training"]["class_weight_cap"]
    counts = np.bincount(ds.labels.numpy(), minlength=len(CLASSES)).astype(np.float64)
    w = counts.sum() / np.maximum(counts, 1)
    w = np.minimum(w, cap * w.min())
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def make_loaders(arm: str, seed: int, batch_size: int | None = None,
                 d: dict | None = None):
    """train / val / test loaders for one arm. Only train is augmented."""
    d = d if d is not None else load_beats()
    bs = batch_size or MCFG["training"]["batch_size"]
    g = torch.Generator().manual_seed(seed)

    train = BeatDataset(d, "train", arm, augment=True, seed=seed)
    val = BeatDataset(d, "val", arm)
    test = BeatDataset(d, "test", arm)
    return {
        "train": DataLoader(train, bs, shuffle=True, generator=g, drop_last=False),
        "val": DataLoader(val, bs, shuffle=False),
        "test": DataLoader(test, bs, shuffle=False),
        "datasets": {"train": train, "val": val, "test": test},
        "class_weights": class_weights(train),
    }


if __name__ == "__main__":
    d = load_beats()
    print(f"beats.npz  {d['windows'].shape}  slice [{SLICE_LO}:{SLICE_HI}]")
    print()
    print(f"{'arm':5}{'split':8}{'shape':>16}{'beats':>10}   class counts")
    for arm in MCFG["arms"]:
        for split in ("train", "val", "test"):
            ds = BeatDataset(d, split, arm)
            x, y = ds[0]
            c = np.bincount(ds.labels.numpy(), minlength=4)
            print(f"{arm:5}{split:8}{str(tuple(x.shape)):>16}{len(ds):>10,}   "
                  + " ".join(f"{k}:{v:,}" for k, v in zip(CLASSES, c)))
    print()
    tr = BeatDataset(d, "train", "C", augment=True, seed=0)
    print("class weights (train):",
          {c: round(float(w), 2) for c, w in zip(CLASSES, class_weights(tr))})
    rr = rr_channels(d, np.arange(len(d["labels"])))
    print(f"RR channels   shape {rr.shape}   range [{rr.min():+.2f}, {rr.max():+.2f}]   "
          f"NaN {int(np.isnan(rr).sum())}")
