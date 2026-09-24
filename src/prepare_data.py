'''
Preprocessing pipeline

step 1: raw 360 Hz -> casual filter -> decimate to 180 Hz -> [normalise, segment]
step 2: filter + resample only
'''

from __future__ import annotations
import numpy as np
from functools import lru_cache
from pathlib import Path
from scipy import signal as sig
from io_utils import CFG, CLASSES, all_records, ds1_records, ds2_records, load_record

@lru_cache(maxsize=4)
def design_filters(fs: int):
    f = CFG["filtering"]
    order = f["order"]
    hp = sig.butter(order, f["highpass_hz"], "highpass", fs=fs, output="sos")
    lp = sig.butter(order, f["lowpass_hz"], "lowpass", fs=fs, output="sos")
    b, a = sig.iirnotch(f["notch_hz"], f["notch_q"], fs=fs)
    notch = sig.tf2sos(b, a)
    return (hp, notch, lp)

def filter_signal(x: np.ndarray, fs: int):
    """Causal filter chain, applied in order."""
    for sos in design_filters(fs):
        x = sig.sosfilt(sos, x)
    return x

def resample_signal(x: np.ndarray, fs_in: int, fs_out: int):
    """Anti-aliased decimation."""
    q, rem = divmod(fs_in, fs_out)
    if rem:
        raise ValueError(f"{fs_in} Hz is not an integer multiple of {fs_out} Hz")
    return x if q == 1 else sig.decimate(x, q, ftype="iir", zero_phase=False)

def measure_group_delay(raw: np.ndarray, filtered: np.ndarray,
                        r_peaks: np.ndarray, search: int = 30):
    """How far the causal filter moved each R-peak, in samples."""
    offs = []
    for p in r_peaks:
        lo, hi = p - search, p + search + 1
        if lo < 0 or hi > len(filtered):
            continue
        offs.append(int(np.argmax(np.abs(filtered[lo:hi]))) - search)
    offs = np.asarray(offs)
    return {"n": len(offs), "median": float(np.median(offs)),
            "mean": float(offs.mean()), "p5": float(np.percentile(offs, 5)),
            "p95": float(np.percentile(offs, 95))}


def normalise(x: np.ndarray):
    """Robust per-record scaling, then clip."""
    n = CFG["normalisation"]
    med = np.median(x)
    d = np.abs(x - med)
    scale = np.percentile(d, n["scale_percentile"])
    return np.clip((x - med) / scale, -n["clip"], n["clip"])

def compute_rr(r_peaks: np.ndarray, fs: int):
    """RR intervals in seconds for every beat."""
    rr = np.diff(r_peaks) / fs
    n = len(r_peaks)
    rr_prev = np.full(n, np.nan)
    rr_next = np.full(n, np.nan)
    rr_prev[1:] = rr
    rr_next[:-1] = rr
    w = CFG["rr"]["local_window"]
    rr_local = np.full(n, np.nan)
    for i in range(1, n):
        window = rr[max(0, i - w):i]
        if len(window):
            rr_local[i] = np.median(window)
    valid = ~(np.isnan(rr_prev) | np.isnan(rr_next) | np.isnan(rr_local))
    return rr_prev, rr_next, rr_local, valid

def rescale_peaks(r_peaks: np.ndarray, fs_in: int, fs_out: int,
                  delay_samples: int | None = None) -> np.ndarray:
    """Move R-peak indices from fs_in to fs_out, compensating the causal
    filtering delay (our chain + decimate's anti-alias filter)."""
    if delay_samples is None:
        delay_samples = CFG["resample"]["group_delay_samples"]
    return (r_peaks + delay_samples) * fs_out // fs_in

def segment(x: np.ndarray, peaks: np.ndarray, pre: int, post: int):
    """Cut fixed windows around each peak."""
    keep = (peaks >= pre) & (peaks + post <= len(x))
    idx = peaks[keep]
    offs = np.arange(-pre, post)
    return x[idx[:, None] + offs[None, :]], keep

def prepare_record(record_id: str | int) -> dict:
    """Full pipeline for one record. Returns arrays ready to concatenate."""
    fs_in = CFG["data"]["fs_original"]
    fs_out = CFG["resample"]["fs_target"]
    pre, post = CFG["segmentation"]["pre"], CFG["segmentation"]["post"]
    rec = load_record(record_id, keep_q=True)
    x = normalise(resample_signal(filter_signal(rec.signal, fs_in), fs_in, fs_out))
    rr_prev, rr_next, rr_local, rr_ok = compute_rr(rec.r_peaks, fs_in)
    is_target = np.isin(rec.labels, CLASSES)
    sel = is_target & rr_ok
    peaks = rescale_peaks(rec.r_peaks[sel], fs_in, fs_out)
    windows, fits = segment(x, peaks, pre, post)
    keep = np.flatnonzero(sel)[fits]
    return {
        "record": rec.record_id,
        "windows": windows.astype(np.float32),
        "labels": rec.labels[keep],
        "rr_prev": rr_prev[keep].astype(np.float32),
        "rr_next": rr_next[keep].astype(np.float32),
        "rr_local": rr_local[keep].astype(np.float32),
        "n_target": int(is_target.sum()),
        "n_dropped_rr": int((is_target & ~rr_ok).sum()),
        "n_dropped_edge": int((~fits).sum()),
    }

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "data" / "processed"

def split_of(record_id: int | str) -> str:
    """'train' | 'val' | 'test' for a record. DS2 is always test."""
    rid = int(record_id)
    if rid in CFG["data"]["ds2"]:
        return "test"
    return "val" if rid in CFG["data"]["val"] else "train"

def prepare_all(out_path: str | Path | None = None) -> dict:
    """Run the pipeline over every record and save one array set."""
    ds1, ds2 = set(ds1_records()), set(ds2_records())
    parts, rows = [], []
    for rid in all_records():
        o = prepare_record(rid)
        o["split"] = "DS1" if rid in ds1 else "DS2"
        o["subset"] = split_of(rid)
        parts.append(o)
        counts = {c: int((o["labels"] == c).sum()) for c in CLASSES}
        rows.append({"record": int(rid), "split": o["split"], "subset": o["subset"], **counts,
                     "kept": len(o["windows"]),
                     "dropped_rr": o["n_dropped_rr"],
                     "dropped_edge": o["n_dropped_edge"],
                     "audit_target": o["n_target"]})
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    data = {
        "windows": np.concatenate([p["windows"] for p in parts]),
        "labels": np.concatenate([[cls_idx[c] for c in p["labels"]] for p in parts]).astype(np.int8),
        "record": np.concatenate([np.full(len(p["windows"]), int(p["record"])) for p in parts]).astype(np.int16),
        "is_ds1": np.concatenate([np.full(len(p["windows"]), p["split"] == "DS1") for p in parts]),
        "subset": np.concatenate([np.full(len(p["windows"]), p["subset"]) for p in parts]),
        "rr_prev": np.concatenate([p["rr_prev"] for p in parts]),
        "rr_next": np.concatenate([p["rr_next"] for p in parts]),
        "rr_local": np.concatenate([p["rr_local"] for p in parts]),
    }
    out = Path(out_path or _OUT / "beats.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, classes=np.array(CLASSES), **data)
    return {"data": data, "per_record": rows, "path": out}

if __name__ == "__main__":
    rid = "101"
    fs_in = CFG["data"]["fs_original"]
    fs_out = CFG["resample"]["fs_target"]
    rec = load_record(rid, keep_q=False)
    raw = rec.signal
    filt = filter_signal(raw, fs_in)
    ds = resample_signal(filt, fs_in, fs_out)
    print(f"record {rid}   lead {rec.lead_name}   {len(rec)} beats")
    print()
    print(f"{'stage':<26}{'samples':>10}{'fs':>6}{'min':>9}{'max':>9}{'std':>9}")
    for name, x, fs in [("raw", raw, fs_in),
                        ("filtered", filt, fs_in),
                        ("filtered + decimated", ds, fs_out)]:
        print(f"{name:<26}{len(x):>10,}{fs:>6}{x.min():>9.3f}{x.max():>9.3f}{x.std():>9.3f}")
    print()
    print("baseline drift removed:")
    for name, x in [("raw", raw), ("filtered", filt)]:
        w = 2 * fs_in
        med = np.median(x[: (len(x) // w) * w].reshape(-1, w), axis=1)
        print(f"   {name:<10} baseline swing = {med.max() - med.min():.3f} mV")
    print()
    gd = measure_group_delay(raw, filt, rec.r_peaks)
    print(f"group delay (causal filter shifts the R-peak):")
    print(f"   median {gd['median']:+.0f} samples "
          f"= {1000*gd['median']/fs_in:+.1f} ms at {fs_in} Hz   "
          f"(5th-95th pct: {gd['p5']:+.0f} to {gd['p95']:+.0f})")
    print()
    print("--- full pipeline, one record ---")
    out = prepare_record(rid)
    w = out["windows"]
    print(f'  windows            {w.shape}  {w.dtype}')
    print(f'  target-class beats  {out["n_target"]:,}')
    print(f'  dropped (no RR)     {out["n_dropped_rr"]}   (first/last beat of the record)')
    print(f'  dropped (edge)      {out["n_dropped_edge"]}   (too close to record start/end)')
    print(f'  kept                {len(w):,}')
    print(f'  labels              {dict(zip(*np.unique(out["labels"], return_counts=True)))}')
    print(f'  amplitude           min {w.min():+.2f}  max {w.max():+.2f}  '
          f'mean {w.mean():+.3f}  std {w.std():.3f}')
    print(f'  R-peak at index     {int(np.median(np.argmax(w, axis=1)))}   (target {CFG["segmentation"]["pre"]})')
    print(f'  NaN / inf           {int(np.isnan(w).sum())} / {int(np.isinf(w).sum())}')
    print(f'  RR (s)              prev {np.median(out["rr_prev"]):.2f}  '
          f'next {np.median(out["rr_next"]):.2f}  local {np.median(out["rr_local"]):.2f}')