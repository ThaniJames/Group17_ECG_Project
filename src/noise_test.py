"""
Noise for the robustness test (report 5.4)

Real noise from the MIT-BIH Noise Stress Test Database (NSTDB) is added to the
raw 360 Hz ECG, and the result goes through exactly the same preprocessing as
the clean data (filter, resample to 180 Hz, normalise, cut into beats).

    BW  baseline wander        EM  electrode motion artefact        MA  muscle artefact (EMG)

Rules, from configs/model.yaml (noise:)
  * test noise = the SECOND half of each noise recording, channel noise2.
    The first half is kept for training augmentation, so the two never overlap.
  * the pre-built 118e* / 119e* records are never used (they are training records).
  * SNR = power of the clean record / power of the noise, power = variance.
    The noise is repeated to cover the record (a recording half is shorter).
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import numpy as np
import wfdb
import yaml
from scipy.signal import detrend

from io_utils import CFG, CLASSES, all_records, ensure_database, load_record
from prepare_data import (compute_rr, filter_signal, normalise, rescale_peaks,
                          resample_signal, segment, split_of)

_ROOT = Path(__file__).resolve().parent.parent
with open(_ROOT / "configs" / "model.yaml") as _fh:
    NCFG = yaml.safe_load(_fh)["noise"]

NOISE_TYPES = ("BW", "MA", "EM")
SNR_DB = tuple(NCFG["snr_db_eval"])                       # 12, 6, 0
CONDITIONS = [(None, None)] + [(n, s) for n in NOISE_TYPES for s in SNR_DB]   # clean first
NOISY_DIR = _ROOT / "data" / "noisy"

FS_IN = CFG["data"]["fs_original"]                        # 360 Hz
FS_OUT = CFG["resample"]["fs_target"]                     # 180 Hz
STORE_PRE = CFG["segmentation"]["store_pre"]
STORE_POST = CFG["segmentation"]["store_post"]


@lru_cache(maxsize=None)
def load_noise(noise_type: str) -> np.ndarray:
    """Test noise in mV: channel noise2, second half of the recording."""
    name = noise_type.lower()
    if name not in NCFG["records"]:
        raise ValueError(f"{noise_type!r} is not one of {NCFG['records']}")
    folder = ensure_database(NCFG["database"], _ROOT / NCFG["local_dir"])   # downloads once
    rec = wfdb.rdrecord(str(folder / name))
    if int(rec.fs) != FS_IN:
        raise ValueError(f"{name}: sampling rate {rec.fs}, expected {FS_IN}")
    x = rec.p_signal[:, list(rec.sig_name).index(NCFG["eval_channel"])]
    return np.asarray(x[len(x) // 2:], dtype=np.float64)


def add_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """signal + noise scaled so that var(signal) / var(noise) = 10^(snr_db / 10)."""
    noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))[: len(signal)]
    noise = detrend(noise)                                # remove the mean and any linear trend
    noise_power = np.var(signal) / 10 ** (snr_db / 10)
    return signal + noise * np.sqrt(noise_power) / np.std(noise)


def achieved_snr_db(signal: np.ndarray, noisy: np.ndarray) -> float:
    """The SNR that was actually produced (a check)."""
    return 10 * np.log10(np.var(signal) / np.var(noisy - signal))


def preprocess(rec, signal: np.ndarray) -> dict:
    """The steps of prepare_data.prepare_record, starting from a given 360 Hz signal.

    rec supplies the beat positions and labels (load_record). With the clean
    signal the result is identical to prepare_record.
    """
    x = normalise(resample_signal(filter_signal(signal, FS_IN), FS_IN, FS_OUT))
    rr_prev, rr_next, rr_local, rr_ok = compute_rr(rec.r_peaks, FS_IN)
    sel = np.isin(rec.labels, CLASSES) & rr_ok
    peaks = rescale_peaks(rec.r_peaks[sel], FS_IN, FS_OUT)
    windows, fits = segment(x, peaks, STORE_PRE, STORE_POST)
    keep = np.flatnonzero(sel)[fits]
    return {"windows": windows.astype(np.float32),
            "labels": np.array([CLASSES.index(c) for c in rec.labels[keep]], dtype=np.int8),
            "record": np.full(len(keep), int(rec.record_id), dtype=np.int16),
            "rr_prev": rr_prev[keep].astype(np.float32),
            "rr_next": rr_next[keep].astype(np.float32),
            "rr_local": rr_local[keep].astype(np.float32)}


def make_dataset(subset: str, noise_type: str | None = None, snr_db: float | None = None) -> dict:
    """All beats of one split ('val' or 'test'), clean or with noise.
    Same keys as data/processed/beats.npz."""
    parts = []
    for rid in all_records():
        if split_of(rid) != subset:
            continue
        rec = load_record(rid, keep_q=True)
        signal = rec.signal
        if noise_type is not None:
            signal = add_noise(signal, load_noise(noise_type), snr_db)
        parts.append(preprocess(rec, signal))
    if not parts:
        raise ValueError(f"no records in split {subset!r}; check configs/prep.yaml")
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d["subset"] = np.full(len(d["labels"]), subset)
    d["is_ds1"] = np.full(len(d["labels"]), subset != "test")
    return d


def dataset_path(subset: str, noise_type: str | None = None, snr_db: float | None = None) -> Path:
    stem = "clean" if noise_type is None else f"{noise_type}_{int(snr_db)}dB"
    return NOISY_DIR / f"{subset}_{stem}.npz"


def load_dataset(subset: str, noise_type: str | None = None, snr_db: float | None = None) -> dict:
    path = dataset_path(subset, noise_type, snr_db)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/run_noise.py"
                                + (" --final" if subset == "test" else ""))
    return dict(np.load(path))
