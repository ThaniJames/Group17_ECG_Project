from __future__ import annotations
from pathlib import Path

import numpy as np
import wfdb
import yaml

from io_utils import CFG, CLASSES, load_record
from prepare_data import (filter_signal, resample_signal, normalise, compute_rr,
                          rescale_peaks, segment)

ROOT = Path(__file__).resolve().parent.parent


def mix_at_snr(clean, noise, snr_db):
    clean, noise = np.asarray(clean, dtype=float), np.asarray(noise, dtype=float)
    noise = noise - noise.mean()
    signal_power = np.mean((clean - clean.mean()) ** 2)
    noise_power = np.mean(noise ** 2)
    if not np.isfinite(snr_db) or min(signal_power, noise_power) <= 0:
        raise ValueError("SNR must be finite and both inputs must have positive power")
    addition = noise * np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    measured = 10 * np.log10(signal_power / np.mean(addition ** 2))
    return clean + addition, float(measured)


def noisy_windows(d, rows, noise_name, snr_db, seed):
    with open(ROOT / "configs/model.yaml") as stream:
        ncfg = yaml.safe_load(stream)["noise"]
    if noise_name not in ncfg["records"]:
        raise ValueError("Use an NSTDB noise-only record, not the 118/119 ECG mixtures")
    target = ROOT / ncfg["local_dir"]
    target.mkdir(parents=True, exist_ok=True)
    if not all((target / f"{noise_name}.{ext}").exists() for ext in ("hea", "dat")):
        wfdb.dl_database(ncfg["database"], str(target), records=[noise_name],
                         annotators=[], keep_subdirs=False)
    noise_record = wfdb.rdrecord(str(target / noise_name))
    channel_name = ncfg["eval_channel"]
    if channel_name not in noise_record.sig_name:
        raise ValueError(f"Missing configured evaluation noise channel {channel_name}")
    if ncfg["eval_half"] != "second":
        raise ValueError("This experiment reserves the second noise half for evaluation")
    noise = noise_record.p_signal[len(noise_record.p_signal) // 2:,
                                  noise_record.sig_name.index(channel_name)]
    fs_in, fs_out = CFG["data"]["fs_original"], CFG["resample"]["fs_target"]
    if int(noise_record.fs) != fs_in:
        raise ValueError("Noise and ECG sampling frequencies must agree")
    pre, post = CFG["segmentation"]["store_pre"], CFG["segmentation"]["store_post"]
    out = np.empty((len(rows), pre + post), dtype=np.float32)
    details = []
    for rid in np.unique(d["record"][rows]):
        positions = np.flatnonzero(d["record"][rows] == rid)
        rec = load_record(int(rid), keep_q=True)
        rng = np.random.default_rng(seed + int(rid))
        offset = int(rng.integers(len(noise)))
        trace = noise[(np.arange(len(rec.signal)) + offset) % len(noise)]
        noisy_raw, achieved = mix_at_snr(rec.signal, trace, snr_db)
        _, _, _, rr_ok = compute_rr(rec.r_peaks, fs_in)
        sel = np.isin(rec.labels, CLASSES) & rr_ok
        peaks = rescale_peaks(rec.r_peaks[sel], fs_in, fs_out)
        clean_filtered = resample_signal(filter_signal(rec.signal, fs_in), fs_in, fs_out)
        noisy_filtered = resample_signal(filter_signal(noisy_raw, fs_in), fs_in, fs_out)
        clean_w, fits = segment(normalise(clean_filtered), peaks, pre, post)
        noisy_w, _ = segment(normalise(noisy_filtered), peaks, pre, post)
        expected_labels = np.array([CLASSES.index(c) for c in rec.labels[sel][fits]])
        selected_rows = rows[positions]
        if not np.array_equal(expected_labels, d["labels"][selected_rows]):
            raise ValueError(f"Record {rid}: annotation order differs from prepared dataset")
        if clean_w.shape != d["windows"][selected_rows].shape or not np.allclose(
                clean_w, d["windows"][selected_rows], atol=2e-6):
            raise ValueError(f"Record {rid}: clean pipeline cannot reproduce stored windows")
        out[positions] = noisy_w.astype(np.float32)
        details.append({"record": int(rid), "noise": noise_name, "channel": channel_name,
                        "half": "second", "offset_in_half": offset,
                        "raw_snr_db": achieved,
                        "noise_repeated": bool(len(rec.signal) > len(noise)),
                        "normalisation": "Recomputed on each noisy record, as in clean pipeline"})
        print(f"Noise: record {rid}, {len(noisy_w)} beats, SNR {achieved:.6f} dB", flush=True)
    return out, details
