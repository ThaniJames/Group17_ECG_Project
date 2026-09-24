"""
All record reading goes through here i.e lead selection, annotation filtering and AAMI mapping
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import wfdb
import yaml

_ROOT = Path(__file__).resolve().parent.parent
with open(_ROOT / "configs" / "prep.yaml") as _fh:
    CFG: dict = yaml.safe_load(_fh)

#AAMI EC57 grouping (de Chazal et al. 2004)
#L and R are bundle branch blocks with wide QRS but are calss N
#lowercase f (pcaemaker class) we are not taking count on them
#uppercase F is our class
AAMI_MAP: dict[str, str] = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",          # paced / unclassifiable, excluded
}

CLASSES: tuple[str, ...] = ("N", "S", "V", "F")
BEAT_SYMBOLS: frozenset[str] = frozenset("NLRBAaJSVrFejnE/fQ?")

@dataclass
class RecordData:
    record_id: str
    signal: np.ndarray            # 1-D, the selected lead, in mV
    r_peaks: np.ndarray           # sample indices AT fs (360 Hz) — rescale after decimation
    labels: np.ndarray            # 'N' | 'S' | 'V' | 'F' | 'Q'
    symbols: np.ndarray           # original MIT-BIH symbol
    lead_name: str
    fs: int
    substituted_lead: bool
    all_lead_names: list[str] = field(default_factory=list)
    n_non_beat_dropped: int = 0
    unmapped_symbols: dict[str, int] = field(default_factory=dict)

    def __len__(self):
        return len(self.r_peaks)
    
def ensure_database(db: str | None = None, local_dir: str | Path | None = None):
    """Download the database"""
    db = db or CFG["data"]["database"]
    target = Path(local_dir or _ROOT / CFG["data"]["local_dir"])
    marker = target / ".complete"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{db}' to {target} (one time, a few minutes)…")
    wfdb.dl_database(db, str(target))
    marker.touch()
    return target

def _record_path(record_id: str):
    """Prefer the local copy; fall back to streaming from PhysioNet."""
    local = _ROOT / CFG["data"]["local_dir"] / str(record_id)
    if local.with_suffix(".hea").exists():
        return str(local), None
    return str(record_id), CFG["data"]["database"]

def select_lead(sig_names: list[str]):
    """Select a channel by name"""
    preferred = CFG["lead"]["preferred"]
    if preferred in sig_names:
        return sig_names.index(preferred), preferred, False
    for alt in CFG["lead"]["fallback"]:
        if alt in sig_names:
            return sig_names.index(alt), alt, True
    return 0, sig_names[0], True

def load_record(record_id: str | int, keep_q: bool = True):
    """Load one record with its beat annotations.
    keep_q=True keeps paced/unclassifiable beats as class 'Q' so the audit can
    count the exclusions. The preprocessing pipeline should pass keep_q=False.
    """
    rid = str(record_id)
    path, pn_dir = _record_path(rid)
    rec = wfdb.rdrecord(path, pn_dir=pn_dir)
    ann = wfdb.rdann(path, "atr", pn_dir=pn_dir)
    ch, lead_name, substituted = select_lead(list(rec.sig_name))
    signal = np.asarray(rec.p_signal[:, ch], dtype=np.float64)
    symbols = np.asarray(ann.symbol)
    samples = np.asarray(ann.sample, dtype=np.int64)
    is_beat = np.array([s in BEAT_SYMBOLS for s in symbols])
    n_dropped = int((~is_beat).sum())
    symbols, samples = symbols[is_beat], samples[is_beat]
    labels = np.array([AAMI_MAP.get(s, "?") for s in symbols])
    unmapped = {str(s): int((symbols == s).sum())
                for s in np.unique(symbols[labels == "?"])}
    if not keep_q:
        keep = np.isin(labels, CLASSES)
        symbols, samples, labels = symbols[keep], samples[keep], labels[keep]
    return RecordData(
        record_id=rid,
        signal=signal,
        r_peaks=samples,
        labels=labels,
        symbols=symbols,
        lead_name=lead_name,
        fs=int(rec.fs),
        substituted_lead=substituted,
        all_lead_names=list(rec.sig_name),
        n_non_beat_dropped=n_dropped,
        unmapped_symbols=unmapped,
    )

def ds1_records():
    """training, validation data set"""
    return [str(r) for r in CFG["data"]["ds1"]]

def ds2_records():
    """Final test data set."""
    return [str(r) for r in CFG["data"]["ds2"]]

def all_records():
    return ds1_records() + ds2_records()

if __name__ == "__main__":
    ensure_database()
    r = load_record(100)
    print(f"record {r.record_id}: lead={r.lead_name} fs={r.fs} "
          f"n_beats={len(r)} dropped_non_beat={r.n_non_beat_dropped}")
    print("  class counts:", {c: int((r.labels == c).sum()) for c in CLASSES + ("Q",)})