'''
Data Audit.

Three things we will check here

1. Class mapping correctness
2. Lead availability
3. Minority-class concentrations
'''

from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import (
    CLASSES,
    CFG,
    all_records,
    ds1_records,
    ds2_records,
    ensure_database,
    load_record,
)
_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results"
# Commonly published inter-patient counts (de Chazal protocol). We compare
# against these as a sanity check only — our own counts are what we report.
EXPECTED = {
    "DS1": {"N": 45800, "S": 950, "V": 3780, "F": 415},
    "DS2": {"N": 44200, "S": 1840, "V": 3220, "F": 390},
}

def audit():
    ensure_database()
    rows, leads = [], []
    unmapped_total: Counter = Counter()
    non_beat_total = 0
    for rid in all_records():
        rec = load_record(rid, keep_q=True)
        split = "DS1" if rid in ds1_records() else "DS2"
        counts = {c: int((rec.labels == c).sum()) for c in CLASSES}
        counts["Q"] = int((rec.labels == "Q").sum())
        sym_counts = Counter(rec.symbols.tolist())
        rows.append(
            {
                "record": rid,
                "split": split,
                **counts,
                "total_4class": sum(counts[c] for c in CLASSES),
                "duration_min": round(len(rec.signal) / rec.fs / 60, 1),
                "symbols": " ".join(
                    f"{s}:{n}" for s, n in sorted(sym_counts.items(), key=lambda x: -x[1])
                ),
            }
        )
        leads.append(
            {
                "record": rid,
                "split": split,
                "channels": " | ".join(rec.all_lead_names),
                "lead_used": rec.lead_name,
                "substituted": rec.substituted_lead,
                "fs": rec.fs,
            }
        )
        unmapped_total.update(rec.unmapped_symbols)
        non_beat_total += rec.n_non_beat_dropped
    counts_df = pd.DataFrame(rows)
    leads_df = pd.DataFrame(leads)
    _RESULTS.mkdir(exist_ok=True)
    counts_df.to_csv(_RESULTS / "beat_counts.csv", index=False)
    leads_df.to_csv(_RESULTS / "lead_audit.csv", index=False)
    report(counts_df, leads_df, unmapped_total, non_beat_total)
    return counts_df, leads_df

def report(counts, leads, unmapped, non_beat_total):
    out: list[str] = []
    def w(line: str = "") -> None:
        out.append(line)
        print(line)
    w("-" * 74)
    w("DATA AUDIT — MIT-BIH Arrhythmia Database")
    w("-" * 74)
    w(f"records audited        : {len(counts)}  ({len(counts[counts.split=='DS1'])} DS1"
      f" + {len(counts[counts.split=='DS2'])} DS2)")
    w(f"non-beat annotations dropped : {non_beat_total:,}"
      "   (rhythm '+', quality '~', artefact '|', …)")
    
    # check 1: mapping correctness
    w("")
    w("-" * 74)
    w("1. CLASS MAPPING CHECK")
    w("-" * 74)
    w(f"{'split':6}{'class':>7}{'ours':>10}{'expected':>11}{'diff':>9}{'':>4}")
    ok = True
    for split in ("DS1", "DS2"):
        sub = counts[counts.split == split]
        for c in CLASSES:
            ours = int(sub[c].sum())
            exp = EXPECTED[split][c]
            diff = ours - exp
            flag = "ok" if abs(diff) <= max(60, 0.05 * exp) else "CHECK"
            ok &= flag == "ok"
            w(f"{split:6}{c:>7}{ours:>10,}{exp:>11,}{diff:>+9,}{flag:>7}")
    w("")
    if unmapped:
        w(f"!! UNMAPPED BEAT SYMBOLS: {dict(unmapped)}")
        w("   These are beats our AAMI_MAP does not cover. Fix io_utils.AAMI_MAP")
        w("   before anyone builds on this data.")
    else:
        w("No unmapped beat symbols. Every beat annotation fell into N/S/V/F/Q.")
    w(f"Q beats (paced / unclassifiable) excluded from the 4-class task: "
      f"{int(counts['Q'].sum()):,}")

    # check 2: leads
    w("")
    w("-" * 74)
    w("2. LEAD AVAILABILITY")
    w("-" * 74)
    w(f"preferred lead: {CFG['lead']['preferred']}")
    subs = leads[leads.substituted]
    if len(subs) == 0:
        w("Every audited record provides the preferred lead. No substitutions.")
    else:
        w(f"{len(subs)} record(s) required a substitute lead:")
        for _, r in subs.iterrows():
            w(f"   record {r.record}  ({r.split})  channels: {r.channels}"
              f"   -> using {r.lead_used}")
    order = leads.channels.value_counts()
    w("")
    w("channel orders present:")
    for combo, n in order.items():
        recs = leads[leads.channels == combo].record.tolist()
        w(f"   {combo:22} x{n:<3} {'' if n > 4 else ' '.join(recs)}")
    w("Note: channel order is NOT uniform — this is why the loader selects by name.")

    # check 3: minority concentration
    w("")
    w("-" * 74)
    w("3. MINORITY-CLASS CONCENTRATION")
    w("-" * 74)
    for c in ("S", "F"):
        for split in ("DS1", "DS2"):
            sub = counts[counts.split == split]
            tot = int(sub[c].sum())
            top = sub.nlargest(4, c)[["record", c]]
            share = 100 * int(top[c].sum()) / max(tot, 1)
            nz = int((sub[c] > 0).sum())
            w(f"{split}  class {c}: {tot:,} beats across {nz}/{len(sub)} records; "
              f"top 4 records hold {share:.0f}%")
            w("        " + "  ".join(f"{r.record}:{int(r[c])}" for _, r in top.iterrows()))
    w("")
    w("   Full per-record counts are in results/beat_counts.csv.")
    w("")
    w("=" * 74)
    w("Result: " + ("mapping looks correct — safe to build on."
                     if ok and not unmapped else
                     "DISCREPANCY FOUND — resolve before building on this data."))
    w("=" * 74)
    (_RESULTS / "audit_summary.txt").write_text("\n".join(out) + "\n")

if __name__ == "__main__":
    audit()
