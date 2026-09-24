"""Run the pipeline over all 44 records and verify the totals
and make sure we are not loosing lots of data from the original data"""
import sys, time; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from io_utils import CFG, CLASSES
from prepare_data import prepare_all
t0 = time.perf_counter()
res = prepare_all()
d, rows = res["data"], pd.DataFrame(res["per_record"])
rows.to_csv("results/dataset_stats.csv", index=False)
print(f"prepared in {time.perf_counter()-t0:.1f}s -> {res['path']}")
audit = pd.read_csv("results/beat_counts.csv").set_index("record")
rows = rows.set_index("record")
print()
print("-" * 70)
print("1. RECONCILIATION")
print("-" * 70)
bad = []
for r, row in rows.iterrows():
    exp = int(audit.loc[r, list(CLASSES)].sum())
    got = row.kept + row.dropped_rr + row.dropped_edge
    if got != exp:
        bad.append((r, exp, got))
print(f"   records checked : {len(rows)}")
print(f"   mismatches      : {len(bad)}  {bad if bad else ''}")

print()
print("2. CLASS TOTALS")
print("-" * 70)
print(f"{'split':6}{'class':>7}{'audit':>10}{'prepared':>11}{'diff':>8}")
for sp, mask in [("DS1", d["is_ds1"]), ("DS2", ~d["is_ds1"])]:
    sub = audit[audit.index.isin(rows[rows.split == sp].index)]
    for i, c in enumerate(CLASSES):
        a, p = int(sub[c].sum()), int((d["labels"][mask] == i).sum())
        print(f"{sp:6}{c:>7}{a:>10,}{p:>11,}{p-a:>+8,}")

print()
print("3. DROPS")
print("-" * 70)
print(f"   no RR (first/last beat of a record) : {rows.dropped_rr.sum():,}")
print(f"   too close to record start/end       : {rows.dropped_edge.sum():,}")
print(f"   total dropped                       : {rows.dropped_rr.sum()+rows.dropped_edge.sum():,}"
      f"  ({100*(rows.dropped_rr.sum()+rows.dropped_edge.sum())/int(audit[list(CLASSES)].sum().sum()):.3f}%)")
print()
print("4. ARRAY HEALTH")
print("=" * 70)
w = d["windows"]
print(f"   windows          {w.shape}  {w.dtype}   {w.nbytes/1e6:.0f} MB in memory")
print(f"   NaN / inf        {int(np.isnan(w).sum())} / {int(np.isinf(w).sum())}")
print(f"   amplitude        min {w.min():+.2f}   max {w.max():+.2f}   "
      f"clip at {CFG['normalisation']['clip']}  ->  {100*np.mean(np.abs(w)>=CFG['normalisation']['clip']-1e-6):.4f}% clipped")
nmask = d["labels"] == 0
print(f"   R-peak index     median {int(np.median(np.argmax(np.abs(w[nmask]), axis=1)))}  "
      f"(N beats; target {CFG['segmentation']['pre']})")
print(f"   RR seconds       prev {np.median(d['rr_prev']):.2f}   next {np.median(d['rr_next']):.2f}   "
      f"local {np.median(d['rr_local']):.2f}   NaN {int(np.isnan(d['rr_prev']).sum())}")
print()
print("5. FINAL DATASET  (section 3.8)")
print("=" * 70)
tot = len(w)
sub = d["subset"]
print(f"{'split':12}{'records':>9}{'N':>9}{'S':>8}{'V':>8}{'F':>7}{'total':>9}{'%F':>7}")
table = []
for name in ("train", "val", "test"):
    m = sub == name
    c = [int((d["labels"][m] == i).sum()) for i in range(4)]
    nrec = int((rows.subset == name).sum())
    table.append({"split": name, "records": nrec, **dict(zip(CLASSES, c)), "total": sum(c)})
    print(f"{name:12}{nrec:>9}{c[0]:>9,}{c[1]:>8,}{c[2]:>8,}{c[3]:>7,}{sum(c):>9,}"
          f"{100*c[3]/max(sum(c),1):>7.2f}")
c = [int((d["labels"] == i).sum()) for i in range(4)]
print(f"{'TOTAL':12}{len(rows):>9}{c[0]:>9,}{c[1]:>8,}{c[2]:>8,}{c[3]:>7,}{tot:>9,}")
print(f"{'% of all':12}{'':>9}" + "".join(f"{100*x/tot:>8.1f}" for x in c[:3]) + f"{100*c[3]/tot:>7.1f}")
pd.DataFrame(table).to_csv("results/split_stats.csv", index=False)
print()
print("   records per subset:")
for name in ("train", "val", "test"):
    print(f"      {name:6} {sorted(rows[rows.subset==name].index.tolist())}")
print()