# D7043E — Edge-Ready ECG Classification (MIT-BIH → MAX78002)

**Group 17.** Four-class heartbeat classification (N / S / V / F) with a compact 1D CNN designed
for the Analog Devices MAX78002 Edge-AI accelerator, following CRISP-DM.

This submission covers CRISP-DM stages 1–3 — business understanding, data understanding and data
preparation. No model results are included at this stage.

| | |
|---|---|
| Report | [`report.pdf`](report.pdf) — stages 1–3 |
| Exploratory analysis | [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) |
| Data preparation code | [`src/`](src/), [`scripts/`](scripts/) |
| Generated statistics | [`results/`](results/) |

---

## Setup

```bash
bash run.sh
```

`run.sh` creates `.venv`, installs [`requirements.txt`](requirements.txt), and runs the data preparation
below. Requirements:

- **Python 3.14.** The pinned versions in `requirements.txt` only support it, and `run.sh` stops with
  an error if it cannot find it.
- **bash.** On Windows use Git Bash.
- **Internet access** on the first run, for the packages and the PhysioNet data.

Without bash (Windows PowerShell):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_prepare.py
```


## Reproducing the preprocessing

```bash
bash run.sh                                  # or: .venv/bin/python scripts/run_prepare.py
```

The first run takes a few minutes, because it installs the packages (torch is large) and downloads
about 90 MB from PhysioNet into `data/`. Every run after that takes a few seconds.

The script:

1. downloads the MIT-BIH Arrhythmia Database if `data/mitdb/` is not already populated,
2. applies the preprocessing pipeline to all 44 records of the DS1 + DS2 partition,
3. saves the prepared dataset to `data/processed/beats.npz`,
4. reconciles its own output against `results/beat_counts.csv`,
5. writes `results/dataset_stats.csv` and `results/split_stats.csv`.

### What a correct run looks like

```
prepared in 2.7s -> data/processed/beats.npz
   records checked : 44
   mismatches      : 0
```

`mismatches : 0` is the check that matters. It means every beat counted in the audit is accounted
for — either kept, or dropped for a stated reason. A non-zero value means beats were silently lost
and the output should not be used.

### The two supporting scripts

```bash
.venv/bin/python src/data_audit.py            # results/beat_counts.csv, lead_audit.csv, audit_summary.txt
.venv/bin/python scripts/verify_alignment.py  # figures/1_alignment_check.png
```

Neither is needed for a normal run, because their outputs are included in the repository.
`data_audit.py` only needs re-running if the class mapping in `src/io_utils.py` changes, since
`run_prepare.py` reconciles against its CSV.

`configs/prep.yaml`, `src/io_utils.py` and `src/prepare_data.py` are imported by the scripts above
rather than run directly.

---

## Changing the train/validation split

Edit one line in [`configs/prep.yaml`](configs/prep.yaml):

```yaml
val: [223, 124, 201, 106, 112]   # DS1 records used for validation
```

Any DS1 record not listed becomes training data; DS2 is always the test set. Then regenerate the
statistics:

```bash
.venv/bin/python scripts/run_prepare.py
```

No other file needs editing — every script reads this list at startup.

---

## Using the prepared dataset

```python
import numpy as np
d = np.load("data/processed/beats.npz", allow_pickle=False)

d["windows"]    # (100630, 128) float32 — 48 samples before the R-peak, 80 after, at 180 Hz
d["labels"]     # (100630,)     int8    — 0=N 1=S 2=V 3=F
d["record"]     # (100630,)     int16   — which patient each beat came from
d["subset"]     # (100630,)             — 'train' | 'val' | 'test'
d["rr_prev"]    # seconds, float32 — gap to the previous beat
d["rr_next"]    # seconds, float32 — gap to the next beat
d["rr_local"]   # seconds, float32 — median of the previous 4 gaps
```

Every beat keeps its record number, so the data can be re-split without re-running preprocessing.

---

## Final dataset

| Split | Records | N | S | V | F | Total |
|---|---|---|---|---|---|---|
| Train | 17 | 36,574 | 709 | 2,550 | 393 | 40,226 |
| Validation | 5 | 9,250 | 234 | 1,238 | 21 | 10,743 |
| Test (DS2) | 22 | 44,218 | 1,836 | 3,219 | 388 | 49,661 |
| **Total** | **44** | **90,042** | **2,779** | **7,007** | **802** | **100,630** |

Records are patients; no patient appears in more than one split. Of 100,718 beats in the four
target classes, 100,630 were extracted and 88 dropped (0.087 %) — the first and last beat of each
recording, which have no neighbour on one side and therefore no measurable RR interval.

Full detail in report section 3.8.

---

## Conventions

Every configurable value lives in [`configs/prep.yaml`](configs/prep.yaml): the random seed, the
DS1/DS2 record lists, lead preference and fallback order, filter cutoffs, resampling target,
group-delay compensation, normalisation parameters, window lengths and the RR window. Nothing is
hard-coded in the source.

All record loading goes through `src.io_utils.load_record()`:

```python
from src.io_utils import load_record, ds1_records, ds2_records

rec = load_record(100)
rec.signal      # 1-D float array, the selected lead, in mV
rec.r_peaks     # sample indices of annotated R-peaks, at 360 Hz
rec.labels      # 'N' | 'S' | 'V' | 'F' | 'Q' per R-peak
rec.symbols     # the original MIT-BIH annotation symbol
rec.lead_name   # which lead was actually used
rec.fs          # 360
```

Three rules are enforced in code, each of which would silently corrupt results if bypassed:

- **Leads are selected by name, never by channel index.** Record 114 stores its channels as
  `V5 | MLII`, the reverse of the usual order — reading channel 0 would load the wrong lead for
  that patient while every other record supplied MLII.
- **Non-beat annotations are filtered out.** The `.atr` files mix rhythm (`+`), signal-quality
  (`~`) and artefact (`|`) markers in with the beat labels; 2,991 of them across DS1 + DS2.
  Counting them as beats would inflate every class total.
- **Filtering is causal.** `sosfilt`, never `filtfilt` — zero-phase filtering needs samples from
  the future and cannot run on a device classifying a live signal. The cost is a measured 8-sample
  group delay, which `rescale_peaks()` compensates before windowing.

---

## Repository layout

```
report.pdf       CRISP-DM report, stages 1–3
reports/         the report sections as written by each member
notebooks/       EDA.ipynb — exploratory data analysis (report section 2)
configs/         prep.yaml — the shared configuration
src/             io_utils.py (loader) · prepare_data.py (pipeline) · data_audit.py (audit)
scripts/         run_prepare.py · verify_alignment.py
results/         generated statistics
figures/         generated figures
data/            PhysioNet data and processed arrays — not included, see above
run.sh           sets up .venv, then runs scripts/run_prepare.py
```

`data/` and `.venv/` are excluded by [`.gitignore`](.gitignore). The project brief does not permit
submitting the raw PhysioNet datasets, and both are regenerated by the commands above.

---

## Rules observed, from the project brief

- Beats from one recording never span train and test — the split is **by record**, following the
  inter-patient partition of de Chazal et al. (2004).
- **DS2 is evaluated once per submission**, on a frozen model, and is never used for model
  selection.
- Accuracy is never reported alone; macro-F1 and per-class metrics are required.
- No claim of clinical validation is made, and no MAX78002 hardware measurement is claimed —
  no physical board was used.

## Data source

Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng in Med and Biol
20(3):45-50, 2001. Available from PhysioNet: https://physionet.org/content/mitdb/

Downloaded automatically by the commands above, or manually:

```python
import wfdb
wfdb.dl_database('mitdb', 'data/mitdb')   # Arrhythmia Database        (90 MB)
wfdb.dl_database('nstdb', 'data/nstdb')   # Noise Stress Test Database
```