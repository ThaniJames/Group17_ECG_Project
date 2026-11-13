# D7043E — Edge-Ready ECG Classification (MIT-BIH → MAX78002)

**Group 17.** Four-class heartbeat classification (N / S / V / F) with a compact 1D CNN designed
for the Analog Devices MAX78002 Edge-AI accelerator, following CRISP-DM.

| | |
|---|---|
| Report | [`reports/Group17_Final_Report.pdf`](reports/Group17_Final_Report.pdf) |
| Presentation | [slides](PresenationVideo/Group17_ECG_Project.pdf) and [videos](PresenationVideo/) |
| Exploratory analysis | [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) |
| Data preparation | [`src/prepare_data.py`](src/prepare_data.py), [`scripts/run_prepare.py`](scripts/run_prepare.py) |
| Model and training | [`src/model.py`](src/model.py), [`src/train.py`](src/train.py) |
| Generated statistics | [`results/`](results/) |
| Training runs | [`results/runs/`](results/runs/) |
| MAX78002 deployment | [`max78002/`](max78002/) |

---

## Setup

```bash
bash run.sh
```

`run.sh` creates `.venv`, installs [`requirements.txt`](requirements.txt), and runs the data preparation
below. Requirements:

- **Python 3.12 or newer.** `numpy` and `scipy` require 3.12; `run.sh` stops with an error if it
  cannot find a suitable interpreter.
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

## Training

```bash
.venv/bin/python src/model.py                        # layer tables for every arm, no training
.venv/bin/python src/dataset.py                      # shapes, class counts, class weights
.venv/bin/python src/train.py --arm B2 --seed 20260906   # one run, about a minute
.venv/bin/python src/train.py --all                  # the full ablation, 4 arms x 5 seeds, ~12 min
```

`src/train.py` **never loads the test split.** DS2 is evaluated once, on a frozen model, by
`src/evaluate.py` — not here.

### The four input arms

The same architecture trained on four views of the same beats, so the comparison isolates the input
representation. S beats are defined by *prematurity* — a property of the rhythm, not the waveform —
so a single-beat window cannot contain the evidence needed to identify one.

| arm | input | what it tests |
|---|---|---|
| A | 1 × 128 — the beat alone | control: morphology only, rhythm-blind by construction |
| B | 1 × 256 — a longer window | prematurity visible as geometry |
| B2 | 1 × 256, pooled once more | **selected** — arm B off the MAX78002 FC ceiling |
| C | 5 × 128 — beat + 4 RR ratios | prematurity supplied as explicit features |

Arms are defined in [`configs/model.yaml`](configs/model.yaml). An arm may override individual
architecture keys; B2 uses this to pool once more, which is why its flatten is 512 rather than 1024.

### Results on the validation split

| arm | macro-F1 (N/S/V) | S recall | accuracy | parameters | flatten |
|---|---|---|---|---|---|
| A | 0.615 ±0.031 | 0.091 | 0.911 | 35,588 | 512 |
| B | 0.721 ±0.011 | 0.277 | 0.950 | 37,636 | 1024 — at the limit |
| **B2** | **0.717 ±0.015** | **0.275** | **0.951** | **35,588** | **512** |
| C | 0.670 ±0.008 | 0.440 | 0.906 | 35,908 | 512 |

Mean ± standard deviation over five seeds. Runs are deterministic: re-running `--all` reproduces
these figures exactly, not approximately.

**B2 is the selected model.** B and B2 are statistically indistinguishable (paired t-test over
matched seeds, p = 0.46), so the selection metric cannot separate them; the MAX78002 flatten limit
of 1024 breaks the tie. Arm C's higher S recall comes from predicting S 3.2× more often than S
occurs — it sits on B2's own precision–recall curve rather than above it.

### The selected architecture

```
input 1 x 256                                           (1.42 s at 180 Hz)
 1  Conv1d  1->16  k=5 p=2, ReLU                16 x 256         96
 2  MaxPool2 -> Conv1d 16->32  k=5 p=2, ReLU    32 x 128      2,592
 3  MaxPool2 -> Conv1d 32->64  k=5 p=2, ReLU    64 x  64     10,304
 4  MaxPool2 -> Conv1d 64->64  k=5 p=2, ReLU    64 x  32     20,544
 5  MaxPool4 -> Flatten -> Linear 512->4               4      2,052
                                                   TOTAL     35,588
```

35 KiB of the 2,340 KiB weight budget (1.5 %). Four hardware rules shape the design, and
`src/model.py` raises an error rather than building a model that breaks them: conv stride is fixed
at 1 so pooling does all downsampling, pooling precedes the convolution, ReLU is the only
activation, and the flattened vector must be ≤ 1024. There is no softmax layer — `CrossEntropyLoss`
applies it internally and the chip predicts with argmax over the raw logits.

### What each run writes

```
results/runs/<arm>_<seed>/
    best.pt        checkpoint at the best validation macro-F1(N/S/V)
    history.json   per-epoch loss, validation metrics, learning rate
    report.txt     per-class precision/recall/F1 and the confusion matrix
    run.json       the full configuration, environment and validation metrics
```

`run.json` records the exact config used, so a result can always be traced back to the settings
that produced it.

### Cross-validation and the class-weight sweep

```
results/cv2/     5-fold cross-validation inside DS1, arm B2 (folds balanced by record count,
                 then by S count) + per_record.json, the per-patient breakdown
results/sweep/   class_weight_cap swept over 20 / 40 / 60 / 93
```

Cross-validation exists because the single validation split rests on five patients. Pooled over all
943 DS1 S beats, S recall is **0.455 ±0.029**; restricted to the seven patients with 30 or more S
beats, it is **0.266 ±0.294**, and across those seven it ranges from 0.000 to 0.786 — two patients
with 106 and 73 S beats are missed entirely. S detection is strongly patient-dependent, and
`results/cv2/per_record.json` has the full table. Patients with only a handful of S beats are
excluded from that figure because a recall of 0 or 1 out of 2 carries no information.

---

## Final evaluation on DS2

```bash
.venv/bin/python scripts/run_noise.py --final      # noisy DS2 datasets (once)
.venv/bin/python scripts/run_compare.py --final    # FP32 vs INT8 under noise -> results/noise_test/
.venv/bin/python src/evaluate.py                   # FP32 and INT8 on DS2, about a minute
```

`src/evaluate.py` tests the five final B2 models and their INT8 copies on all 49,656 DS2 beats.
It prints the results tables, the S recall for each patient and the success criteria check, and
saves the numbers to `results/` and the plots to `figures/`. The logistic regression baseline is
tested separately with `src/evaluate_baseline_logreg.py`.

| DS2, mean ± sd over 5 seeds | FP32 | INT8 |
|---|---|---|
| Accuracy | 0.863 ± 0.024 | 0.867 ± 0.024 |
| Macro-F1 | 0.468 ± 0.027 | 0.470 ± 0.028 |

---

## MAX78002 deployment

The selected model (B2) is retrained, quantized and synthesized with Analog Devices'
[ai8x-training](https://github.com/analogdevicesinc/ai8x-training) and
[ai8x-synthesis](https://github.com/analogdevicesinc/ai8x-synthesis). These tools run on Linux
(on Windows, inside WSL2) in their own Python environment, separate from `.venv`. Install them as
described in the ai8x-training README, with both repositories in one folder:

```
<AI8X>/ai8x-training/
<AI8X>/ai8x-synthesis/
<AI8X>/.venv/
```

Then, from the project root, with `data/processed/beats.npz` already prepared:

```bash
export AI8X=/path/to/ai8x VENV=$AI8X/.venv     # defaults: /root/max78002, $AI8X/.venv
bash max78002/run_toolchain.sh all              # or one stage at a time, in this order:
bash max78002/run_toolchain.sh install          #   copy model, data loader, policies, YAML, beats.npz
bash max78002/run_toolchain.sh train            #   60 FP32 epochs + 40 QAT epochs, ~5 min on a GPU
bash max78002/run_toolchain.sh quantize         #   INT8 checkpoint
bash max78002/run_toolchain.sh evaluate         #   simulated 8-bit evaluation on the validation records
bash max78002/run_toolchain.sh synth            #   C code for the MAX78002
python max78002/verify_toolchain.py             # independent checks, run with the ai8x environment
```

The data loader (`max78002/model_configuration/ecg.py`) keeps the same five validation records,
class weights and training-only augmentation as `src/dataset.py`. It replaces ai8x's random
beat-level validation split with the record-level one, and refuses to load DS2 for training. Inputs
are divided by 4 to map the normalised range [−4, 4] onto the accelerator's 8-bit input.

| | |
|---|---|
| Hardware layers | 6 — 4 × Conv1d (pooling fused), 1 × MaxPool4, 1 × Linear 512→4 |
| Parameters | 35,588 |
| Weight memory | 35,408 B of 2,396,160 B (1.5 %) + 180 B bias |
| Compute | 1,660,928 MACs per beat |
| Validation, simulated INT8 | accuracy 0.944, macro-F1(N/S/V) 0.681 — identical to the QAT model |

The last line is on the five validation records, not DS2. `verify_toolchain.py` passes 11 checks
(`max78002/synthesis_output/verification.txt`), including that the synthesizer's own simulation of
the generated network is bit-exact with the PyTorch INT8 logits for 40 validation beats of all four
classes. All results are software verified; no board was used.

```
max78002/
    run_toolchain.sh        install | train | quantize | evaluate | synth | all
    verify_toolchain.py     independent checks of the port
    model_configuration/    ai87ecgnet.py (model) · ecg.py (data loader) · ecg-b2.yaml (network)
                            qat_policy_ecg.yaml · schedule_ecg.yaml · sample_ecg.npy
    quantization_output/    training log + QAT checkpoint · INT8 checkpoint · quantize and evaluation logs
    synthesis_output/       ecg_b2/ (cnn.c, cnn.h, weights.h, main.c, ... , log.txt) · verification.txt
```

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

d["windows"]    # (100622, 256) float32 — 96 samples before the R-peak, 160 after, at 180 Hz
d["labels"]     # (100622,)     int8    — 0=N 1=S 2=V 3=F
d["record"]     # (100622,)     int16   — which patient each beat came from
d["subset"]     # (100622,)             — 'train' | 'val' | 'test'
d["is_ds1"]     # (100622,)     bool
d["rr_prev"]    # seconds, float32 — gap to the previous beat
d["rr_next"]    # seconds, float32 — gap to the next beat
d["rr_local"]   # seconds, float32 — median of the previous 4 gaps
```

**The stored window is 256 samples; the network input is a centred slice of it.** Storing wide
means the long-window arm and the time-shift augmentation both use real recorded signal rather than
padding, and no arm requires re-running preprocessing. The 128-sample input is `x[48:176]`.
`src/dataset.py` does the slicing — prefer it over reading the file directly:

```python
from dataset import load_beats, make_loaders
loaders = make_loaders("B2", seed=20260906)   # train / val / test DataLoaders for one arm
```

Every beat keeps its record number, so the data can be re-split without re-running preprocessing.

---

## Final dataset

| Split | Records | N | S | V | F | Total |
|---|---|---|---|---|---|---|
| Train | 17 | 36,571 | 709 | 2,550 | 393 | 40,223 |
| Validation | 5 | 9,250 | 234 | 1,238 | 21 | 10,743 |
| Test (DS2) | 22 | 44,213 | 1,836 | 3,219 | 388 | 49,656 |
| **Total** | **44** | **90,034** | **2,779** | **7,007** | **802** | **100,622** |

Records are patients; no patient appears in more than one split. Of 100,718 beats in the four
target classes, 100,622 were extracted and 96 dropped (0.095 %) — beats too close to the start or
end of a recording to fill the stored window, or with no neighbour on one side and therefore no
measurable RR interval.

The classes are severely imbalanced, and unevenly spread across patients. One DS1 record (209)
holds 41 % of its S beats; one DS2 record (232) holds 75 % of the test set's. Aggregate per-beat
metrics are therefore dominated by a small number of patients, which is why per-class and
per-record figures are reported alongside them.

Full detail in report section 3.8.

---

## Conventions

Every configurable value lives in [`configs/prep.yaml`](configs/prep.yaml) (data) and
[`configs/model.yaml`](configs/model.yaml) (modelling): the random seed, the DS1/DS2 record lists,
lead preference and fallback order, filter cutoffs, resampling target, group-delay compensation,
normalisation parameters, window lengths, the RR window, the input arms, the architecture, every
training hyperparameter and the seed list. Nothing is hard-coded in the source.

The two files are kept separate so a modelling change can never invalidate the prepared dataset.

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

Two more apply to modelling:

- **Model selection uses macro-F1 over N/S/V only.** The validation split holds 21 F beats, far too
  few for a stable F1 — including it would make selection follow noise. F is reported everywhere.
- **The metric definitions live in one place.** `src/metrics.py` is imported by both training and
  evaluation, so a macro-F1 quoted in report section 4.4 means exactly the same thing as one in 5.1.

---

## Repository layout

```
reports/         final report (Group17_Final_Report.pdf / .docx) and each member's sections
PresenationVideo/ presentation slides and recorded videos (final video stored with Git LFS)
notebooks/       EDA.ipynb — exploratory data analysis (report section 2)
configs/         prep.yaml (data) · model.yaml (arms, architecture, training)
src/             io_utils.py (loader) · prepare_data.py (pipeline) · data_audit.py (audit)
                 dataset.py (input arms) · model.py (EdgeCNN) · metrics.py · train.py
                 evaluate.py (final DS2 evaluation) · noise_test.py · quantize.py
scripts/         run_prepare.py · verify_alignment.py
results/         generated statistics · runs/ (ablation) · cv2/ (cross-validation) · sweep/
max78002/        ai8x model configuration, quantization and synthesis output (report section 6)
figures/         generated figures
data/            PhysioNet data and processed arrays — not included, see above
run.sh           sets up .venv, then runs scripts/run_prepare.py
```

`data/`, `.venv/` and `__pycache__/` are excluded by [`.gitignore`](.gitignore). The project brief
does not permit submitting the raw PhysioNet datasets, and both are regenerated by the commands
above.

---

## Rules observed, from the project brief

- Beats from one recording never span train and test — the split is **by record**, following the
  inter-patient partition of de Chazal et al. (2004).
- **DS2 is evaluated once per submission**, on a frozen model, and is never used for model
  selection. `src/train.py` does not load it.
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