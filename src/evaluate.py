"""Final evaluation on the test set DS2  (report 5.1, 5.2, 5.6)

    python src/evaluate.py

Tests the FP32 models and the INT8 models on DS2 (5 seeds each)
and prints the tables for the report.
Saves metrics.csv, confusion_matrix.csv and noise_results.csv in results/
and confusion_matrix.png, per_class_f1.png in figures/.

For the noise table, run scripts/run_compare.py final first.

TODO: add the baseline model here when it is ready (table 5.1 and the
      "better than the baseline" check in the success criteria).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import BeatDataset, load_beats
from io_utils import CLASSES
from metrics import confusion, summary
from quantize import ARM, CPU, SEEDS, load_models, size_kib
from train import predict

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
MODELS = ['FP32', 'INT8']


# 1. test data
beats = load_beats()
test_set = BeatDataset(beats, 'test', ARM)
test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)
records = test_set.record # record number of each test beat

print('Final evaluation on DS2, seeds:', SEEDS)

# 2. predictions of every model and seed
predictions = {}                    # (model, seed) -> predicted classes
rows = []                           # one row per model and seed for metrics.csv
for seed in SEEDS:
    fp32_model, int8_model = load_models(ARM, seed, beats)
    y_true, pred_fp32 = predict(fp32_model, test_loader, CPU)
    y_true, pred_int8 = predict(int8_model, test_loader, CPU)
    predictions['FP32', seed] = pred_fp32
    predictions['INT8', seed] = pred_int8

    for model in MODELS:
        s = summary(y_true, predictions[model, seed])
        row = {'model': model, 'seed': seed,
               'accuracy': s['accuracy'], 'macro_f1': s['macro_f1'], 'macro_f1_nsv': s['macro_f1_nsv']}
        for c in CLASSES:
            row['precision_' + c] = s['per_class'][c]['precision']
            row['recall_' + c] = s['per_class'][c]['recall']
            row['f1_' + c] = s['per_class'][c]['f1']
        rows.append(row)
        print(f'  seed {seed}  {model:8}  accuracy {s["accuracy"]:.3f}  macro-F1 {s["macro_f1"]:.3f}')

df = pd.DataFrame(rows)
df.to_csv(RESULTS / 'metrics.csv', index=False)

# mean and standard deviation over the 5 seeds
mean = df.groupby('model').mean(numeric_only=True)
std = df.groupby('model').std(ddof=0, numeric_only=True)
n_params = sum(p.numel() for p in fp32_model.parameters())

# 3. table 5.1
print('\nTABLE 5.1  clean test results on DS2 (mean +- sd over 5 seeds)')
header = f'{"":18}'
for model in MODELS:
    header += f'{model:>18}'
print(header)
for metric in ['accuracy', 'macro_f1', 'macro_f1_nsv']:
    line = f'{metric:18}'
    for model in MODELS:
        line += f'{mean.loc[model, metric]:>11.3f} ± {std.loc[model, metric]:.3f}'
    print(line)
print(f'{"parameters":18}{n_params:>18,}{n_params:>18,}')
print(f'{"saved size":18}{size_kib(fp32_model):>14.0f} KiB{size_kib(int8_model):>14.0f} KiB')

# 4. table 5.2
print('\nTABLE 5.2  per-class results on DS2 (mean +- sd over 5 seeds)')
for model in MODELS:
    print(f'\n{model}')
    print(f'{"class":8}{"precision":>16}{"recall":>16}{"F1":>16}')
    for c in CLASSES:
        line = f'{c:8}'
        for metric in ['precision_', 'recall_', 'f1_']:
            line += f'{mean.loc[model, metric + c]:>9.3f} ± {std.loc[model, metric + c]:.3f}'
        print(line)

# 5. confusion matrices
# csv: all 5 seeds added together
# figure: the seed with the middle macro-F1, so it is a typical model
csv_rows = []
middle = {}
for model in MODELS:
    total = np.zeros((4, 4), dtype=int)
    for seed in SEEDS:
        total += confusion(y_true, predictions[model, seed])
    for i, c in enumerate(CLASSES):
        csv_rows.append([model, c] + list(total[i]))

    scores = df[df.model == model].macro_f1.values
    middle_seed = SEEDS[np.argsort(scores)[2]]
    middle[model] = (middle_seed, confusion(y_true, predictions[model, middle_seed]))

columns = ['model', 'true'] + ['pred_' + c for c in CLASSES]
pd.DataFrame(csv_rows, columns=columns).to_csv(RESULTS / 'confusion_matrix.csv', index=False)

for model in MODELS:
    seed, cm = middle[model]
    print(f'\nconfusion matrix {model} (seed {seed}, rows = true class)')
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

# 6. S recall for every patient (INT8 model)
print('\nS recall per DS2 patient (INT8)')
print(f'{"record":>8}{"S beats":>10}{"recall":>8}')
is_s = y_true == 1
for rec in np.unique(records[is_s]):
    beats_of_rec = is_s & (records == rec)
    recall = np.mean([np.mean(predictions['INT8', s][beats_of_rec] == 1) for s in SEEDS])
    print(f'{rec:>8}{beats_of_rec.sum():>10}{recall:>8.3f}')

# record 232 has 75 % of all S beats, so we also check the others without it
others = is_s & (records != 232)
recall = np.mean([np.mean(predictions['INT8', s][others] == 1) for s in SEEDS])
print(f'without record 232:  S recall {recall:.3f}')

# 7. noise results (from scripts/run_compare.py --final)
noise_file = RESULTS / 'noise_test' / 'noise_results_test.csv'
noise = None
if noise_file.exists():
    res = pd.read_csv(noise_file)
    res['condition'] = 'clean'
    res.loc[res.snr_db == 12, 'condition'] = '12 dB'
    res.loc[res.snr_db == 6, 'condition'] = '6 dB'
    res.loc[res.snr_db == 0, 'condition'] = '0 dB'
    # first average the 3 noise types, then mean and sd over the 5 seeds
    per_seed = res.groupby(['kind', 'seed', 'condition'])[['accuracy', 'macro_f1']].mean().reset_index()
    noise_mean = per_seed.groupby(['kind', 'condition'])[['accuracy', 'macro_f1']].mean()
    noise_std = per_seed.groupby(['kind', 'condition'])[['accuracy', 'macro_f1']].std(ddof=0)
    noise = noise_mean.join(noise_std, rsuffix='_sd')
    noise.to_csv(RESULTS / 'noise_results.csv')
    print('\nNOISE on DS2 (average of the 3 noise types)')
    print(noise.round(3))
else:
    print('\nno noise results, run scripts/run_compare.py --final first')

# 8. success criteria from section 1.4
print('\nSUCCESS CRITERIA')
int8, fp32 = mean.loc['INT8'], mean.loc['FP32']

# TODO: also check "better than the baseline" when the baseline is added
if int8.macro_f1 >= 0.55:
    result = 'Achieved'
else:
    result = 'Not achieved'
print(f'Predictive performance: INT8 macro-F1 {int8.macro_f1:.3f} (target 0.55) -> {result}')

ok_s = int8.recall_S >= 0.60
ok_v = int8.recall_V >= 0.80
if ok_s and ok_v:
    result = 'Achieved'
elif ok_s or ok_v:
    result = 'Partially achieved'
else:
    result = 'Not achieved'
print(f'Minority classes: S recall {int8.recall_S:.3f}, V recall {int8.recall_V:.3f} -> {result}')

if noise is not None:
    clean = noise.loc[('INT8', 'clean'), 'macro_f1']
    at_12 = noise.loc[('INT8', '12 dB'), 'macro_f1'] / clean
    at_6 = noise.loc[('INT8', '6 dB'), 'macro_f1'] / clean
    at_0 = noise.loc[('INT8', '0 dB'), 'macro_f1'] / clean
    if at_12 >= 0.90 and at_6 >= 0.75:
        result = 'Achieved'
    else:
        result = 'Not achieved'
    print(f'Noise robustness: 12 dB {at_12:.1%}, 6 dB {at_6:.1%}, 0 dB {at_0:.1%} of clean -> {result}')

drop_f1 = fp32.macro_f1 - int8.macro_f1
drop_s = fp32.recall_S - int8.recall_S
drop_v = fp32.recall_V - int8.recall_V
if drop_f1 <= 0.03 and drop_s <= 0.05 and drop_v <= 0.05:
    result = 'Achieved'
else:
    result = 'Not achieved'
print(f'Quantization: macro-F1 drop {drop_f1:+.4f}, S drop {drop_s:+.3f}, V drop {drop_v:+.3f} -> {result}')

if n_params < 100_000 and size_kib(int8_model) <= 2048:
    result = 'Achieved'
else:
    result = 'Not achieved'
print(f'Model complexity: {n_params:,} parameters, {size_kib(int8_model):.0f} KiB -> {result}')

verification = ROOT / 'max78002' / 'synthesis_output' / 'verification.txt'
if verification.exists() and '11/11 checks passed' in verification.read_text():
    result = 'Achieved'
else:
    result = 'Not achieved'
print(f'MAX78002 compatibility: see max78002/synthesis_output/verification.txt -> {result}')

# 9. figures
# confusion matrices of the middle seed, numbers + % of each true class
fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 3.6))
for ax, model in zip(axes, MODELS):
    seed, cm = middle[model]
    percent = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(percent, cmap='Blues', vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            color = 'white' if percent[i, j] > 0.55 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}\n{percent[i, j]:.0%}', ha='center', va='center', fontsize=7, color=color)
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel('predicted')
    ax.set_title(f'{model} (seed {seed})')
axes[0].set_ylabel('true')
fig.suptitle('Confusion matrices on DS2 — count and row % (median model of five)', y=1.03)
fig.savefig(FIGURES / 'confusion_matrix.png', dpi=160, bbox_inches='tight')
plt.close(fig)

# F1 of each class, one bar per model
fig, ax = plt.subplots(figsize=(6.5, 3.2))
x = np.arange(4)
width = 0.8 / len(MODELS)
for k, model in enumerate(MODELS):
    f1 = [mean.loc[model, 'f1_' + c] for c in CLASSES]
    err = [std.loc[model, 'f1_' + c] for c in CLASSES]
    ax.bar(x + (k - (len(MODELS) - 1) / 2) * width, f1, width=width, yerr=err, capsize=2, label=model)
ax.set_xticks(x, CLASSES)
ax.set_ylabel('F1 on DS2')
ax.set_ylim(0, 1)
ax.legend(frameon=False)
ax.set_title('Per-class F1 on DS2 (mean ± sd over five models)')
fig.savefig(FIGURES / 'per_class_f1.png', dpi=160, bbox_inches='tight')
plt.close(fig)

print('\nsaved results/metrics.csv, results/confusion_matrix.csv, results/noise_results.csv')
print('saved figures/confusion_matrix.png, figures/per_class_f1.png')
