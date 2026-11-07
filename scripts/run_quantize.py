"""Static post-training INT8 quantisation of the trained models  (report 4.5, 5.3)

    python scripts/run_quantize.py

Quantises the five trained B2 models and compares each INT8 model with its FP32
original on the VALIDATION split (the test split is not used here).
Writes results/quant/int8_val_B2.csv.
"""
import sys, time; sys.path.insert(0, 'src')
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from dataset import BeatDataset, load_beats
from metrics import summary
from quantize import ARM, BACKEND, CPU, SEEDS, load_models, size_kib
from train import predict

Path('results/quant').mkdir(parents=True, exist_ok=True)
beats = load_beats()
val = DataLoader(BeatDataset(beats, 'val', ARM), 1024, shuffle=False)

print('-' * 70)
print(f'QUANTISATION  arm {ARM}   backend {BACKEND}   seeds {SEEDS}')
print('-' * 70)
rows = []
for seed in SEEDS:
    t0 = time.perf_counter()
    fp32, int8 = load_models(ARM, seed, beats)
    if seed == SEEDS[0]:
        first = int8
    y, p32 = predict(fp32, val, CPU)
    _, p8 = predict(int8, val, CPU)
    a, b = summary(y, p32), summary(y, p8)
    rows.append({'seed': seed,
                 'fp32_macro_f1': a['macro_f1'], 'int8_macro_f1': b['macro_f1'],
                 'fp32_accuracy': a['accuracy'], 'int8_accuracy': b['accuracy'],
                 'fp32_S_recall': a['per_class']['S']['recall'], 'int8_S_recall': b['per_class']['S']['recall'],
                 'fp32_F_recall': a['per_class']['F']['recall'], 'int8_F_recall': b['per_class']['F']['recall'],
                 'changed_predictions': int((p32 != p8).sum()), 'val_beats': len(y),
                 'fp32_kib': size_kib(fp32), 'int8_kib': size_kib(int8)})
    r = rows[-1]
    print(f"seed {seed}   macro-F1 {r['fp32_macro_f1']:.4f} -> {r['int8_macro_f1']:.4f}   "
          f"{r['changed_predictions']:>3} of {r['val_beats']:,} beats changed   "
          f"{r['fp32_kib']:.0f} -> {r['int8_kib']:.0f} KiB   ({time.perf_counter()-t0:.0f}s)")

df = pd.DataFrame(rows)
df.to_csv('results/quant/int8_val_B2.csv', index=False)

f, i = df.fp32_macro_f1.to_numpy(), df.int8_macro_f1.to_numpy()
print()
print(f'validation macro-F1, mean ± sd over {len(df)} models')
print(f'   FP32 {f.mean():.4f} ± {f.std():.4f}     INT8 {i.mean():.4f} ± {i.std():.4f}     '
      f'difference {(i - f).mean():+.4f}')
print(f"   model size      {df.fp32_kib.mean():.0f} KiB -> {df.int8_kib.mean():.0f} KiB "
      f"({df.fp32_kib.mean() / df.int8_kib.mean():.1f}x smaller)")
print(f'   changed beats   {df.changed_predictions.mean():.0f} of {int(df.val_beats[0]):,} per model')

print()
print('the INT8 model (seed %d):' % SEEDS[0])
print(first.model.net)
print('written to results/quant/int8_val_B2.csv')
