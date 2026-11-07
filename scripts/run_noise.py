"""Download the noise, build the noisy datasets, plot examples  (report 5.4)

    python scripts/run_noise.py            # validation records
    python scripts/run_noise.py --final    # also the test records (DS2); only once the models are frozen

Writes data/noisy/<split>_clean.npz and <split>_<BW|MA|EM>_<12|6|0>dB.npz,
and the figures noise_signals.png and noise_beats.png.
"""
import sys, time; sys.path.insert(0, 'src')
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io_utils import CLASSES, all_records, load_record
from noise_test import (NOISE_TYPES, SNR_DB, NOISY_DIR, add_noise, achieved_snr_db,
                        dataset_path, load_noise, make_dataset)
from prepare_data import split_of
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 160, 'savefig.bbox': 'tight',
                     'axes.grid': True, 'grid.alpha': .25})

ap = argparse.ArgumentParser()
ap.add_argument('--final', action='store_true', help='also build the test split (DS2)')
args = ap.parse_args()
subsets = ['val'] + (['test'] if args.final else [])
Path('figures').mkdir(exist_ok=True)
NOISY_DIR.mkdir(parents=True, exist_ok=True)

print('-' * 70)
print('1. NOISE RECORDINGS  (second half, channel noise2)')
print('-' * 70)
for n in NOISE_TYPES:
    x = load_noise(n)
    print(f'   {n}   {len(x):>8,} samples   {len(x)/360:6.0f} s   std {x.std():.3f} mV')

rid = [r for r in all_records() if split_of(r) == 'val'][0]
rec = load_record(rid)
print()
print(f'2. SNR CHECK  (record {rid}, whole record)')
print('-' * 70)
print(f"   {'':4}" + ''.join(f'{s:>10} dB' for s in SNR_DB))
for n in NOISE_TYPES:
    got = [achieved_snr_db(rec.signal, add_noise(rec.signal, load_noise(n), s)) for s in SNR_DB]
    print(f'   {n:4}' + ''.join(f'{g:>13.3f}' for g in got))

print()
print('3. DATASETS')
print('-' * 70)
sets = {}
for sp in subsets:
    t0 = time.perf_counter()
    conditions = [(None, None)] + [(n, s) for n in NOISE_TYPES for s in SNR_DB]
    for n, s in conditions:
        d = make_dataset(sp, n, s)
        np.savez_compressed(dataset_path(sp, n, s), **d)
        sets[(sp, n, s)] = d
        if n is None:
            ref = d
        elif not np.array_equal(d['labels'], ref['labels']):
            raise SystemExit('labels differ between clean and noisy data: something is wrong')
    print(f"   {sp:5} {len(ref['labels']):>7,} beats in {len(conditions)} files   {time.perf_counter()-t0:.0f}s")
    beats = Path('data/processed/beats.npz')
    if beats.exists():
        b = np.load(beats)
        same = np.array_equal(sets[(sp, None, None)]['windows'], b['windows'][b['subset'] == sp])
        print(f'         clean file identical to data/processed/beats.npz : {same}')
        if not same:
            print('         -> beats.npz is out of date: run  python scripts/run_prepare.py')
print(f'   saved in {NOISY_DIR}')

print()
print('4. FIGURES')
print('-' * 70)
# raw signal: 6 s of one validation record, every noise type and level
sl = slice(10 * 360, 16 * 360)
t = np.arange(sl.stop - sl.start) / 360
fig, axes = plt.subplots(len(NOISE_TYPES), 1 + len(SNR_DB), figsize=(13, 6.5), sharex=True, sharey=True)
for i, n in enumerate(NOISE_TYPES):
    for j, s in enumerate((None,) + SNR_DB):
        y = rec.signal if s is None else add_noise(rec.signal, load_noise(n), s)
        axes[i, j].plot(t, y[sl], lw=.8)
        axes[i, j].set_title('clean' if s is None else f'{n}, {s} dB')
for ax in axes[-1]:
    ax.set_xlabel('seconds')
for ax in axes[:, 0]:
    ax.set_ylabel('mV')
fig.suptitle(f'Record {rid}: noise added to the raw signal', y=1.0)
fig.savefig('figures/noise_signals.png')
plt.close(fig)
print('   figures/noise_signals.png')

# what the model sees: the same N, S and V beat, after preprocessing
clean = sets[('val', None, None)]
picks = []
for c in (0, 1, 2):
    idx = np.flatnonzero(clean['labels'] == c)
    picks.append(int(idx[min(200, len(idx) - 1)]))
fig, axes = plt.subplots(len(NOISE_TYPES), 3, figsize=(11, 7), sharex=True)
for i, n in enumerate(NOISE_TYPES):
    for j, b in enumerate(picks):
        ax = axes[i, j]
        for s in SNR_DB:
            ax.plot(sets[('val', n, s)]['windows'][b], lw=.9, label=f'{s} dB')
        ax.plot(clean['windows'][b], 'k', lw=1.5, label='clean')
        ax.set_title(f'{n}, {CLASSES[clean["labels"][b]]} beat')
axes[0, 0].legend(fontsize=7, ncol=2)
for ax in axes[-1]:
    ax.set_xlabel('sample (180 Hz)')
fig.suptitle('Beats after preprocessing, clean and noisy')
fig.savefig('figures/noise_beats.png')
plt.close(fig)
print('   figures/noise_beats.png')
