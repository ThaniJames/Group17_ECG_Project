"""Compare the FP32 and INT8 models on clean and noisy data  (report 5.3, 5.4)

    python scripts/run_compare.py            # validation split
    python scripts/run_compare.py --final    # validation and test split (DS2)

--final evaluates the test split: do that once, on frozen models.
Needs the noisy datasets from  python scripts/run_noise.py  (with --final for the test split).
Writes tables to results/noise_test/ and figures to figures/.
"""
import sys, time; sys.path.insert(0, 'src')
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import BeatDataset, load_beats
from io_utils import CLASSES
from metrics import summary
from noise_test import CONDITIONS, NOISE_TYPES, SNR_DB, load_dataset
from quantize import ARM, CPU, SEEDS, load_models
from train import predict
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 160, 'savefig.bbox': 'tight',
                     'axes.grid': True, 'grid.alpha': .25})

RECALL = [f'recall_{c}' for c in CLASSES]
METRICS = ['accuracy', 'macro_f1'] + RECALL
ORDER = [np.inf] + sorted(SNR_DB, reverse=True)           # clean, 12, 6, 0
LABELS = ['clean'] + [f'{s} dB' for s in sorted(SNR_DB, reverse=True)]


def evaluate(split, models):
    """Every model on every condition. models = {seed: (fp32, int8)}."""
    rows = []
    for noise, snr in CONDITIONS:
        d = load_dataset(split, noise, snr)
        loader = DataLoader(BeatDataset(d, split, ARM), 1024, shuffle=False)
        for seed, (fp32, int8) in models.items():
            y, p32 = predict(fp32, loader, CPU)
            _, p8 = predict(int8, loader, CPU)
            for kind, p in (('FP32', p32), ('INT8', p8)):
                s = summary(y, p)
                rows.append({'split': split, 'kind': kind, 'seed': seed,
                             'noise': noise or 'clean', 'snr_db': np.inf if snr is None else float(snr),
                             'n_beats': len(y), 'accuracy': s['accuracy'], 'macro_f1': s['macro_f1'],
                             **{f'recall_{c}': s['per_class'][c]['recall'] for c in CLASSES},
                             'changed_vs_fp32': 0 if kind == 'FP32' else int((p32 != p8).sum())})
        print(f"   {noise or 'clean':5} {'' if snr is None else str(snr) + ' dB'}")
    return pd.DataFrame(rows)


def per_seed(res):
    """One row per model and noise level: the three noise types averaged."""
    return res.groupby(['kind', 'seed', 'snr_db'])[METRICS].mean().reset_index()


def mean_sd(df, keys=('kind', 'snr_db')):
    """Mean and standard deviation over the five trained models."""
    g = df.groupby(list(keys))[METRICS]
    return g.mean(), g.std(ddof=0)


def pm(m, s):
    return f'{m:.3f} ± {s:.3f}'


def label(snr):
    return 'clean' if np.isinf(snr) else f'{snr:g} dB'


def to_md(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines) + '\n'


def save(df, out, name):
    df.to_csv(out / f'{name}.csv', index=False)
    (out / f'{name}.md').write_text(to_md(df))


def make_tables(res, split, out):
    ps = per_seed(res)
    mean, sd = mean_sd(ps)

    # 1. the table for the report: clean / 12 / 6 / 0 dB, noise types averaged
    i8 = res[res.kind == 'INT8'].groupby('snr_db')
    changed = 100 * i8.changed_vs_fp32.mean() / i8.n_beats.first()     # % of beats whose class differs from FP32
    rows = []
    for kind in ('FP32', 'INT8'):
        for snr in ORDER:
            m, s = mean.loc[(kind, snr)], sd.loc[(kind, snr)]
            rows.append({'model': kind, 'condition': label(snr), 'accuracy': pm(m.accuracy, s.accuracy),
                         'macro-F1': pm(m.macro_f1, s.macro_f1),
                         'beats changed vs FP32': '' if kind == 'FP32' else f'{changed[snr]:.2f} %'})
    overall = pd.DataFrame(rows)
    save(overall, out, f'table_overall_{split}')

    diff = (mean.loc['INT8'] - mean.loc['FP32']).reindex(ORDER)[['accuracy', 'macro_f1']]
    diff = diff.round(4).reset_index().rename(columns={'snr_db': 'condition'})
    diff['condition'] = diff.condition.map(label)
    save(diff, out, f'table_difference_{split}')

    # 2. per noise type
    rows = []
    for n in NOISE_TYPES:
        m_, s_ = mean_sd(res[(res.noise == n) | (res.noise == 'clean')])
        for kind in ('FP32', 'INT8'):
            for snr in ORDER[1:]:
                m, s = m_.loc[(kind, snr)], s_.loc[(kind, snr)]
                rows.append({'noise': n, 'model': kind, 'condition': label(snr),
                             'accuracy': pm(m.accuracy, s.accuracy), 'macro-F1': pm(m.macro_f1, s.macro_f1)})
    by_noise = pd.DataFrame(rows)
    save(by_noise, out, f'table_by_noise_{split}')

    # 3. recall of each class
    rows = []
    for kind in ('FP32', 'INT8'):
        for snr in ORDER:
            m, s = mean.loc[(kind, snr)], sd.loc[(kind, snr)]
            rows.append({'model': kind, 'condition': label(snr),
                         **{f'recall {c}': pm(m[f'recall_{c}'], s[f'recall_{c}']) for c in CLASSES}})
    per_class = pd.DataFrame(rows)
    save(per_class, out, f'table_per_class_{split}')

    print(f'\n{split.upper()}: report table (noise types averaged, mean ± sd over {ps.seed.nunique()} models)')
    print(overall.to_string(index=False))
    print('\nINT8 minus FP32:')
    print(diff.to_string(index=False))
    print('\nrecall of each class:')
    print(per_class.to_string(index=False))


def make_figures(res, split):
    x = range(len(ORDER))
    # macro-F1 against noise level, one panel per noise type
    fig, axes = plt.subplots(1, len(NOISE_TYPES), figsize=(13, 3.8), sharey=True)
    for ax, n in zip(axes, NOISE_TYPES):
        d = res[(res.noise == n) | (res.noise == 'clean')]
        for kind in ('FP32', 'INT8'):
            g = d[d.kind == kind].groupby('snr_db').macro_f1
            ax.errorbar(x, g.mean().reindex(ORDER), yerr=g.std(ddof=0).reindex(ORDER),
                        marker='o', capsize=3, label=kind)
        ax.set_xticks(list(x), LABELS)
        ax.set_title(n)
    axes[0].set_ylabel('macro-F1')
    axes[0].legend()
    fig.suptitle(f'FP32 vs INT8 under noise ({split} split, mean ± sd over the trained models)', y=1.04)
    fig.savefig(f'figures/compare_{split}_macro_f1.png')
    plt.close(fig)

    # recall of each class, noise types averaged, FP32 vs INT8
    ps = per_seed(res)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(13, 3.6), sharey=True)
    for ax, c in zip(axes, CLASSES):
        for kind in ('FP32', 'INT8'):
            g = ps[ps.kind == kind].groupby('snr_db')[f'recall_{c}']
            ax.errorbar(x, g.mean().reindex(ORDER), yerr=g.std(ddof=0).reindex(ORDER),
                        marker='o', capsize=3, label=kind)
        ax.set_xticks(list(x), LABELS)
        ax.set_title(f'class {c}')
    axes[0].set_ylabel('recall')
    axes[0].legend()
    fig.suptitle(f'Recall of each class under noise ({split} split)', y=1.04)
    fig.savefig(f'figures/compare_{split}_recall.png')
    plt.close(fig)
    print(f'figures/compare_{split}_macro_f1.png, figures/compare_{split}_recall.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true', help='also evaluate the test split (DS2), once')
    args = ap.parse_args()
    out = Path('results/noise_test')
    out.mkdir(parents=True, exist_ok=True)
    Path('figures').mkdir(exist_ok=True)

    beats = load_beats()
    print(f'quantising the {len(SEEDS)} trained {ARM} models ...')
    models = {seed: load_models(ARM, seed, beats) for seed in SEEDS}
    for split in ['val'] + (['test'] if args.final else []):
        print('-' * 70)
        print(f'EVALUATING  {split}')
        print('-' * 70)
        t0 = time.perf_counter()
        res = evaluate(split, models)
        res.to_csv(out / f'noise_results_{split}.csv', index=False)
        make_tables(res, split, out)
        make_figures(res, split)
        print(f'({time.perf_counter() - t0:.0f}s)  results in {out}')
