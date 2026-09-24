"""Confirm the R-peak lands where we think after filtering + decimation"""
import sys; sys.path.insert(0, 'src')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io_utils import CFG, load_record, all_records
from prepare_data import filter_signal, resample_signal, rescale_peaks, segment
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 160, 'savefig.bbox': 'tight',
                     'axes.grid': True, 'grid.alpha': .25})
FS_IN = CFG['data']['fs_original']
FS_OUT = CFG['resample']['fs_target']
PRE, POST = CFG['segmentation']['pre'], CFG['segmentation']['post']
DELAY = CFG['resample']['group_delay_samples']
LABEL = f'delay=+{DELAY}'
print('Where does the R-PEAK actually land?  (target index = %d)' % PRE)
print(f"{'rec':>5}{'no compensation':>18}{LABEL:>12}{'n(N beats)':>12}")
rows = []
for rid in all_records()[:12]:
    r = load_record(rid, keep_q=False)
    ds = resample_signal(filter_signal(r.signal, FS_IN), FS_IN, FS_OUT)
    npk = r.r_peaks[r.labels == 'N']
    if len(npk) < 50:
        continue
    res = []
    for d in (0, DELAY):
        w, keep = segment(ds, rescale_peaks(npk, FS_IN, FS_OUT, d), PRE, POST)
        res.append(np.median(np.argmax(w, axis=1)))
    rows.append(res)
    print(f'{rid:>5}{res[0]:>18.0f}{res[1]:>12.0f}{len(npk):>12,}')
rows = np.array(rows)
print(f"\n  median position: no compensation = {np.median(rows[:,0]):.0f}, "
      f"{LABEL} = {np.median(rows[:,1]):.0f}   (target {PRE})")

rid = '101'
r = load_record(rid, keep_q=False)
filt = filter_signal(r.signal, FS_IN)
ds = resample_signal(filt, FS_IN, FS_OUT)
npk = r.r_peaks[r.labels == 'N']

fig, ax = plt.subplots(2, 2, figsize=(12, 6.4))

# raw vs filtered, 5 s
a = ax[0, 0]; lo = 60 * FS_IN; n = 5 * FS_IN
t = np.arange(n) / FS_IN
a.plot(t, r.signal[lo:lo+n], lw=.8, color='#a0aec0', label='raw')
a.plot(t, filt[lo:lo+n], lw=.9, color='#1a202c', label='filtered')
pk = r.r_peaks[(r.r_peaks >= lo) & (r.r_peaks < lo+n)]
a.plot((pk-lo)/FS_IN, r.signal[pk], 'v', ms=5, color='#e53e3e')
a.legend(frameon=False, fontsize=8); a.set_title('raw vs causal-filtered, 5 s', loc='left')
a.set_xlabel('s'); a.set_ylabel('mV')

# zoom on one beat: the shift
a = ax[0, 1]; c = npk[40]; w = 40
t = np.arange(-w, w) / FS_IN * 1000
a.plot(t, r.signal[c-w:c+w], lw=1.5, color='#a0aec0', label='raw')
a.plot(t, filt[c-w:c+w], lw=1.5, color='#1a202c', label='filtered')
a.axvline(0, color='#e53e3e', lw=1, ls='--', label='annotation')
a.axvline(1000*3/FS_IN, color='#2b6cb0', lw=1, ls=':', label='+3 (filter only)')
a.legend(frameon=False, fontsize=8)
a.set_title('group delay: filter chain alone shifts the peak +3 samples (8.3 ms).\n'
            'decimate adds another 4.5 -> total 8 used for windowing.', loc='left', fontsize=8.5)
a.set_xlabel('ms from annotation')

# overlaid windows, both variants
for j, d in enumerate((0, DELAY)):
    a = ax[1, j]
    w_, keep = segment(ds, rescale_peaks(npk, FS_IN, FS_OUT, d), PRE, POST)
    sel = w_[np.linspace(0, len(w_)-1, 150).astype(int)]
    a.plot(sel.T, lw=.4, alpha=.35, color='#2b6cb0')
    a.plot(sel.mean(0), lw=2, color='#1a202c', label='mean')
    a.axvline(PRE, color='#e53e3e', lw=1.4, ls='--', label=f'target index {PRE}')
    med = np.median(np.argmax(w_, axis=1))
    a.axvline(med, color='#2f855a', lw=1.4, ls=':', label=f'actual peak {med:.0f}')
    a.legend(frameon=False, fontsize=8, loc='upper right')
    a.set_title(f'{"no compensation" if d == 0 else f"delay = +{d} samples"}'
                f'  —  150 N beats, record {rid}', loc='left')
    a.set_xlabel(f'sample index in the {PRE+POST}-sample window at {FS_OUT} Hz')
fig.tight_layout()
fig.savefig('figures/1_alignment_check.png')
print('\nsaved figures/1_alignment_check.png')