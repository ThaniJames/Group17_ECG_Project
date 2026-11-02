"""
MIT-BIH beats (data/processed/beats.npz) for ai8x-training, input arm B2.

Copied into ai8x-training/datasets/ by max78002/run_toolchain.sh; expects
<--data>/ECG/beats.npz.

Mirrors src/dataset.py: the same 5 validation records (by record, never by
beat), the same augmentation on the training split only, and the same capped
inverse-frequency class weights. DS2 is only reachable through 'ECG_DS2',
which refuses to load for training.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ('N', 'S', 'V', 'F')
CLIP = 4.0                      # configs/prep.yaml normalisation.clip
AMP_LO, AMP_HI = 0.9, 1.1       # configs/model.yaml augmentation
MAX_SHIFT = 4
P_APPLY = 0.5
WEIGHT_CAP = 20                 # configs/model.yaml training.class_weight_cap
# Values src/dataset.class_weights() gives on the train split; checked at load time.
WEIGHTS = (0.0723, 1.4456, 1.0366, 1.4456)


def class_weights(labels):
    """Same formula as src/dataset.class_weights."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    w = counts.sum() / np.maximum(counts, 1)
    w = np.minimum(w, WEIGHT_CAP * w.min())
    return w / w.mean()


class ECGBeats(Dataset):
    """Beats from the given subsets; augmentation only on rows from augment_subsets."""
    def __init__(self, path, subsets, args, augment_subsets=()):
        d = np.load(path, allow_pickle=False)
        subset = d['subset']
        idx = np.flatnonzero(np.isin(subset, subsets))
        # [-4, 4] -> [-1, 1], the accelerator's Q7 input range
        self.x = (d['windows'][idx] / CLIP).astype(np.float32)
        self.y = d['labels'][idx].astype(np.int64)
        self.record = d['record'][idx]
        self.subset = subset[idx]
        self.augment = np.isin(self.subset, augment_subsets)
        self.args = args

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment[i]:
            if torch.rand(1).item() < P_APPLY:
                x = np.roll(x, int(torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (1,))))
            if torch.rand(1).item() < P_APPLY:
                x = x * (AMP_LO + (AMP_HI - AMP_LO) * torch.rand(1).item())
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, :]
        # same quantisation as ai8x.normalize: Q7 integers in 8-bit mode, /128 otherwise
        t = t.mul(128.).round().clamp(min=-128, max=127)
        if not self.args.act_mode_8bit:
            t = t.div(128.)
        return t, int(self.y[i])


def ecg_get_datasets(data, load_train=True, load_test=True):
    """Train = train + val records with val marked by valid_indices; test = val records."""
    data_dir, args = data
    path = os.path.join(data_dir, 'ECG', 'beats.npz')

    train_dataset = None
    if load_train:
        train_dataset = ECGBeats(path, ('train', 'val'), args, augment_subsets=('train', ))
        # distiller's get_data_loaders uses this instead of a random beat-level split
        train_dataset.valid_indices = np.flatnonzero(train_dataset.subset == 'val').tolist()
        assert train_dataset.x.shape[1] == 256, 'beats.npz is stale; rerun scripts/run_prepare.py'
        w = class_weights(train_dataset.y[train_dataset.subset == 'train'])
        assert np.allclose(w, WEIGHTS, atol=1e-3), f'class weights drifted: {w}'

    test_dataset = ECGBeats(path, ('val', ), args) if load_test else None
    return train_dataset, test_dataset


def ecg_ds2_get_datasets(data, load_train=True, load_test=True):
    """DS2 final test set. Evaluation only (train.py --evaluate)."""
    if load_train:
        raise RuntimeError('ECG_DS2 is the held-out test set and must never be trained on')
    data_dir, args = data
    path = os.path.join(data_dir, 'ECG', 'beats.npz')
    return None, ECGBeats(path, ('test', ), args) if load_test else None


datasets = [
    {
        'name': 'ECG',
        'input': (1, 256),
        'output': CLASSES,
        'weight': WEIGHTS,
        'loader': ecg_get_datasets,
    },
    {
        'name': 'ECG_DS2',
        'input': (1, 256),
        'output': CLASSES,
        'weight': WEIGHTS,
        'loader': ecg_ds2_get_datasets,
    },
]
