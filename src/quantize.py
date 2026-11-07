"""
Static post-training INT8 quantisation (report 4.5 and 5.3)

PyTorch's eager-mode workflow, no retraining:
  1. stubs      QuantStub / DeQuantStub mark where float becomes int8 and back
  2. fuse       every Conv1d + ReLU becomes one fused layer
  3. prepare    observers record the value range of every layer
  4. calibrate  20 batches of TRAINING beats go through the model
  5. convert    float layers are replaced by int8 layers

Quantised models run on the CPU only. The INT8 model is rebuilt from the FP32
checkpoint whenever it is needed: calibration is seeded, so the result is the
same every time and nothing has to be saved.

This is a software result from PyTorch's own quantiser (weights int8,
activations uint8). It is not a hardware measurement.
"""

from __future__ import annotations
import copy
import io
import platform
import warnings

# torch.ao.quantization is marked deprecated in recent PyTorch (it still works).
warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
warnings.filterwarnings("ignore", message=".*quantized tensor creation functions.*")

import torch
import torch.nn as nn
from torch.ao.quantization import (DeQuantStub, QuantStub, convert, fuse_modules,
                                   get_default_qconfig, prepare)
from torch.utils.data import DataLoader

from dataset import MCFG, BeatDataset
from model import build
from train import RUNS

ARM = "B2"                                   # the model selected by Task 1
SEEDS = MCFG["training"]["seeds"]            # the five trained models
CPU = torch.device("cpu")
CALIB_BATCHES = 20
BACKEND = "qnnpack" if platform.machine().lower() in ("arm64", "aarch64") else "fbgemm"


class QuantWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


def fuse_pairs(m: QuantWrapper) -> list[list[str]]:
    """Every Conv1d that is directly followed by a ReLU."""
    net = m.model.net
    return [[f"model.net.{i}", f"model.net.{i + 1}"] for i in range(len(net) - 1)
            if isinstance(net[i], nn.Conv1d) and isinstance(net[i + 1], nn.ReLU)]


def load_fp32(arm: str, seed: int) -> nn.Module:
    ck = torch.load(RUNS / f"{arm}_{seed}" / "best.pt", map_location=CPU)
    model = build(arm)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


def quantize_int8(fp32: nn.Module, calib_loader: DataLoader) -> nn.Module:
    torch.backends.quantized.engine = BACKEND
    m = QuantWrapper(copy.deepcopy(fp32)).eval()          # the copy keeps fp32 untouched
    m = fuse_modules(m, fuse_pairs(m))
    m.qconfig = get_default_qconfig(BACKEND)
    m = prepare(m)
    with torch.no_grad():
        for i, (xb, _) in enumerate(calib_loader):
            if i == CALIB_BATCHES:
                break
            m(xb)
    return convert(m).eval()


def load_models(arm: str, seed: int, beats: dict):
    """(FP32 model, INT8 model) for one trained seed. beats = data/processed/beats.npz as a dict."""
    train = BeatDataset(beats, "train", arm, augment=False)
    loader = DataLoader(train, 256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    fp32 = load_fp32(arm, seed)
    return fp32, quantize_int8(fp32, loader)


def size_kib(model: nn.Module) -> float:
    """Size of the saved weights in KiB (includes a little file overhead)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024
