"""
The edge model

One architecture, three input arms.

Four hardware rules, all from the MAX78002 CNN accelerator:
  1. Conv stride is fixed at 1, so every downsample is a pooling layer.
  2. Pooling precedes the convolution. ai8x fuses the pair as
     FusedMaxPoolConv1dReLU, and synthesis rejects the opposite order.
  3. ReLU is the only supported activation.
  4. The flattened vector feeding the classifier must be <= 1024.

No softmax layer: CrossEntropyLoss applies it internally, and the chip
predicts with argmax over the raw logits. Adding one here would flatten the
gradients and change nothing at inference.
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import yaml

_ROOT = Path(__file__).resolve().parent.parent
with open(_ROOT / "configs" / "model.yaml") as _fh:
    MCFG: dict = yaml.safe_load(_fh)

FC_IN_LIMIT = 1024 # MAX78002: maximum fully-connected input width
KERNEL_LIMIT = 9 # MAX78002: maximum Conv1d kernel size


class EdgeCNN(nn.Module):
    """1D CNN for N/S/V/F, sized by the input arm."""
    def __init__(self, channels: int, length: int, cfg: dict | None = None):
        super().__init__()
        a = cfg or MCFG["architecture"]
        k, pad = a["kernel_size"], a["padding"]
        if k > KERNEL_LIMIT:
            raise ValueError(f"kernel {k} exceeds the MAX78002 limit of {KERNEL_LIMIT}")

        layers: list[nn.Module] = []
        ch, ln = channels, length
        self.layout: list[tuple] = []

        for out, pool in a["blocks"]:
            if pool:
                layers.append(nn.MaxPool1d(pool))
                ln //= pool
            layers += [nn.Conv1d(ch, out, k, padding=pad), nn.ReLU()]
            self.layout.append((ch, out, pool, ln))
            ch = out

        layers.append(nn.MaxPool1d(a["final_pool"]))
        ln //= a["final_pool"]

        self.fc_in = ch * ln
        if self.fc_in > FC_IN_LIMIT:
            raise ValueError(
                f"flatten is {self.fc_in}, above the MAX78002 limit of {FC_IN_LIMIT}. "
                "Add a pooling stage or reduce the channel count."
            )

        layers += [nn.Flatten(), nn.Linear(self.fc_in, a["n_classes"])]
        self.net = nn.Sequential(*layers)
        self.channels, self.length = channels, length

    def forward(self, x: torch.Tensor):
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def arch(arm: str) -> dict:
    """The architecture for arms"""
    a = MCFG["arms"][arm]
    return {**MCFG["architecture"], **a.get("architecture", {})}


def build(arm: str):
    """Construct the model for one named input arm."""
    a = MCFG["arms"][arm]
    return EdgeCNN(a["channels"], a["length"], arch(arm))


def layer_table(arm: str):
    """one row per operation, with shapes and parameters."""
    m = build(arm)
    a = arch(arm)
    k, pad = a["kernel_size"], a["padding"]
    rows, n = [], 1

    for cin, cout, pool, ln in m.layout:
        op = f"Conv1d {cin}->{cout} k={k} p={pad}, ReLU"
        if pool:
            op = f"MaxPool{pool} -> " + op
        rows.append({"#": n, "operation": op,
                     "output": f"{cout} x {ln}",
                     "params": cin * cout * k + cout})
        n += 1

    rows.append({"#": n,
                 "operation": f"MaxPool{a['final_pool']} -> Flatten -> "
                              f"Linear {m.fc_in}->{a['n_classes']}",
                 "output": str(a["n_classes"]),
                 "params": m.fc_in * a["n_classes"] + a["n_classes"]})
    return rows


if __name__ == "__main__":
    for arm in MCFG["arms"]:
        m = build(arm)
        a = MCFG["arms"][arm]
        print("=" * 72)
        print(f"ARM {arm}  —  input {a['channels']} x {a['length']}")
        print("=" * 72)
        print(f"{'#':>3} {'operation':<46}{'output':>12}{'params':>10}")
        rows = layer_table(arm)
        for r in rows:
            print(f"{r['#']:>3} {r['operation']:<46}{r['output']:>12}{r['params']:>10,}")
        total = sum(r["params"] for r in rows)
        assert total == m.n_params, f"table {total} != model {m.n_params}"
        print(f"{'':>3} {'TOTAL':<46}{'':>12}{total:>10,}")
        print(f"    INT8 weights   {total/1024:.1f} KiB "
              f"({100*total/1024/2340:.2f} % of the 2,340 KiB budget)")
        print(f"    flatten        {m.fc_in}  (limit {FC_IN_LIMIT})"
              f"{'   <-- no headroom' if m.fc_in == FC_IN_LIMIT else ''}")
        x = torch.randn(2, a["channels"], a["length"])
        print(f"    forward        {tuple(x.shape)} -> {tuple(m(x).shape)}")
        print()
