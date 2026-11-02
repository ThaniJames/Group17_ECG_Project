"""
Arm B2 of src/model.py rebuilt from ai8x layers for MAX78002 (AI87).

Copied into ai8x-training/models/ by max78002/run_toolchain.sh.
Hardware layers: 4 x Conv1d, 1 pooling-only passthrough, 1 Linear = 6.
The final MaxPool is its own layer because the MAX78002 cannot pool and
flatten in the same layer.
"""
from torch import nn

import ai8x


class AI87ECGNet(nn.Module):
    """1 x 256 ECG window -> 4 logits (N, S, V, F). 35,588 parameters with bias."""
    def __init__(
            self,
            num_classes=4,
            num_channels=1,
            dimensions=(256, ),
            bias=False,
            **kwargs
    ):
        super().__init__()
        length = dimensions[0]
        # L 256
        self.conv1 = ai8x.FusedConv1dReLU(num_channels, 16, 5, padding=2,
                                          bias=bias, **kwargs)
        # L 128
        self.conv2 = ai8x.FusedMaxPoolConv1dReLU(16, 32, 5, pool_size=2, pool_stride=2,
                                                 padding=2, bias=bias, **kwargs)
        # L 64
        self.conv3 = ai8x.FusedMaxPoolConv1dReLU(32, 64, 5, pool_size=2, pool_stride=2,
                                                 padding=2, bias=bias, **kwargs)
        # L 32
        self.conv4 = ai8x.FusedMaxPoolConv1dReLU(64, 64, 5, pool_size=2, pool_stride=2,
                                                 padding=2, bias=bias, **kwargs)
        # L 8
        self.pool = ai8x.MaxPool1d(4, 4, **kwargs)
        self.fc = ai8x.Linear(64 * (length // 32), num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def ai87ecgnet(pretrained=False, **kwargs):
    """Constructs the Group 17 ECG network."""
    assert not pretrained
    return AI87ECGNet(**kwargs)


models = [
    {
        'name': 'ai87ecgnet',
        'min_input': 1,
        'dim': 1,
    },
]
