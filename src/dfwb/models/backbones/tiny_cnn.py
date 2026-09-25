"""``tiny-cnn``: a tiny convolutional backbone for CPU tests and toy training runs."""

from __future__ import annotations

from torch import Tensor, nn

from dfwb.core.detector import InputSpec
from dfwb.models.backbone import Backbone, BackboneOutput, FreezeSpec

__all__ = ["TinyCNN"]

# Channels of each of the 3 conv blocks; the first block's input is always 3 (RGB).
_CHANNELS: tuple[int, int, int] = (16, 32, 32)

_DEFAULT_FREEZE = FreezeSpec()


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv - BatchNorm - ReLU, then a 2x downsample."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TinyCNN(Backbone):
    """3 conv blocks (conv, batchnorm, relu, then downsample) and global average pooling.

    Native input is 64x64 RGB in ``[0, 1]`` (no normalisation): 3 halvings take it to an 8x8
    feature map of 32 channels, which global average pooling reduces to a ``[N, 32]`` vector.
    """

    kind = "image"
    out_dim = 32
    native_input = InputSpec(size=(64, 64), value_range=(0.0, 1.0), mean=None, std=None)

    def __init__(self, freeze: FreezeSpec = _DEFAULT_FREEZE) -> None:
        super().__init__()
        in_channels = (3, *_CHANNELS[:-1])
        self.block0 = _conv_block(in_channels[0], _CHANNELS[0])
        self.block1 = _conv_block(in_channels[1], _CHANNELS[1])
        self.block2 = _conv_block(in_channels[2], _CHANNELS[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.apply_freeze(freeze)

    def forward(self, x: Tensor) -> BackboneOutput:
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        pooled = self.pool(x).flatten(1)
        return BackboneOutput(pooled=pooled, tokens=None)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "blocks.0": list(self.block0.parameters()),
            "blocks.1": list(self.block1.parameters()),
            "blocks.2": list(self.block2.parameters()),
        }
