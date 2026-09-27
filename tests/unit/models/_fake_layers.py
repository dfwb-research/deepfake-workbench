"""Fake ``layers`` registry targets for stem-hook tests (no dfwb-torch dependency)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from torch import Tensor, nn


class FakeHP(nn.Module):
    """A stand-in high-pass stem layer whose channel count isn't 3."""

    def __init__(self, in_channels: int = 3, out_channels: int = 9) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Identity3(nn.Module):
    """A stand-in stem layer that already outputs 3 channels."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.out_channels = 3
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def not_a_module(in_channels: int = 3) -> int:
    """A registry target that doesn't build an ``nn.Module`` at all (a contract violation)."""
    return in_channels


class NoOutChannels(nn.Module):
    """A stand-in layer that forgets to expose ``out_channels`` (a contract violation)."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class FakeVideoBackbone(nn.Module):
    """A stand-in ``kind="video"`` backbone for assembly checks: sees ``[B,T,C,H,W]`` whole."""

    kind = "video"
    out_dim = 6

    def __init__(self) -> None:
        super().__init__()
        from dfwb.core.detector import InputSpec

        self.native_input = InputSpec(size=(8, 8), frames=4, value_range=(0.0, 1.0))
        self.proj = nn.Linear(3 * 8 * 8 * 4, 6)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x.reshape(x.shape[0], -1))

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"proj": list(self.proj.parameters())}
