"""Temporal pools (``dfwb.models.temporal_pools``): ``[B,T,D] -> [B,D]``, image backbones only.

A ``kind="video"`` backbone already returns one feature vector per clip, so no pool applies to it;
the assembler wires a pool in only for image backbones, always passing ``dim`` (the backbone's
``out_dim``) so every built-in pool shares one constructor shape, whether or not it has its own
learnable parameters.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["AttentionPool", "MaxPool", "MeanPool", "TemporalPool"]


class TemporalPool(nn.Module):
    """``[B,T,D] -> [B,D]``. ``dim`` is given by the assembler, from the backbone's ``out_dim``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class MeanPool(TemporalPool):
    """The mean feature over time."""

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1)


class MaxPool(TemporalPool):
    """The per-channel maximum feature over time."""

    def forward(self, x: Tensor) -> Tensor:
        return x.max(dim=1).values


class AttentionPool(TemporalPool):
    """A small learned pool: one linear score per frame, softmax-weighted over time.

    Takes no parameters besides ``dim``, which the assembler supplies.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.score(x), dim=1)  # [B,T,1]
        return (x * weights).sum(dim=1)
