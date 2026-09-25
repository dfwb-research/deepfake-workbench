"""Heads (``dfwb.models.heads``): ``[B,D] -> [B]`` for ``num_classes=1`` (the default), or
``[B,K]``. Both built-ins use the old heads' initialisation: every ``Linear``'s weight is drawn
from a truncated normal with ``std=0.02``, its bias set to zero.
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["Head", "LinearHead", "MLPHead"]

_INIT_STD = 0.02


def _init_linear(module: nn.Linear) -> None:
    nn.init.trunc_normal_(module.weight, std=_INIT_STD)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class Head(nn.Module):
    """``[B,D] -> [B]`` (``num_classes=1``) or ``[B,K]``. ``dim`` is given by the assembler.

    ``num_classes`` is part of the interface (every head sets it in ``__init__``): scoring
    (``AssembledDetector.predict``) only supports ``num_classes == 1``, so it can check this
    upfront without knowing the concrete head class.
    """

    num_classes: int

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class LinearHead(Head):
    """Dropout, then one ``Linear`` layer."""

    def __init__(self, dim: int, num_classes: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, num_classes)
        _init_linear(self.linear)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.linear(self.dropout(x))
        return out.squeeze(-1) if self.num_classes == 1 else out


class MLPHead(Head):
    """``Linear -> [BatchNorm] -> GELU -> Dropout -> Linear``.

    ``norm=True`` puts a ``BatchNorm1d`` after the first ``Linear``, as the old MLP-BN head did.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int = 1,
        hidden: int = 512,
        dropout: float = 0.0,
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(dim, hidden)
        self.norm = nn.BatchNorm1d(hidden) if norm else None
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.fc1(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1) if self.num_classes == 1 else out
