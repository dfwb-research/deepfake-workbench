"""Losses (registry ``losses``): ``forward(out, batch) -> LossOutput(total, parts)``.

Every built-in shares this one contract, so a training loop dispatches through the registry
rather than special-casing loss names.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dfwb.core.detector import ClipBatch, DetectorOutput
from dfwb.core.errors import ConfigError, ContractError

__all__ = [
    "BCELoss",
    "CELoss",
    "FocalLoss",
    "LabelSmoothingBCELoss",
    "LossOutput",
]


@dataclass(frozen=True)
class LossOutput:
    """A loss's ``total`` (backpropagated) plus its named ``parts`` (for logging)."""

    total: Tensor
    parts: dict[str, Tensor]


def _labels(batch: ClipBatch, *, where: str) -> Tensor:
    if batch.labels is None:
        raise ContractError(
            f"{where}: batch.labels is required to compute a training loss",
            hint="build the batch with labels (train and val batches carry them)",
        )
    return batch.labels


def _binary_logit(out: DetectorOutput, *, where: str) -> Tensor:
    if out.logit is None:
        raise ContractError(
            f"{where}: DetectorOutput.logit is required to compute this loss",
            hint="run the detector's forward() (not predict()) before computing a training loss",
        )
    return out.logit


class BCELoss(nn.Module):  # type: ignore[misc, unused-ignore]  # Any without torch
    """Binary cross-entropy on the detector's logit ``[B]`` against float labels."""

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, out: DetectorOutput, batch: ClipBatch) -> LossOutput:
        labels = _labels(batch, where="bce").float()
        logit = _binary_logit(out, where="bce")
        pos_weight = (
            torch.as_tensor(self.pos_weight, dtype=logit.dtype, device=logit.device)
            if self.pos_weight is not None
            else None
        )
        loss = F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_weight)
        return LossOutput(total=loss, parts={"bce": loss})


class CELoss(nn.Module):  # type: ignore[misc, unused-ignore]  # Any without torch
    """Categorical cross-entropy on ``[B,K]`` logits, for multi-class (``K>1``) heads."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, out: DetectorOutput, batch: ClipBatch) -> LossOutput:
        labels = _labels(batch, where="ce")
        logit = _binary_logit(out, where="ce")
        if logit.ndim == 1:
            raise ConfigError(
                "ce: the detector's logit is 1-D (a binary head); use 'bce' for binary heads",
                hint="'ce' needs a multi-class head whose logit is [B, num_classes]",
            )
        loss = F.cross_entropy(logit, labels.long(), label_smoothing=self.label_smoothing)
        return LossOutput(total=loss, parts={"ce": loss})


class FocalLoss(nn.Module):  # type: ignore[misc, unused-ignore]  # Any without torch
    """Binary focal loss on the logit. ``gamma=0`` and ``alpha=None`` is exactly BCE."""

    def __init__(self, alpha: float | None = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, out: DetectorOutput, batch: ClipBatch) -> LossOutput:
        labels = _labels(batch, where="focal").float()
        logit = _binary_logit(out, where="focal")
        p = torch.sigmoid(logit)
        p_t = p * labels + (1 - p) * (1 - labels)
        per_sample = F.binary_cross_entropy_with_logits(logit, labels, reduction="none")
        per_sample = ((1 - p_t) ** self.gamma) * per_sample
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            per_sample = alpha_t * per_sample
        loss = per_sample.mean()
        return LossOutput(total=loss, parts={"focal": loss})


class LabelSmoothingBCELoss(nn.Module):  # type: ignore[misc, unused-ignore]  # Any without torch
    """BCE with smoothed targets: ``y(1-eps) + eps/2``."""

    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, out: DetectorOutput, batch: ClipBatch) -> LossOutput:
        labels = _labels(batch, where="label-smoothing-bce").float()
        logit = _binary_logit(out, where="label-smoothing-bce")
        smoothed = labels * (1 - self.eps) + self.eps / 2
        loss = F.binary_cross_entropy_with_logits(logit, smoothed)
        return LossOutput(total=loss, parts={"label-smoothing-bce": loss})
