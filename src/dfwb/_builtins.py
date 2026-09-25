"""Registers the framework's built-in components through the same plugin API as any plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dfwb.core.plugins import PluginAPI


def register(api: PluginAPI) -> None:
    """Register built-in components.

    Built-ins are added here as their layers are implemented; each one is an import-path target,
    so registering stays cheap and imports nothing heavy.
    """
    api.metrics.add(
        "auc",
        target="dfwb.eval.metrics:auc",
        summary="ROC AUC via the rank statistic, ties averaged; undefined for a single class",
    )
    api.metrics.add(
        "ap",
        target="dfwb.eval.metrics:ap",
        summary="Average precision, the step-wise area under the precision-recall curve",
    )
    api.metrics.add(
        "eer",
        target="dfwb.eval.metrics:eer",
        summary="Equal error rate: the FPR=FNR crossing of the ROC curve, linearly interpolated",
    )
    api.metrics.add(
        "acc",
        target="dfwb.eval.metrics:acc",
        summary="Accuracy at a threshold on P(fake) (thr=0.5 by default)",
    )
    api.metrics.add(
        "tpr",
        target="dfwb.eval.metrics:tpr",
        summary="Largest TPR achievable with FPR at or below fpr= (interp=true to interpolate)",
    )
    api.metrics.add(
        "fpr",
        target="dfwb.eval.metrics:fpr",
        summary="Smallest FPR needed to reach at least tpr= (interp=true to interpolate)",
    )
    api.metrics.add(
        "ece",
        target="dfwb.eval.metrics:ece",
        summary="Expected calibration error over bins= bins (adaptive=true for equal-mass bins)",
    )
    api.metrics.add(
        "brier",
        target="dfwb.eval.metrics:brier",
        summary="Brier score: the mean squared error of P(fake) against the label",
    )
    api.metrics.add(
        "nll",
        target="dfwb.eval.metrics:nll",
        summary="Binary cross-entropy of P(fake) against the label, clipped at 1e-7",
    )
    api.metrics.add(
        "aurc",
        target="dfwb.eval.metrics:aurc",
        summary="Area under the risk-coverage curve for selective prediction",
    )
    api.backbones.add(
        "tiny-cnn",
        target="dfwb.models.backbones.tiny_cnn:TinyCNN",
        summary="A tiny 3-block CNN backbone for CPU tests and toy training runs",
        requires=("torch",),
    )
