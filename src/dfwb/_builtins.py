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
    api.backbones.add(
        "timm",
        target="dfwb.models.backbones.timm_backbone:TimmBackbone",
        summary="Any timm image classification backbone, pooled the model's own way",
        requires=("torch", "timm"),
    )
    api.backbones.add(
        "hf-vision",
        target="dfwb.models.backbones.hf_vision:HFVisionBackbone",
        summary="Any Hugging Face vision transformer (CLIP, DINOv2, SigLIP 2, ...) via AutoModel",
        requires=("torch", "transformers"),
    )
    api.temporal_pools.add(
        "mean",
        target="dfwb.models.pools:MeanPool",
        summary="The mean feature over time",
        requires=("torch",),
    )
    api.temporal_pools.add(
        "max",
        target="dfwb.models.pools:MaxPool",
        summary="The per-channel maximum feature over time",
        requires=("torch",),
    )
    api.temporal_pools.add(
        "attention",
        target="dfwb.models.pools:AttentionPool",
        summary="A small learned pool: a linear score per frame, softmax-weighted over time",
        requires=("torch",),
    )
    api.heads.add(
        "linear",
        target="dfwb.models.heads:LinearHead",
        summary="Dropout, then a single Linear layer",
        requires=("torch",),
    )
    api.heads.add(
        "mlp",
        target="dfwb.models.heads:MLPHead",
        summary="Linear, an optional BatchNorm, GELU, Dropout, then a Linear output layer",
        requires=("torch",),
    )
    api.losses.add(
        "bce",
        target="dfwb.train.losses:BCELoss",
        summary="Binary cross-entropy on the logit, with an optional pos_weight",
        requires=("torch",),
    )
    api.losses.add(
        "ce",
        target="dfwb.train.losses:CELoss",
        summary="Categorical cross-entropy for multi-class heads, with label smoothing",
        requires=("torch",),
    )
    api.losses.add(
        "focal",
        target="dfwb.train.losses:FocalLoss",
        summary="Binary focal loss on the logit (alpha, gamma); gamma=0, alpha=None is bce",
        requires=("torch",),
    )
    api.losses.add(
        "label-smoothing-bce",
        target="dfwb.train.losses:LabelSmoothingBCELoss",
        summary="Binary cross-entropy with smoothed targets y(1-eps) + eps/2",
        requires=("torch",),
    )
    api.detector_sources.add(
        "run",
        target="dfwb.models.source:load_run",
        summary="Rebuild a detector saved by training, from its run directory",
        requires=("torch",),
    )
