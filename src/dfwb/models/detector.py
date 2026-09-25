"""``AssembledDetector``: backbone -> (stem ->) (temporal pool ->) head, wired to the detector
contract. ``build_detector`` resolves a resolved ``model:`` config through the registries into
one of these.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from importlib import metadata
from typing import Any

import torch
from torch import Tensor, nn

from dfwb.core.config.schema import ModelSection
from dfwb.core.detector import (
    DETECTOR_CONTRACT_VERSION,
    ClipBatch,
    DetectorMeta,
    DetectorOutput,
)
from dfwb.core.errors import ContractError
from dfwb.core.plugins import api
from dfwb.models.backbone import Backbone
from dfwb.models.heads import Head
from dfwb.models.pools import TemporalPool
from dfwb.models.stem import build_stem

__all__ = ["AssembledDetector", "build_detector"]

_FRAMEWORK_DISTRIBUTION = "deepfake-workbench"


class AssembledDetector(nn.Module):
    """backbone -> (stem ->) (pool ->) head. Implements the ``Detector`` contract: ``meta`` plus
    ``predict``, and ``.to()`` inherited from :class:`torch.nn.Module`.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        stem: nn.Module | None,
        pool: TemporalPool | None,
        head: Head,
        meta: DetectorMeta,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.stem = stem
        self.pool = pool
        self.head = head
        self.meta = meta

    def _pooled_and_per_frame(self, batch: ClipBatch) -> tuple[Tensor, Tensor | None]:
        """``(pooled [B,D], per_frame [B,T,D] or None)``; ``per_frame`` is ``None`` for a
        ``kind="video"`` backbone, which never exposes individual frame features."""
        clips = batch.clips
        b, t = clips.shape[0], clips.shape[1]
        if self.backbone.kind == "video":
            out = self.backbone(clips)
            return out.pooled, None
        x = clips.reshape(b * t, *clips.shape[2:])
        if self.stem is not None:
            x = self.stem(x)
        out = self.backbone(x)
        per_frame = out.pooled.reshape(b, t, -1)
        if self.pool is None:
            raise RuntimeError(
                "AssembledDetector: an image backbone must be assembled with a temporal pool "
                "(this instance was built with pool=None)"
            )
        pooled = self.pool(per_frame)
        return pooled, per_frame

    def forward(self, batch: ClipBatch) -> DetectorOutput:
        """Training path: logits, not probabilities."""
        pooled, _ = self._pooled_and_per_frame(batch)
        logit = self.head(pooled)
        return DetectorOutput(score=torch.sigmoid(logit), logit=logit, features=pooled)

    def predict(self, batch: ClipBatch) -> DetectorOutput:
        """Inference path: switches to eval mode for the duration (restoring whatever mode this
        module was in beforehand, even on error), and runs under ``torch.inference_mode()``.
        Scores are in ``[0, 1]``; ``frame_scores`` is filled for image backbones.

        Raises:
            ContractError: The head has ``num_classes != 1``. Scoring is defined for binary
                heads only (one logit per clip); a multi-class head still trains through
                ``forward()``.
        """
        if self.head.num_classes != 1:
            raise ContractError(
                "predict(): scoring supports binary heads only (one logit per clip), but this "
                f"head has num_classes={self.head.num_classes}",
                hint="train a multi-class head through forward(); predict() is for scoring",
            )
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                pooled, per_frame = self._pooled_and_per_frame(batch)
                logit = self.head(pooled)
                score = torch.sigmoid(logit)
                frame_scores = None
                if per_frame is not None:
                    b, t, d = per_frame.shape
                    frame_logit = self.head(per_frame.reshape(b * t, d)).reshape(b, t)
                    frame_scores = torch.sigmoid(frame_logit)
                return DetectorOutput(
                    score=score, logit=logit, frame_scores=frame_scores, features=pooled
                )
        finally:
            self.train(was_training)


def _framework_version() -> str:
    from dfwb import __version__

    return __version__


def _framework_license() -> str:
    meta = metadata.metadata(_FRAMEWORK_DISTRIBUTION)
    return meta.get("License-Expression") or meta.get("License") or "UNKNOWN"


def build_detector(
    model_cfg: ModelSection | Mapping[str, Any],
    *,
    input_spec_overrides: Mapping[str, Any] | None = None,
    source: str | None = None,
) -> AssembledDetector:
    """Build backbone -> (stem ->) (pool ->) head from a resolved ``model:`` config.

    ``input_spec_overrides`` is applied on top of the backbone's own ``native_input`` (via
    ``dataclasses.replace``). ``source`` becomes ``DetectorMeta.source`` verbatim, e.g.
    ``"run:<fingerprint>"`` once a checkpoint records where it came from.
    """
    section = (
        model_cfg if isinstance(model_cfg, ModelSection) else ModelSection.model_validate(model_cfg)
    )
    backbone = api.backbones.build(section.backbone.name, **section.backbone.params)
    stem = build_stem(section.stem) if section.stem is not None else None
    pool: TemporalPool | None = None
    if backbone.kind != "video":
        pool = api.temporal_pools.build(
            section.temporal_pool.name, dim=backbone.out_dim, **section.temporal_pool.params
        )
    head = api.heads.build(section.head.name, dim=backbone.out_dim, **section.head.params)

    input_spec = backbone.native_input
    if input_spec_overrides:
        input_spec = dataclasses.replace(input_spec, **input_spec_overrides)

    detector_meta = DetectorMeta(
        name=f"{section.backbone.name}-{section.temporal_pool.name}-{section.head.name}",
        version=_framework_version(),
        contract_version=DETECTOR_CONTRACT_VERSION,
        input=input_spec,
        license=_framework_license(),
        weights_license=None,
        citation=None,
        source=source,
    )
    return AssembledDetector(backbone=backbone, stem=stem, pool=pool, head=head, meta=detector_meta)
