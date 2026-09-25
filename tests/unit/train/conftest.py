"""A small real ``AssembledDetector`` (``tiny-cnn``) for loss, optimiser and schedule tests.

Built directly from the model classes rather than through ``build_detector``/the registries: these
tests care about parameter groups and gradients, not config parsing or plugin discovery.
"""

from __future__ import annotations

import pytest
from torch import nn

from dfwb.core.detector import DETECTOR_CONTRACT_VERSION, DetectorMeta, InputSpec
from dfwb.models.backbone import FreezeSpec
from dfwb.models.backbones.tiny_cnn import TinyCNN
from dfwb.models.detector import AssembledDetector
from dfwb.models.heads import LinearHead
from dfwb.models.pools import MeanPool, TemporalPool


def _meta() -> DetectorMeta:
    return DetectorMeta(
        name="test-detector",
        version="0",
        contract_version=DETECTOR_CONTRACT_VERSION,
        input=InputSpec(),
        license="MIT",
        weights_license=None,
        citation=None,
        source=None,
    )


def make_detector(
    *,
    freeze: FreezeSpec | None = None,
    pool: TemporalPool | None = None,
    stem: nn.Module | None = None,
) -> AssembledDetector:
    """``tiny-cnn`` (3 blocks, 32-dim) -> ``pool`` (mean, by default) -> a linear head."""
    backbone = TinyCNN(freeze=freeze or FreezeSpec())
    resolved_pool = pool if pool is not None else MeanPool(dim=backbone.out_dim)
    head = LinearHead(dim=backbone.out_dim)
    return AssembledDetector(
        backbone=backbone, stem=stem, pool=resolved_pool, head=head, meta=_meta()
    )


@pytest.fixture
def detector() -> AssembledDetector:
    return make_detector()
