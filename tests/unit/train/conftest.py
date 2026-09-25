"""Fixtures for the train layer's tests.

- :func:`make_detector`: a small real ``AssembledDetector`` (``tiny-cnn``) for loss, optimiser and
  schedule tests, built directly from the model classes rather than through ``build_detector``/the
  registries: those tests care about parameter groups and gradients, not config parsing or plugin
  discovery.
- ``toy_pack``/``toy_work_root``: the ``toytrain`` protocol pack and processed store from
  :mod:`tests.unit.train._toy`, for the Lightning module, data module, callback and logger tests.
  They import it lazily, so the loss/optimiser/schedule tests never need Lightning.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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


@pytest.fixture
def deterministic_torch(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Lightning's ``deterministic=True`` switches torch's process-wide deterministic mode on and
    sets ``CUBLAS_WORKSPACE_CONFIG``; put both back so later tests in this worker see neither."""
    import torch

    before = torch.are_deterministic_algorithms_enabled()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    yield
    torch.use_deterministic_algorithms(before)


@pytest.fixture
def toy_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The ``toytrain`` protocol pack, installed; returns its dataset directory."""
    from tests.unit.train._toy import install_toytrain_pack

    return install_toytrain_pack(tmp_path, monkeypatch)


@pytest.fixture
def toy_work_root(tmp_path: Path, toy_pack: Path, deterministic_torch: None) -> Path:
    """A work root holding the ``toytrain`` processed store (every video processed)."""
    from tests.unit.train._toy import write_toy_store

    work_root = tmp_path / "work"
    write_toy_store(work_root)
    return work_root
