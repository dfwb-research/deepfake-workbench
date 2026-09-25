"""``AssembledDetector``: backbone -> stem -> temporal pool -> head, wired to the detector
contract, and ``build_detector`` resolving a ``model:`` config through the registries."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dfwb.core.config.schema import ComponentSpec, ModelSection
from dfwb.core.detector import DETECTOR_CONTRACT_VERSION, ClipBatch, DetectorMeta, InputSpec
from dfwb.core.errors import ContractError
from dfwb.core.plugins import api
from dfwb.models.backbone import Backbone, BackboneOutput
from dfwb.models.backbones.tiny_cnn import TinyCNN
from dfwb.models.detector import AssembledDetector, build_detector
from dfwb.models.heads import LinearHead

_POOLS = ["mean", "max", "attention"]
_HEADS = ["linear", "mlp"]
_TINY_CNN_OUT_DIM = 32


def _model_cfg(pool: str = "mean", head: str = "linear", stem: str | None = None) -> ModelSection:
    return ModelSection(
        backbone=ComponentSpec(name="tiny-cnn"),
        temporal_pool=ComponentSpec(name=pool),
        head=ComponentSpec(name=head),
        stem=ComponentSpec(name=stem) if stem else None,
    )


def _batch(b: int, t: int, size: int = 64) -> ClipBatch:
    return ClipBatch(
        clips=torch.rand(b, t, 3, size, size),
        keys=[f"k{i}" for i in range(b)],
        dataset_ids=["toy"] * b,
        compressions=[None] * b,
        clip_index=torch.zeros(b, dtype=torch.long),
        frame_indices=torch.zeros(b, t, dtype=torch.long),
    )


@pytest.mark.parametrize("pool", _POOLS)
@pytest.mark.parametrize("head", _HEADS)
@pytest.mark.parametrize("t", [1, 8])
def test_forward_and_predict_shapes_for_every_pool_and_head(pool, head, t):
    detector = build_detector(_model_cfg(pool, head))
    batch = _batch(3, t)

    out = detector.forward(batch)
    assert out.logit is not None
    assert out.logit.shape == (3,)
    assert out.score.shape == (3,)
    assert out.features is not None
    assert out.features.shape == (3, _TINY_CNN_OUT_DIM)

    prediction = detector.predict(batch)
    assert prediction.score.shape == (3,)
    assert torch.all(prediction.score >= 0)
    assert torch.all(prediction.score <= 1)
    assert prediction.frame_scores is not None
    assert prediction.frame_scores.shape == (3, t)


def test_forward_gives_unconstrained_logits_predict_gives_probabilities():
    detector = build_detector(_model_cfg())
    detector.eval()
    batch = _batch(2, 4)
    logits = detector.forward(batch).logit
    scores = detector.predict(batch).score
    assert torch.allclose(torch.sigmoid(logits), scores, atol=1e-5)
    assert not torch.equal(logits, scores)


def test_predict_runs_under_inference_mode():
    detector = build_detector(_model_cfg())
    batch = _batch(2, 2)
    prediction = detector.predict(batch)
    assert prediction.score.requires_grad is False
    assert prediction.frame_scores is not None
    assert prediction.frame_scores.requires_grad is False


def test_predict_with_high_dropout_gives_identical_scores_across_calls():
    cfg = ModelSection(
        backbone=ComponentSpec(name="tiny-cnn"),
        temporal_pool=ComponentSpec(name="mean"),
        head=ComponentSpec(name="linear", dropout=0.9),
    )
    detector = build_detector(cfg)
    detector.train()  # deliberately left in train mode; predict() must switch out of it
    batch = _batch(2, 3)
    first = detector.predict(batch)
    second = detector.predict(batch)
    assert torch.allclose(first.score, second.score)
    assert torch.allclose(first.logit, second.logit)


def test_predict_with_a_batchnorm_head_works_at_batch_size_one():
    cfg = ModelSection(
        backbone=ComponentSpec(name="tiny-cnn"),
        temporal_pool=ComponentSpec(name="mean"),
        head=ComponentSpec(name="mlp", norm=True),
    )
    detector = build_detector(cfg)
    detector.train()  # BatchNorm1d in train mode cannot compute batch statistics for size 1
    batch = _batch(1, 2)
    prediction = detector.predict(batch)
    assert prediction.score.shape == (1,)


@pytest.mark.parametrize("was_training", [True, False])
def test_predict_restores_the_previous_training_mode(was_training):
    detector = build_detector(_model_cfg())
    detector.train(was_training)
    detector.predict(_batch(2, 2))
    assert detector.training is was_training


def test_predict_raises_contract_error_for_a_multiclass_head():
    cfg = ModelSection(
        backbone=ComponentSpec(name="tiny-cnn"),
        temporal_pool=ComponentSpec(name="mean"),
        head=ComponentSpec(name="linear", num_classes=3),
    )
    detector = build_detector(cfg)
    batch = _batch(2, 2)
    with pytest.raises(ContractError, match="binary"):
        detector.predict(batch)
    # forward() still serves multi-class training
    out = detector.forward(batch)
    assert out.logit.shape == (2, 3)


def test_missing_pool_on_an_image_backbone_raises_a_clear_runtime_error():
    backbone = TinyCNN()
    head = LinearHead(dim=backbone.out_dim)
    meta = DetectorMeta(
        name="tiny-cnn-none-linear",
        version="0.0.0",
        contract_version=DETECTOR_CONTRACT_VERSION,
        input=backbone.native_input,
        license="MIT",
        weights_license=None,
        citation=None,
        source=None,
    )
    detector = AssembledDetector(backbone=backbone, stem=None, pool=None, head=head, meta=meta)
    with pytest.raises(RuntimeError):
        detector.forward(_batch(2, 2))


def test_detector_meta_is_filled():
    detector = build_detector(_model_cfg(), source="run:deadbeef")
    meta = detector.meta
    assert meta.name == "tiny-cnn-mean-linear"
    assert meta.contract_version == DETECTOR_CONTRACT_VERSION
    assert isinstance(meta.input, InputSpec)
    assert meta.input.size == (64, 64)
    assert meta.source == "run:deadbeef"
    assert meta.weights_license is None
    assert meta.citation is None
    assert meta.license  # a non-empty SPDX identifier
    assert meta.version  # the framework's own version


def test_detector_meta_source_defaults_to_none():
    detector = build_detector(_model_cfg())
    assert detector.meta.source is None


def test_input_spec_overrides_are_applied_on_top_of_the_backbones_native_input():
    detector = build_detector(_model_cfg(), input_spec_overrides={"frames": 8})
    assert detector.meta.input.frames == 8
    assert detector.meta.input.size == (64, 64)


def test_model_cfg_accepts_a_plain_mapping():
    detector = build_detector(
        {
            "backbone": {"name": "tiny-cnn"},
            "temporal_pool": {"name": "mean"},
            "head": {"name": "linear"},
        }
    )
    assert detector.meta.name == "tiny-cnn-mean-linear"


@pytest.fixture
def _fake_stem_layer():
    api.layers.add(
        "fake-stem-hook",
        target="tests.unit.models._fake_layers:FakeHP",
        summary="fake stem for detector-assembly tests",
    )


@pytest.mark.usefixtures("_fake_stem_layer")
def test_stem_is_applied_before_the_backbone():
    detector = build_detector(_model_cfg(stem="fake-stem-hook"))
    batch = _batch(2, 2)
    out = detector.forward(batch)
    assert out.logit.shape == (2,)


class _FakeVideoBackbone(Backbone):
    """A stand-in ``kind="video"`` backbone: sees ``[B,T,C,H,W]`` whole, never per-frame."""

    kind = "video"
    out_dim = 6
    native_input = InputSpec(size=(8, 8), frames=4, value_range=(0.0, 1.0), mean=None, std=None)

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8 * 4, 6)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        pooled = self.proj(x.reshape(x.shape[0], -1))
        return BackboneOutput(pooled=pooled, tokens=None)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"proj": list(self.proj.parameters())}


def test_video_backbone_skips_the_pool_and_has_no_frame_scores():
    backbone = _FakeVideoBackbone()
    head = LinearHead(dim=backbone.out_dim)
    meta = DetectorMeta(
        name="fake-video-none-linear",
        version="0.0.0",
        contract_version=DETECTOR_CONTRACT_VERSION,
        input=backbone.native_input,
        license="MIT",
        weights_license=None,
        citation=None,
        source=None,
    )
    detector = AssembledDetector(backbone=backbone, stem=None, pool=None, head=head, meta=meta)
    batch = _batch(2, 4, size=8)

    out = detector.forward(batch)
    assert out.logit is not None
    assert out.logit.shape == (2,)

    prediction = detector.predict(batch)
    assert prediction.frame_scores is None
    assert prediction.score.shape == (2,)
