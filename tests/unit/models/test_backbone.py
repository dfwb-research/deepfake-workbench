"""``FreezeSpec`` validation and ``Backbone.apply_freeze`` semantics, exercised on ``tiny-cnn``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from torch import nn

from dfwb.core.errors import ConfigError
from dfwb.models.backbone import FreezeSpec
from dfwb.models.backbones.tiny_cnn import TinyCNN


class _Detector(nn.Module):
    """A minimal stand-in for the assembled detector: a backbone plus an unrelated head."""

    def __init__(self, backbone: TinyCNN) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_dim, 1)


# --------------------------------------------------------------------------- FreezeSpec itself


def test_freeze_spec_defaults():
    freeze = FreezeSpec()
    assert freeze.mode == "none"
    assert freeze.trainable_blocks is None
    assert (freeze.r, freeze.alpha, freeze.dropout) == (16, 32, 0.05)
    assert freeze.targets == ["q_proj", "v_proj"]


def test_partial_requires_trainable_blocks():
    with pytest.raises(ValidationError, match="trainable_blocks"):
        FreezeSpec(mode="partial")


def test_partial_rejects_trainable_blocks_below_one():
    with pytest.raises(ValidationError, match="trainable_blocks"):
        FreezeSpec(mode="partial", trainable_blocks=0)


@pytest.mark.parametrize("mode", ["none", "full", "norm-only", "lora"])
def test_trainable_blocks_rejected_outside_partial(mode):
    with pytest.raises(ValidationError, match="trainable_blocks"):
        FreezeSpec(mode=mode, trainable_blocks=1)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        FreezeSpec(mode="none", bogus=True)


# --------------------------------------------------------------------------- apply_freeze modes


def test_freeze_none_trains_every_backbone_and_head_param():
    detector = _Detector(TinyCNN(freeze=FreezeSpec(mode="none")))
    assert all(p.requires_grad for p in detector.backbone.parameters())
    assert all(p.requires_grad for p in detector.head.parameters())


def test_freeze_full_trains_nothing_in_the_backbone_but_leaves_the_head_alone():
    detector = _Detector(TinyCNN(freeze=FreezeSpec(mode="full")))
    assert not any(p.requires_grad for p in detector.backbone.parameters())
    assert all(p.requires_grad for p in detector.head.parameters())


def test_freeze_partial_trains_only_the_last_n_blocks():
    detector = _Detector(TinyCNN(freeze=FreezeSpec(mode="partial", trainable_blocks=1)))
    groups = detector.backbone.param_groups()
    for name, params in groups.items():
        expected = name == "blocks.2"
        assert all(p.requires_grad is expected for p in params), name
    assert all(p.requires_grad for p in detector.head.parameters())


def test_freeze_partial_two_blocks_trains_the_last_two():
    detector = _Detector(TinyCNN(freeze=FreezeSpec(mode="partial", trainable_blocks=2)))
    groups = detector.backbone.param_groups()
    for name, params in groups.items():
        expected = name in {"blocks.1", "blocks.2"}
        assert all(p.requires_grad is expected for p in params), name


def test_freeze_partial_rejects_n_greater_than_block_count():
    with pytest.raises(ConfigError, match="exceeds"):
        TinyCNN(freeze=FreezeSpec(mode="partial", trainable_blocks=4))


def test_freeze_norm_only_trains_only_normalisation_parameters():
    detector = _Detector(TinyCNN(freeze=FreezeSpec(mode="norm-only")))
    backbone = detector.backbone
    norm_param_ids = {
        id(p)
        for module in backbone.modules()
        if isinstance(module, nn.BatchNorm2d)
        for p in module.parameters(recurse=False)
    }
    assert norm_param_ids  # tiny-cnn has BatchNorm2d in every block
    for p in backbone.parameters():
        assert p.requires_grad == (id(p) in norm_param_ids)
    assert all(p.requires_grad for p in detector.head.parameters())


def test_freeze_lora_raises_config_error_naming_lora_support():
    with pytest.raises(ConfigError, match="LoRA"):
        TinyCNN(freeze=FreezeSpec(mode="lora"))
