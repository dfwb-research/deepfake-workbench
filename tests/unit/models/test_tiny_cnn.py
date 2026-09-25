"""``tiny-cnn``: shape, contract and registration checks."""

from __future__ import annotations

import pytest
import torch

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError
from dfwb.core.plugins import api, get_registry
from dfwb.models.backbone import BackboneOutput
from dfwb.models.backbones.tiny_cnn import TinyCNN


def test_kind_out_dim_and_native_input():
    backbone = TinyCNN()
    assert backbone.kind == "image"
    assert backbone.out_dim == 32
    assert backbone.native_input == InputSpec(
        size=(64, 64), value_range=(0.0, 1.0), mean=None, std=None
    )


def test_param_groups_partition_every_parameter_exactly_once():
    backbone = TinyCNN()
    groups = backbone.param_groups()
    assert set(groups) == {"blocks.0", "blocks.1", "blocks.2"}
    all_group_ids = [id(p) for params in groups.values() for p in params]
    assert len(all_group_ids) == len(set(all_group_ids))  # no parameter counted twice
    assert sorted(all_group_ids) == sorted(id(p) for p in backbone.parameters())


def test_forward_returns_backbone_output_of_the_right_shape():
    backbone = TinyCNN()
    x = torch.rand(2, 3, 64, 64)
    out = backbone(x)
    assert isinstance(out, BackboneOutput)
    assert out.pooled.shape == (2, 32)
    assert out.tokens is None


def test_registered_as_a_builtin_backbone():
    entry = get_registry("backbones").entry("tiny-cnn")
    assert entry.provider == "dfwb"
    assert entry.requires == ("torch",)
    backbone = api.backbones.build("tiny-cnn")
    assert isinstance(backbone, TinyCNN)


def test_registry_build_rejects_bad_freeze_mode_naming_the_dotted_path():
    with pytest.raises(ConfigError) as info:
        api.backbones.build("tiny-cnn", freeze={"mode": "partail"})
    assert "freeze.mode" in info.value.message
