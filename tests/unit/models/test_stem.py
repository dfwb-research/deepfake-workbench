"""The stem hook: builds a registered layer and adapts its channel count back to 3."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dfwb.core.config.schema import ComponentSpec
from dfwb.core.errors import ContractError
from dfwb.core.plugins import api
from dfwb.models.backbones.tiny_cnn import TinyCNN
from dfwb.models.stem import build_stem

_T = "tests.unit.models._fake_layers"


@pytest.fixture(autouse=True)
def _fake_layers():
    api.layers.add("fake-hp", target=f"{_T}:FakeHP", summary="fake stem for tests")
    api.layers.add("identity3", target=f"{_T}:Identity3", summary="already 3 channels")
    api.layers.add("no-out-channels", target=f"{_T}:NoOutChannels", summary="missing out_channels")
    api.layers.add("not-a-module", target=f"{_T}:not_a_module", summary="not an nn.Module")


def test_stem_adds_a_1x1_adapter_when_out_channels_differs():
    module = build_stem(ComponentSpec(name="fake-hp"))
    assert isinstance(module, nn.Sequential)
    x = torch.rand(2, 3, 16, 16)
    y = module(x)
    assert y.shape == (2, 3, 16, 16)


def test_stem_then_backbone_forward_pass():
    stem = build_stem(ComponentSpec(name="fake-hp"))
    backbone = TinyCNN()
    x = torch.rand(2, 3, 64, 64)
    out = backbone(stem(x))
    assert out.pooled.shape == (2, 32)


def test_stem_returns_the_layer_itself_when_channels_already_match():
    module = build_stem(ComponentSpec(name="identity3"))
    assert not isinstance(module, nn.Sequential)
    x = torch.rand(2, 3, 16, 16)
    assert module(x).shape == (2, 3, 16, 16)


def test_stem_requires_an_out_channels_attribute():
    with pytest.raises(ContractError, match="layers/no-out-channels"):
        build_stem(ComponentSpec(name="no-out-channels"))


def test_stem_requires_the_layer_to_be_an_nn_module():
    with pytest.raises(ContractError, match="layers/not-a-module"):
        build_stem(ComponentSpec(name="not-a-module"))


def test_stem_passes_in_channels_unless_the_spec_sets_it():
    default = build_stem(ComponentSpec(name="fake-hp"), in_channels=3)
    assert default[0].in_channels == 3

    overridden = build_stem(ComponentSpec(name="fake-hp", in_channels=1), in_channels=3)
    assert overridden[0].in_channels == 1
