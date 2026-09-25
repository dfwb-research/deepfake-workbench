"""``timm``: shape, contract, freeze and LoRA checks on ``timm/test_vit`` (no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("timm")

import torch

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError
from dfwb.core.plugins import api, get_registry
from dfwb.models.backbone import BackboneOutput, FreezeSpec
from dfwb.models.backbones.timm_backbone import TimmBackbone

_MODEL = "test_vit"


def _backbone(**freeze_kwargs: object) -> TimmBackbone:
    freeze = FreezeSpec(**freeze_kwargs) if freeze_kwargs else FreezeSpec()  # type: ignore[arg-type]
    return TimmBackbone(model=_MODEL, pretrained=False, freeze=freeze)


def test_kind_and_out_dim():
    backbone = _backbone()
    assert backbone.kind == "image"
    assert backbone.out_dim == backbone.model.num_features
    assert backbone.out_dim > 0


def test_forward_returns_pooled_and_tokens_of_the_right_shape():
    backbone = _backbone()
    height, width = backbone.native_input.size
    x = torch.rand(2, 3, height, width)
    out = backbone(x)
    assert isinstance(out, BackboneOutput)
    assert out.pooled.shape == (2, backbone.out_dim)
    assert out.tokens is not None
    assert out.tokens.shape[0] == 2
    assert out.tokens.shape[-1] == backbone.out_dim
    assert out.tokens.ndim == 3


def test_native_input_values():
    backbone = _backbone()
    native = backbone.native_input
    assert isinstance(native, InputSpec)
    assert native.value_range == (0.0, 1.0)
    assert native.mean is not None
    assert native.std is not None
    assert len(native.mean) == 3
    assert len(native.std) == 3
    assert len(native.size) == 2
    assert all(s > 0 for s in native.size)


def test_param_groups_partition_every_parameter_exactly_once():
    backbone = _backbone()
    groups = backbone.param_groups()
    all_ids = [id(p) for params in groups.values() for p in params]
    assert len(all_ids) == len(set(all_ids))  # no parameter counted twice
    assert sorted(all_ids) == sorted(id(p) for p in backbone.model.parameters())
    block_names = [n for n in groups if n.startswith("blocks.")]
    assert block_names  # test_vit has blocks
    assert "embed" in groups
    assert "norm" in groups


def test_partial_trains_exactly_the_last_block():
    backbone = _backbone(mode="partial", trainable_blocks=1)
    groups = backbone.param_groups()
    block_names = sorted(
        (n for n in groups if n.startswith("blocks.")), key=lambda n: int(n.split(".")[1])
    )
    last = block_names[-1]
    for name, params in groups.items():
        expected = name == last
        assert all(p.requires_grad is expected for p in params), name


def test_lora_makes_only_lora_params_trainable():
    pytest.importorskip("peft")
    backbone = _backbone(mode="lora", targets=["qkv"])
    trainable = [n for n, p in backbone.model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in backbone.model.named_parameters() if not p.requires_grad]
    assert trainable
    assert frozen
    assert all("lora" in n for n in trainable)
    assert all("lora" not in n for n in frozen)


def test_registered_as_a_builtin_backbone():
    entry = get_registry("backbones").entry("timm")
    assert entry.provider == "dfwb"
    assert entry.requires == ("torch", "timm")
    backbone = api.backbones.build("timm", model=_MODEL, pretrained=False)
    assert isinstance(backbone, TimmBackbone)


def test_registry_build_rejects_bad_freeze_mode_naming_the_dotted_path():
    with pytest.raises(ConfigError) as info:
        api.backbones.build("timm", model=_MODEL, pretrained=False, freeze={"mode": "partail"})
    assert "freeze.mode" in info.value.message
