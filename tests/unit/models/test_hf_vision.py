"""``hf-vision``: shape, contract, freeze and LoRA checks on tiny random HF configs.

No network: every model here is built from a tiny config saved to ``tmp_path`` (``pretrained``
is always ``False``), with an image processor saved alongside it, and loaded as a local
directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("transformers")

import torch

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError
from dfwb.core.plugins import api, get_registry
from dfwb.models.backbone import BackboneOutput, FreezeSpec
from dfwb.models.backbones.hf_vision import HFVisionBackbone

_HIDDEN_SIZE = 32
_NUM_LAYERS = 3
_IMAGE_SIZE = 32
_PATCH_SIZE = 8


def _save_processor(path: Path) -> None:
    from transformers import ViTImageProcessor

    ViTImageProcessor(
        size={"height": _IMAGE_SIZE, "width": _IMAGE_SIZE},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.25, 0.25, 0.25],
    ).save_pretrained(path)


def _save_vit(tmp_path: Path) -> Path:
    from transformers import ViTConfig

    path = tmp_path / "vit"
    path.mkdir()
    ViTConfig(
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=_IMAGE_SIZE,
        patch_size=_PATCH_SIZE,
    ).save_pretrained(path)
    _save_processor(path)
    return path


def _save_clip(tmp_path: Path) -> Path:
    from transformers import CLIPVisionConfig

    path = tmp_path / "clip"
    path.mkdir()
    CLIPVisionConfig(
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=_IMAGE_SIZE,
        patch_size=_PATCH_SIZE,
        projection_dim=_HIDDEN_SIZE,
    ).save_pretrained(path)
    _save_processor(path)
    return path


@pytest.fixture(params=["vit", "clip"])
def hf_dir(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    builder = _save_vit if request.param == "vit" else _save_clip
    return builder(tmp_path)


def _backbone(model_dir: Path, **freeze_kwargs: object) -> HFVisionBackbone:
    freeze = FreezeSpec(**freeze_kwargs) if freeze_kwargs else FreezeSpec()  # type: ignore[arg-type]
    return HFVisionBackbone(model=str(model_dir), pretrained=False, freeze=freeze)


def test_kind_and_out_dim(hf_dir: Path):
    backbone = _backbone(hf_dir)
    assert backbone.kind == "image"
    assert backbone.out_dim == _HIDDEN_SIZE


def test_forward_cls_pool_returns_pooled_and_tokens(hf_dir: Path):
    backbone = _backbone(hf_dir)
    height, width = backbone.native_input.size
    out = backbone(torch.rand(2, 3, height, width))
    assert isinstance(out, BackboneOutput)
    assert out.pooled.shape == (2, _HIDDEN_SIZE)
    assert out.tokens is not None
    assert out.tokens.shape[0] == 2
    assert out.tokens.shape[-1] == _HIDDEN_SIZE


def test_forward_mean_pool(hf_dir: Path):
    backbone = HFVisionBackbone(model=str(hf_dir), pretrained=False, pool="mean")
    height, width = backbone.native_input.size
    out = backbone(torch.rand(2, 3, height, width))
    assert out.pooled.shape == (2, _HIDDEN_SIZE)


def test_native_input_values(hf_dir: Path):
    backbone = _backbone(hf_dir)
    native = backbone.native_input
    assert isinstance(native, InputSpec)
    assert native.size == (_IMAGE_SIZE, _IMAGE_SIZE)
    assert native.mean == (0.5, 0.5, 0.5)
    assert native.std == (0.25, 0.25, 0.25)
    assert native.value_range == (0.0, 1.0)


def test_native_input_falls_back_to_shortest_edge(tmp_path: Path):
    from transformers import ViTConfig, ViTImageProcessor

    path = tmp_path / "square-edge"
    path.mkdir()
    ViTConfig(
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=_IMAGE_SIZE,
        patch_size=_PATCH_SIZE,
    ).save_pretrained(path)
    ViTImageProcessor(
        size={"shortest_edge": _IMAGE_SIZE},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.25, 0.25, 0.25],
    ).save_pretrained(path)
    backbone = _backbone(path)
    assert backbone.native_input.size == (_IMAGE_SIZE, _IMAGE_SIZE)


def test_param_groups_partition_every_parameter_exactly_once(hf_dir: Path):
    backbone = _backbone(hf_dir)
    groups = backbone.param_groups()
    all_ids = [id(p) for params in groups.values() for p in params]
    assert len(all_ids) == len(set(all_ids))
    assert sorted(all_ids) == sorted(id(p) for p in backbone.model.parameters())
    block_names = [n for n in groups if n.startswith("blocks.")]
    assert len(block_names) == _NUM_LAYERS


def test_partial_trains_exactly_one_encoder_layer(hf_dir: Path):
    backbone = _backbone(hf_dir, mode="partial", trainable_blocks=1)
    groups = backbone.param_groups()
    block_names = sorted(
        (n for n in groups if n.startswith("blocks.")), key=lambda n: int(n.split(".")[1])
    )
    assert len(block_names) == _NUM_LAYERS
    last = block_names[-1]
    for name, params in groups.items():
        expected = name == last
        assert all(p.requires_grad is expected for p in params), name
    trainable_blocks = {
        name for name in block_names if all(p.requires_grad for p in groups[name]) and groups[name]
    }
    assert trainable_blocks == {last}


def test_lora_makes_only_lora_params_trainable(tmp_path: Path):
    pytest.importorskip("peft")
    path = _save_clip(tmp_path)
    backbone = _backbone(path, mode="lora", targets=["q_proj", "v_proj"])
    trainable = [n for n, p in backbone.model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in backbone.model.named_parameters() if not p.requires_grad]
    assert trainable
    assert frozen
    assert all("lora" in n for n in trainable)
    assert all("lora" not in n for n in frozen)


def test_missing_image_processor_raises_config_error(tmp_path: Path):
    from transformers import ViTConfig

    path = tmp_path / "no-processor"
    path.mkdir()
    ViTConfig(
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=_IMAGE_SIZE,
        patch_size=_PATCH_SIZE,
    ).save_pretrained(path)
    with pytest.raises(ConfigError, match="image processor"):
        HFVisionBackbone(model=str(path), pretrained=False)


def test_registered_as_a_builtin_backbone():
    entry = get_registry("backbones").entry("hf-vision")
    assert entry.provider == "dfwb"
    assert entry.requires == ("torch", "transformers")


def test_registry_build_rejects_bad_freeze_mode_naming_the_dotted_path(hf_dir: Path):
    with pytest.raises(ConfigError) as info:
        api.backbones.build(
            "hf-vision", model=str(hf_dir), pretrained=False, freeze={"mode": "partail"}
        )
    assert "freeze.mode" in info.value.message


def test_registry_build_rejects_lora_targets_matching_nothing(tmp_path: Path):
    pytest.importorskip("peft")
    path = _save_clip(tmp_path)
    with pytest.raises(ConfigError) as info:
        api.backbones.build(
            "hf-vision",
            model=str(path),
            pretrained=False,
            freeze={"mode": "lora", "targets": ["nope"]},
        )
    assert "nope" in info.value.message
