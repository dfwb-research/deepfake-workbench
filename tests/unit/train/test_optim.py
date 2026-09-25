"""``adamw`` and ``sgd``: named parameter groups, layer-wise LR decay, frozen exclusion.

Optimisers are not a plugin registry: :func:`build_optimizer` takes the resolved
``AssembledDetector`` directly, since group names come from the backbone's own
``param_groups()`` plus ``head``, ``pool`` and ``stem``.
"""

from __future__ import annotations

import pytest
import torch
from tests.unit.train.conftest import make_detector
from torch import nn

from dfwb.core.config.schema import ComponentSpec
from dfwb.core.errors import ConfigError
from dfwb.models.backbone import FreezeSpec
from dfwb.models.pools import AttentionPool
from dfwb.train.optim import _layer_multiplier, build_optimizer


def _lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {g["name"]: g["lr"] for g in optimizer.param_groups}


def _wds(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {g["name"]: g["weight_decay"] for g in optimizer.param_groups}


# --------------------------------------------------------------------------- basic construction


def test_default_adamw_covers_every_trainable_param_at_the_base_lr():
    detector = make_detector()
    optimizer = build_optimizer(ComponentSpec(name="adamw", lr=1e-3), detector)
    assert isinstance(optimizer, torch.optim.AdamW)
    # tiny-cnn's 3 blocks plus the head; the default MeanPool has no parameters of its own, and
    # there is no stem, so both are dropped rather than appearing as empty groups.
    assert {g["name"] for g in optimizer.param_groups} == {
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "head",
    }
    covered = sum(len(g["params"]) for g in optimizer.param_groups)
    assert covered == sum(1 for _ in detector.parameters())
    for g in optimizer.param_groups:
        assert g["lr"] == pytest.approx(1e-3)
        assert g["weight_decay"] == pytest.approx(0.01)  # adamw's default


def test_sgd_default_weight_decay_is_zero_and_takes_momentum_and_nesterov():
    detector = make_detector()
    optimizer = build_optimizer(
        ComponentSpec(name="sgd", lr=0.1, momentum=0.8, nesterov=True), detector
    )
    assert isinstance(optimizer, torch.optim.SGD)
    for g in optimizer.param_groups:
        assert g["weight_decay"] == pytest.approx(0.0)
        assert g["momentum"] == pytest.approx(0.8)
        assert g["nesterov"] is True


def test_pool_and_stem_groups_appear_only_when_they_have_parameters():
    stem = nn.Conv2d(3, 3, kernel_size=1)
    detector = make_detector(pool=AttentionPool(dim=32), stem=stem)
    optimizer = build_optimizer(ComponentSpec(name="adamw", lr=1e-3), detector)
    names = {g["name"] for g in optimizer.param_groups}
    assert {"pool", "stem"} <= names


def test_group_name_is_recorded_on_every_torch_param_group():
    detector = make_detector()
    optimizer = build_optimizer(ComponentSpec(name="adamw", lr=1e-3), detector)
    for g in optimizer.param_groups:
        assert isinstance(g["name"], str)


# --------------------------------------------------------------------------- groups


def test_group_lr_scale_and_weight_decay_override_the_top_level_defaults():
    detector = make_detector()
    optimizer = build_optimizer(
        ComponentSpec(
            name="adamw",
            lr=1.0,
            weight_decay=0.01,
            groups={"head": {"lr_scale": 0.5, "weight_decay": 0.0}},
        ),
        detector,
    )
    lrs, wds = _lrs(optimizer), _wds(optimizer)
    assert lrs["head"] == pytest.approx(0.5)
    assert wds["head"] == pytest.approx(0.0)
    assert lrs["blocks.0"] == pytest.approx(1.0)  # untouched group keeps the top-level lr
    assert wds["blocks.0"] == pytest.approx(0.01)


def test_group_weight_decay_defaults_to_the_top_level_value_when_omitted():
    detector = make_detector()
    optimizer = build_optimizer(
        ComponentSpec(name="adamw", lr=1.0, weight_decay=0.05, groups={"head": {"lr_scale": 2.0}}),
        detector,
    )
    assert _wds(optimizer)["head"] == pytest.approx(0.05)


# --------------------------------------------------------------------------- layer_decay


def test_layer_decay_gives_geometric_multipliers_on_tiny_cnns_three_blocks():
    detector = make_detector()
    gamma = 0.5
    optimizer = build_optimizer(ComponentSpec(name="adamw", lr=1.0, layer_decay=gamma), detector)
    lrs = _lrs(optimizer)
    # K=3 blocks, no embed/norm group (tiny-cnn's every parameter lives inside a block):
    # block i -> gamma ** (K - i).
    assert lrs["blocks.0"] == pytest.approx(gamma**3)
    assert lrs["blocks.1"] == pytest.approx(gamma**2)
    assert lrs["blocks.2"] == pytest.approx(gamma**1)
    # head/pool/stem are never touched by layer_decay.
    assert lrs["head"] == pytest.approx(1.0)


def test_layer_decay_composes_multiplicatively_with_lr_scale():
    detector = make_detector()
    optimizer = build_optimizer(
        ComponentSpec(
            name="adamw", lr=2.0, layer_decay=0.5, groups={"blocks.2": {"lr_scale": 0.1}}
        ),
        detector,
    )
    assert _lrs(optimizer)["blocks.2"] == pytest.approx(2.0 * 0.1 * (0.5**1))


def test_no_layer_decay_means_every_group_multiplier_is_one():
    detector = make_detector()
    optimizer = build_optimizer(ComponentSpec(name="adamw", lr=1.0), detector)
    assert all(lr == pytest.approx(1.0) for lr in _lrs(optimizer).values())


def test_layer_multiplier_embed_group_is_gamma_to_the_k_plus_1():
    # tiny-cnn has no 'embed' group (every parameter is inside a block); a backbone that does
    # (timm, hf-vision) puts its patch/stem embedding one step further back than block 0.
    assert _layer_multiplier("embed", 0.5, num_blocks=3) == pytest.approx(0.5**4)
    assert _layer_multiplier("norm", 0.5, num_blocks=3) == pytest.approx(1.0)


# --------------------------------------------------------------------------- frozen parameters


def test_frozen_parameters_are_excluded_and_their_group_is_dropped():
    detector = make_detector(freeze=FreezeSpec(mode="partial", trainable_blocks=1))
    optimizer = build_optimizer(ComponentSpec(name="adamw", lr=1e-3), detector)
    names = {g["name"] for g in optimizer.param_groups}
    assert "blocks.0" not in names
    assert "blocks.1" not in names
    assert "blocks.2" in names
    trainable_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    for name, params in detector.backbone.param_groups().items():
        for p in params:
            assert (id(p) in trainable_ids) == (name == "blocks.2")


def test_no_trainable_parameters_at_all_is_a_config_error():
    detector = make_detector(freeze=FreezeSpec(mode="full"))
    for p in detector.head.parameters():
        p.requires_grad = False
    with pytest.raises(ConfigError, match="no trainable parameters"):
        build_optimizer(ComponentSpec(name="adamw", lr=1e-3), detector)


# --------------------------------------------------------------------------- config errors


def test_unknown_optimiser_name_is_a_config_error_with_did_you_mean():
    detector = make_detector()
    with pytest.raises(ConfigError, match=r"optim\.name") as info:
        build_optimizer(ComponentSpec(name="adam", lr=1e-3), detector)
    assert "adamw" in info.value.message


def test_unknown_top_level_param_is_a_config_error():
    detector = make_detector()
    with pytest.raises(ConfigError) as info:
        build_optimizer(ComponentSpec(name="adamw", lr=1e-3, foo=1), detector)
    assert "optim" in info.value.message
    assert "foo" in info.value.message


def test_unknown_group_field_is_a_config_error_with_did_you_mean():
    detector = make_detector()
    with pytest.raises(ConfigError) as info:
        build_optimizer(
            ComponentSpec(name="adamw", lr=1e-3, groups={"head": {"lr_sacle": 1.0}}), detector
        )
    assert "lr_scale" in info.value.message


def test_unknown_group_name_is_a_config_error_naming_the_dotted_path():
    detector = make_detector()
    with pytest.raises(ConfigError, match=r"optim\.groups\.blockz") as info:
        build_optimizer(
            ComponentSpec(name="adamw", lr=1e-3, groups={"blockz": {"lr_scale": 1.0}}), detector
        )
    assert "blocks.2" in info.value.message or "blocks" in info.value.hint
