"""Heads: ``linear`` and ``mlp``, ``[B,D] -> [B]`` for ``num_classes=1``, ``[B,K]`` otherwise."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dfwb.core.plugins import api, get_registry
from dfwb.models.heads import Head, LinearHead, MLPHead

_HEAD_CLASSES = [LinearHead, MLPHead]


@pytest.mark.parametrize("head_cls", _HEAD_CLASSES)
def test_binary_head_returns_one_logit_per_sample(head_cls):
    head = head_cls(dim=8)
    assert isinstance(head, Head)
    out = head(torch.rand(5, 8))
    assert out.shape == (5,)


@pytest.mark.parametrize("head_cls", _HEAD_CLASSES)
def test_multiclass_head_returns_k_logits_per_sample(head_cls):
    head = head_cls(dim=8, num_classes=3)
    out = head(torch.rand(5, 8))
    assert out.shape == (5, 3)


def test_linear_head_is_dropout_then_linear():
    head = LinearHead(dim=8, dropout=0.3)
    assert isinstance(head.dropout, nn.Dropout)
    assert head.dropout.p == 0.3
    assert isinstance(head.linear, nn.Linear)
    assert head.linear.in_features == 8
    assert head.linear.out_features == 1


def test_linear_head_defaults():
    head = LinearHead(dim=8)
    assert head.dropout.p == 0.0


def test_mlp_head_defaults():
    head = MLPHead(dim=8)
    assert head.fc1.out_features == 512
    assert head.norm is None
    assert isinstance(head.act, nn.GELU)
    assert head.dropout.p == 0.0


def test_mlp_head_hidden_and_dropout_are_configurable():
    head = MLPHead(dim=8, hidden=16, dropout=0.5)
    assert head.fc1.out_features == 16
    assert head.fc2.in_features == 16
    assert head.dropout.p == 0.5


def test_mlp_head_norm_adds_batchnorm_after_the_first_linear():
    head = MLPHead(dim=8, hidden=16, norm=True)
    assert isinstance(head.norm, nn.BatchNorm1d)
    assert head.norm.num_features == 16


def test_mlp_head_order_is_linear_norm_gelu_dropout_linear():
    head = MLPHead(dim=8, hidden=16, norm=True, dropout=0.1)
    modules = [type(m).__name__ for m in head.children()]
    assert modules == ["Linear", "BatchNorm1d", "GELU", "Dropout", "Linear"]


@pytest.mark.parametrize(
    ("head_cls", "linear_names", "extra_kwargs"),
    [(LinearHead, ["linear"], {}), (MLPHead, ["fc1", "fc2"], {"hidden": 256})],
)
def test_weights_are_trunc_normal_std_0_02_and_bias_is_zero(head_cls, linear_names, extra_kwargs):
    torch.manual_seed(0)
    head = head_cls(dim=256, **extra_kwargs)
    for name in linear_names:
        linear = getattr(head, name)
        assert torch.all(linear.bias == 0)
        std = linear.weight.std().item()
        assert 0.0 < std < 0.05


@pytest.mark.parametrize(("key", "cls"), [("linear", LinearHead), ("mlp", MLPHead)])
def test_registered_as_a_builtin_head(key, cls):
    entry = get_registry("heads").entry(key)
    assert entry.provider == "dfwb"
    assert entry.requires == ("torch",)
    head = api.heads.build(key, dim=8)
    assert isinstance(head, cls)
