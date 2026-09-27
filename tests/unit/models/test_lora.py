"""``apply_lora``: the one shared PEFT helper every LoRA-capable backbone calls."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("peft")

import torch
from torch import nn

from dfwb.core.errors import ConfigError, InstallationError
from dfwb.models.backbone import FreezeSpec
from dfwb.models.lora import apply_lora


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attn()
        self.norm = nn.LayerNorm(8)


def test_apply_lora_makes_only_lora_params_trainable():
    module = _Toy()
    apply_lora(module, FreezeSpec(mode="lora", targets=["q_proj", "v_proj"]))
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    frozen = [n for n, p in module.named_parameters() if not p.requires_grad]
    assert trainable
    assert frozen
    assert all("lora" in n for n in trainable)
    assert all("lora" not in n for n in frozen)


def test_apply_lora_uses_the_freeze_spec_hyperparameters():
    module = _Toy()
    apply_lora(module, FreezeSpec(mode="lora", targets=["q_proj"], r=4, alpha=8, dropout=0.0))
    lora_a = module.attn.q_proj.lora_A["default"]
    assert lora_a.out_features == 4


def test_apply_lora_leaves_the_module_usable():
    module = _Toy()
    apply_lora(module, FreezeSpec(mode="lora", targets=["q_proj", "v_proj"]))
    y = module.attn.q_proj(torch.rand(2, 8))
    assert y.shape == (2, 8)


def test_apply_lora_returns_the_same_module():
    module = _Toy()
    result = apply_lora(module, FreezeSpec(mode="lora", targets=["q_proj"]))
    assert result is module


def test_apply_lora_rejects_targets_matching_nothing():
    module = _Toy()
    with pytest.raises(ConfigError) as info:
        apply_lora(module, FreezeSpec(mode="lora", targets=["nope"]))
    assert "nope" in info.value.message
    assert "q_proj" in info.value.hint


def test_apply_lora_rejects_targets_matching_nothing_without_mutating_the_module():
    module = _Toy()
    names_before = [name for name, _ in module.named_modules()]
    param_ids_before = {id(p) for p in module.parameters()}
    grads_before = [p.requires_grad for p in module.parameters()]
    with pytest.raises(ConfigError):
        apply_lora(module, FreezeSpec(mode="lora", targets=["nope"]))
    assert [name for name, _ in module.named_modules()] == names_before
    assert {id(p) for p in module.parameters()} == param_ids_before
    assert [p.requires_grad for p in module.parameters()] == grads_before
    assert all(p.requires_grad for p in module.parameters())


def test_apply_lora_raises_installation_error_without_peft(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "peft", None)
    module = _Toy()
    with pytest.raises(InstallationError) as info:
        apply_lora(module, FreezeSpec(mode="lora", targets=["q_proj"]))
    assert "peft" in info.value.hint
