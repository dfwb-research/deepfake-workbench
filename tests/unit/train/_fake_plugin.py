"""A plugin's registry targets, as a separately installed distribution would provide them: a
pairwise loss that reads ``extras["dfwb/pair_id"]`` and the detector's features, and a stateful
callback that records which training hooks ran.

:func:`install_fake_plugin` registers both under a fake ``dfwb.plugins`` entry point, the way an
installed plugin is discovered, so a training config names them like any other component.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("lightning")

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch import nn

from dfwb.core import plugins
from dfwb.core.detector import ClipBatch, DetectorOutput
from dfwb.train.losses import LossOutput

__all__ = ["HookRecorder", "PairLoss", "install_fake_plugin"]

LOSS = "fixture-pair-loss"
CALLBACK = "fixture-hook-recorder"


class PairLoss(nn.Module):
    """BCE on the logit, plus a pull between the features of each pair's rows.

    Validation batches are never paired, and ``val/loss`` is computed with the training loss, so
    in eval mode it is the BCE alone."""

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, out: DetectorOutput, batch: ClipBatch) -> LossOutput:
        assert batch.labels is not None
        assert out.logit is not None
        assert out.features is not None
        bce = F.binary_cross_entropy_with_logits(out.logit, batch.labels.float())
        if not self.training:
            return LossOutput(total=bce, parts={"bce": bce})
        pair_id = batch.extras["dfwb/pair_id"]  # a KeyError here: the pairs never arrived
        same_pair = (pair_id[:, None] == pair_id[None, :]).float()
        distance = torch.cdist(out.features, out.features)
        pair = (distance * same_pair).sum() / same_pair.sum()
        return LossOutput(total=bce + self.weight * pair, parts={"bce": bce, "pair": pair})


class HookRecorder(Callback):
    """Writes ``path`` (JSON) with how many times each hook ran; its count of finished epochs is
    state, so a resumed run carries it on."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.calls: dict[str, int] = {}
        self.epochs = 0

    def _record(self, hook: str) -> None:
        self.calls[hook] = self.calls.get(hook, 0) + 1
        self.path.write_text(json.dumps({"calls": self.calls, "epochs": self.epochs}))

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._record("on_fit_start")

    def on_train_batch_end(self, *args: Any) -> None:
        self._record("on_train_batch_end")

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.epochs += 1
        self._record("on_train_epoch_end")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._record("on_validation_end")

    def state_dict(self) -> dict[str, Any]:
        return {"epochs": self.epochs}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.epochs = int(state_dict["epochs"])


def not_a_callback(path: str) -> object:
    """A callbacks target that builds something other than a Lightning callback."""
    return object()


class _FakeEntryPoint:
    def __init__(self, register: Callable[[Any], None]) -> None:
        self.name = "fixture-plugin"
        self.value = "fixture_plugin:register"
        self.dist = SimpleNamespace(name="fixture-plugin", version="0.0.0")
        self._register = register

    def load(self) -> Callable[[Any], None]:
        return self._register


def install_fake_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the plugin discoverable, beside whatever entry points are already there."""

    def _register(api: Any) -> None:
        target = "tests.unit.train._fake_plugin"
        api.losses.add(LOSS, target=f"{target}:PairLoss", summary="fixture pairwise loss")
        api.callbacks.add(CALLBACK, target=f"{target}:HookRecorder", summary="fixture callback")
        api.callbacks.add(
            "fixture-not-a-callback", target=f"{target}:not_a_callback", summary="broken"
        )

    earlier = plugins._entry_points

    def _entry_points(group: str) -> list[Any]:
        found = earlier(group)
        if group == plugins.ENTRY_POINT_GROUP:
            return [*found, _FakeEntryPoint(_register)]
        return found

    monkeypatch.setattr(plugins, "_entry_points", _entry_points)
    plugins.reset()
