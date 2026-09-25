"""The training callbacks: best/last safetensors checkpoints, early stop, the non-finite loss
guard, the heartbeat file, and the LR monitor."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("lightning")

import torch
from tests.unit.train._toy import fit_toy, read_metrics_csv, toy_config

from dfwb.core.errors import ContractError
from dfwb.models import checkpoint
from dfwb.train.callbacks import (
    EarlyStop,
    Heartbeat,
    NonFiniteGuard,
    NonFiniteLoss,
    SafetensorsCheckpoint,
    ValScoreDump,
    build_callbacks,
)
from dfwb.train.losses import BCELoss, LossOutput

# ----------------------------------------------------------------------------- checkpoints


def test_best_and_last_checkpoints_are_safetensors_only(tmp_path, toy_work_root):
    run = fit_toy(tmp_path / "run", toy_config(train={"max_epochs": 2}), work_root=toy_work_root)
    checkpoints = run.run_dir / "checkpoints"

    assert sorted(p.name for p in checkpoints.iterdir()) == ["best", "last"]
    for name in ("best", "last"):
        files = sorted(p.name for p in (checkpoints / name).iterdir())
        assert files == ["detector.json", "model.safetensors"]
    every_file = [p for p in run.run_dir.rglob("*") if p.is_file()]
    assert not [p for p in every_file if p.suffix in {".ckpt", ".pt", ".pth", ".pkl"}]

    # "last" is the final weights; both load back through the registries.
    last = checkpoint.load(checkpoints / "last")
    live = run.module.detector.state_dict()
    for key, value in last.state_dict().items():
        assert torch.equal(value, live[key].cpu())
    checkpoint.load(checkpoints / "best")


def test_best_follows_the_monitor(tmp_path, toy_work_root):
    config = toy_config(train={"max_epochs": 3, "monitor": "val/loss", "mode": "min"})
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)

    (saver,) = [cb for cb in run.trainer.callbacks if isinstance(cb, SafetensorsCheckpoint)]
    state = saver.state_dict()
    assert state["monitor"] == "val/loss"
    assert state["best_value"] == pytest.approx(min(saver.history))
    assert len(saver.history) == 3


def test_lightning_checkpointing_is_refused(tmp_path, toy_work_root):
    with pytest.raises(ContractError, match="pickles") as caught:
        fit_toy(tmp_path / "run", toy_config(), work_root=toy_work_root, enable_checkpointing=True)
    assert "enable_checkpointing=False" in caught.value.hint


def test_checkpoint_state_round_trips(tmp_path):
    saver = SafetensorsCheckpoint(tmp_path, toy_config().model)
    saver.load_state_dict({"monitor": "val/loss", "best_value": 0.25, "best_epoch": 3})
    assert saver.state_dict() == {"monitor": "val/loss", "best_value": 0.25, "best_epoch": 3}


# ------------------------------------------------------------------------------ early stop


def test_early_stop_stops_after_patience_epochs_without_improvement(tmp_path, toy_work_root):
    config = toy_config(train={"max_epochs": 5})
    run = fit_toy(
        tmp_path / "run",
        config,
        work_root=toy_work_root,
        # nothing can improve by more than 10 on any metric, so only the first epoch counts.
        callback_options={"early_stop_patience": 2, "early_stop_min_delta": 10.0},
    )
    assert run.trainer.current_epoch == 3
    (stopper,) = [cb for cb in run.trainer.callbacks if isinstance(cb, EarlyStop)]
    assert stopper.state_dict()["wait"] == 2


def test_early_stop_state_round_trips():
    stopper = EarlyStop(patience=3)
    stopper.load_state_dict({"monitor": "val/loss", "best_value": 1.5, "wait": 2})
    assert stopper.state_dict() == {"monitor": "val/loss", "best_value": 1.5, "wait": 2}


def test_early_stop_is_off_unless_configured(tmp_path):
    from lightning.pytorch.callbacks import LearningRateMonitor

    kinds = {type(cb) for cb in build_callbacks(tmp_path, toy_config().model)}
    assert kinds == {
        SafetensorsCheckpoint,
        ValScoreDump,
        NonFiniteGuard,
        Heartbeat,
        LearningRateMonitor,
    }
    with pytest.raises(ValueError, match="patience"):
        EarlyStop(patience=0)


# ------------------------------------------------------------------------------ NaN guard


class _NonFiniteFor(BCELoss):
    """BCE, except the loss is NaN for the first ``n`` training batches."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.remaining = n

    def forward(self, out, batch):
        result = super().forward(out, batch)
        if self.training and self.remaining > 0:
            self.remaining -= 1
            nan = result.total * float("nan")
            return LossOutput(total=nan, parts={"bce": nan})
        return result


def test_nan_guard_stops_after_the_tolerance_naming_the_step(tmp_path, toy_work_root):
    modules = []
    before: dict[str, torch.Tensor] = {}

    def _poison(module):
        module.loss = _NonFiniteFor(100)
        modules.append(module)
        before.update({n: p.detach().clone() for n, p in module.detector.named_parameters()})

    with pytest.raises(NonFiniteLoss, match=r"3 consecutive.*step") as caught:
        fit_toy(
            tmp_path / "run",
            toy_config(train={"max_epochs": 2}),
            work_root=toy_work_root,
            module_hook=_poison,
        )
    assert "batch 2 of epoch 0" in caught.value.message

    # the three skipped updates never touched the weights.
    (module,) = modules
    for name, param in module.detector.named_parameters():
        assert torch.equal(param.detach(), before[name]), name


def test_nan_guard_tolerates_fewer_non_finite_steps_than_its_tolerance(tmp_path, toy_work_root):
    run = fit_toy(
        tmp_path / "run",
        toy_config(),
        work_root=toy_work_root,
        module_hook=lambda module: setattr(module, "loss", _NonFiniteFor(2)),
    )
    assert run.trainer.current_epoch == 1


def test_nan_guard_tolerance_is_configurable(tmp_path):
    callbacks = build_callbacks(tmp_path, toy_config().model, nan_tolerance=5)
    (guard,) = [cb for cb in callbacks if isinstance(cb, NonFiniteGuard)]
    assert guard.tolerance == 5
    with pytest.raises(ValueError, match="tolerance"):
        NonFiniteGuard(tolerance=0)


# ------------------------------------------------------------------------------ heartbeat


def test_heartbeat_file_is_updated_with_step_epoch_and_time(tmp_path, toy_work_root):
    run = fit_toy(
        tmp_path / "run",
        toy_config(train={"max_epochs": 2}),
        work_root=toy_work_root,
        callback_options={"heartbeat_every": 3},
    )
    beat = json.loads((run.run_dir / "heartbeat.json").read_text("utf-8"))
    assert set(beat) == {"step", "epoch", "time"}
    assert (beat["step"], beat["epoch"]) == (6, 1)  # 8 batches in all, a beat every 3
    assert beat["time"].endswith("Z")
    with pytest.raises(ValueError, match="every_n_steps"):
        Heartbeat(run.run_dir / "heartbeat.json", every_n_steps=0)


# ------------------------------------------------------------------------------ LR monitor


def test_lr_monitor_logs_through_the_logger(tmp_path, toy_work_root):
    run = fit_toy(tmp_path / "run", toy_config(), work_root=toy_work_root)
    header = read_metrics_csv(run.run_dir)[0].keys()
    assert {"lr-AdamW/blocks.0", "lr-AdamW/head"} <= set(header)
