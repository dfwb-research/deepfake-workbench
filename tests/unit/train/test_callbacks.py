"""The training callbacks: best/last safetensors checkpoints, early stop, the non-finite loss
guard, the heartbeat file, and the LR monitor."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("lightning")

import torch
from lightning.pytorch.callbacks import Callback
from tests.unit.train._toy import fit_toy, make_parts, read_metrics_csv, toy_config

from dfwb.core.errors import ContractError
from dfwb.models import checkpoint
from dfwb.train import callbacks as callbacks_module
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


def test_without_monitoring_best_is_a_copy_of_last(tmp_path, toy_work_root):
    config = toy_config(train={"max_epochs": 2}, data={"val": []})
    run = fit_toy(
        tmp_path / "run", config, work_root=toy_work_root, callback_options={"best_is_last": True}
    )
    checkpoints = run.run_dir / "checkpoints"
    assert _hashes(checkpoints / "best") == _hashes(checkpoints / "last")
    assert run.module.val_metrics == {}  # no validation ran at all


def test_lightning_checkpointing_is_refused(tmp_path, toy_work_root):
    with pytest.raises(ContractError, match="pickles") as caught:
        fit_toy(tmp_path / "run", toy_config(), work_root=toy_work_root, enable_checkpointing=True)
    assert "enable_checkpointing=False" in caught.value.hint


def test_checkpoint_state_round_trips(tmp_path):
    saver = SafetensorsCheckpoint(tmp_path, toy_config().model)
    saver.load_state_dict({"monitor": "val/loss", "best_value": 0.25, "best_epoch": 3})
    assert saver.state_dict() == {"monitor": "val/loss", "best_value": 0.25, "best_epoch": 3}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(directory: Path) -> dict[str, str]:
    return {p.name: _sha(p) for p in directory.iterdir()}


def test_best_is_rewritten_only_on_improvement_and_last_every_epoch(tmp_path, toy_work_root):
    config = toy_config(train={"monitor": "val/loss", "mode": "min"})
    _, _, module = make_parts(config, work_root=toy_work_root)
    saver = SafetensorsCheckpoint(tmp_path / "ck", config.model)
    saved: list[tuple[int, str]] = []
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=0)
    saver._save = lambda name, pl_module: saved.append((trainer.current_epoch, name))  # type: ignore[method-assign]
    for epoch, value in enumerate([0.5, 0.4, 0.45, 0.4, 0.3, float("nan"), 0.2]):
        module.val_metrics = {"val/loss": value}
        trainer.current_epoch = epoch
        saver.on_validation_end(trainer, module)
        saver.on_train_epoch_end(trainer, module)
    assert [e for e, name in saved if name == "best"] == [0, 1, 4, 6]  # ties and NaN never win
    assert [e for e, name in saved if name == "last"] == list(range(7))


def test_a_run_rewrites_last_every_epoch_and_leaves_no_staging(tmp_path, toy_work_root):
    checkpoints = tmp_path / "run" / "checkpoints"
    seen: list[str] = []

    class _LastHash(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            seen.append(_sha(checkpoints / "last" / "model.safetensors"))

    fit_toy(
        tmp_path / "run",
        toy_config(train={"max_epochs": 3}),
        work_root=toy_work_root,
        callbacks=[_LastHash()],
    )
    assert len(set(seen)) == 3
    assert not list(checkpoints.glob(".*"))


def _crashing_save(directory, detector, model_cfg):
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / "model.safetensors").write_bytes(b"half written")
    raise OSError("disk full")


def test_a_failed_save_leaves_the_previous_checkpoint_in_place(
    tmp_path, toy_work_root, monkeypatch
):
    config = toy_config()
    _, _, module = make_parts(config, work_root=toy_work_root)
    saver = SafetensorsCheckpoint(tmp_path / "ck", config.model)
    saver._save("best", module)
    before = _hashes(tmp_path / "ck" / "best")

    with torch.no_grad():
        for param in module.detector.parameters():
            param.add_(1.0)
    monkeypatch.setattr(callbacks_module.checkpoint, "save", _crashing_save)
    with pytest.raises(OSError, match="disk full"):
        saver._save("best", module)

    assert _hashes(tmp_path / "ck" / "best") == before
    checkpoint.load(tmp_path / "ck" / "best")
    monkeypatch.undo()
    saver._save("best", module)
    assert _hashes(tmp_path / "ck" / "best") != before
    assert not list((tmp_path / "ck").glob(".*"))


def _interrupted_swap(tmp_path: Path, name: str, module, model_cfg) -> dict[str, str]:
    """A checkpoint directory as a crash between the two renames of a swap leaves it: the
    previous ``<name>`` moved aside to ``.<name>.old``, nothing at ``<name>`` yet."""
    directory = tmp_path / "ck"
    checkpoint.save(directory / f".{name}.old", module.detector, model_cfg)
    return _hashes(directory / f".{name}.old")


def test_setup_restores_a_checkpoint_left_aside_by_an_interrupted_swap(tmp_path, toy_work_root):
    config = toy_config()
    _, _, module = make_parts(config, work_root=toy_work_root)
    old_best = _interrupted_swap(tmp_path, "best", module, config.model)
    old_last = _interrupted_swap(tmp_path, "last", module, config.model)
    saver = SafetensorsCheckpoint(tmp_path / "ck", config.model)

    saver.setup(SimpleNamespace(callbacks=[saver]), module, "fit")

    assert _hashes(tmp_path / "ck" / "best") == old_best
    assert _hashes(tmp_path / "ck" / "last") == old_last
    assert not list((tmp_path / "ck").glob(".*"))


def test_a_save_first_restores_a_checkpoint_left_aside(tmp_path, toy_work_root, monkeypatch):
    config = toy_config()
    _, _, module = make_parts(config, work_root=toy_work_root)
    old_best = _interrupted_swap(tmp_path, "best", module, config.model)
    saver = SafetensorsCheckpoint(tmp_path / "ck", config.model)

    monkeypatch.setattr(callbacks_module.checkpoint, "save", _crashing_save)
    with pytest.raises(OSError, match="disk full"):
        saver._save("best", module)

    # the save failed, but the checkpoint it would have replaced is back in place.
    assert _hashes(tmp_path / "ck" / "best") == old_best


def test_setup_leaves_complete_checkpoints_alone(tmp_path, toy_work_root):
    config = toy_config()
    _, _, module = make_parts(config, work_root=toy_work_root)
    saver = SafetensorsCheckpoint(tmp_path / "ck", config.model)
    saver._save("best", module)
    before = _hashes(tmp_path / "ck" / "best")
    shutil.copytree(tmp_path / "ck" / "best", tmp_path / "ck" / ".best.old")  # a stale leftover

    saver.setup(SimpleNamespace(callbacks=[saver]), module, "fit")

    assert _hashes(tmp_path / "ck" / "best") == before
    assert not (tmp_path / "ck" / ".best.old").exists()


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


def test_nan_guard_counts_only_consecutive_non_finite_losses(toy_work_root):
    _, _, module = make_parts(toy_config(), work_root=toy_work_root)
    trainer = SimpleNamespace(current_epoch=0, global_step=0)
    guard = NonFiniteGuard(tolerance=3)
    nan, inf, finite = torch.tensor(float("nan")), torch.tensor(float("inf")), torch.tensor(0.3)

    # two in a row, then a finite loss resets the count -- never three in a row.
    for i, loss in enumerate([nan, nan, finite, nan, nan, finite, inf, nan, finite]):
        module.last_loss = loss
        guard.on_train_batch_end(trainer, module, None, None, i)

    for i, loss in enumerate([nan, torch.tensor(float("-inf"))]):
        module.last_loss = loss
        trainer.global_step = 40 + i
        guard.on_train_batch_end(trainer, module, None, None, i)
    module.last_loss = nan
    trainer.global_step = 42
    with pytest.raises(NonFiniteLoss, match="global step 42"):
        guard.on_train_batch_end(trainer, module, None, None, 2)


def test_nan_guard_state_round_trips():
    guard = NonFiniteGuard(tolerance=3)
    assert guard.state_dict() == {"count": 0}
    guard.load_state_dict({"count": 2})
    assert guard.state_dict() == {"count": 2}


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
    beat = json.loads((run.run_dir / "logs" / "heartbeat.json").read_text("utf-8"))
    assert set(beat) == {"step", "epoch", "time"}
    assert (beat["step"], beat["epoch"]) == (6, 1)  # 8 batches in all, a beat every 3
    assert beat["time"].endswith("Z")
    assert not (run.run_dir / "heartbeat.json").exists()  # logs/ holds it, beside the other logs
    with pytest.raises(ValueError, match="every_n_steps"):
        Heartbeat(run.run_dir / "heartbeat.json", every_n_steps=0)


def test_heartbeat_state_round_trips(tmp_path):
    beat = Heartbeat(tmp_path / "heartbeat.json", every_n_steps=3)
    beat.load_state_dict({"batches": 7})
    assert beat.state_dict() == {"batches": 7}


# ------------------------------------------------------------------------------ LR monitor


def test_lr_monitor_logs_through_the_logger(tmp_path, toy_work_root):
    run = fit_toy(tmp_path / "run", toy_config(), work_root=toy_work_root)
    header = read_metrics_csv(run.run_dir)[0].keys()
    assert {"lr-AdamW/blocks.0", "lr-AdamW/head"} <= set(header)
