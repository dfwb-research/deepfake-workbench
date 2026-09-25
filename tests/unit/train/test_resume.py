"""Resume: the ``resume/`` state (safetensors and JSON, never a pickle) written at every epoch end,
and a resumed run finishing exactly where an uninterrupted one does."""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

import pytest

pytest.importorskip("lightning")

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from safetensors.torch import load_file
from tests.unit.train._toy import read_metrics_csv, toy_config, write_toy_config

from dfwb.core.config import load_config
from dfwb.core.config.loader import LoadedConfig
from dfwb.core.config.schema import TrainConfig
from dfwb.core.errors import ConfigError, ContractError
from dfwb.train import resume as resume_module
from dfwb.train import run as run_module
from dfwb.train.run import RunResult, resume_run, run_experiment


def load(directory: Path, config: TrainConfig) -> LoadedConfig:
    return load_config(write_toy_config(directory, config))


class _Crash(Exception):
    """Stands in for the process dying mid-run."""


class _CrashAtEpochStart(Callback):
    def __init__(self, epoch: int) -> None:
        self.epoch = epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.epoch:
            raise _Crash(f"interrupted at the start of epoch {self.epoch}")


class _CrashAtTrainEnd(Callback):
    def on_train_end(self, trainer, pl_module):
        raise _Crash("interrupted after the last epoch")


def _config() -> TrainConfig:
    # warmup then cosine decay: the learning rate at every step depends on where the run is, so a
    # resume that lost its place in the schedule (or the optimiser's moments) shows up.
    return toy_config(
        train={"max_epochs": 2, "lightning": {"deterministic": True}},
        schedule={"name": "cosine", "warmup_epochs": 0.5, "min_lr": 0.001},
        optim={"name": "adamw", "lr": 0.01, "weight_decay": 0.01},
    )


def _train(directory: Path, work_root: Path, config: TrainConfig) -> RunResult:
    directory.mkdir(parents=True, exist_ok=True)
    (result,) = run_experiment(
        load(directory, config), work_root=work_root, runs_root=directory / "runs", progress=False
    )
    return result


def _interrupted(
    directory: Path, work_root: Path, config: TrainConfig, monkeypatch, crash: Callback
) -> Path:
    """Train ``config`` until ``crash`` fires; returns the run directory it left behind."""
    real = run_module.build_callbacks
    monkeypatch.setattr(
        run_module, "build_callbacks", lambda *args, **kwargs: [*real(*args, **kwargs), crash]
    )
    directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(_Crash):
        run_experiment(
            load(directory, config),
            work_root=work_root,
            runs_root=directory / "runs",
            progress=False,
        )
    monkeypatch.setattr(run_module, "build_callbacks", real)
    (run_dir,) = [p for p in (directory / "runs" / "toy").iterdir() if not p.is_symlink()]
    return run_dir


def _no_pickles(monkeypatch) -> None:
    def _refuse(*args, **kwargs):
        raise AssertionError("torch.load was called")

    monkeypatch.setattr(torch, "load", _refuse)


def test_resume_matches_uninterrupted(tmp_path, toy_work_root, monkeypatch):
    config = _config()
    uninterrupted = _train(tmp_path / "a", toy_work_root, config)
    run_dir = _interrupted(
        tmp_path / "b", toy_work_root, config, monkeypatch, _CrashAtEpochStart(1)
    )
    assert not (run_dir / "metrics.json").exists()
    assert json.loads((run_dir / "resume" / "state.json").read_text("utf-8"))["epoch"] == 1

    _no_pickles(monkeypatch)
    resumed = resume_run(run_dir, work_root=toy_work_root, progress=False)

    assert resumed.run_dir == run_dir
    expected, got = uninterrupted.metrics, resumed.metrics
    assert (got["epochs"], got["global_step"]) == (expected["epochs"], expected["global_step"])
    assert got["val"].keys() == expected["val"].keys()
    for key, value in expected["val"].items():
        assert got["val"][key] == pytest.approx(value, abs=1e-6), key
    best = got["monitor"].pop("best")
    assert best == pytest.approx(expected["monitor"].pop("best"), abs=1e-6)
    assert got["monitor"] == expected["monitor"]
    got["monitor"]["best"] = best

    a = load_file(uninterrupted.run_dir / "checkpoints" / "last" / "model.safetensors")
    b = load_file(run_dir / "checkpoints" / "last" / "model.safetensors")
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), key

    # env.json records the resume beside the start (the machine or versions may differ).
    env = json.loads((run_dir / "env.json").read_text("utf-8"))
    (resumed_at,) = env["resumed"]
    assert resumed_at["epoch"] == 1
    assert {"created", "command", "env", "git", "plugins"} <= set(resumed_at)

    # a finished run keeps no resume state, and its log covers both epochs.
    assert not (run_dir / "resume").exists()
    assert json.loads((run_dir / "metrics.json").read_text("utf-8")) == got
    epochs = {int(row["epoch"]) for row in read_metrics_csv(run_dir) if row.get("train/loss_epoch")}
    assert epochs == {0, 1}
    assert len([row for row in read_metrics_csv(run_dir) if row.get("val/loss")]) == 2


def test_resume_state_is_safetensors_and_json(tmp_path, toy_work_root, monkeypatch):
    config = toy_config(
        train={"max_epochs": 2, "early_stop": {"patience": 3}, "monitor": "val/loss", "mode": "min"}
    )
    run_dir = _interrupted(tmp_path, toy_work_root, config, monkeypatch, _CrashAtEpochStart(1))
    resume_dir = run_dir / "resume"

    assert sorted(p.name for p in resume_dir.iterdir()) == [
        "model.safetensors",
        "optimizer.safetensors",
        "state.json",
    ]
    state = json.loads((resume_dir / "state.json").read_text("utf-8"))
    assert (state["seed"], state["epoch"], state["global_step"]) == (0, 1, 4)
    assert state["data"] == {"epoch": 0}
    assert {"python", "numpy", "torch"} <= set(state["rng"])
    assert state["module"]["monitor"] == {"key": "val/loss", "mode": "min", "fallback": False}
    assert "val/loss" in state["module"]["val_metrics"]
    callbacks = state["callbacks"]
    assert any(key.startswith("SafetensorsCheckpoint") for key in callbacks)
    assert any(key.startswith("EarlyStop") for key in callbacks)
    assert any(key.startswith("NonFiniteGuard") for key in callbacks)
    groups = state["optimizer"]["param_groups"]
    assert [group["name"] for group in groups] == ["blocks.0", "blocks.1", "blocks.2", "head"]
    assert state["scheduler"]["last_epoch"] == 4

    model = load_file(resume_dir / "model.safetensors")
    last = load_file(run_dir / "checkpoints" / "last" / "model.safetensors")
    assert all(torch.equal(model[key], last[key]) for key in last)
    moments = load_file(resume_dir / "optimizer.safetensors")
    assert any(key.endswith(".exp_avg") for key in moments)


def test_resume_after_the_last_epoch_only_finishes_the_run(tmp_path, toy_work_root, monkeypatch):
    config = toy_config(train={"max_epochs": 2})
    run_dir = _interrupted(tmp_path, toy_work_root, config, monkeypatch, _CrashAtTrainEnd())
    state = json.loads((run_dir / "resume" / "state.json").read_text("utf-8"))
    assert state["epoch"] == 2

    resumed = resume_run(run_dir, work_root=toy_work_root, progress=False)
    assert (resumed.metrics["epochs"], resumed.metrics["global_step"]) == (2, 8)
    assert resumed.metrics["val"] == pytest.approx(state["module"]["val_metrics"])
    assert not (run_dir / "resume").exists()


def test_a_run_early_stopping_had_ended_stays_ended(tmp_path, toy_work_root, monkeypatch):
    config = toy_config(train={"max_epochs": 5, "early_stop": {"patience": 1, "min_delta": 10.0}})
    run_dir = _interrupted(tmp_path, toy_work_root, config, monkeypatch, _CrashAtTrainEnd())
    state = json.loads((run_dir / "resume" / "state.json").read_text("utf-8"))
    assert (state["epoch"], state["should_stop"]) == (2, True)

    resumed = resume_run(run_dir, work_root=toy_work_root, progress=False)
    assert resumed.metrics["epochs"] == 2  # not a third epoch


def test_a_resume_warns_when_the_joined_data_changed(tmp_path, toy_work_root, monkeypatch, caplog):
    config = toy_config(train={"max_epochs": 2})
    run_dir = _interrupted(tmp_path, toy_work_root, config, monkeypatch, _CrashAtEpochStart(1))
    index = toy_work_root / "toytrain" / "processed"
    (store,) = index.iterdir()
    rows = (store / "index.jsonl").read_text("utf-8").splitlines()
    (store / "index.jsonl").write_text("\n".join(r for r in rows if "f11" not in r) + "\n")

    with caplog.at_level(logging.WARNING, logger="dfwb"):
        resume_run(run_dir, work_root=toy_work_root, progress=False)
    assert any("joined data differs" in r.getMessage() for r in caplog.records)


def test_a_finished_run_has_nothing_to_resume(tmp_path, toy_work_root):
    result = _train(tmp_path, toy_work_root, toy_config())
    with pytest.raises(ConfigError, match="nothing to resume") as caught:
        resume_run(result.run_dir, work_root=toy_work_root, progress=False)
    assert "finished" in caught.value.message


def test_a_directory_that_is_not_a_run_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="not a run directory"):
        resume_run(tmp_path, progress=False)


def test_an_edited_config_is_refused(tmp_path, toy_work_root, monkeypatch):
    config = toy_config(train={"max_epochs": 2})
    run_dir = _interrupted(tmp_path, toy_work_root, config, monkeypatch, _CrashAtEpochStart(1))
    resolved = run_dir / "config.resolved.yaml"
    resolved.write_text(resolved.read_text("utf-8").replace("lr: 0.01", "lr: 0.02"), "utf-8")
    with pytest.raises(ContractError, match="fingerprint"):
        resume_run(run_dir, work_root=toy_work_root, progress=False)


def test_a_resume_restores_a_state_left_aside_by_an_interrupted_write(tmp_path):
    directory = tmp_path / "resume"
    resume_module.write_state(directory, {"w": torch.ones(2)}, {}, {"format": 1, "epoch": 1})
    directory.rename(tmp_path / ".resume.old")  # a crash between the two renames of a swap

    state = resume_module.read_state(directory)
    assert state is not None
    assert state.payload["epoch"] == 1
    assert torch.equal(state.model["w"], torch.ones(2))
    assert not (tmp_path / ".resume.old").exists()
    assert resume_module.read_state(tmp_path / "elsewhere") is None


def test_a_state_in_another_format_is_refused(tmp_path):
    resume_module.write_state(tmp_path / "resume", {}, {}, {"format": 99})
    with pytest.raises(ContractError, match="format 99"):
        resume_module.read_state(tmp_path / "resume")


# ------------------------------------------------------------------------------ encoding


def test_nested_state_round_trips_through_json_and_tensors():
    value = {
        "state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.arange(4.0)}},
        "param_groups": [{"betas": (0.9, 0.999), "params": [0, 1], "name": "head", "x": None}],
        "floats": [math.inf, -math.inf, 0.5],
        "flag": True,
        "np": np.float64(0.25),
        "npi": np.int64(7),
    }
    tensors: dict[str, torch.Tensor] = {}
    encoded = resume_module.encode(value, tensors, "optimizer")
    text = json.dumps(encoded, allow_nan=False)  # strict JSON: no NaN/Infinity literals
    decoded = resume_module.decode(json.loads(text), tensors)

    assert decoded["param_groups"] == [
        {"betas": (0.9, 0.999), "params": [0, 1], "name": "head", "x": None}
    ]
    assert list(decoded["state"]) == [0]  # integer keys survive
    assert torch.equal(decoded["state"][0]["exp_avg"], torch.arange(4.0))
    assert decoded["floats"][:2] == [math.inf, -math.inf]
    assert (decoded["np"], decoded["npi"]) == (0.25, 7)
    assert set(tensors) == {"optimizer.state.0.step", "optimizer.state.0.exp_avg"}

    nan = resume_module.decode(json.loads(json.dumps(resume_module.encode(math.nan, {}, "x"))), {})
    assert math.isnan(nan)
    with pytest.raises(ContractError, match="cannot store"):
        resume_module.encode(object(), {}, "somewhere")


def test_random_states_round_trip():
    # the legacy global NumPy generator is exactly what seeding sets, so it is what a resume
    # must put back.
    random.seed(1)
    np.random.seed(2)  # noqa: NPY002
    torch.manual_seed(3)
    state = json.loads(json.dumps(resume_module.capture_rng()))
    expected = (random.random(), np.random.rand(), torch.rand(1).item())  # noqa: NPY002

    random.seed(9)
    np.random.seed(9)  # noqa: NPY002
    torch.manual_seed(9)
    resume_module.restore_rng(state)
    again = (random.random(), np.random.rand(), torch.rand(1).item())  # noqa: NPY002
    assert again == expected
