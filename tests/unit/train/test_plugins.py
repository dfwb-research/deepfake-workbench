"""Plugins in training: a loss and a callback from an installed plugin, named in a config the way
built-ins are, trained with, and carried across a resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("lightning")

from tests.unit.train._fake_plugin import CALLBACK, LOSS, install_fake_plugin
from tests.unit.train._toy import read_metrics_csv, toy_config, write_toy_config

from dfwb.core.config import load_config
from dfwb.core.errors import ConfigError, ContractError
from dfwb.train import run as run_module
from dfwb.train.resume import ResumeCheckpoint
from dfwb.train.run import resume_run, run_experiment


@pytest.fixture
def plugin(toy_work_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    install_fake_plugin(monkeypatch)
    return toy_work_root


def _config(tmp_path: Path, **train: object):
    return toy_config(
        data={"pairs": True},
        loss={"name": LOSS, "weight": 0.5},
        train={
            "max_epochs": 2,
            "callbacks": [{"name": CALLBACK, "path": str(tmp_path / "hooks.json")}],
            **train,
        },
    )


def _run(tmp_path: Path, work_root: Path, config) -> list:
    return run_experiment(
        load_config(write_toy_config(tmp_path, config)),
        work_root=work_root,
        runs_root=tmp_path / "runs",
        progress=False,
    )


def test_a_plugin_loss_and_callback_train_a_run(tmp_path, plugin):
    (result,) = _run(tmp_path, plugin, _config(tmp_path))
    assert result.metrics["epochs"] == 2

    hooks = json.loads((tmp_path / "hooks.json").read_text())
    assert hooks["calls"]["on_fit_start"] == 1
    assert hooks["calls"]["on_train_epoch_end"] == 2
    assert hooks["calls"]["on_validation_end"] >= 2
    assert hooks["calls"]["on_train_batch_end"] > 0
    assert hooks["epochs"] == 2

    # the plugin's loss trained the run: its own parts were logged, pair term included
    rows = read_metrics_csv(result.run_dir)
    assert len([row for row in rows if row.get("train/pair_epoch")]) == 2
    assert len([row for row in rows if row.get("train/bce_epoch")]) == 2


def test_plugin_callbacks_run_before_the_resume_state_is_saved(tmp_path, plugin, monkeypatch):
    seen: list[list[str]] = []
    real = run_module._trainer

    def _spy(plan, run_dir, callbacks, loggers):
        seen.append([type(callback).__name__ for callback in callbacks])
        return real(plan, run_dir, callbacks, loggers)

    monkeypatch.setattr(run_module, "_trainer", _spy)
    _run(tmp_path, plugin, _config(tmp_path, max_epochs=1))
    (names,) = seen
    assert names[-1] == ResumeCheckpoint.__name__  # last, so it saves every other one's state
    assert names[-2] == "HookRecorder"


def test_a_plugin_callbacks_state_survives_a_resume(tmp_path, plugin, monkeypatch):
    from tests.unit.train.test_resume import _Crash, _CrashAtEpochStart

    real = run_module.build_callbacks
    monkeypatch.setattr(
        run_module,
        "build_callbacks",
        lambda *args, **kwargs: [*real(*args, **kwargs), _CrashAtEpochStart(1)],
    )
    with pytest.raises(_Crash):
        _run(tmp_path, plugin, _config(tmp_path))
    monkeypatch.setattr(run_module, "build_callbacks", real)
    (run_dir,) = [p for p in (tmp_path / "runs" / "toy").iterdir() if not p.is_symlink()]
    state = json.loads((run_dir / "resume" / "state.json").read_text())
    assert state["callbacks"]["HookRecorder"] == {"epochs": 1}

    (tmp_path / "hooks.json").unlink()
    resume_run(run_dir, work_root=plugin, progress=False)
    hooks = json.loads((tmp_path / "hooks.json").read_text())
    assert hooks["epochs"] == 2  # 1 carried over, 1 run after the resume
    assert hooks["calls"]["on_train_epoch_end"] == 1


def test_a_callback_param_typo_is_named_before_any_data_loads(tmp_path, plugin):
    config = _config(tmp_path)
    config.train.callbacks[0] = config.train.callbacks[0].model_validate(
        {"name": CALLBACK, "pth": "x"}
    )
    with pytest.raises(ConfigError) as caught:
        _run(tmp_path, plugin, config)
    assert "train.callbacks[0].pth: unknown parameter (did you mean 'path'?)" in (
        caught.value.message
    )
    assert not (tmp_path / "runs").exists()


def test_a_callbacks_entry_that_builds_no_callback_is_refused(tmp_path, plugin):
    config = toy_config(train={"callbacks": [{"name": "fixture-not-a-callback", "path": "x"}]})
    with pytest.raises(ContractError, match=r"train\.callbacks\[0\]"):
        _run(tmp_path, plugin, config)
    assert not (tmp_path / "runs").exists()
