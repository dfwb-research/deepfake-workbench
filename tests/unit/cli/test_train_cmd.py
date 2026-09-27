"""``dfwb train``: new runs, resuming one, and refusing a bad config before any data loads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("lightning")

import yaml
from lightning.pytorch.callbacks import Callback
from tests.unit.train._toy import (
    install_toytrain_pack,
    toy_config,
    write_toy_config,
    write_toy_store,
)

from dfwb.train import run as run_module
from dfwb.train.datamodule import ProtocolDataModule


@pytest.fixture
def toy(run, tmp_path, monkeypatch):
    """The toy pack and store, with the work and runs roots pointing at them."""
    import torch

    install_toytrain_pack(tmp_path, monkeypatch)
    work_root = tmp_path / "work"
    write_toy_store(work_root)
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work_root))
    monkeypatch.setenv("DFWB_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # seeding sets it; put it back afterwards
    before = torch.are_deterministic_algorithms_enabled()
    yield tmp_path
    torch.use_deterministic_algorithms(before)


def _run_dirs(root: Path) -> list[Path]:
    return sorted(p for p in (root / "runs" / "toy").iterdir() if not p.is_symlink())


def test_train_writes_a_run_matching_config_show(run, toy):
    path = write_toy_config(toy, toy_config())
    result = run("train", "-c", str(path), "train.max_epochs=2", "--json")
    assert result.code == 0, result.err

    (entry,) = json.loads(result.out)
    run_dir = Path(entry["run_dir"])
    assert run_dir.parent == toy / "runs" / "toy"
    assert (entry["seed"], entry["metrics"]["epochs"]) == (0, 2)

    shown = run("config", "show", "-c", str(path), "train.max_epochs=2")
    header, _extends, *body = shown.out.splitlines(keepends=True)
    fingerprint = (run_dir / "fingerprint.txt").read_text("utf-8")
    assert header == f"# fingerprint: {fingerprint}"
    assert "".join(body) == (run_dir / "config.resolved.yaml").read_text("utf-8")
    as_json = json.loads(run("config", "show", "-c", str(path), "train.max_epochs=2", "--json").out)
    assert as_json["fingerprint"] == fingerprint.strip()
    assert (
        yaml.safe_load((run_dir / "config.resolved.yaml").read_text("utf-8")) == as_json["config"]
    )

    # and `dfwb runs` reads the run back
    (listed,) = json.loads(run("runs", "list", "--json").out)
    assert (listed["name"], listed["run"], listed["status"]) == ("toy", run_dir.name, "completed")
    assert listed["latest"] is True
    shown_run = json.loads(run("runs", "show", "toy", "--json").out)
    assert shown_run["fingerprint"] == fingerprint.strip()
    assert shown_run["path"] == str(run_dir)


def test_train_prints_each_run(run, toy):
    path = write_toy_config(toy, toy_config(run={"name": "toy", "seeds": [0, 1]}))
    result = run("train", "-c", str(path), "--device", "cpu")
    assert result.code == 0, result.err
    # the progress bar and model summary print too (with terminal escapes); each run then gets
    # one line of its own
    run_dirs = _run_dirs(toy)
    assert len(run_dirs) == 2
    for run_dir in run_dirs:
        assert f"{run_dir}: 1 epoch(s), best val/video_auc = " in result.out
        assert "(after epoch 1 of 1)" in result.out
    env = json.loads((_run_dirs(toy)[0] / "env.json").read_text("utf-8"))
    assert env["env"]["device"] == "cpu"


def test_train_rejects_bad_component_params_early(run, toy, monkeypatch):
    def _refuse(self, stage=None):
        raise AssertionError("data was loaded")

    monkeypatch.setattr(ProtocolDataModule, "setup", _refuse)
    model = {
        "backbone": {"name": "tiny-cnn", "freeze": {"mode": "partail"}},
        "temporal_pool": {"name": "mean"},
        "head": {"name": "linear"},
    }
    path = write_toy_config(toy, toy_config(model=model))

    result = run("train", "-c", str(path))

    assert result.code == 2, result.err
    assert "model.backbone.freeze.mode: 'partail' is not one of" in result.err
    assert "(did you mean 'partial'?)" in result.err
    assert "hint: " in result.err
    assert not (toy / "runs").exists()


def test_train_refuses_a_dim_the_framework_supplies(run, toy, monkeypatch):
    def _refuse(self, stage=None):
        raise AssertionError("data was loaded")

    monkeypatch.setattr(ProtocolDataModule, "setup", _refuse)
    path = write_toy_config(toy, toy_config())
    result = run("train", "-c", str(path), "model.head.dim=32")
    assert result.code == 2, result.err
    assert "model.head.dim: set from the backbone's output size; leave it out" in result.err
    assert "unexpected" not in result.err


def test_an_input_the_store_cannot_serve_names_the_built_in_profiles_that_can(run, toy):
    # the toy store is a face crop; a full-frame detector needs a store without face detection
    model = {
        "backbone": {"name": "tiny-cnn"},
        "temporal_pool": {"name": "mean"},
        "head": {"name": "linear"},
        "input": {"crop": "full-frame", "crop_scale": None},
    }
    path = write_toy_config(toy, toy_config(model=model))
    result = run("train", "-c", str(path))
    assert result.code == 4, result.err
    assert "built-in profiles that would serve it: toy-64-center-8f" in result.err
    assert "--profile toy-64-center-8f" in result.err


def test_json_output_stays_json_whatever_the_passthrough_says(run, toy):
    # the progress bar and the model summary print to stdout, which --json keeps for the results
    lightning = {"enable_progress_bar": True, "enable_model_summary": True}
    path = write_toy_config(toy, toy_config(train={"max_epochs": 1, "lightning": lightning}))
    result = run("train", "-c", str(path), "--json")
    assert result.code == 0, result.err
    (entry,) = json.loads(result.out)
    assert entry["metrics"]["epochs"] == 1


def test_resume_help_says_a_finished_run_is_not_extended(run):
    result = run("train", "--help")
    assert result.code == 0
    text = " ".join(result.out.split())
    assert "only an interrupted run" in text
    assert "a finished run cannot be extended" in text
    assert "more epochs is a new experiment" in text


class _Crash(Exception):
    pass


class _CrashAtEpochStart(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 1:
            raise _Crash("interrupted")


def test_train_resumes_an_interrupted_run(run, toy, monkeypatch):
    path = write_toy_config(toy, toy_config(train={"max_epochs": 2}))
    real = run_module.build_callbacks
    monkeypatch.setattr(
        run_module,
        "build_callbacks",
        lambda *args, **kwargs: [*real(*args, **kwargs), _CrashAtEpochStart()],
    )
    assert run("train", "-c", str(path)).code == 1
    monkeypatch.setattr(run_module, "build_callbacks", real)
    (run_dir,) = _run_dirs(toy)
    assert (run_dir / "resume").is_dir()

    other = write_toy_config(toy, toy_config(train={"max_epochs": 3}), name="other.yaml")
    refused = run("train", "-c", str(other), "--resume", str(run_dir))
    assert refused.code == 2
    assert "is not the config of" in refused.err

    listed = json.loads(run("runs", "list", "--json").out)
    assert [(r["status"], r["epochs"]) for r in listed] == [("incomplete", 1)]

    result = run("train", "-c", str(path), "--resume", str(run_dir), "--json")
    assert result.code == 0, result.err
    (entry,) = json.loads(result.out)
    assert (entry["run_dir"], entry["metrics"]["epochs"]) == (str(run_dir), 2)
    assert not (run_dir / "resume").exists()
    again = run("train", "--resume", str(run_dir))
    assert again.code == 2
    assert "nothing to resume" in again.err


def test_train_needs_a_config_or_a_run(run):
    result = run("train")
    assert result.code == 2
    assert "-c/--config" in result.err
    overrides = run("train", "--resume", "runs/x", "train.max_epochs=2")
    assert overrides.code == 2
    assert "overrides need -c/--config" in overrides.err


def test_train_without_torch_names_the_extra(run, monkeypatch):
    from dfwb.cli import train as train_command

    monkeypatch.setattr(train_command.importlib.util, "find_spec", lambda name: None)
    result = run("train", "-c", "exp.yaml")
    assert result.code == 5
    assert "needs 'torch'" in result.err
    assert 'pip install "deepfake-workbench[train]"' in result.err


def test_train_without_validation_says_so(run, toy):
    path = write_toy_config(toy, toy_config(data={"val": []}))
    result = run("train", "-c", str(path))
    assert result.code == 0, result.err
    assert ": 1 epoch(s), no validation (checkpoints/best is the last epoch's)" in result.out


def test_an_unknown_device_is_refused_before_training(run, toy):
    path = write_toy_config(toy, toy_config())
    result = run("train", "-c", str(path), "--device", "cdua")
    assert result.code == 2
    assert "did you mean 'cuda'" in result.err
    assert not (toy / "runs").exists()
