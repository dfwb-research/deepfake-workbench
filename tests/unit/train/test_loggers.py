"""Loggers: CSV always (into ``logs/``), TensorBoard when it is installed, W&B only when
configured and installed."""

from __future__ import annotations

import pytest

pytest.importorskip("lightning")

from lightning.pytorch.loggers import CSVLogger
from tests.unit.train._toy import fit_toy, read_metrics_csv, toy_config

from dfwb.core.errors import InstallationError
from dfwb.train import loggers as loggers_module
from dfwb.train.loggers import build_loggers


def test_csv_logger_writes_the_logged_columns(tmp_path, toy_work_root):
    run = fit_toy(tmp_path / "run", toy_config(train={"max_epochs": 2}), work_root=toy_work_root)

    rows = read_metrics_csv(run.run_dir)
    header = set(rows[0])
    assert {
        "epoch",
        "step",
        "train/loss_step",
        "train/loss_epoch",
        "train/bce_step",
        "val/loss",
        "val/video_auc",
        "val/toytrain-official/auc",
        "val/toytrain-official/eer",
    } <= header
    assert len([row for row in rows if row["val/loss"]]) == 2  # one per validation epoch


def test_csv_is_the_default_and_the_only_logger_without_extras(tmp_path, monkeypatch):
    monkeypatch.setattr(loggers_module, "_installed", lambda name: False)
    (csv_logger,) = build_loggers(tmp_path)
    assert isinstance(csv_logger, CSVLogger)
    assert csv_logger.log_dir.rstrip("/") == str(tmp_path / "logs")


def test_tensorboard_is_added_when_installed(tmp_path, monkeypatch):
    made = {}

    class _FakeTensorBoard:
        def __init__(self, **options):
            made.update(options)

    monkeypatch.setattr(loggers_module, "_installed", lambda name: name == "tensorboard")
    monkeypatch.setattr(loggers_module, "TensorBoardLogger", _FakeTensorBoard)

    csv_logger, tensorboard = build_loggers(tmp_path)

    assert isinstance(csv_logger, CSVLogger)
    assert isinstance(tensorboard, _FakeTensorBoard)
    assert made["save_dir"] == tmp_path / "logs"
    assert made["name"] == "tensorboard"


def test_tensorboard_can_be_left_out(tmp_path, monkeypatch):
    monkeypatch.setattr(loggers_module, "_installed", lambda name: True)
    (csv_logger,) = build_loggers(tmp_path, tensorboard=False)
    assert isinstance(csv_logger, CSVLogger)


def test_wandb_without_the_package_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setattr(loggers_module, "_installed", lambda name: False)
    with pytest.raises(InstallationError, match="wandb") as caught:
        build_loggers(tmp_path, wandb={"project": "toy"})
    assert "[wandb]" in caught.value.hint


def test_wandb_is_added_when_configured_and_installed(tmp_path, monkeypatch):
    made = {}

    class _FakeWandb:
        def __init__(self, **options):
            made.update(options)

    monkeypatch.setattr(loggers_module, "_installed", lambda name: name == "wandb")
    monkeypatch.setattr(loggers_module, "_wandb_logger_class", lambda: _FakeWandb)

    csv_logger, wandb = build_loggers(tmp_path, wandb={"project": "toy"})

    assert isinstance(csv_logger, CSVLogger)
    assert isinstance(wandb, _FakeWandb)
    assert made == {"project": "toy", "save_dir": str(tmp_path / "logs")}


def test_wandb_is_not_used_unless_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(loggers_module, "_installed", lambda name: True)
    monkeypatch.setattr(loggers_module, "TensorBoardLogger", lambda **options: "tb")
    assert len(build_loggers(tmp_path)) == 2  # csv + tensorboard, no wandb


def test_installed_checks_for_an_importable_module():
    assert loggers_module._installed("lightning") is True
    assert loggers_module._installed("dfwb_no_such_module_anywhere") is False
    assert loggers_module._installed("") is False  # find_spec raises on an empty name


def test_the_wandb_logger_class_is_lightnings():
    from lightning.pytorch.loggers import WandbLogger

    assert loggers_module._wandb_logger_class() is WandbLogger
