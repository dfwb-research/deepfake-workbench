"""``DetectorModule``: training steps, and validation that computes its metrics with exactly the
code final evaluation uses (per-video aggregation, then ``dfwb.eval`` metrics)."""

from __future__ import annotations

import logging
import math

import pytest

pytest.importorskip("lightning")

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import Callback
from tests.unit.train._toy import (
    DATASET,
    column,
    fit_toy,
    make_parts,
    read_metrics_csv,
    toy_config,
    toy_source,
    write_toy_store,
)

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import read_scores
from dfwb.eval.metrics import MetricUndefined
from dfwb.eval.report import evaluate
from dfwb.train.module import DetectorModule, Monitor

SOURCE = f"{DATASET}-official"


def _dump(run_dir, name=SOURCE):
    return run_dir / "scores" / "val" / f"{name}.scores.csv"


# -------------------------------------------------------------------------------- training


def test_one_epoch_runs(tmp_path, toy_work_root):
    run = fit_toy(tmp_path / "run", toy_config(), work_root=toy_work_root)

    assert run.trainer.current_epoch == 1
    assert run.trainer.global_step == 4  # 16 videos * 2 clips / batch 8
    metrics = run.module.val_metrics
    assert {"val/loss", "val/video_auc", f"val/{SOURCE}/auc"} <= set(metrics)
    assert all(math.isfinite(value) for value in metrics.values())


def test_loss_decreases_over_three_epochs_on_separable_data(tmp_path, toy_work_root):
    run = fit_toy(tmp_path / "run", toy_config(train={"max_epochs": 3}), work_root=toy_work_root)

    epoch_losses = column(read_metrics_csv(run.run_dir), "train/loss_epoch")
    assert len(epoch_losses) == 3
    assert epoch_losses[-1] < epoch_losses[0]
    assert run.module.val_metrics["val/video_auc"] == 1.0  # dark reals, bright fakes


def test_a_non_finite_loss_skips_the_update(toy_work_root):
    _, datamodule, module = make_parts(toy_config(), work_root=toy_work_root)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    batch.clips[0] = float("nan")

    assert module.training_step(batch, 0) is None
    assert module.last_loss is not None
    assert not torch.isfinite(module.last_loss)


def test_optimizer_and_schedule_come_from_the_config(tmp_path, toy_work_root):
    config = toy_config(
        train={"max_epochs": 2},
        schedule={"name": "constant", "warmup_epochs": 1},
    )
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)

    rows = read_metrics_csv(run.run_dir)
    lr_columns = sorted(name for name in rows[0] if name.startswith("lr-"))
    assert lr_columns  # the LR monitor logs one column per parameter group
    lrs = column(rows, lr_columns[0])
    # linear warmup from 0 over the first epoch's 4 steps, then the constant base LR.
    assert lrs[:4] == pytest.approx([0.0, 0.0025, 0.005, 0.0075])
    assert lrs[-1] == pytest.approx(0.01)


# ------------------------------------------------------------------------------ validation


def test_val_metrics_equal_eval_on_dump(tmp_path, toy_work_root):
    config = toy_config(
        train={"max_epochs": 2},
        data={"val": [toy_source("val", **{"attrs.group": "a"}), toy_source("val")]},
        eval={"metrics": ["auc", "eer", "brier", "ece", "tpr@fpr=0.5"], "aggregate": "mean-prob"},
    )
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)
    logged = run.module.val_metrics
    names = [f"{SOURCE}-a", SOURCE]

    for name in names:
        result = evaluate([_dump(run.run_dir, name)], metrics=config.eval.metrics, bootstrap=10)
        (row,) = result.tables["files"]
        for metric in config.eval.metrics:
            assert logged[f"val/{name}/{metric}"] == row["metrics"][metric]["value"]

    for metric in config.eval.metrics:
        per_source = [logged[f"val/{name}/{metric}"] for name in names]
        assert logged[f"val/video_{metric}"] == sum(per_source) / len(per_source)

    # the logger got the same numbers (at its own float32 precision)
    rows = read_metrics_csv(run.run_dir)
    for key in ("val/video_auc", f"val/{SOURCE}/eer", "val/loss"):
        assert column(rows, key)[-1] == pytest.approx(logged[key], rel=1e-6)


def test_val_dump_is_a_complete_score_file(tmp_path, toy_pack, deterministic_torch):
    work_root = tmp_path / "work"
    write_toy_store(work_root, skip=["FAKE/f11"])  # one val video never processed
    run = fit_toy(tmp_path / "run", toy_config(), work_root=work_root)

    score_file = read_scores(_dump(run.run_dir))
    rows = {row.key: row for row in score_file.rows}
    assert len(rows) == 8  # every val video of the split, scored or not
    assert rows["FAKE/f11"].status == "missing"
    assert rows["FAKE/f11"].score is None
    assert rows["FAKE/f11"].label == 1
    assert rows["REAL/r08"].status == "ok"
    assert rows["REAL/r08"].n_clips == 2
    assert (rows["REAL/r08"].label_key, rows["REAL/r08"].method) == ("TOYTRAIN-REAL", "original")

    meta = score_file.meta
    assert meta is not None
    assert meta.coverage.model_dump() == {"expected": 8, "ok": 7, "missing": 1, "error": 0}
    assert meta.protocol.id == f"{DATASET}/official"
    assert meta.protocol.split == "val"
    assert meta.labels == "binary"
    assert meta.aggregation is not None
    assert (meta.aggregation.clip_to_video, meta.aggregation.clips_per_video) == ("mean-prob", 2)
    assert meta.processing_profile is not None
    assert meta.processing_profile.id == run.datamodule.val_sources[0].profile.profile_id()
    assert meta.seed == 0


def test_single_class_val_warns_and_monitor_falls_back(tmp_path, toy_work_root, caplog):
    config = toy_config(train={"max_epochs": 2}, data={"val": [toy_source("val", task="REAL")]})
    with caplog.at_level(logging.WARNING, logger="dfwb"):
        run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)

    metrics = run.module.val_metrics
    name = f"{SOURCE}-REAL"
    # AUC and EER need both classes: not reported at all -- never a placeholder 0.5.
    assert "val/video_auc" not in metrics
    assert f"val/{name}/auc" not in metrics
    assert f"val/{name}/brier" in metrics  # a single-class-safe metric is still reported
    assert "val/video_auc" not in read_metrics_csv(run.run_dir)[0]

    assert run.module.monitor == Monitor("val/loss", "min", fallback=True)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("auc" in m and "undefined" in m for m in warnings)
    assert any("val/video_auc" in m and "val/loss" in m for m in warnings)
    assert sum("val/video_auc" in m and "val/loss" in m for m in warnings) == 1  # warned once

    # checkpoint selection followed the fallback: a best checkpoint exists.
    assert (run.run_dir / "checkpoints" / "best" / "model.safetensors").is_file()
    # the dump is still written, and eval agrees the AUC is undefined for it.
    with pytest.raises(MetricUndefined):
        evaluate([_dump(run.run_dir, name)], metrics=["auc"], bootstrap=10)


def test_monitor_mean_skips_sources_where_the_metric_is_undefined(tmp_path, toy_work_root):
    config = toy_config(
        data={"val": [toy_source("val", task="REAL"), toy_source("val", **{"attrs.group": "a"})]}
    )
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)

    metrics = run.module.val_metrics
    assert metrics["val/video_auc"] == metrics[f"val/{SOURCE}-a/auc"]
    assert run.module.monitor == Monitor("val/video_auc", "max")


def test_the_sanity_check_pass_neither_logs_nor_dumps(tmp_path, toy_work_root):
    # a single-class first batch must not make the sanity pass decide the monitor's fallback.
    config = toy_config(data={"loader": {"batch_size": 2, "num_workers": 0}})
    seen = {}

    class _AfterSanity(Callback):
        def on_sanity_check_end(self, trainer, pl_module):
            seen["metrics"] = dict(pl_module.val_metrics)
            seen["dumped"] = _dump(tmp_path / "run").exists()

    run = fit_toy(
        tmp_path / "run",
        config,
        work_root=toy_work_root,
        num_sanity_val_steps=1,
        callbacks=[_AfterSanity()],
    )
    assert seen == {"metrics": {}, "dumped": False}
    assert run.module.monitor == Monitor("val/video_auc", "max")
    assert read_scores(_dump(run.run_dir)).meta.coverage.ok == 8


def test_a_multi_class_head_trains_on_val_loss_without_video_scores(tmp_path, toy_work_root):
    config = toy_config(
        model={
            "backbone": {"name": "tiny-cnn"},
            "temporal_pool": {"name": "mean"},
            "head": {"name": "linear", "num_classes": 3},
        },
        loss={"name": "ce"},
    )
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)

    assert set(run.module.val_metrics) == {"val/loss"}
    assert run.module.monitor == Monitor("val/loss", "min", fallback=True)
    assert not (run.run_dir / "scores").exists()  # no video scores, so no score file
    assert (run.run_dir / "checkpoints" / "best" / "model.safetensors").is_file()


# --------------------------------------------------------------------------------- monitor


def test_a_monitor_on_a_metric_not_computed_is_a_config_error(toy_work_root):
    config = toy_config(train={"monitor": "val/video_acu"})
    with pytest.raises(ConfigError, match=r"train\.monitor.*video_auc"):
        make_parts(config, work_root=toy_work_root)


def test_a_monitor_outside_validation_is_a_config_error(toy_work_root):
    config = toy_config(train={"monitor": "train/loss", "mode": "min"})
    with pytest.raises(ConfigError, match=r"train\.monitor"):
        make_parts(config, work_root=toy_work_root)


def test_a_monitor_naming_no_logged_value_fails_at_the_first_validation(tmp_path, toy_work_root):
    config = toy_config(train={"monitor": "val/other-source/auc"})
    with pytest.raises(ConfigError, match=r"train\.monitor.*other-source"):
        fit_toy(tmp_path / "run", config, work_root=toy_work_root)


def test_a_per_source_monitor_is_used_directly(tmp_path, toy_work_root):
    config = toy_config(train={"monitor": f"val/{SOURCE}/eer", "mode": "min"})
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)
    assert run.module.monitor == Monitor(f"val/{SOURCE}/eer", "min")


@pytest.mark.parametrize(
    ("mode", "value", "best", "expected"),
    [
        ("max", 0.9, None, True),
        ("max", 0.9, 0.8, True),
        ("max", 0.8, 0.8, False),
        ("min", 0.1, 0.2, True),
        ("min", 0.3, 0.2, False),
        ("max", float("nan"), None, False),
        ("max", 0.85, 0.8, False),  # not by more than min_delta
    ],
)
def test_monitor_improvement(mode, value, best, expected):
    min_delta = 0.1 if value == 0.85 else 0.0
    assert Monitor("val/x", mode).improved(value, best, min_delta=min_delta) is expected


def test_the_module_needs_the_protocol_data_module(tmp_path, toy_work_root):
    _, datamodule, module = make_parts(toy_config(), work_root=toy_work_root)
    datamodule.setup("fit")
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    loader = datamodule.train_dataloader()
    with pytest.raises(ContractError, match="ProtocolDataModule"):
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=datamodule.val_dataloader())


def test_detector_module_is_a_lightning_module():
    assert issubclass(DetectorModule, L.LightningModule)
