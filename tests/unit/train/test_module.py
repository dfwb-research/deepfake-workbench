"""``DetectorModule``: training steps, and validation that computes its metrics with exactly the
code final evaluation uses (per-video aggregation, then ``dfwb.eval`` metrics)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("lightning")

import lightning.pytorch as L
import numpy as np
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
from dfwb.data.collate import collate_clips
from dfwb.eval.aggregate import aggregate
from dfwb.eval.metrics import MetricUndefined, compute
from dfwb.eval.report import evaluate
from dfwb.train.callbacks import EarlyStop, SafetensorsCheckpoint
from dfwb.train.module import DetectorModule, Monitor
from dfwb.train.optim import build_optimizer
from dfwb.train.schedules import build_schedule

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


def test_val_metrics_match_an_independent_per_video_computation(tmp_path, toy_work_root):
    config = toy_config(
        data={
            "val": [toy_source("val", **{"attrs.group": "a"}), toy_source("val")],
            "loader": {"batch_size": 3, "num_workers": 0},
        },
        eval={"metrics": ["auc", "brier", "nll"], "aggregate": "mean-logit"},
    )
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)
    module = run.module
    module.eval()
    for source in run.datamodule.val_sources:
        # score every clip on its own, outside the validation loop
        rows, labels = [], {}
        with torch.no_grad():
            for i in range(len(source.dataset)):
                sample = source.dataset[i]
                key = (sample.dataset, sample.key, sample.compression)
                rows.append((key, float(module(collate_clips([sample])).score[0])))
                labels[key] = sample.label
        video = aggregate(rows, "mean-logit")
        keys = sorted(video, key=lambda k: (k[0], k[1], k[2] or ""))
        y = np.asarray([labels[k] for k in keys])
        p = np.asarray([video[k] for k in keys], dtype=np.float64)
        for metric in config.eval.metrics:
            logged = module.val_metrics[f"val/{source.name}/{metric}"]
            assert logged == pytest.approx(compute(metric, y, p), rel=1e-6, abs=1e-9)
        dump = read_scores(_dump(run.run_dir, source.name))
        ok = [row for row in dump.rows if row.status == "ok"]
        assert len(ok) == len(source.index.items)
        assert {row.n_clips for row in ok} == {2}


def test_the_fallback_monitor_is_what_checkpoints_and_early_stop_follow(tmp_path, toy_work_root):
    config = toy_config(train={"max_epochs": 3}, data={"val": [toy_source("val", task="REAL")]})
    run = fit_toy(
        tmp_path / "run",
        config,
        work_root=toy_work_root,
        callback_options={"early_stop_patience": 5},
    )
    (saver,) = [c for c in run.trainer.callbacks if isinstance(c, SafetensorsCheckpoint)]
    (stopper,) = [c for c in run.trainer.callbacks if isinstance(c, EarlyStop)]
    assert saver.monitor == stopper.monitor == "val/loss"
    assert len(saver.history) == 3
    assert saver.best_value == stopper.best_value == min(saver.history)
    assert "val/loss" in run.trainer.callback_metrics
    assert "val/video_auc" not in run.trainer.callback_metrics


def test_a_batch_norm_head_never_sees_a_batch_of_one(tmp_path, toy_work_root):
    config = toy_config(
        model={
            "backbone": {"name": "tiny-cnn"},
            "temporal_pool": {"name": "mean"},
            "head": {"name": "mlp", "norm": True, "hidden": 8},
        },
        data={"loader": {"batch_size": 31, "num_workers": 0}},  # 32 train clips: 31 + 1
    )
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)
    assert run.trainer.global_step == 1  # the trailing single clip was dropped


# ----------------------------------------------------------------------------- determinism


def _deterministic_run(run_dir: Path, work_root: Path, *, num_workers: int):
    config = toy_config(
        train={"max_epochs": 2},
        data={
            "transforms": {"train": [{"name": "hflip", "p": 0.5}]},
            "loader": {"batch_size": 8, "num_workers": num_workers, "balance": "video-label"},
            "clip": {
                "frames": 2,
                "sampling": "random-window",
                "clips_per_video": {"train": 2, "eval": 2},
            },
        },
    )
    run = fit_toy(run_dir, config, work_root=work_root)
    weights = {k: v.clone() for k, v in run.module.detector.state_dict().items()}
    losses = column(read_metrics_csv(run.run_dir), "train/loss_step")
    dump = [(row.key, row.score) for row in read_scores(_dump(run.run_dir)).rows]
    return dict(run.module.val_metrics), weights, losses, dump


def test_two_runs_with_the_same_seed_are_identical(tmp_path, toy_work_root):
    metrics_a, weights_a, losses_a, dump_a = _deterministic_run(
        tmp_path / "a", toy_work_root, num_workers=0
    )
    metrics_b, weights_b, losses_b, dump_b = _deterministic_run(
        tmp_path / "b", toy_work_root, num_workers=0
    )
    assert metrics_a == metrics_b
    assert weights_a.keys() == weights_b.keys()
    assert all(torch.equal(weights_a[k], weights_b[k]) for k in weights_a)
    assert losses_a == losses_b
    assert dump_a == dump_b


def test_loader_workers_do_not_change_the_draws(tmp_path, toy_work_root):
    metrics_a, _, losses_a, _ = _deterministic_run(tmp_path / "a", toy_work_root, num_workers=0)
    metrics_b, _, losses_b, _ = _deterministic_run(tmp_path / "b", toy_work_root, num_workers=2)
    assert losses_a == losses_b
    assert metrics_a == metrics_b


# -------------------------------------------------------------------------------- schedule


def _expected_head_lrs(config, work_root: Path, *, steps_per_epoch: int, epochs: int):
    detector, _, _ = make_parts(config, work_root=work_root)
    optimizer = build_optimizer(config.optim, detector)
    scheduler = build_schedule(
        config.schedule, optimizer, steps_per_epoch=steps_per_epoch, epochs=epochs
    )
    (head,) = [i for i, g in enumerate(optimizer.param_groups) if g.get("name") == "head"]
    lrs = []
    for _ in range(steps_per_epoch * epochs):
        lrs.append(optimizer.param_groups[head]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


@pytest.mark.parametrize(
    "schedule",
    [
        {"name": "cosine", "warmup_epochs": 1, "min_lr": 0.001},
        {"name": "step", "step_epochs": 1, "gamma": 0.5},
    ],
)
def test_the_logged_lr_sequence_is_the_built_schedule(tmp_path, toy_work_root, schedule):
    config = toy_config(train={"max_epochs": 3}, schedule=schedule)
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root)

    logged = column(read_metrics_csv(run.run_dir), "lr-AdamW/head")
    assert run.trainer.global_step == 12
    expected = _expected_head_lrs(config, toy_work_root, steps_per_epoch=4, epochs=3)
    assert logged == pytest.approx(expected, rel=1e-12, abs=0)


def test_the_schedule_spans_the_trainers_epochs(tmp_path, toy_work_root):
    # the Trainer runs 2 epochs of a config that says 5: the schedule decays over the 2 run.
    config = toy_config(
        train={"max_epochs": 5}, schedule={"name": "cosine", "warmup_epochs": 0, "min_lr": 0.0}
    )
    run = fit_toy(tmp_path / "run", config, work_root=toy_work_root, max_epochs=2)

    logged = column(read_metrics_csv(run.run_dir), "lr-AdamW/head")
    expected = _expected_head_lrs(config, toy_work_root, steps_per_epoch=4, epochs=2)
    assert logged == pytest.approx(expected, rel=1e-12, abs=0)


def test_an_endless_run_has_no_schedule_length(tmp_path, toy_work_root):
    with pytest.raises(ConfigError, match="finite"):
        fit_toy(tmp_path / "run", toy_config(), work_root=toy_work_root, max_epochs=-1)


# --------------------------------------------------------------------------- one process


def test_more_than_one_process_is_refused(toy_work_root):
    _, _, module = make_parts(toy_config(), work_root=toy_work_root)
    module.trainer = SimpleNamespace(world_size=2)
    with pytest.raises(ContractError, match="single process") as caught:
        module.setup("fit")
    assert "devices" in caught.value.hint


# --------------------------------------------------------------------------------- monitor


def test_a_monitor_on_a_metric_not_computed_is_a_config_error(toy_work_root):
    config = toy_config(train={"monitor": "val/video_acu"})
    with pytest.raises(ConfigError, match=r"train\.monitor.*video_auc"):
        make_parts(config, work_root=toy_work_root)


def test_a_monitor_outside_validation_is_a_config_error(toy_work_root):
    config = toy_config(train={"monitor": "train/loss", "mode": "min"})
    with pytest.raises(ConfigError, match=r"train\.monitor"):
        make_parts(config, work_root=toy_work_root)


class _CountBatches(Callback):
    def __init__(self) -> None:
        self.batches = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batches += 1


def test_a_per_source_monitor_naming_no_source_fails_before_training(tmp_path, toy_work_root):
    counter = _CountBatches()
    config = toy_config(train={"monitor": "val/toytrain-oficial/auc"})
    with pytest.raises(ConfigError, match=r"train\.monitor") as caught:
        fit_toy(tmp_path / "run", config, work_root=toy_work_root, callbacks=[counter])
    assert f"did you mean 'val/{SOURCE}/auc'" in caught.value.message
    assert counter.batches == 0


def test_a_per_source_monitor_on_a_metric_not_computed_fails_before_training(
    tmp_path, toy_work_root
):
    counter = _CountBatches()
    config = toy_config(train={"monitor": f"val/{SOURCE}/ap"})
    with pytest.raises(ConfigError, match=r"train\.monitor"):
        fit_toy(tmp_path / "run", config, work_root=toy_work_root, callbacks=[counter])
    assert counter.batches == 0


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
