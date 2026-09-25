"""The callbacks every training run uses.

- :class:`SafetensorsCheckpoint`: ``checkpoints/best`` (by the module's monitor) and
  ``checkpoints/last``, each ``model.safetensors`` plus ``detector.json`` -- weights and JSON
  only, never a pickled module. Lightning's own checkpointing pickles, so it must be off
  (``enable_checkpointing=False``); this callback refuses to run alongside it.
- :class:`ValScoreDump`: after each validation epoch, one score file per validation source,
  ``scores/val/<source>.scores.{csv,meta.json}``, holding exactly the video scores the logged
  metrics were computed from.
- :class:`EarlyStop`: stop once the monitor has not improved for ``patience`` validations.
- :class:`NonFiniteGuard`: stop, with an error naming the step, once the loss has not been finite
  for ``tolerance`` steps in a row (each such step's update is skipped by the module).
- :class:`Heartbeat`: ``heartbeat.json`` with the step, epoch and time, every few steps, so a
  stalled run can be told apart from a slow one.
- Lightning's ``LearningRateMonitor``, which logs every parameter group's learning rate.

All of them follow :attr:`DetectorModule.monitor <dfwb.train.module.DetectorModule.monitor>`, not
the config, so they switch together when the module falls back to ``val/loss``.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from dfwb.core.config.schema import ModelSection
from dfwb.core.errors import ContractError, DFWBError
from dfwb.core.records import ScoreMeta, ScoreRow, write_scores
from dfwb.core.runmeta import capture_env, utc_now
from dfwb.models import checkpoint
from dfwb.train.datamodule import ProtocolDataModule, SourceData
from dfwb.train.module import DetectorModule, SourceResult

__all__ = [
    "EarlyStop",
    "Heartbeat",
    "NonFiniteGuard",
    "NonFiniteLoss",
    "SafetensorsCheckpoint",
    "ValScoreDump",
    "build_callbacks",
]

_log = logging.getLogger(__name__)

_CHECKPOINTS_DIR = "checkpoints"
_CHECKPOINT_NAMES = ("best", "last")
_SCORES_DIR = Path("scores") / "val"
_HEARTBEAT_FILE = "heartbeat.json"
_UNKNOWN_SOURCE = "unknown"


class NonFiniteLoss(DFWBError):
    """Training stopped because the loss was NaN or infinite for too many steps in a row."""


def _module(pl_module: L.LightningModule) -> DetectorModule | None:
    return pl_module if isinstance(pl_module, DetectorModule) else None


def _monitored(pl_module: L.LightningModule) -> tuple[DetectorModule, float] | None:
    """The module and its monitor's value from the last validation, if both exist."""
    module = _module(pl_module)
    if module is None:
        return None
    value = module.val_metrics.get(module.monitor.key)
    return None if value is None else (module, value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------------------ checkpoints


class SafetensorsCheckpoint(Callback):
    """Keeps ``<directory>/best`` and ``<directory>/last``, each a
    :func:`dfwb.models.checkpoint.save` directory (``model.safetensors`` + ``detector.json``).

    ``best`` is rewritten whenever the module's monitor improves after a validation; ``last`` at
    the end of every training epoch. Each is written beside its target first (``.<name>.tmp``)
    and swapped in only when complete: the previous copy is moved aside to ``.<name>.old``, the
    new one renamed into place, and the old one removed. So an interrupted save never leaves a
    half-written checkpoint in place, and one interrupted between the two renames is undone --
    the previous copy moved back -- at setup and before the next save.

    Raises:
        ContractError: At setup, when Lightning's own (pickling) checkpointing is also enabled.
    """

    def __init__(self, directory: Path, model_cfg: ModelSection) -> None:
        self.directory = Path(directory)
        self.model_cfg = model_cfg
        self.monitor: str | None = None
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        #: The monitor's value after each validation, in order.
        self.history: list[float] = []

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        callbacks = getattr(trainer, "callbacks", [])
        if any(isinstance(callback, ModelCheckpoint) for callback in callbacks):
            raise ContractError(
                "Lightning's own checkpointing is enabled; it pickles the whole module into "
                ".ckpt files",
                hint="build the Trainer with enable_checkpointing=False: best and last are "
                "already saved as safetensors",
            )
        for name in _CHECKPOINT_NAMES:
            self._recover(name)
            shutil.rmtree(self.directory / f".{name}.tmp", ignore_errors=True)

    def _recover(self, name: str) -> None:
        """Undo a swap interrupted between its two renames: when ``<name>`` is missing but its
        previous copy is still at ``.<name>.old``, move it back. A leftover ``.<name>.old``
        beside a complete ``<name>`` (interrupted after the new copy was in place) is removed."""
        target = self.directory / name
        retired = self.directory / f".{name}.old"
        if not retired.exists():
            return
        if target.exists():
            shutil.rmtree(retired)
        else:
            retired.rename(target)
            _log.warning("%s: restored the previous checkpoint after an interrupted save", target)

    def _save(self, name: str, pl_module: L.LightningModule) -> None:
        module = _module(pl_module)
        if module is None:
            return
        self._recover(name)
        target = self.directory / name
        staging = self.directory / f".{name}.tmp"
        retired = self.directory / f".{name}.old"
        shutil.rmtree(staging, ignore_errors=True)
        checkpoint.save(staging, module.detector, self.model_cfg)
        shutil.rmtree(retired, ignore_errors=True)
        if target.exists():
            target.rename(retired)
        staging.rename(target)
        shutil.rmtree(retired, ignore_errors=True)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        monitored = _monitored(pl_module)
        if monitored is None:
            return
        module, value = monitored
        if module.monitor.key != self.monitor:
            # The monitor changed (the module fell back to val/loss): values are not comparable.
            self.monitor, self.best_value, self.best_epoch = module.monitor.key, None, None
            self.history = []
        self.history.append(value)
        if module.monitor.improved(value, self.best_value):
            self.best_value, self.best_epoch = value, trainer.current_epoch
            self._save("best", pl_module)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._save("last", pl_module)

    def state_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.monitor = state_dict["monitor"]
        self.best_value = state_dict["best_value"]
        self.best_epoch = state_dict["best_epoch"]


# ------------------------------------------------------------------------------ score dump


def _missing_rows(source: SourceData, labels: str) -> list[ScoreRow]:
    """A ``missing`` row for every video of the source's split that the store could not serve
    (never processed, failed, or frameless), so the score file covers the whole split. Videos
    the label mapping excludes are not part of the split under that mapping, so get no row."""
    unscored = {
        (key, compression)
        for _, key, compression, reason in source.index.excluded
        if reason != "label-excluded"
    }
    if not unscored:
        return []
    mapping = source.protocol.labels(labels)
    rows: list[ScoreRow] = []
    for record in source.protocol.records(split=source.entry.split, where=source.entry.where):
        if (record.key, record.compression) not in unscored:
            continue
        label = mapping(record.label_key)
        if not isinstance(label, int):
            continue
        rows.append(
            ScoreRow(
                dataset=source.protocol.dataset,
                key=record.key,
                compression=record.compression,
                label=label,
                score=None,
                status="missing",
                label_key=record.label_key,
                method=record.method,
            )
        )
    return rows


class ValScoreDump(Callback):
    """Writes ``<directory>/<source>.scores.{csv,meta.json}`` (C5) for every validation source
    after each validation epoch, replacing the previous epoch's files.

    Each file has one row per video of the source's split: an ``ok`` row with the aggregated
    video score the module's metrics were computed from, or a ``missing`` row for a video the
    processed store could not serve. Re-running ``dfwb eval`` on it therefore reproduces the
    logged ``val/<source>/<metric>`` values exactly.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._missing: dict[str, list[ScoreRow]] = {}

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        module = _module(pl_module)
        datamodule = ProtocolDataModule.attached_to(trainer)
        if trainer.sanity_checking or module is None or datamodule is None:
            return
        if not module.val_results:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        for source, result in zip(datamodule.val_sources, module.val_results, strict=True):
            rows = self._rows(source, result, datamodule.data.labels)
            meta = self._meta(source, rows, module, datamodule)
            write_scores(self.directory / f"{source.name}.scores.csv", rows, meta)

    def _rows(self, source: SourceData, result: SourceResult, labels: str) -> list[ScoreRow]:
        if source.name not in self._missing:
            self._missing[source.name] = _missing_rows(source, labels)
        items = {(i.dataset, i.key, i.compression): i for i in source.index.items}
        rows = [
            ScoreRow(
                dataset=video.dataset,
                key=video.key,
                compression=video.compression,
                label=video.label,
                score=video.score,
                status="ok",
                label_key=items[(video.dataset, video.key, video.compression)].label_key,
                method=items[(video.dataset, video.key, video.compression)].method,
                n_clips=video.n_clips,
            )
            for video in result.videos
        ]
        rows.extend(self._missing[source.name])
        return sorted(rows, key=lambda row: (row.dataset, row.key, row.compression or ""))

    @staticmethod
    def _meta(
        source: SourceData,
        rows: list[ScoreRow],
        module: DetectorModule,
        datamodule: ProtocolDataModule,
    ) -> ScoreMeta:
        detector_meta = module.detector.meta
        summary = source.index.summary()["sources"][0]
        statuses = [row.status for row in rows]
        return ScoreMeta.model_validate(
            {
                "detector": {
                    "name": detector_meta.name,
                    "version": detector_meta.version,
                    "source": detector_meta.source or _UNKNOWN_SOURCE,
                    "contract_version": detector_meta.contract_version,
                },
                "protocol": summary["protocol"],
                "labels": datamodule.data.labels,
                "processing_profile": {
                    "id": source.profile.profile_id(),
                    "sha256": source.profile.sha256(),
                },
                "input_adaptation": {
                    "derived_crop": source.adaptation.derived_crop,
                    "mismatch_override": source.adaptation.mismatch,
                },
                "aggregation": {
                    "clip_to_video": module.eval_cfg.aggregate,
                    "clips_per_video": datamodule.data.clip.clips_per_video.eval,
                },
                "coverage": {
                    "expected": len(rows),
                    "ok": statuses.count("ok"),
                    "missing": statuses.count("missing"),
                    "error": statuses.count("error"),
                },
                "seed": datamodule.seed,
                "env": capture_env(),
                "created": datetime.datetime.now(datetime.UTC),
            }
        )


# ------------------------------------------------------------------------------ early stop


class EarlyStop(Callback):
    """Stops training once the module's monitor has gone ``patience`` validations without
    improving by more than ``min_delta``."""

    def __init__(self, patience: int, *, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError(f"EarlyStop: patience must be at least 1, got {patience}")
        self.patience = patience
        self.min_delta = min_delta
        self.monitor: str | None = None
        self.best_value: float | None = None
        self.wait = 0

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        monitored = _monitored(pl_module)
        if monitored is None:
            return
        module, value = monitored
        if module.monitor.key != self.monitor:
            self.monitor, self.best_value, self.wait = module.monitor.key, None, 0
        if module.monitor.improved(value, self.best_value, min_delta=self.min_delta):
            self.best_value, self.wait = value, 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            _log.info(
                "early stop: %s has not improved for %d validation(s); best %s",
                self.monitor,
                self.wait,
                self.best_value,
            )
            trainer.should_stop = True

    def state_dict(self) -> dict[str, Any]:
        return {"monitor": self.monitor, "best_value": self.best_value, "wait": self.wait}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.monitor = state_dict["monitor"]
        self.best_value = state_dict["best_value"]
        self.wait = state_dict["wait"]


# ------------------------------------------------------------------------------- NaN guard


class NonFiniteGuard(Callback):
    """Raises :class:`NonFiniteLoss` once ``tolerance`` training steps in a row had a loss that
    was NaN or infinite. The module skips each such step's update, so a short run of them (a
    rare bad batch) passes without harm; a finite loss resets the count."""

    def __init__(self, tolerance: int = 3) -> None:
        if tolerance < 1:
            raise ValueError(f"NonFiniteGuard: tolerance must be at least 1, got {tolerance}")
        self.tolerance = tolerance
        self._count = 0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        module = _module(pl_module)
        if module is None or module.last_loss is None:
            return
        if bool(torch.isfinite(module.last_loss).all()):
            self._count = 0
            return
        self._count += 1
        if self._count >= self.tolerance:
            raise NonFiniteLoss(
                f"the loss was not finite for {self._count} consecutive steps (the last at "
                f"batch {batch_idx} of epoch {trainer.current_epoch}, global step "
                f"{trainer.global_step}; its value: {module.last_loss.item()})",
                hint="lower the learning rate, check the inputs for NaN/Inf, or train in "
                "32-true precision",
            )


# ------------------------------------------------------------------------------- heartbeat


class Heartbeat(Callback):
    """Rewrites ``path`` (JSON: ``step``, ``epoch``, ``time`` in UTC) every ``every_n_steps``
    training batches."""

    def __init__(self, path: Path, *, every_n_steps: int = 50) -> None:
        if every_n_steps < 1:
            raise ValueError(f"Heartbeat: every_n_steps must be at least 1, got {every_n_steps}")
        self.path = Path(path)
        self.every_n_steps = every_n_steps
        self._batches = 0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._batches += 1
        if self._batches % self.every_n_steps == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                self.path,
                {"step": trainer.global_step, "epoch": trainer.current_epoch, "time": utc_now()},
            )


# ---------------------------------------------------------------------------------- builder


def build_callbacks(
    run_dir: Path,
    model_cfg: ModelSection,
    *,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    nan_tolerance: int = 3,
    heartbeat_every: int = 50,
) -> list[Callback]:
    """Every callback of a run whose directory is ``run_dir``: best/last checkpoints under
    ``checkpoints/``, the validation score dump under ``scores/val/``, the non-finite loss guard,
    the heartbeat file, the LR monitor, and -- only when ``early_stop_patience`` is given --
    early stopping on the monitor."""
    run_dir = Path(run_dir)
    callbacks: list[Callback] = [
        SafetensorsCheckpoint(run_dir / _CHECKPOINTS_DIR, model_cfg),
        ValScoreDump(run_dir / _SCORES_DIR),
        NonFiniteGuard(tolerance=nan_tolerance),
        Heartbeat(run_dir / _HEARTBEAT_FILE, every_n_steps=heartbeat_every),
        LearningRateMonitor(logging_interval="step"),
    ]
    if early_stop_patience is not None:
        callbacks.append(EarlyStop(early_stop_patience, min_delta=early_stop_min_delta))
    return callbacks
