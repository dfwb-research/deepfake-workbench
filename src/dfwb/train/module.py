"""``DetectorModule``: trains an ``AssembledDetector`` and validates it the way final evaluation
scores it.

Training is forward -> loss (from the ``losses`` registry) -> optimiser step, with the optimiser
and its per-step schedule built from the config. A step whose loss is not finite is skipped (no
update); :class:`~dfwb.train.callbacks.NonFiniteGuard` stops the run once too many are skipped in
a row.

Validation deliberately reuses the evaluation code rather than a copy of it: every clip score of
a validation source is collected, reduced to one score per video with
:func:`dfwb.eval.aggregate.aggregate` (the config's ``eval.aggregate`` mode), and every
``eval.metrics`` entry is computed with :func:`dfwb.eval.metrics.compute` on those video scores.
So a validation number is exactly what ``dfwb eval`` reports on the same videos' score file.

What gets logged at the end of each validation epoch:

- ``val/<source>/<metric>`` for each source and metric, when the metric is defined there. A
  metric that is not (an AUC over a single-class source, for instance) is left out with a
  warning -- never replaced by a placeholder number.
- ``val/video_<metric>``: the mean of ``val/<source>/<metric>`` over the sources where it is
  defined.
- ``val/loss``: the mean loss over every validation clip.

The checkpoint monitor is ``train.monitor``/``train.mode``, checked against the validation
sources' names when fitting starts, before any training. When it names a metric that no source
defines, the module switches it, once and for the rest of the run, to ``val/loss`` (``min``), with
a warning; the callbacks read :attr:`DetectorModule.monitor` rather than the config, so they
follow the switch.

**One process only.** Validation keeps every clip score in this process, and the score dump and
the checkpoints are written from it, so a run spread over several processes (DDP) would compute
each rank's metrics on its own share of the videos and write conflicting files. Fitting with more
than one process is therefore refused at setup, rather than producing numbers that look right.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import lightning.pytorch as L
import numpy as np
import torch
from torch import Tensor

from dfwb.core.config.schema import ComponentSpec, EvalSection, TrainSection
from dfwb.core.detector import ClipBatch, DetectorOutput
from dfwb.core.errors import ConfigError, ContractError, did_you_mean
from dfwb.core.plugins import api
from dfwb.eval.aggregate import aggregate
from dfwb.eval.metrics import MetricUndefined, compute
from dfwb.models.detector import AssembledDetector
from dfwb.train.datamodule import ProtocolDataModule
from dfwb.train.losses import LossOutput
from dfwb.train.optim import build_optimizer
from dfwb.train.schedules import build_schedule

__all__ = [
    "DetectorModule",
    "Monitor",
    "SourceResult",
    "VideoScore",
    "check_monitor",
    "check_monitor_syntax",
]

_log = logging.getLogger(__name__)

_VAL_PREFIX = "val/"
_MEAN_PREFIX = "val/video_"
_LOSS_KEY = "val/loss"

VideoKey = tuple[str, str, str | None]  # (dataset, key, compression): one score-file row


def _video_order(key: VideoKey) -> tuple[str, str, str]:
    return key[0], key[1], key[2] or ""


@dataclass(frozen=True)
class Monitor:
    """The logged value checkpoint selection and early stopping follow, and which way is better.

    ``fallback`` is ``True`` once the configured metric turned out to be undefined on every
    validation source and ``val/loss`` stands in for it.
    """

    key: str
    mode: Literal["max", "min"]
    fallback: bool = False

    def improved(self, value: float, best: float | None, *, min_delta: float = 0.0) -> bool:
        """Whether ``value`` beats ``best`` by more than ``min_delta`` (any finite value beats
        ``None``; a non-finite value never improves on anything)."""
        if not math.isfinite(value):
            return False
        if best is None:
            return True
        if self.mode == "max":
            return value > best + min_delta
        return value < best - min_delta


@dataclass(frozen=True)
class VideoScore:
    """One validation video: its aggregated score and how many clip scores went into it."""

    dataset: str
    key: str
    compression: str | None
    label: int
    score: float
    n_clips: int


@dataclass(frozen=True)
class SourceResult:
    """One validation source after an epoch: its videos (in score-file order) and metrics."""

    name: str
    videos: tuple[VideoScore, ...]
    metrics: dict[str, float]


def check_monitor_syntax(key: str, metrics: Sequence[str]) -> None:
    """Check a monitor key that needs no source names: it must be a validation value, and a
    ``val/video_<metric>`` must name a metric in ``eval.metrics``.

    Raises:
        ConfigError: It is neither.
    """
    if not key.startswith(_VAL_PREFIX):
        raise ConfigError(
            f"train.monitor: {key!r} is not a validation value",
            hint=f"monitor {_LOSS_KEY} or {_MEAN_PREFIX}<metric>, e.g. {_MEAN_PREFIX}auc",
        )
    if key.startswith(_MEAN_PREFIX) and key.removeprefix(_MEAN_PREFIX) not in metrics:
        known = [f"{_MEAN_PREFIX}{m}" for m in metrics]
        raise ConfigError(
            f"train.monitor: {key!r} is not computed{did_you_mean(key, known)}",
            hint="add its metric to eval.metrics, or monitor one of: "
            + ", ".join([*known, _LOSS_KEY]),
        )


def check_monitor(key: str, metrics: Sequence[str], source_names: Sequence[str]) -> None:
    """Check ``train.monitor`` against every value validation over sources named
    ``source_names`` logs: ``val/loss``, ``val/video_<metric>`` and ``val/<source>/<metric>``.

    Raises:
        ConfigError: ``key`` names no value validation will log.
    """
    check_monitor_syntax(key, metrics)
    known = {f"{_MEAN_PREFIX}{m}" for m in metrics} | {_LOSS_KEY}
    known |= {f"{_VAL_PREFIX}{name}/{m}" for name in source_names for m in metrics}
    if key not in known:
        raise ConfigError(
            f"train.monitor: {key!r} is not a value validation logs{did_you_mean(key, known)}",
            hint="validation logs: " + ", ".join(sorted(known)),
        )


def _build_loss(spec: ComponentSpec) -> torch.nn.Module:
    loss: torch.nn.Module = api.losses.build(spec.name, **spec.params)
    return loss


class DetectorModule(L.LightningModule):  # type: ignore[misc, unused-ignore]  # Any w/o torch
    """The Lightning module of one training run.

    Args:
        detector: The detector to train.
        loss: ``loss:`` (a ``losses`` registry component).
        optim: ``optim:`` (see :func:`~dfwb.train.optim.build_optimizer`).
        schedule: ``schedule:`` (see :func:`~dfwb.train.schedules.build_schedule`); stepped once
            per optimiser step, over the epochs the trainer runs.
        train: ``train:``; ``monitor`` and ``mode`` choose the checkpoint monitor (the trainer
            itself is built from the rest of it).
        eval: ``eval:``; ``metrics`` and ``aggregate`` are exactly what validation computes.

    Raises:
        ConfigError: ``train.monitor`` is not a validation value (``val/...``), or names a
            ``val/video_<metric>`` whose metric is not in ``eval.metrics``.
    """

    def __init__(
        self,
        detector: AssembledDetector,
        *,
        loss: ComponentSpec,
        optim: ComponentSpec,
        schedule: ComponentSpec,
        train: TrainSection,
        eval: EvalSection,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.loss = _build_loss(loss)
        self.optim_spec = optim
        self.schedule_spec = schedule
        self.eval_cfg = eval
        self.monitor = Monitor(train.monitor, train.mode)
        check_monitor_syntax(self.monitor.key, eval.metrics)
        #: The last training step's loss (detached), finite or not.
        self.last_loss: Tensor | None = None
        #: Every value logged by the last validation epoch, at full precision.
        self.val_metrics: dict[str, float] = {}
        #: Per-source videos and metrics of the last validation epoch, in source order.
        self.val_results: list[SourceResult] = []
        self._scores: dict[int, dict[VideoKey, list[float]]] = {}
        self._labels: dict[int, dict[VideoKey, int]] = {}
        self._loss_sum = 0.0
        self._loss_count = 0
        self._warned: set[str] = set()

    def setup(self, stage: str) -> None:
        """Refuse more than one process, and check the monitor against the validation sources'
        names -- both before any training happens.

        Raises:
            ContractError: The trainer runs more than one process, or has no
                ``ProtocolDataModule`` to name the validation sources.
            ConfigError: ``train.monitor`` names no value validation will log.
        """
        if self.trainer.world_size > 1:
            raise ContractError(
                f"training runs in a single process only, but this trainer has "
                f"{self.trainer.world_size}: validation scores, the score dump and the "
                "checkpoints are all produced by one process",
                hint="train with devices: 1 (one GPU, or the CPU)",
            )
        self.check_monitor(self._source_names())

    def check_monitor(self, source_names: Sequence[str]) -> None:
        """:func:`check_monitor` for this module's monitor and metrics."""
        check_monitor(self.monitor.key, self.eval_cfg.metrics, source_names)

    def _warn_once(self, message: str, *args: object) -> None:
        text = message % args
        if text not in self._warned:
            self._warned.add(text)
            _log.warning("%s", text)

    # -------------------------------------------------------------------------------- training

    def forward(self, batch: ClipBatch) -> DetectorOutput:
        out: DetectorOutput = self.detector(batch)
        return out

    def training_step(self, batch: ClipBatch, batch_idx: int) -> Tensor | None:
        out: DetectorOutput = self(batch)
        result: LossOutput = self.loss(out, batch)
        total = result.total
        self.last_loss = total.detach()
        if not bool(torch.isfinite(self.last_loss).all()):
            return None  # skip this update; NonFiniteGuard decides when to stop
        batch_size = len(batch.keys)
        self.log("train/loss", total, on_step=True, on_epoch=True, batch_size=batch_size)
        for name, value in result.parts.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, batch_size=batch_size)
        return total

    def on_train_epoch_start(self) -> None:
        datamodule = ProtocolDataModule.attached_to(self.trainer)
        if datamodule is not None:
            datamodule.set_epoch(self.current_epoch)

    def configure_optimizers(self) -> Any:
        """The optimiser, and its schedule stepped once per optimiser step.

        The schedule's length comes from the trainer alone -- the epochs it will actually run and
        the optimiser steps in each -- so it always ends exactly when training does.

        Raises:
            ConfigError: The trainer has no finite length (no epoch limit), so a schedule over it
                cannot be laid out.
        """
        optimizer = build_optimizer(self.optim_spec, self.detector)
        total_steps = self.trainer.estimated_stepping_batches
        epochs = self.trainer.max_epochs
        if not math.isfinite(total_steps) or epochs is None or epochs < 1:
            raise ConfigError(
                "the learning-rate schedule needs a finite run, but the trainer has no epoch "
                f"limit (max_epochs={epochs})",
                hint="set train.max_epochs to a positive number of epochs",
            )
        steps_per_epoch = max(1, math.ceil(total_steps / epochs))
        scheduler = build_schedule(
            self.schedule_spec, optimizer, steps_per_epoch=steps_per_epoch, epochs=epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ------------------------------------------------------------------------------ validation

    def on_validation_epoch_start(self) -> None:
        self._scores = {}
        self._labels = {}
        self._loss_sum = 0.0
        self._loss_count = 0

    def validation_step(self, batch: ClipBatch, batch_idx: int, dataloader_idx: int = 0) -> None:
        out: DetectorOutput = self(batch)
        result: LossOutput = self.loss(out, batch)
        size = len(batch.keys)
        self._loss_sum += float(result.total) * size
        self._loss_count += size
        if self.detector.head.num_classes != 1:
            return  # video scores (and so the metrics) are defined for binary heads only
        if batch.labels is None:
            raise ContractError(
                "validation batches need labels", hint="validation data always carries labels"
            )
        scores = out.score.detach().float().cpu().tolist()
        labels = batch.labels.tolist()
        per_video = self._scores.setdefault(dataloader_idx, {})
        per_label = self._labels.setdefault(dataloader_idx, {})
        for i, score in enumerate(scores):
            key: VideoKey = (batch.dataset_ids[i], batch.keys[i], batch.compressions[i])
            per_video.setdefault(key, []).append(float(score))
            per_label[key] = int(labels[i])

    def _source_names(self) -> list[str]:
        datamodule = ProtocolDataModule.attached_to(self.trainer)
        if datamodule is None:
            raise ContractError(
                "DetectorModule validates through a ProtocolDataModule, which names each "
                "validation source",
                hint="pass datamodule=ProtocolDataModule(...) to Trainer.fit",
            )
        return [source.name for source in datamodule.val_sources]

    def _source_result(self, index: int, name: str) -> SourceResult:
        clips = self._scores.get(index, {})
        labels = self._labels.get(index, {})
        rows: Iterable[tuple[Hashable, float]] = (
            (key, score) for key, values in clips.items() for score in values
        )
        video_scores = aggregate(rows, self.eval_cfg.aggregate)
        keys: list[VideoKey] = sorted(clips, key=_video_order)
        videos = tuple(
            VideoScore(*key, label=labels[key], score=video_scores[key], n_clips=len(clips[key]))
            for key in keys
        )
        metrics: dict[str, float] = {}
        if not videos:
            self._warn_once("val/%s: no video was scored; it has no metrics", name)
            return SourceResult(name, videos, metrics)
        y = np.asarray([video.label for video in videos], dtype=np.int64)
        p = np.asarray([video.score for video in videos], dtype=np.float64)
        for metric in self.eval_cfg.metrics:
            try:
                metrics[metric] = compute(metric, y, p)
            except MetricUndefined as exc:
                self._warn_once(
                    "val/%s: %s is undefined here (%s); it is left out of val/video_%s",
                    name,
                    metric,
                    exc.message,
                    metric,
                )
        return SourceResult(name, videos, metrics)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        names = self._source_names()
        results: list[SourceResult] = []
        if self.detector.head.num_classes == 1:
            results = [self._source_result(index, name) for index, name in enumerate(names)]
        else:
            self._warn_once("video scores need a binary head; only val/loss is computed")

        logged: dict[str, float] = {}
        for result in results:
            for metric, value in result.metrics.items():
                logged[f"{_VAL_PREFIX}{result.name}/{metric}"] = value
        for metric in self.eval_cfg.metrics:
            defined = [r.metrics[metric] for r in results if metric in r.metrics]
            if defined:
                logged[f"{_MEAN_PREFIX}{metric}"] = sum(defined) / len(defined)
        logged[_LOSS_KEY] = self._loss_sum / self._loss_count if self._loss_count else math.nan

        self._resolve_monitor(logged)
        for key, value in logged.items():
            self.log(key, value, prog_bar=key == self.monitor.key)
        self.val_results = results
        self.val_metrics = logged

    def _resolve_monitor(self, logged: dict[str, float]) -> None:
        """Fall back to ``val/loss`` when the monitored metric (already checked to be one that
        validation logs, at setup) was not defined on any source this epoch."""
        key = self.monitor.key
        if key in logged or self.monitor.fallback:
            return
        _log.warning(
            "%s is not defined on any validation source; checkpoints and early stopping "
            "monitor %s (min) instead for the rest of this run",
            key,
            _LOSS_KEY,
        )
        self.monitor = Monitor(_LOSS_KEY, "min", fallback=True)
