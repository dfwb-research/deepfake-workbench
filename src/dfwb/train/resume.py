"""Resuming an interrupted run, from plain tensors and JSON -- never a pickle.

At the end of every training epoch a run writes ``resume/`` in its run directory, whole or not at
all (it is built beside its final place and swapped in):

- ``model.safetensors``: the detector's weights and buffers;
- ``optimizer.safetensors``: every tensor of the optimiser's state (its moments and step counts);
- ``state.json``: everything else -- the epochs and optimiser steps done, and whether early
  stopping had already ended the run; the optimiser's
  parameter groups and non-tensor state; the learning-rate schedule's state; every callback's
  state (the best monitored value, the early-stop wait, the non-finite loss count, ...); the
  module's monitor and last validation values; the data's epoch; the precision plugin's state;
  and the Python, NumPy and torch random states.

:class:`ResumeCheckpoint` writes it, and, given a saved state, puts all of it back when fitting
starts: the trainer then starts at the next epoch, with the optimiser, schedule, callbacks and
random states exactly where the interrupted run left them, so it finishes where an
uninterrupted run would have.

Lightning's own resume reads a pickled ``.ckpt``, so it is not used: the trainer's epoch and step
counters are set directly instead, as if the finished epochs had just run in this process.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import numbers
import random
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from safetensors.torch import load_file, save_file
from torch import Tensor

from dfwb.core.errors import ContractError
from dfwb.train.datamodule import ProtocolDataModule
from dfwb.train.module import DetectorModule, Monitor
from dfwb.train.rundir import RESUME_STATE_FILE

__all__ = [
    "ResumeCheckpoint",
    "SavedState",
    "capture_rng",
    "decode",
    "encode",
    "read_state",
    "restore_rng",
    "write_state",
]

_log = logging.getLogger(__name__)

STATE_FORMAT = 1
_MODEL_FILE = "model.safetensors"
_TENSORS_FILE = "optimizer.safetensors"

# Markers of the lossless JSON encoding: a tensor (by its key in the tensor file), a tuple, a
# mapping with non-string keys, and a float JSON cannot hold (NaN, +-infinity).
_TENSOR = "__tensor__"
_TUPLE = "__tuple__"
_ITEMS = "__items__"
_FLOAT = "__float__"
_MARKERS = frozenset({_TENSOR, _TUPLE, _ITEMS, _FLOAT})


# ---------------------------------------------------------------------------------- encoding


def encode(value: Any, tensors: dict[str, Tensor], path: str) -> Any:
    """``value`` as strict JSON, with every tensor moved into ``tensors`` (keyed by where it was,
    e.g. ``optimizer.state.0.exp_avg``) and replaced by a reference to it.

    Tuples, mappings with non-string keys (an optimiser's per-parameter state is keyed by
    integers) and non-finite floats are marked so :func:`decode` gives back exactly ``value``.

    Raises:
        ContractError: ``value`` holds something other than tensors, mappings, lists, tuples,
            strings, numbers, booleans and ``None``.
    """
    if isinstance(value, Tensor):
        tensors[path] = value.detach().cpu().clone().contiguous()
        return {_TENSOR: path}
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else {_FLOAT: repr(number)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value) and not _MARKERS & set(value):
            return {key: encode(item, tensors, f"{path}.{key}") for key, item in value.items()}
        return {
            _ITEMS: [
                [encode(key, tensors, f"{path}.key"), encode(item, tensors, f"{path}.{key}")]
                for key, item in value.items()
            ]
        }
    if isinstance(value, tuple):
        return {_TUPLE: [encode(item, tensors, f"{path}.{i}") for i, item in enumerate(value)]}
    if isinstance(value, list):
        return [encode(item, tensors, f"{path}.{i}") for i, item in enumerate(value)]
    raise ContractError(
        f"resume state: cannot store a {type(value).__name__} (at {path})",
        hint="a callback's state_dict() must hold only tensors, numbers, strings and containers",
    )


def decode(value: Any, tensors: Mapping[str, Tensor]) -> Any:
    """The inverse of :func:`encode`."""
    if isinstance(value, list):
        return [decode(item, tensors) for item in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        (marker, content) = next(iter(value.items()))
        if marker == _TENSOR:
            return tensors[content]
        if marker == _TUPLE:
            return tuple(decode(item, tensors) for item in content)
        if marker == _ITEMS:
            return {decode(key, tensors): decode(item, tensors) for key, item in content}
        if marker == _FLOAT:
            return float(content)
    return {key: decode(item, tensors) for key, item in value.items()}


# ------------------------------------------------------------------------------- random state


def _bytes(tensor: Tensor) -> str:
    return base64.b64encode(tensor.numpy().tobytes()).decode("ascii")


def _tensor(text: str) -> Tensor:
    return torch.frombuffer(bytearray(base64.b64decode(text)), dtype=torch.uint8)


def capture_rng() -> dict[str, Any]:
    """The Python, NumPy (global, legacy) and torch random states, as JSON."""
    name, keys, pos, has_gauss, cached = np.random.get_state(legacy=True)  # noqa: NPY002
    state: dict[str, Any] = {
        "python": encode(random.getstate(), {}, "python"),
        "numpy": {
            "bit_generator": str(name),
            "keys": [int(key) for key in keys],
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        "torch": _bytes(torch.get_rng_state()),
    }
    if torch.cuda.is_available():
        state["cuda"] = [_bytes(s) for s in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state: Mapping[str, Any]) -> None:
    """Put back the random states :func:`capture_rng` captured."""
    random.setstate(decode(state["python"], {}))
    numpy_state = state["numpy"]
    np.random.set_state(  # noqa: NPY002
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["pos"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(_tensor(state["torch"]))
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_tensor(s) for s in state["cuda"]])


# ---------------------------------------------------------------------------------- on disk


@dataclass(frozen=True)
class SavedState:
    """A ``resume/`` directory, read back."""

    payload: dict[str, Any]
    model: dict[str, Tensor]
    tensors: dict[str, Tensor]


def _recover(directory: Path) -> None:
    """Undo a swap interrupted between its two renames (the previous state moved aside to
    ``.<name>.old``, nothing in its place yet), and drop a leftover ``.old`` or ``.tmp``."""
    retired = directory.with_name(f".{directory.name}.old")
    if retired.exists():
        if directory.exists():
            shutil.rmtree(retired)
        else:
            retired.rename(directory)
            _log.warning(
                "%s: restored the previous resume state after an interrupted save", directory
            )
    shutil.rmtree(directory.with_name(f".{directory.name}.tmp"), ignore_errors=True)


def write_state(
    directory: Path,
    model: Mapping[str, Tensor],
    tensors: Mapping[str, Tensor],
    payload: Mapping[str, Any],
) -> None:
    """Write ``model.safetensors``, ``optimizer.safetensors`` and ``state.json`` to
    ``directory``, replacing what is there only once all three are complete."""
    _recover(directory)
    staging = directory.with_name(f".{directory.name}.tmp")
    retired = directory.with_name(f".{directory.name}.old")
    staging.mkdir(parents=True)
    save_file(dict(model), str(staging / _MODEL_FILE))
    save_file(dict(tensors), str(staging / _TENSORS_FILE))
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False)
    (staging / RESUME_STATE_FILE).write_text(text + "\n", encoding="utf-8")
    if directory.exists():
        directory.rename(retired)
    staging.rename(directory)
    shutil.rmtree(retired, ignore_errors=True)


def read_state(directory: Path) -> SavedState | None:
    """The state saved in ``directory``, or ``None`` when there is none.

    Raises:
        ContractError: The state was written in a format this version does not read.
    """
    _recover(directory)
    if not (directory / RESUME_STATE_FILE).is_file():
        return None
    payload = json.loads((directory / RESUME_STATE_FILE).read_text("utf-8"))
    if payload.get("format") != STATE_FORMAT:
        raise ContractError(
            f"{directory}: resume state format {payload.get('format')!r} is not "
            f"{STATE_FORMAT}, the one this version reads",
            hint="resume it with the dfwb version that started the run",
        )
    return SavedState(
        payload=payload,
        model=load_file(str(directory / _MODEL_FILE)),
        tensors=load_file(str(directory / _TENSORS_FILE)),
    )


# ---------------------------------------------------------------------------------- callback


def _callbacks(trainer: L.Trainer) -> list[Callback]:
    return cast(list[Callback], getattr(trainer, "callbacks", []))


def _set_epochs_done(trainer: L.Trainer, epochs: int) -> None:
    """Count ``epochs`` epochs as fully run (ready, started, processed and completed)."""
    progress = trainer.fit_loop.epoch_progress
    for tracker in (progress.total, progress.current):
        for stage in ("ready", "started", "processed", "completed"):
            setattr(tracker, stage, epochs)


class ResumeCheckpoint(Callback):  # type: ignore[misc, unused-ignore]  # Any w/o torch
    """Writes ``directory`` (see the module docstring) at the end of every training epoch; given
    ``restore``, puts that state back when fitting starts.

    It must be the last callback, so the states it saves are the ones every other callback holds
    once the epoch is over.
    """

    def __init__(self, directory: Path, *, seed: int, restore: SavedState | None = None) -> None:
        self.directory = Path(directory)
        self.seed = seed
        self._restore = restore
        self._rng: Mapping[str, Any] | None = None

    # ------------------------------------------------------------------------------- saving

    def _payload(
        self, trainer: L.Trainer, module: DetectorModule, tensors: dict[str, Tensor]
    ) -> dict[str, Any]:
        epoch_loop = trainer.fit_loop.epoch_loop
        (optimizer,) = trainer.optimizers
        schedulers = [config.scheduler for config in trainer.lr_scheduler_configs]
        callbacks = {
            callback.state_key: encode(state, tensors, f"callbacks.{callback.state_key}")
            for callback in _callbacks(trainer)
            if callback is not self and (state := callback.state_dict())
        }
        datamodule = ProtocolDataModule.attached_to(trainer)
        return {
            "format": STATE_FORMAT,
            "seed": self.seed,
            # the trainer counts an epoch as complete only after this hook returns
            "epoch": trainer.current_epoch + 1,
            "global_step": trainer.global_step,
            # set by early stopping, before this hook: the run ends after this epoch
            "should_stop": bool(trainer.should_stop),
            "loop": {
                "optimizer_progress": epoch_loop.automatic_optimization.optim_progress.state_dict(),
                "batch_progress": epoch_loop.batch_progress.state_dict(),
                "batches_that_stepped": epoch_loop._batches_that_stepped,
            },
            "optimizer": encode(optimizer.state_dict(), tensors, "optimizer"),
            "scheduler": encode(
                schedulers[0].state_dict() if schedulers else None, tensors, "scheduler"
            ),
            "precision": encode(trainer.precision_plugin.state_dict(), tensors, "precision"),
            "callbacks": callbacks,
            "module": {
                "monitor": {
                    "key": module.monitor.key,
                    "mode": module.monitor.mode,
                    "fallback": module.monitor.fallback,
                },
                "val_metrics": encode(module.val_metrics, tensors, "val_metrics"),
            },
            "data": {"epoch": datamodule.epoch if datamodule is not None else None},
            "rng": capture_rng(),
        }

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not isinstance(pl_module, DetectorModule):
            return
        tensors: dict[str, Tensor] = {}
        payload = self._payload(trainer, pl_module, tensors)
        model = {
            name: tensor.detach().cpu().clone().contiguous()
            for name, tensor in pl_module.detector.state_dict().items()
        }
        write_state(self.directory, model, tensors, payload)

    # ---------------------------------------------------------------------------- restoring

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        saved = self._restore
        if saved is None or not isinstance(pl_module, DetectorModule):
            return
        self._restore = None
        payload, tensors = saved.payload, saved.tensors
        pl_module.detector.load_state_dict(saved.model, strict=True)
        (optimizer,) = trainer.optimizers
        optimizer.load_state_dict(decode(payload["optimizer"], tensors))
        scheduler_state = decode(payload["scheduler"], tensors)
        if scheduler_state is not None:
            trainer.lr_scheduler_configs[0].scheduler.load_state_dict(scheduler_state)
        trainer.precision_plugin.load_state_dict(decode(payload["precision"], tensors))
        states = payload["callbacks"]
        for callback in _callbacks(trainer):
            if callback.state_key in states:
                callback.load_state_dict(decode(states[callback.state_key], tensors))
        pl_module.monitor = Monitor(**payload["module"]["monitor"])
        pl_module.val_metrics = decode(payload["module"]["val_metrics"], tensors)

        _set_epochs_done(trainer, payload["epoch"])
        trainer.should_stop = bool(payload["should_stop"])
        epoch_loop = trainer.fit_loop.epoch_loop
        loop = payload["loop"]
        epoch_loop.automatic_optimization.optim_progress.load_state_dict(loop["optimizer_progress"])
        epoch_loop.batch_progress.load_state_dict(loop["batch_progress"])
        epoch_loop._batches_that_stepped = loop["batches_that_stepped"]
        datamodule = ProtocolDataModule.attached_to(trainer)
        if datamodule is not None and payload["data"]["epoch"] is not None:
            datamodule.set_epoch(payload["data"]["epoch"])
        # put back at the start of the next epoch, where the uninterrupted run captured them
        self._rng = payload["rng"]

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._rng is not None:
            restore_rng(self._rng)
            self._rng = None
