"""Training loggers: CSV always, TensorBoard when it is installed, W&B only when asked for.

Everything goes under the run directory's ``logs/``: ``logs/metrics.csv`` (one column per logged
value, one row per logging step), ``logs/tensorboard/`` for TensorBoard's event files, and W&B's
local files. The CSV log is always written, so a run's numbers never depend on an optional extra
being installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger

from dfwb.core.errors import InstallationError

__all__ = ["build_loggers"]

_LOGS_DIR = "logs"


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _wandb_logger_class() -> type[Logger]:
    # Imported only once W&B is actually wanted: the module pulls in wandb itself.
    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger  # type: ignore[no-any-return, unused-ignore]  # Any w/o torch


def build_loggers(
    run_dir: Path, *, tensorboard: bool = True, wandb: Mapping[str, Any] | None = None
) -> list[Logger]:
    """The loggers of one run, all writing under ``run_dir / "logs"``.

    - CSV, always: ``logs/metrics.csv``.
    - TensorBoard, when ``tensorboard`` is asked for and installed: ``logs/tensorboard/``.
    - W&B, only when ``wandb`` is given (its entries are passed to Lightning's ``WandbLogger``,
      e.g. ``{"project": ..., "entity": ...}``).

    Raises:
        InstallationError: ``wandb`` is given but the ``wandb`` package is not installed.
    """
    logs = Path(run_dir) / _LOGS_DIR
    loggers: list[Logger] = [CSVLogger(save_dir=logs, name="", version="")]
    if tensorboard and _installed("tensorboard"):
        loggers.append(TensorBoardLogger(save_dir=logs, name="tensorboard", version=""))
    if wandb is not None:
        if not _installed("wandb"):
            raise InstallationError(
                "W&B logging is configured, but the wandb package is not installed",
                hint='pip install "deepfake-workbench[wandb]"',
            )
        loggers.append(_wandb_logger_class()(**{**wandb, "save_dir": str(logs)}))
    return loggers
