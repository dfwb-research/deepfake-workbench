"""Logging setup for the ``dfwb`` logger tree. The root logger is never touched."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys

__all__ = ["LOGGER_NAME", "get_logger", "setup_logging"]

LOGGER_NAME = "dfwb"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_MARK = "_dfwb_handler"


def get_logger(name: str) -> logging.Logger:
    """A logger inside the ``dfwb`` tree (``get_logger("train")`` -> ``dfwb.train``)."""
    return logging.getLogger(
        name
        if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}.")
        else f"{LOGGER_NAME}.{name}"
    )


def _rich_available() -> bool:
    try:
        return importlib.util.find_spec("rich") is not None
    except (ImportError, ValueError):
        return False


def setup_logging(level: int | str = "INFO", *, use_rich: bool | None = None) -> logging.Logger:
    """Send ``dfwb`` log records to stderr; safe to call more than once.

    Args:
        level: Logging level for the ``dfwb`` logger.
        use_rich: Use ``rich`` for pretty output. ``None`` means: if installed and stderr is a TTY.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for existing in list(logger.handlers):
        if getattr(existing, _MARK, False):
            logger.removeHandler(existing)
    if use_rich is None:
        use_rich = _rich_available() and sys.stderr.isatty()
    handler: logging.Handler
    if use_rich:
        rich_logging = importlib.import_module("rich.logging")
        handler = rich_logging.RichHandler(show_path=False)
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
    setattr(handler, _MARK, True)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
