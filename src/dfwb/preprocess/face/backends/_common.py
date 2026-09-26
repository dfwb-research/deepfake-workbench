"""What the shipped face backends share: argument checks, model files, and the licence gate
(``require_accepted``, re-exported here from :mod:`dfwb.core.licenses`, which owns the one
implementation shared with the zoo).

numpy is imported inside the frame checks rather than at module scope, as in the rest of the face
pipeline.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from dfwb.core.errors import ConfigError
from dfwb.core.hashing import sha256_file
from dfwb.core.licenses import require_accepted
from dfwb.core.paths import require_root, resolve_roots

__all__ = [
    "cache_models_dir",
    "check_frame",
    "check_frames",
    "find_verified",
    "parse_device",
    "require_accepted",
]

_log = logging.getLogger(__name__)

_CUDA_DEVICE = re.compile(r"cuda:([0-9]+)")


def parse_device(device: str) -> int | None:
    """The CUDA device index that ``device`` names (``"cuda:<index>"``), or ``None`` for
    ``"cpu"``.

    Raises:
        ConfigError: ``device`` is neither.
    """
    if device == "cpu":
        return None
    match = _CUDA_DEVICE.fullmatch(device) if isinstance(device, str) else None
    if match is None:
        raise ConfigError(
            f"unknown device {device!r}",
            hint="use --device cpu or --device cuda:<index>, e.g. cuda:0",
        )
    return int(match.group(1))


def _describe(value: object) -> str:
    import numpy as np

    if isinstance(value, np.ndarray):
        return f"a {value.dtype} array of shape {value.shape}"
    return f"a {type(value).__name__}"


def check_frames(frames: object) -> None:
    """Raise ``ValueError`` unless ``frames`` is an ``[N, H, W, 3]`` ``uint8`` array."""
    import numpy as np

    if not (
        isinstance(frames, np.ndarray)
        and frames.ndim == 4
        and frames.shape[-1] == 3
        and frames.dtype == np.uint8
    ):
        raise ValueError(
            f"frames must be a uint8 RGB array of shape [N, H, W, 3], got {_describe(frames)}"
        )


def check_frame(frame: object) -> None:
    """Raise ``ValueError`` unless ``frame`` is one ``[H, W, 3]`` ``uint8`` image."""
    import numpy as np

    if not (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[-1] == 3
        and frame.dtype == np.uint8
    ):
        raise ValueError(
            f"a frame must be a uint8 RGB array of shape [H, W, 3], got {_describe(frame)}"
        )


def cache_models_dir() -> Path:
    """``<cache root>/models``, where downloaded model files are kept."""
    return require_root("cache", resolve_roots()) / "models"


def find_verified(name: str, sha256: str, directories: Sequence[Path]) -> Path | None:
    """The first ``directory / name`` that exists and hashes to ``sha256``, or ``None``.

    A copy that exists but hashes to something else is skipped, with a warning, so a stale or
    damaged file is never used.
    """
    for directory in directories:
        path = directory / name
        if not path.is_file():
            continue
        got = sha256_file(path)
        if got == sha256:
            return path
        _log.warning("%s: sha256 is %s, expected %s; not using this copy", path, got, sha256)
    return None
