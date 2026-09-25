"""The ``run:`` detector source: rebuilds a saved checkpoint from a run directory.

Registered under ``detector_sources`` as ``run``; the score layer resolves it with
``get_registry("detector_sources").load("run")(ref)``, never importing this module directly.
"""

from __future__ import annotations

from pathlib import Path

from dfwb.core.errors import ConfigError
from dfwb.models import checkpoint
from dfwb.models.detector import AssembledDetector

__all__ = ["load_run"]

_TAGS = ("best", "last")
_DEFAULT_TAG = "best"


def _resolve_checkpoint_dir(root: Path, tag: str) -> Path:
    """Find ``checkpoints/<tag>`` under ``root``, a ``latest`` symlink of ``root``, or a
    ``latest`` symlink inside ``root``, in that order."""
    tried: list[Path] = []
    candidates = [root]
    named_latest = root / "latest"
    if named_latest.exists():
        candidates.append(named_latest)
    for candidate in candidates:
        target = candidate.resolve() if candidate.is_symlink() else candidate
        checkpoint_dir = target / "checkpoints" / tag
        tried.append(checkpoint_dir)
        if checkpoint_dir.is_dir():
            return checkpoint_dir
    raise ConfigError(
        f"run: no checkpoints/{tag} found for {root}",
        hint="tried: " + ", ".join(str(path) for path in tried),
    )


def load_run(ref: str) -> AssembledDetector:
    """Rebuild the detector saved at ``<dir>[#best|#last]`` (default ``#best``).

    ``<dir>`` may be a run directory (holding ``checkpoints/<tag>/``), a ``latest`` symlink, or a
    run-name directory with a ``latest`` symlink inside it.
    """
    dir_part, sep, tag = ref.partition("#")
    tag = tag if sep else _DEFAULT_TAG
    if tag not in _TAGS:
        raise ConfigError(
            f"run: unknown checkpoint tag {tag!r} in {ref!r}",
            hint="use '#best' or '#last' (default: best)",
        )
    checkpoint_dir = _resolve_checkpoint_dir(Path(dir_part), tag)
    return checkpoint.load(checkpoint_dir)
