"""The ``run:`` detector source: rebuilds a saved checkpoint from a run directory.

Registered under ``detector_sources`` as ``run``; the score layer resolves it with
``get_registry("detector_sources").load("run")(ref)``, never importing this module directly.

Sets two attributes on the detector it returns, beyond contract C4: ``checkpoint_sha256`` (the
sha256 of the exact ``model.safetensors`` loaded, via :func:`dfwb.core.hashing.sha256_file`) and
``training_seed`` (the run's seed, read directly from its ``env.json`` -- this module never
imports ``dfwb.train``, the layer above it, the same way :mod:`dfwb.data.index` reads a processed
store's ``index.jsonl`` directly rather than importing ``dfwb.preprocess``). ``training_seed`` is
``None`` when the run directory has no readable ``env.json`` (a checkpoint built by hand, as
tests do, rather than by a real training run).
"""

from __future__ import annotations

import json
from pathlib import Path

from dfwb.core.errors import ConfigError
from dfwb.core.hashing import sha256_file
from dfwb.models import checkpoint
from dfwb.models.detector import AssembledDetector

__all__ = ["load_run"]

_ENV_FILE = "env.json"

_TAGS = ("best", "last")
_DEFAULT_TAG = "best"


def _follow(candidate: Path) -> Path:
    """Resolve ``candidate`` if it is a symlink, raising a clear error if it dangles.

    ``Path.exists()`` follows symlinks and silently reports ``False`` for a broken one, which
    would otherwise make a dangling ``latest`` look just like a missing directory; this checks
    for that case specifically so the error names the missing target.
    """
    if candidate.is_symlink() and not candidate.exists():
        raise ConfigError(
            f"run: {candidate} is a symlink to {candidate.readlink()}, which does not exist",
            hint="the run directory may have been moved, renamed or partially cleaned up",
        )
    return candidate.resolve() if candidate.is_symlink() else candidate


def _resolve_checkpoint_dir(root: Path, tag: str) -> Path:
    """Find ``checkpoints/<tag>`` under ``root``, a ``latest`` symlink of ``root``, or a
    ``latest`` symlink inside ``root``, in that order."""
    tried: list[Path] = []
    candidates = [root]
    named_latest = root / "latest"
    if named_latest.is_symlink() or named_latest.exists():
        candidates.append(named_latest)
    for candidate in candidates:
        target = _follow(candidate)
        checkpoint_dir = target / "checkpoints" / tag
        tried.append(checkpoint_dir)
        if checkpoint_dir.is_dir():
            return checkpoint_dir
    raise ConfigError(
        f"run: no checkpoints/{tag} found for {root}",
        hint="tried: " + ", ".join(str(path) for path in tried),
    )


def _read_training_seed(run_dir: Path) -> int | None:
    """The run's seed from ``run_dir/env.json``, or ``None`` if it has none (missing, unreadable,
    or without an integer ``"seed"`` key)."""
    env_file = run_dir / _ENV_FILE
    if not env_file.is_file():
        return None
    try:
        data = json.loads(env_file.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    seed = data.get("seed") if isinstance(data, dict) else None
    return seed if isinstance(seed, int) and not isinstance(seed, bool) else None


def load_run(ref: str) -> AssembledDetector:
    """Rebuild the detector saved at ``<dir>[#best|#last]`` (default ``#best``).

    ``<dir>`` may be a run directory (holding ``checkpoints/<tag>/``), a ``latest`` symlink, or a
    run-name directory with a ``latest`` symlink inside it. Sets ``checkpoint_sha256`` and
    ``training_seed`` on the returned detector (see the module docstring).
    """
    dir_part, sep, tag = ref.partition("#")
    tag = tag if sep else _DEFAULT_TAG
    if tag not in _TAGS:
        raise ConfigError(
            f"run: unknown checkpoint tag {tag!r} in {ref!r}",
            hint="use '#best' or '#last' (default: best)",
        )
    checkpoint_dir = _resolve_checkpoint_dir(Path(dir_part), tag)
    detector = checkpoint.load(checkpoint_dir)
    detector.checkpoint_sha256 = sha256_file(checkpoint.weights_path(checkpoint_dir))
    detector.training_seed = _read_training_seed(checkpoint_dir.parent.parent)
    return detector
