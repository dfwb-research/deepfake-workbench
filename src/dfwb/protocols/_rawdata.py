"""Raw data is read-only: refuse to write under a datasets root, wherever the work root points."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dfwb.core.errors import ConfigError
from dfwb.core.paths import resolve_roots

__all__ = ["check_outside_datasets_roots"]


def _is_inside(path: Path, parent: Path) -> bool:
    return path.is_relative_to(parent) or path.resolve().is_relative_to(parent.resolve())


def check_outside_datasets_roots(
    path: Path, datasets_roots: Sequence[Path] | None, *, what: str
) -> None:
    """Raise :class:`ConfigError` if ``path`` is inside a datasets root.

    ``datasets_roots`` defaults to the resolved ``datasets`` root(s). ``what`` names what would
    have been written, for the message.
    """
    roots = resolve_roots()["datasets"].paths if datasets_roots is None else datasets_roots
    for root in roots:
        if _is_inside(path, root):
            raise ConfigError(
                f"refusing to write {what} to {path}: it is inside the datasets root {root}",
                hint="raw data is read-only; set DFWB_WORK_ROOT to a folder outside every "
                "datasets root",
            )
