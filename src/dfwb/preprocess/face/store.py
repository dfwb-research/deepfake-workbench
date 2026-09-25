"""The lossless processed store: ``profile.json``, ``index.jsonl``, and per-video directories.

A store is one directory per hashed profile, holding:

- ``profile.json``, written once, on first use, recording the profile itself and the backend that
  produced its faces;
- ``index.jsonl``, one line appended per processed video, so a run that stops partway through and
  resumes later only has to redo the videos whose last row is not ``ok``;
- one output directory per video, ``<key>/<compression or "_">/`` (keys carry a ``/``, so this
  nests one directory per task inside one per dataset), holding its frames and ``clip.json``.

Nothing here decodes a video or writes a frame -- that is :mod:`dfwb.preprocess.face.process`.
This module only owns the directory layout and the two files that describe it, and the rule that
nothing is ever written under a datasets root: raw data is read-only.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.hashing import canonical_json
from dfwb.core.paths import ResolvedRoot, RootName, resolve_roots
from dfwb.core.records.io import read_jsonl
from dfwb.core.records.local import ProcessedRecord, ProcessingProfile

__all__ = ["Store", "recover_video_dir", "video_relpath"]

_log = logging.getLogger(__name__)

_INDEX_FILE = "index.jsonl"
_PROFILE_FILE = "profile.json"
_TMP_SUFFIX = re.compile(r"\.tmp-\d+$")
_OLD_SUFFIX = re.compile(r"\.old-\d+$")

Key = tuple[str, str | None]


def video_relpath(key: str, compression: str | None) -> str:
    """``<key>/<compression or "_">``, one video's output directory relative to a store root.

    Keys carry a ``/`` of their own (``<task>/<id>``), so this nests one directory per task
    inside one per dataset. Shared by :meth:`Store.video_dir` and
    :func:`dfwb.preprocess.face.process.process_video`, which needs the same path but has no
    :class:`Store` of its own to ask.
    """
    return f"{key}/{compression or '_'}"


def _stray_siblings(out_dir: Path) -> tuple[list[Path], list[Path]]:
    """``(old dirs, tmp dirs)`` next to ``out_dir`` left by an atomic swap that never finished."""
    parent = out_dir.parent
    if not parent.is_dir():
        return [], []
    old_dirs = [
        path
        for path in parent.glob(f"{out_dir.name}.old-*")
        if path.is_dir() and _OLD_SUFFIX.search(path.name)
    ]
    tmp_dirs = [
        path
        for path in parent.glob(f"{out_dir.name}.tmp-*")
        if path.is_dir() and _TMP_SUFFIX.search(path.name)
    ]
    return old_dirs, tmp_dirs


def recover_video_dir(out_dir: Path) -> None:
    """Repair or clean up whatever ``out_dir.old-*``/``out_dir.tmp-*`` siblings a run that
    stopped mid-swap left behind.

    A successful redo replaces ``out_dir`` in two steps: the previous, good directory is renamed
    to a private ``.old-<pid>`` sibling, then the freshly written ``.tmp-<pid>`` directory is
    renamed into ``out_dir``'s place, and only then is the ``.old-<pid>`` sibling deleted. A crash
    between the first and second of those steps leaves ``out_dir`` missing with its last good
    content sitting in ``.old-<pid>``; this restores it, since a redo that never finished must
    never look like data was lost. Any other leftover -- a ``.tmp-<pid>`` whose swap never
    happened, or an ``.old-<pid>`` whose final delete never ran after a swap that did complete --
    holds nothing a reader should see and is simply removed.
    """
    old_dirs, tmp_dirs = _stray_siblings(out_dir)
    if not out_dir.exists() and old_dirs:
        old_dirs[0].rename(out_dir)
        old_dirs = old_dirs[1:]
    for stray in (*old_dirs, *tmp_dirs):
        shutil.rmtree(stray, ignore_errors=True)


def _is_inside(path: Path, parent: Path) -> bool:
    return path.is_relative_to(parent) or path.resolve().is_relative_to(parent.resolve())


def _check_outside_datasets_roots(root: Path, datasets_roots: Sequence[Path]) -> None:
    """Refuse a store root that sits inside any datasets root: raw data is never written to.

    This mirrors the check the inventory builder makes on its own output folder, adapted for a
    store, which has no dataset folder of its own to check against -- only the datasets roots.
    """
    for place in datasets_roots:
        if _is_inside(root, place):
            raise ConfigError(
                f"the processed store {root} would be written inside the datasets root {place}",
                hint="set DFWB_WORK_ROOT to a folder outside every datasets root",
            )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


class Store:
    """The processed store for one hashed profile, rooted at ``root``.

    Args:
        root: The profile's directory, ``<work root>/<dataset>/processed/<profile_id>/``.
            Callers compute it; this class only checks that it is not under a datasets root.
        profile: The profile this store was, or will be, built with.
        roots: Resolved roots to check ``root`` against (default:
            :func:`~dfwb.core.paths.resolve_roots`). Passing this in lets a caller check a store
            root without touching the real environment or config files.

    Raises:
        ConfigError: ``root`` is inside a datasets root.
    """

    def __init__(
        self,
        root: Path,
        profile: ProcessingProfile,
        *,
        roots: Mapping[RootName, ResolvedRoot] | None = None,
    ) -> None:
        resolved = resolve_roots() if roots is None else roots
        datasets = resolved.get("datasets")
        _check_outside_datasets_roots(root, datasets.paths if datasets is not None else ())
        self.root = root
        self.profile = profile

    @property
    def index_path(self) -> Path:
        """``<root>/index.jsonl``."""
        return self.root / _INDEX_FILE

    def video_dir(self, record: ProcessedRecord) -> Path:
        """``<root>/<key>/<compression or "_">/``, the directory ``record`` is (or would be)
        written into."""
        return self.root / video_relpath(record.key, record.compression)

    def append(self, record: ProcessedRecord) -> None:
        """Append ``record`` to ``index.jsonl`` as one canonical JSON line, then flush and fsync.

        Every outcome is appended, not only ``ok`` ones, so the index always says what happened to
        every video that was attempted.

        Raises:
            ValueError: a field of ``record`` is NaN or infinite, which is not valid JSON;
                nothing is written.
        """
        line = canonical_json(dataclasses.asdict(record)) + "\n"
        self.root.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> list[ProcessedRecord]:
        """The latest row per ``(key, compression)``, sorted by ``(key, compression)``.

        Empty when nothing has been appended yet.
        """
        if not self.index_path.is_file():
            return []
        latest: dict[Key, ProcessedRecord] = {}
        for record in read_jsonl(self.index_path, ProcessedRecord):
            latest[(record.key, record.compression)] = record
        return [latest[key] for key in sorted(latest, key=lambda k: (k[0], k[1] or ""))]

    def done_keys(self) -> set[Key]:
        """``(key, compression)`` pairs whose latest row is ``ok``."""
        return {(r.key, r.compression) for r in self.records() if r.status == "ok"}

    def write_profile(self, backend_meta: Mapping[str, Any]) -> None:
        """Write ``profile.json`` on first use; a no-op (besides the checks below) after that.

        Args:
            backend_meta: The backend section to record, shaped as ``{"name", "version",
                "license", "meta"}``.

        Raises:
            ContractError: ``profile.json`` already exists and holds a different profile (a
                different sha256), naming both profile ids.
        """
        path = self.root / _PROFILE_FILE
        payload = {
            "profile": self.profile.model_dump(mode="json"),
            "sha256": self.profile.sha256(),
            "profile_id": self.profile.profile_id(),
            "backend": dict(backend_meta),
        }
        if not path.is_file():
            _write_json_atomic(path, payload)
            return

        existing = json.loads(path.read_text("utf-8"))
        if existing.get("sha256") != payload["sha256"]:
            raise ContractError(
                f"{path}: already holds profile {existing.get('profile_id')!r}, "
                f"not {payload['profile_id']!r}",
                hint="each processed store is for exactly one profile; "
                "use a different store root, or delete the existing store to rebuild it",
            )
        if existing.get("backend") != payload["backend"]:
            _log.warning(
                "%s: recorded backend metadata %r differs from this run's %r; keeping the "
                "recorded one",
                path,
                existing.get("backend"),
                payload["backend"],
            )

    def cleanup_partial(self) -> None:
        """Repair or clean up every video directory under this store left mid-swap by a crash.

        Every distinct ``.tmp-<pid>``/``.old-<pid>`` sibling found is resolved with
        :func:`recover_video_dir`: a video whose last good directory is stranded in ``.old-*``
        gets it restored, and every other stray tmp or old directory is simply removed.
        """
        if not self.root.is_dir():
            return
        slots: set[Path] = set()
        for path in self.root.rglob("*"):
            if not path.is_dir():
                continue
            match = _TMP_SUFFIX.search(path.name) or _OLD_SUFFIX.search(path.name)
            if match:
                slots.add(path.with_name(path.name[: match.start()]))
        for out_dir in slots:
            recover_video_dir(out_dir)
