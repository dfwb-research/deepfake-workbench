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
from dfwb.core.paths import ResolvedRoot, RootName, resolve_roots
from dfwb.core.records.io import read_jsonl
from dfwb.core.records.local import ProcessedRecord, ProcessingProfile

__all__ = ["Store"]

_log = logging.getLogger(__name__)

_INDEX_FILE = "index.jsonl"
_PROFILE_FILE = "profile.json"
_TMP_SUFFIX = re.compile(r"\.tmp-\d+$")

Key = tuple[str, str | None]


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
        return self.root / record.key / (record.compression or "_")

    def append(self, record: ProcessedRecord) -> None:
        """Append ``record`` to ``index.jsonl`` as one canonical JSON line, then flush and fsync.

        Every outcome is appended, not only ``ok`` ones, so the index always says what happened to
        every video that was attempted.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        line = (
            json.dumps(
                dataclasses.asdict(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
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
        """Remove every leftover ``*.tmp-<pid>`` video directory under this store.

        A directory in this shape is always the tmp sibling of a video directory that a crashed
        run never finished renaming into place; it holds no output any reader should see.
        """
        if not self.root.is_dir():
            return
        stale = [
            path
            for path in self.root.rglob("*.tmp-*")
            if path.is_dir() and _TMP_SUFFIX.search(path.name)
        ]
        for path in stale:
            shutil.rmtree(path, ignore_errors=True)
