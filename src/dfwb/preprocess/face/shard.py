"""Combining shards, and reporting a processed store's status: ``merge`` and ``status``.

A sharded :func:`~dfwb.preprocess.face.runner.run` writes each shard's outcomes into its own
``index.shard-<i>-of-<n>.jsonl`` (see ``shard=`` on :class:`~dfwb.preprocess.face.store.Store`),
so that two machines processing different shards of one dataset never write the same file at once.
:func:`merge` folds every shard file of one store back into a single ``index.jsonl``, after which
the store reads exactly as an unsharded run's would.

:func:`status` reports, for a dataset and profile, how many videos in scope hold each outcome --
including videos in scope that hold no row at all (``"not-processed"``) -- without writing
anything. It reuses :func:`~dfwb.preprocess.face.runner.in_scope`, the same scoping
:func:`~dfwb.preprocess.face.runner.run` uses for a protocol split or a ``where`` filter, so its
counts always describe exactly the videos a matching ``run`` would touch.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from dfwb.core.errors import ContractError
from dfwb.core.paths import ResolvedRoot, RootName, require_root, resolve_roots
from dfwb.core.records.io import read_jsonl, write_jsonl
from dfwb.core.records.local import ProcessedRecord
from dfwb.preprocess.face import runner
from dfwb.preprocess.face.profiles import load_profile
from dfwb.preprocess.face.store import INDEX_FILE, Store, parse_shard_filename
from dfwb.preprocess.inventory.runner import read_inventory
from dfwb.protocols.rules import task_of

__all__ = ["MergeSummary", "StatusRow", "StatusTable", "merge", "status", "store_root"]

Key = tuple[str, str | None]

_NOT_PROCESSED = "not-processed"


def store_root(
    dataset_id: str, profile: str, *, roots: Mapping[RootName, ResolvedRoot] | None = None
) -> Path:
    """``<work root>/<dataset_id>/processed/<profile's id>``: where ``run``'s store for this
    dataset and profile lives, whether or not anything has been written there yet.
    """
    resolved = resolve_roots() if roots is None else roots
    work_root = require_root("work", resolved)
    processing = load_profile(profile)
    return work_root / dataset_id / "processed" / processing.profile_id()


# ------------------------------------------------------------------------------------ merge


@dataclasses.dataclass(frozen=True)
class MergeSummary:
    """What one :func:`merge` did.

    Attributes:
        store: The store directory merged.
        n_records: How many videos the merged ``index.jsonl`` now holds.
        n_shards: How many shard files were combined in (and removed).
    """

    store: Path
    n_records: int
    n_shards: int


def _shard_files(store: Path) -> list[tuple[Path, int, int]]:
    """Every ``index.shard-*-of-*.jsonl`` under ``store``, each with its parsed ``(i, n)``."""
    if not store.is_dir():
        return []
    found: list[tuple[Path, int, int]] = []
    for path in sorted(store.glob("index.shard-*-of-*.jsonl")):
        parsed = parse_shard_filename(path.name)
        if parsed is not None:
            found.append((path, *parsed))
    return found


def _resolve(old: ProcessedRecord | None, new: ProcessedRecord) -> ProcessedRecord:
    """The row to keep for one video: ``old`` (already in ``index.jsonl``) is kept only when it is
    ``ok`` and ``new`` (from a shard file, always the later attempt) is not; otherwise ``new``
    wins."""
    if old is not None and old.status == "ok" and new.status != "ok":
        return old
    return new


def merge(store: Path) -> MergeSummary:
    """Combine every shard file of ``store`` with any existing ``index.jsonl`` into
    ``index.jsonl``, then remove the shard files.

    One line per video, sorted by ``(key, compression or "")``. Across duplicates -- a video
    ``index.jsonl`` already held a row for, that a shard also processed -- an ``ok`` row wins over
    a failing one; otherwise the shard's row wins, since it is always the later attempt. The write
    is atomic (a temporary file renamed into place); the shard files are removed only once it has
    succeeded, so a merge interrupted partway leaves either the old ``index.jsonl`` and every shard
    file untouched, or the new ``index.jsonl`` and no shard files -- never a mix.

    Args:
        store: A processed store's directory (``<work root>/<dataset>/processed/<profile id>/``),
            as returned by :func:`store_root` or a ``RunSummary``.

    Returns:
        How many videos the merged index now holds, and how many shard files were combined.

    Raises:
        ContractError: the shard files in ``store`` do not all share the same ``n`` (e.g. some
            ``-of-2`` and some ``-of-3``); nothing is changed.
    """
    shards = _shard_files(store)
    counts = {count for _, _, count in shards}
    if len(counts) > 1:
        raise ContractError(
            f"{store}: shard files disagree on their count: {sorted(counts)}",
            hint="merge only the shards of one sharded run (matching -of-<n>); finish or remove "
            "the shards of the other n first",
        )

    index_path = store / INDEX_FILE
    if not shards and not index_path.is_file():
        return MergeSummary(store=store, n_records=0, n_shards=0)

    combined: dict[Key, ProcessedRecord] = {}
    if index_path.is_file():
        for record in read_jsonl(index_path, ProcessedRecord):
            combined[(record.key, record.compression)] = record

    for path, _, _ in shards:
        shard_latest: dict[Key, ProcessedRecord] = {}
        for record in read_jsonl(path, ProcessedRecord):
            shard_latest[(record.key, record.compression)] = record
        for key, record in shard_latest.items():
            combined[key] = _resolve(combined.get(key), record)

    ordered = [combined[key] for key in sorted(combined, key=lambda k: (k[0], k[1] or ""))]
    write_jsonl(index_path, ordered)
    for path, _, _ in shards:
        path.unlink()

    return MergeSummary(store=store, n_records=len(ordered), n_shards=len(shards))


# ----------------------------------------------------------------------------------- status


@dataclasses.dataclass(frozen=True)
class StatusRow:
    """One row of a :class:`StatusTable`: how many videos of ``task`` (and, with a protocol,
    ``split``) hold ``status`` -- ``"not-processed"`` for one in scope with no row yet."""

    task: str
    split: str | None
    status: str
    count: int


@dataclasses.dataclass(frozen=True)
class StatusTable:
    """The result of :func:`status`: counts by status x task, and x split with a protocol."""

    rows: tuple[StatusRow, ...]


def status(
    dataset: str,
    profile: str,
    protocol: str | None = None,
    split: str | None = None,
    *,
    roots: Mapping[RootName, ResolvedRoot] | None = None,
) -> StatusTable:
    """Count, by status and task (and split when ``protocol`` is given), the videos in scope.

    Scope is exactly what a matching :func:`~dfwb.preprocess.face.runner.run` would process (see
    :func:`~dfwb.preprocess.face.runner.in_scope`): every inventory video, or a protocol split's,
    with ``task`` the key's prefix before ``/``. A video in scope with no row yet in the store's
    ``index.jsonl`` counts as ``"not-processed"``. This never writes to the store, and does not
    fold in a sharded run's not-yet-merged shard files -- run :func:`merge` first to include them.

    Args:
        dataset: The dataset, whose inventory must already be built.
        profile: A shipped profile's name, or the path of a profile YAML file.
        protocol: A protocol reference to take the videos from; also breaks counts down by split.
        split: The protocol split to take; needs ``protocol``.
        roots: Resolved roots (default: :func:`~dfwb.core.paths.resolve_roots`).

    Raises:
        ConfigError: ``split`` was given without ``protocol``, a ``where`` field, split or
            protocol is unknown, or the protocol is another dataset's.
        UnknownKeyError: the profile is unknown.
    """
    resolved = resolve_roots() if roots is None else roots
    work_root = require_root("work", resolved)
    processing = load_profile(profile)
    inventory = read_inventory(dataset, work_root)
    scoped = runner.in_scope(
        dataset, inventory, protocol=protocol, split=split, where=None, work_root=work_root
    )
    store = Store(
        work_root / dataset / "processed" / processing.profile_id(), processing, roots=resolved
    )
    latest = {(record.key, record.compression): record.status for record in store.records()}

    with_split = protocol is not None
    counts: Counter[tuple[str, str | None, str]] = Counter()
    for record, split_name in scoped:
        video_status = latest.get((record.key, record.compression), _NOT_PROCESSED)
        counts[(task_of(record.key), split_name if with_split else None, video_status)] += 1

    rows = tuple(
        StatusRow(task=task, split=split_name, status=video_status, count=count)
        for (task, split_name, video_status), count in sorted(
            counts.items(), key=lambda item: (item[0][0], item[0][1] or "", item[0][2])
        )
    )
    return StatusTable(rows=rows)
