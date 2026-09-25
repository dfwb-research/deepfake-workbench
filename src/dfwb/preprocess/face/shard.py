"""Combining shards, and reporting a processed store's status: ``merge`` and ``status``.

A sharded :func:`~dfwb.preprocess.face.runner.run` writes each shard's outcomes into its own
``index.shard-<i>-of-<n>.jsonl`` (see ``shard=`` on :class:`~dfwb.preprocess.face.store.Store`),
so that two machines processing different shards of one dataset never write the same file at once.
:func:`merge` folds every shard file of one store back into a single ``index.jsonl``, after which
the store reads exactly as an unsharded run's would; it refuses while any shard's run may still be
appending to its file (a ``.running`` marker, see :meth:`~dfwb.preprocess.face.store.Store.running`
and :func:`~dfwb.preprocess.face.store.running_markers`).

:func:`status` reports, for a dataset and profile, how many videos in scope hold each outcome --
including videos in scope that hold no row at all (``"not-processed"``) -- without writing
anything. It reuses :func:`~dfwb.preprocess.face.runner.in_scope`, the same scoping
:func:`~dfwb.preprocess.face.runner.run` uses for a protocol split or a ``where`` filter, so its
counts always describe exactly the videos a matching ``run`` would touch, and it folds in every
shard file the same way :func:`merge` would (:func:`_combined_records`, the two functions' shared
combining logic), so a still-sharded, not-yet-merged store is reported accurately too.
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
from dfwb.preprocess.face.store import (
    INDEX_FILE,
    Store,
    parse_shard_filename,
    read_running_marker,
    running_markers,
)
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


def _combined_records(store: Path) -> dict[Key, ProcessedRecord]:
    """``index.jsonl`` folded with every current shard file, one row per video: an ``ok`` row
    already in ``index.jsonl`` wins over a failing shard row, otherwise the shard's (later) row
    wins. The same precedence :func:`merge` writes out, computed here without touching the store
    -- :func:`status` reads it this way too, so it reports a still-sharded store accurately."""
    combined: dict[Key, ProcessedRecord] = {}
    index_path = store / INDEX_FILE
    if index_path.is_file():
        for record in read_jsonl(index_path, ProcessedRecord):
            combined[(record.key, record.compression)] = record

    for path, _, _ in _shard_files(store):
        shard_latest: dict[Key, ProcessedRecord] = {}
        for record in read_jsonl(path, ProcessedRecord):
            shard_latest[(record.key, record.compression)] = record
        for key, record in shard_latest.items():
            combined[key] = _resolve(combined.get(key), record)

    return combined


def _check_no_live_shard_runs(store: Path) -> None:
    """Refuse to go on while any shard's ``.running`` marker says its run may still be appending
    to its file: merging now could read that file mid-write, or race its own removal once the run
    finishes and calls :func:`merge` itself.

    Raises:
        ContractError: naming the host, pid and marker file of the first live run found.
    """
    markers = running_markers(store)
    if not markers:
        return
    marker = markers[0]
    info = read_running_marker(marker)
    host = info.get("host", "an unknown host")
    pid = info.get("pid", "an unknown pid")
    raise ContractError(
        f"{store}: a sharded run is still live on {host} (pid {pid}, {marker.name})",
        hint="wait for it to finish and merge again; if it crashed, delete the marker file and "
        "merge again",
    )


def merge(store: Path) -> MergeSummary:
    """Combine every shard file of ``store`` with any existing ``index.jsonl`` into
    ``index.jsonl``, then remove the shard files.

    One line per video, sorted by ``(key, compression or "")``, combined by
    :func:`_combined_records` (an ``ok`` row already in ``index.jsonl`` wins over a failing shard
    row; otherwise the shard's row wins, since it is always the later attempt). The write is
    atomic (a temporary file renamed into place); the shard files are removed only once it has
    succeeded. A crash between that rename and finishing the shard-file removals can leave the new
    ``index.jsonl`` in place with one or more shard files not yet deleted -- their rows are
    already folded into that index, so running ``merge`` again is safe: it writes the same result
    and finishes removing them.

    Args:
        store: A processed store's directory (``<work root>/<dataset>/processed/<profile id>/``),
            as returned by :func:`store_root` or a ``RunSummary``.

    Returns:
        How many videos the merged index now holds, and how many shard files were combined.

    Raises:
        ContractError: a shard's run may still be appending to its file (its ``.running`` marker
            exists), naming the host and pid; or the shard files in ``store`` do not all share
            the same ``n`` (e.g. some ``-of-2`` and some ``-of-3``). Nothing is changed.
    """
    _check_no_live_shard_runs(store)

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

    combined = _combined_records(store)
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
    with ``task`` the key's prefix before ``/``. A video in scope with no row yet counts as
    ``"not-processed"``. The store's ``index.jsonl`` is folded with every current shard file
    (:func:`_combined_records`, the same combining :func:`merge` would do -- an ``ok`` row wins
    over a failing one, otherwise the latest), so a sharded run's outcomes show up here even
    before anyone runs :func:`merge`. This never writes to the store, and reads happily while a
    shard's run is still live (a ``.running`` marker existing only stops :func:`merge`).

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
    latest = {key: record.status for key, record in _combined_records(store.root).items()}

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
