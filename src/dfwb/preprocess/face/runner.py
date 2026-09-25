"""Running the face pipeline over a dataset: which videos, which to skip, and on how many workers.

:func:`run` reads a dataset's inventory, narrows it to the videos asked for (a protocol split, a
``where`` filter, a limit, a shard), leaves out every video the processed store already holds an
outcome for (unless that outcome is one to redo), and hands the rest to
:func:`~dfwb.preprocess.face.process.process_video`, either in this process or in a pool of
worker processes.

The process that calls :func:`run` is the only one that ever writes ``index.jsonl``: workers
return each video's :class:`~dfwb.core.records.ProcessedRecord`, and the caller appends it as it
arrives. The index is therefore never interleaved, and a run stopped at any point leaves an index
that names exactly the videos that finished. Each video is scheduled once per run, so no two
workers ever write the same output directory; a video that was being written when a run was
killed has no index row, so the next run cleans up after it and does it again.

The backend is built once in the calling process too, before any video is touched: building it
enforces the licence acknowledgement its weights may need, its description goes into
``profile.json``, and its optional ``prepare()`` fetches and verifies its model files, so that
workers never download anything. Each worker then builds its own backend, lazily, on its first
video, keeps it for the rest of the run and closes it when the worker exits. Workers are started
with the ``spawn`` method, asked for explicitly: nothing here changes the process-wide start
method, the environment or the selected devices, since the device is simply handed to each
backend.

Progress is shown on stderr with ``rich`` when it is installed and stderr is a terminal, and not
at all otherwise.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import logging
import multiprocessing
import sys
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Final, get_args, get_type_hints

from dfwb.core import licenses
from dfwb.core.errors import ConfigError, did_you_mean
from dfwb.core.hashing import canonical_json
from dfwb.core.paths import (
    ResolvedRoot,
    RootName,
    dataset_overrides,
    locate_dataset,
    require_root,
    resolve_roots,
)
from dfwb.core.plugins import get_registry
from dfwb.core.records import InventoryRecord, ProcessedRecord, ProcessingProfile
from dfwb.preprocess.face.backends import FaceBackend
from dfwb.preprocess.face.process import process_video
from dfwb.preprocess.face.profiles import load_profile
from dfwb.preprocess.face.store import Store, video_relpath
from dfwb.preprocess.inventory.runner import get_builder, read_inventory, resolve_video_path
from dfwb.protocols.protocol import Protocol, _matches_where
from dfwb.protocols.protocol import load as load_protocol

__all__ = ["STATUSES", "RunSummary", "run"]

_log = logging.getLogger(__name__)

# Every outcome a video can have, in the order ProcessedRecord declares them.
STATUSES: Final[tuple[str, ...]] = get_args(get_type_hints(ProcessedRecord)["status"])

# A protocol split holding videos a scheme leaves out on purpose; a protocol run without a split
# takes every other split.
_EXCLUDE_SPLIT = "exclude"

# The inventory columns a ``where`` filter can name without a protocol: the same ones a protocol's
# own records carry (``task`` and ``attrs.<name>`` are derived from ``key`` and ``attrs``).
_WHERE_COLUMNS = (
    "key",
    "compression",
    "label_key",
    "method",
    "identity",
    "source_id",
    "target_id",
    "pair_key",
    "attrs",
)

# How many videos are handed to the pool per worker ahead of time: enough to keep every worker
# busy, few enough that stopping early leaves little queued work to wait for.
_QUEUED_PER_WORKER = 2

_NOT_FOUND = "source not found"

Key = tuple[str, str | None]


@dataclasses.dataclass(frozen=True)
class RunSummary:
    """What one :func:`run` did.

    Attributes:
        counts_by_status: How many videos this run processed, by outcome (only outcomes that
            occurred, in :data:`STATUSES` order); a video whose file was not found counts as
            ``decode_error``.
        n_skipped: Videos in scope that were not processed because the store already holds an
            outcome for them that is not one to redo.
        store: The processed store's directory, ``<work root>/<dataset>/processed/<profile id>``.
    """

    counts_by_status: dict[str, int]
    n_skipped: int
    store: Path


def _sort_key(record: InventoryRecord) -> tuple[str, str]:
    return (record.key, record.compression or "")


# ------------------------------------------------------------------------------------ checks


def _check_redo(redo: Collection[str]) -> frozenset[str]:
    for status in redo:
        if status not in STATUSES:
            raise ConfigError(
                f"unknown status {status!r} to redo{did_you_mean(status, STATUSES)}",
                hint="statuses: " + ", ".join(STATUSES),
            )
    return frozenset(redo)


def _check_shard(shard: tuple[int, int] | None) -> tuple[int, int] | None:
    if shard is None:
        return None
    index, count = shard
    if count < 1 or not 0 <= index < count:
        raise ConfigError(
            f"shard {index}/{count} does not exist",
            hint="a shard is i/n with n at least 1 and 0 <= i < n, e.g. --shard 0/2 and 1/2",
        )
    return index, count


def _check_counts(workers: int, limit: int | None) -> None:
    if workers < 0:
        raise ConfigError(
            f"workers must be 0 or more, got {workers}",
            hint="use --workers 0 to process in this process, or N for a pool of N processes",
        )
    if limit is not None and limit < 1:
        raise ConfigError(f"limit must be at least 1, got {limit}", hint="use --limit 1 or more")


# ----------------------------------------------------------------------------------- scoping


def _protocol_scope(
    dataset_id: str,
    inventory: Sequence[InventoryRecord],
    ref: str,
    split: str | None,
    where: Mapping[str, Any] | None,
    work_root: Path,
) -> list[InventoryRecord]:
    """The inventory records the protocol's split (and ``where``) name, joined on
    ``(key, compression)``; the protocol's own videos this inventory lacks are only counted."""
    protocol = load_protocol(ref, work_root=work_root)
    if protocol.dataset != dataset_id:
        raise ConfigError(
            f"protocol {protocol.ref} is for dataset {protocol.dataset!r}, not {dataset_id!r}",
            hint=f"use a protocol of {dataset_id}, e.g. {dataset_id}/<scheme>",
        )
    splits = sorted({row.split for row in protocol.split_rows()})
    if split is not None and split not in splits:
        raise ConfigError(
            f"{protocol.ref}: split {split!r} is not in this scheme{did_you_mean(split, splits)}",
            hint="splits in this scheme: " + ", ".join(splits),
        )
    wanted_splits = [split] if split is not None else [s for s in splits if s != _EXCLUDE_SPLIT]
    wanted = {
        (video.key, video.compression)
        for video in protocol.records(split=wanted_splits, where=where)
    }
    by_key = {(record.key, record.compression): record for record in inventory}
    missing = len(wanted - by_key.keys())
    if missing:
        _log.warning(
            "%s: %d video(s) of %s%s are not in the inventory, so they are left out",
            dataset_id,
            missing,
            protocol.ref,
            f" ({split})" if split is not None else "",
        )
    return [by_key[key] for key in wanted if key in by_key]


def _where_scope(
    dataset_id: str, inventory: Sequence[InventoryRecord], where: Mapping[str, Any]
) -> list[InventoryRecord]:
    """The inventory records ``where`` matches, with exactly a protocol's ``where`` rules."""
    attr_names = Protocol._check_where_fields(where)
    seen_attrs: set[str] = set()
    kept: list[InventoryRecord] = []
    for record in inventory:
        seen_attrs.update(record.attrs)
        row = {column: getattr(record, column) for column in _WHERE_COLUMNS}
        if _matches_where(row, where):
            kept.append(record)
    unknown = sorted(attr_names - seen_attrs)
    if unknown:
        raise ConfigError(
            f"unknown attribute {unknown[0]!r} for dataset {dataset_id!r}"
            f"{did_you_mean(unknown[0], seen_attrs)}",
            hint="attributes in this inventory: " + (", ".join(sorted(seen_attrs)) or "none"),
        )
    return kept


def _shard_index(record: InventoryRecord, count: int) -> int:
    """Which of ``count`` shards ``record`` belongs to: stable across machines and runs."""
    text = f"{record.key}\t{record.compression or ''}"
    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) % count


def _scope(
    dataset_id: str,
    inventory: Sequence[InventoryRecord],
    *,
    protocol: str | None,
    split: str | None,
    where: Mapping[str, Any] | None,
    limit: int | None,
    shard: tuple[int, int] | None,
    work_root: Path,
) -> list[InventoryRecord]:
    """The videos this run is about, sorted by ``(key, compression)``.

    The protocol (or ``where``) narrows the inventory first; ``limit`` then keeps the first N of
    what is left, and ``shard`` last of all keeps this shard's share of those. Limiting before
    sharding means the shards of a limited run still add up to exactly the unsharded run.
    """
    if protocol is not None:
        records = _protocol_scope(dataset_id, inventory, protocol, split, where, work_root)
    elif split is not None:
        raise ConfigError(
            f"split {split!r} was given without a protocol",
            hint="a split belongs to a protocol: pass --protocol <ref> with --split",
        )
    elif where:
        records = _where_scope(dataset_id, inventory, where)
    else:
        records = list(inventory)
    records.sort(key=_sort_key)
    if limit is not None:
        records = records[:limit]
    if shard is not None:
        index, count = shard
        records = [record for record in records if _shard_index(record, count) == index]
    return records


def _to_do(
    records: Sequence[InventoryRecord], store: Store, redo: frozenset[str]
) -> tuple[list[InventoryRecord], int]:
    """``(the records to process, how many were skipped)``: a record is skipped when the store's
    latest row for it has a status that is not in ``redo``; one with no row is always done."""
    latest = {(row.key, row.compression): row.status for row in store.records()}
    to_do: list[InventoryRecord] = []
    skipped = 0
    for record in records:
        status = latest.get((record.key, record.compression))
        if status is None or status in redo:
            to_do.append(record)
        else:
            skipped += 1
    return to_do, skipped


def _dataset_copies(dataset_id: str, roots: Mapping[RootName, ResolvedRoot]) -> tuple[Path, ...]:
    """Every copy of the dataset's folder, in datasets-root order, found the way the inventory
    was built: an override names exactly one; otherwise every root holding the folder."""
    builder = get_builder(dataset_id)
    location = locate_dataset(
        dataset_id, builder.expected_folder, roots, overrides=dataset_overrides()
    )
    return (location.path, *location.also_found)


# ----------------------------------------------------------------------------------- backend


def _backend_params(profile: ProcessingProfile) -> dict[str, Any]:
    params = profile.backend.model_dump(mode="json")
    params.pop("name")
    return params


def _build_backend(name: str, params: Mapping[str, Any], device: str) -> FaceBackend:
    backend: FaceBackend = get_registry("face_backends").build(name, **params, device=device)
    return backend


def _close(backend: object) -> None:
    close = getattr(backend, "close", None)
    if close is not None:
        close()


def _accept_licence(name: str) -> None:
    """Record the acknowledgement backend ``name``'s class says its weights need, if any."""
    target = get_registry("face_backends").load(name)
    gate = getattr(target, "license_gate", None)
    if gate is None or licenses.is_accepted(gate):
        return
    terms = getattr(target, "license_terms", None) or gate
    licenses.accept(gate, license=terms)
    _log.info("%s: licence acknowledged (%s)", gate, terms)


def _backend_record(backend: FaceBackend) -> dict[str, Any]:
    """The backend section of ``profile.json``, as plain JSON values."""
    record = {
        "name": backend.name,
        "version": backend.version,
        "license": backend.license,
        "meta": dict(backend.meta),
    }
    loaded: dict[str, Any] = json.loads(canonical_json(record))
    return loaded


# ----------------------------------------------------------------------------------- workers


@dataclasses.dataclass(frozen=True)
class _Job:
    """One video for a worker: where to read it from, its inventory row, where to write."""

    source: Path
    record: InventoryRecord
    out_dir: Path


@dataclasses.dataclass
class _WorkerState:
    profile: ProcessingProfile
    backend_name: str
    backend_params: dict[str, Any]
    device: str
    backend: FaceBackend | None = None


# Set in each worker process by the pool's initializer; never set in the calling process.
_worker: _WorkerState | None = None


def _start_worker(
    profile: ProcessingProfile, backend_name: str, backend_params: dict[str, Any], device: str
) -> None:
    """The pool's initializer: remember how to build the backend, but build nothing yet."""
    global _worker
    _worker = _WorkerState(profile, backend_name, backend_params, device)


def _work(job: _Job) -> ProcessedRecord:
    """Process one video in a worker, building the worker's backend on its first video."""
    state = _worker
    if state is None:
        raise RuntimeError("a video was sent to a worker the pool did not initialise")
    if state.backend is None:
        state.backend = _build_backend(state.backend_name, state.backend_params, state.device)
        # A worker ends when the pool shuts down; whatever the backend holds open is released
        # then, while the interpreter is still whole.
        atexit.register(_close, state.backend)
    return process_video(job.source, job.record, state.profile, state.backend, job.out_dir)


def _run_pool(
    jobs: Sequence[_Job],
    *,
    workers: int,
    profile: ProcessingProfile,
    backend_name: str,
    backend_params: dict[str, Any],
    device: str,
    finish: Callable[[ProcessedRecord], None],
) -> None:
    """Process ``jobs`` on a pool of ``workers`` spawned processes, finishing each result here.

    Only a few videos per worker are queued at a time. If one raises, nothing more is queued,
    the videos already under way are waited for and finished, and the first error is raised.
    """
    pending: set[Future[ProcessedRecord]] = set()
    queue = iter(jobs)
    error: BaseException | None = None
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(profile, backend_name, backend_params, device),
    ) as pool:
        while True:
            while error is None and len(pending) < workers * _QUEUED_PER_WORKER:
                job = next(queue, None)
                if job is None:
                    break
                pending.add(pool.submit(_work, job))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                failure = future.exception()
                if failure is not None:
                    error = error or failure
                else:
                    finish(future.result())
    if error is not None:
        raise error


# ---------------------------------------------------------------------------------- progress


def _interactive() -> bool:
    """Whether to draw progress: ``rich`` is installed and stderr is a terminal."""
    stream = sys.stderr
    if stream is None or not stream.isatty():
        return False
    try:
        return importlib.util.find_spec("rich") is not None
    except (ImportError, ValueError):
        return False


def _nothing() -> None:
    pass


@contextlib.contextmanager
def _progress(total: int, description: str) -> Iterator[Callable[[], None]]:
    """A function to call once per finished video, drawing a progress bar when interactive."""
    if not _interactive():
        yield _nothing
        return
    progress_module = importlib.import_module("rich.progress")
    console_module = importlib.import_module("rich.console")
    progress = progress_module.Progress(
        progress_module.TextColumn("{task.description}"),
        progress_module.BarColumn(),
        progress_module.MofNCompleteColumn(),
        progress_module.TimeRemainingColumn(),
        console=console_module.Console(stderr=True),
    )
    with progress:
        task = progress.add_task(description, total=total)
        yield lambda: progress.advance(task)


# ------------------------------------------------------------------------------------ run


def _source_not_found(record: InventoryRecord) -> ProcessedRecord:
    return ProcessedRecord(
        key=record.key,
        compression=record.compression,
        status="decode_error",
        n_frames=0,
        frame_indices=[],
        relpath=video_relpath(record.key, record.compression),
        track=None,
        reason=_NOT_FOUND,
    )


def run(
    dataset_id: str,
    *,
    profile: str,
    protocol: str | None = None,
    split: str | None = None,
    where: Mapping[str, Any] | None = None,
    shard: tuple[int, int] | None = None,
    workers: int = 0,
    device: str = "cpu",
    redo: Collection[str] = frozenset(),
    limit: int | None = None,
    accept_license: bool = False,
    roots: Mapping[RootName, ResolvedRoot] | None = None,
) -> RunSummary:
    """Process ``dataset_id``'s videos with ``profile`` into its processed store.

    The store is ``<work root>/<dataset_id>/processed/<profile id>/``. Which videos are in scope:

    - with ``protocol`` (a reference such as ``ffpp/official``), the videos its ``split``
      assigns (every split but ``exclude`` when no split is given), narrowed by ``where`` with
      the protocol's own rules, and joined with the inventory on ``(key, compression)``; the
      protocol's videos missing from the inventory are counted in a warning;
    - without one, every inventory video, or those ``where`` matches, with the same rules
      (``VideoRecord`` fields, ``task``, ``attrs.<name>``; a list means any of its values);
    - then ``limit`` keeps the first N by ``(key, compression)``, and ``shard=(i, n)`` keeps the
      videos whose ``md5("<key>\\t<compression or ''>")`` is ``i`` modulo ``n``.

    Of those, a video the store already has a row for is skipped unless that row's status is in
    ``redo``: a rerun picks up only what never finished, and ``redo={"no_face"}`` retries just
    the videos that found no face. A video whose file is not found in any copy of the dataset
    folder is recorded as ``decode_error`` with reason ``"source not found"``, and the run goes
    on.

    Args:
        dataset_id: The dataset, whose inventory must already be built.
        profile: A shipped profile's name, or the path of a profile YAML file.
        protocol: A protocol reference to take the videos from.
        split: The protocol split to take; needs ``protocol``.
        where: Field filters, as above.
        shard: ``(i, n)``: process only shard ``i`` of ``n``.
        workers: ``0`` processes every video in this process; ``N`` spreads them over a pool of
            ``N`` spawned processes, each building its own backend on its first video.
        device: ``"cpu"`` or ``"cuda:<index>"``, handed to every backend built.
        redo: Statuses (of :data:`STATUSES`) whose videos are processed again.
        limit: Process at most the first N videos in scope.
        accept_license: Record the licence acknowledgement the backend's weights need, if any,
            before building it.
        roots: Resolved roots (default: :func:`~dfwb.core.paths.resolve_roots`).

    Returns:
        What was processed and skipped, and where the store is.

    Raises:
        ConfigError: An argument is out of range, a ``where`` field, split or redo status is
            unknown, the protocol is another dataset's, the work root is unset, the inventory is
            missing, the dataset folder is not found, or the store would land in a datasets root.
        InstallationError: The backend's weights need a licence acknowledgement that has not been
            given (exit code 5), or the backend's extra is not installed. Raised before any video
            is touched.
        UnknownKeyError: The profile or backend is unknown.
        ContractError: The profile, protocol or inventory is invalid.
    """
    redo_statuses = _check_redo(redo)
    shard = _check_shard(shard)
    _check_counts(workers, limit)
    resolved = resolve_roots() if roots is None else roots
    work_root = require_root("work", resolved)
    processing = load_profile(profile)
    inventory = read_inventory(dataset_id, work_root)
    in_scope = _scope(
        dataset_id,
        inventory,
        protocol=protocol,
        split=split,
        where=where,
        limit=limit,
        shard=shard,
        work_root=work_root,
    )
    store = Store(
        work_root / dataset_id / "processed" / processing.profile_id(), processing, roots=resolved
    )
    copies = _dataset_copies(dataset_id, resolved)

    to_do, n_skipped = _to_do(in_scope, store, redo_statuses)

    backend_name = processing.backend.name
    backend_params = _backend_params(processing)
    if accept_license:
        _accept_licence(backend_name)
    backend = _build_backend(backend_name, backend_params, device)
    unclosed: FaceBackend | None = backend
    counts: Counter[str] = Counter()
    try:
        prepare = getattr(backend, "prepare", None)
        if prepare is not None:
            prepare()
        store.cleanup_partial()
        store.write_profile(_backend_record(backend))
        if workers:
            _close(backend)  # each worker builds its own
            unclosed = None

        with _progress(len(to_do), f"{dataset_id} {processing.profile_id()}") as advance:

            def finish(record: ProcessedRecord) -> None:
                store.append(record)
                counts[record.status] += 1
                advance()

            jobs: list[_Job] = []
            missing = 0
            for record in to_do:
                try:
                    source = resolve_video_path(record, copies, datasets_roots=resolved)
                except ConfigError as exc:
                    _log.debug("%s", exc.message)
                    missing += 1
                    finish(_source_not_found(record))
                    continue
                out_dir = store.root / video_relpath(record.key, record.compression)
                jobs.append(_Job(source, record, out_dir))
            if missing:
                _log.warning(
                    "%s: %d video(s) were not found in any copy of the dataset folder, and are "
                    "recorded as decode_error (%s)",
                    dataset_id,
                    missing,
                    _NOT_FOUND,
                )

            if workers:
                _run_pool(
                    jobs,
                    workers=workers,
                    profile=processing,
                    backend_name=backend_name,
                    backend_params=backend_params,
                    device=device,
                    finish=finish,
                )
            else:
                for job in jobs:
                    finish(process_video(job.source, job.record, processing, backend, job.out_dir))
    finally:
        if unclosed is not None:
            _close(unclosed)
    return RunSummary(
        counts_by_status={status: counts[status] for status in STATUSES if counts[status]},
        n_skipped=n_skipped,
        store=store.root,
    )
