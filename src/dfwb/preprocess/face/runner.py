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

One video never stops a run. A video whose processing raises is recorded as ``decode_error``
with the exception as its reason (``--redo decode_error`` retries it), and a video that kills its
worker outright, as a crash in native decoding or inference code does, is pinned down by re-running
each video that was in flight on its own; the one that kills its worker again is recorded as
``decode_error`` too. Only a failure that repeats for video after video -- ten in a row -- stops
the run, since that points at something systemic rather than at the videos. Problems of
configuration, such as a profile needing something its backend cannot do, are checked before any
video is tried.

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
import traceback
from collections import Counter, deque
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Final, get_args, get_type_hints

from dfwb.core import licenses
from dfwb.core.errors import ConfigError, DFWBError, did_you_mean
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
from dfwb.core.records import (
    InventoryRecord,
    ProcessedRecord,
    ProcessingProfile,
    to_video_record,
)
from dfwb.preprocess.face.backends import FaceBackend
from dfwb.preprocess.face.process import check_backend, process_video
from dfwb.preprocess.face.profiles import load_profile
from dfwb.preprocess.face.store import Store, video_relpath
from dfwb.preprocess.inventory.runner import get_builder, read_inventory, resolve_video_path
from dfwb.protocols.protocol import Protocol, _matches_where
from dfwb.protocols.protocol import load as load_protocol

__all__ = ["STATUSES", "RunSummary", "in_scope", "run"]

_log = logging.getLogger(__name__)

# Every outcome a video can have, in the order ProcessedRecord declares them.
STATUSES: Final[tuple[str, ...]] = get_args(get_type_hints(ProcessedRecord)["status"])

# A protocol split holding videos a scheme leaves out on purpose; a protocol run without a split
# takes every other split.
_EXCLUDE_SPLIT = "exclude"

# How many videos are handed to the pool per worker ahead of time: enough to keep every worker
# busy, few enough that stopping early leaves little queued work to wait for.
_QUEUED_PER_WORKER = 2

_NOT_FOUND = "source not found"
_CRASHED = "error: worker crashed"

# This many failures in a row, of videos whose processing raised or whose worker crashed, stop a
# run: a fault that repeats for every video is systemic, and trying the rest would only record
# the same failure for each of them.
_MAX_CONSECUTIVE_ERRORS = 10

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
) -> list[tuple[InventoryRecord, str | None]]:
    """The inventory records the protocol's split (and ``where``) name, joined on
    ``(key, compression)``, each paired with the split the scheme assigns it; the protocol's own
    videos this inventory lacks are only counted."""
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
    split_by_key = {(row.key, row.compression): row.split for row in protocol.split_rows()}
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
    return [(by_key[key], split_by_key.get(key)) for key in wanted if key in by_key]


def _where_scope(
    dataset_id: str, inventory: Sequence[InventoryRecord], where: Mapping[str, Any]
) -> list[InventoryRecord]:
    """The inventory records ``where`` matches, with exactly a protocol's ``where`` rules."""
    attr_names = Protocol._check_where_fields(where)
    seen_attrs: set[str] = set()
    kept: list[InventoryRecord] = []
    for record in inventory:
        seen_attrs.update(record.attrs)
        if _matches_where(dataclasses.asdict(to_video_record(record)), where):
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
    text = f"{record.key}|{record.compression or ''}"
    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) % count


def in_scope(
    dataset_id: str,
    inventory: Sequence[InventoryRecord],
    *,
    protocol: str | None,
    split: str | None,
    where: Mapping[str, Any] | None,
    work_root: Path,
) -> list[tuple[InventoryRecord, str | None]]:
    """The videos ``run`` (or ``status``) is about, each paired with the protocol split it is
    assigned (``None`` without a protocol), sorted by ``(key, compression)``.

    The protocol (or ``where``) narrows the inventory exactly as :func:`run` does; unlike
    :func:`run`'s own scope, this never applies ``limit`` or ``shard``, since ``status`` reports
    on the whole scope, not one run's or one shard's share of it.

    Raises:
        ConfigError: ``split`` was given without ``protocol``, a ``where`` field, split or
            protocol is unknown, or the protocol is another dataset's.
    """
    if protocol is not None:
        pairs = _protocol_scope(dataset_id, inventory, protocol, split, where, work_root)
    elif split is not None:
        raise ConfigError(
            f"split {split!r} was given without a protocol",
            hint="a split belongs to a protocol: pass --protocol <ref> with --split",
        )
    elif where:
        pairs = [(record, None) for record in _where_scope(dataset_id, inventory, where)]
    else:
        pairs = [(record, None) for record in inventory]
    pairs.sort(key=lambda pair: _sort_key(pair[0]))
    return pairs


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
    records = [
        record
        for record, _ in in_scope(
            dataset_id, inventory, protocol=protocol, split=split, where=where, work_root=work_root
        )
    ]
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


# ------------------------------------------------------------------------------ outcomes


def _video(record: InventoryRecord | ProcessedRecord) -> str:
    """How a video is named in messages: its key, and its compression when it has one."""
    return f"{record.key} ({record.compression})" if record.compression else record.key


def _failed(record: InventoryRecord, reason: str) -> ProcessedRecord:
    return ProcessedRecord(
        key=record.key,
        compression=record.compression,
        status="decode_error",
        n_frames=0,
        frame_indices=[],
        relpath=video_relpath(record.key, record.compression),
        track=None,
        reason=reason,
    )


def _error_reason(exc: BaseException) -> str:
    """``error: <Type>: <first line of the message>``, or ``error: <Type>`` without one."""
    lines = str(exc).strip().splitlines()
    name = type(exc).__name__
    return f"error: {name}: {lines[0]}" if lines else f"error: {name}"


@dataclasses.dataclass(frozen=True)
class _Outcome:
    """One video's result: its index row, and, when its processing raised, the error (the row's
    reason) and its traceback."""

    record: ProcessedRecord
    error: str | None = None
    detail: str | None = None


def _raised(record: InventoryRecord, exc: BaseException, detail: str | None) -> _Outcome:
    reason = _error_reason(exc)
    return _Outcome(_failed(record, reason), reason, detail)


class _Recorder:
    """Appends each video's row to the index as it arrives, counts outcomes, and keeps the run of
    consecutive errors that stops a run with a systemic fault."""

    def __init__(self, store: Store, advance: Callable[[], None]) -> None:
        self._store = store
        self._advance = advance
        self.counts: Counter[str] = Counter()
        self._streak = 0
        self._last: _Outcome | None = None

    def add(self, outcome: _Outcome) -> None:
        record = outcome.record
        self._store.append(record)
        self.counts[record.status] += 1
        self._advance()
        if outcome.error is None:
            self._streak = 0
            return
        self._streak += 1
        self._last = outcome
        _log.warning("%s: %s; recorded as decode_error", _video(record), outcome.error)
        if outcome.detail:
            _log.debug("%s: %s", _video(record), outcome.detail)

    def check(self) -> None:
        """Stop the run once :data:`_MAX_CONSECUTIVE_ERRORS` videos in a row have failed.

        Raises:
            DFWBError: naming the last of them and its error.
        """
        if self._streak < _MAX_CONSECUTIVE_ERRORS or self._last is None:
            return
        raise DFWBError(
            f"{self._streak} videos in a row failed with an error, so the run was stopped; "
            f"the last, {_video(self._last.record)}: {self._last.error}",
            hint="a fault that repeats for every video is rarely the videos' own: check the "
            "backend, its libraries, the device and the disk (--debug shows each traceback); "
            "the failed videos are in the index and are retried with --redo decode_error",
        )


def _attempt(job: _Job, profile: ProcessingProfile, backend: Callable[[], FaceBackend]) -> _Outcome:
    """Process one video, turning any exception into a ``decode_error`` outcome for it.

    ``backend`` returns the backend to use; building one can fail too, and that failure is the
    video's like any other. ``KeyboardInterrupt`` and other non-``Exception`` errors propagate.
    """
    try:
        return _Outcome(process_video(job.source, job.record, profile, backend(), job.out_dir))
    except Exception as exc:
        return _raised(job.record, exc, traceback.format_exc())


# ----------------------------------------------------------------------------------- workers


@dataclasses.dataclass(frozen=True)
class _Job:
    """One video for a worker: where to read it from, its inventory row, where to write."""

    source: Path
    record: InventoryRecord
    out_dir: Path


@dataclasses.dataclass(frozen=True)
class _Settings:
    """What every worker needs to process videos, handed over once when it starts."""

    profile: ProcessingProfile
    backend_name: str
    backend_params: dict[str, Any]
    device: str


@dataclasses.dataclass
class _WorkerState:
    settings: _Settings
    backend: FaceBackend | None = None

    def get_backend(self) -> FaceBackend:
        """The worker's backend, built on first use and closed when the worker exits."""
        if self.backend is None:
            settings = self.settings
            self.backend = _build_backend(
                settings.backend_name, settings.backend_params, settings.device
            )
            # A worker ends when its pool shuts down; whatever the backend holds open is
            # released then, while the interpreter is still whole.
            atexit.register(_close, self.backend)
        return self.backend


# Set in each worker process by the pool's initializer; never set in the calling process.
_worker: _WorkerState | None = None


def _start_worker(settings: _Settings) -> None:
    """The pool's initializer: remember how to build the backend, but build nothing yet."""
    global _worker
    _worker = _WorkerState(settings)


def _work(job: _Job) -> _Outcome:
    """Process one video in a worker, building the worker's backend on its first video."""
    state = _worker
    if state is None:
        raise RuntimeError("a video was sent to a worker the pool did not initialise")
    return _attempt(job, state.settings.profile, state.get_backend)


def _interrupted(future: Future[_Outcome]) -> bool:
    """Whether a finished future's worker was interrupted (Ctrl-C reaches every process), rather
    than finishing its video or failing on it."""
    failure = future.exception()
    return failure is not None and not isinstance(failure, Exception)


def _outcome_of(future: Future[_Outcome], job: _Job) -> _Outcome | None:
    """A finished, uninterrupted future's outcome, or ``None`` when its worker died and broke
    the pool."""
    failure = future.exception()
    if isinstance(failure, BrokenProcessPool):
        return None
    if failure is not None:  # the worker could not even report back
        return _raised(job.record, failure, None)
    return future.result()


def _pool_round(
    queue: deque[_Job], workers: int, settings: _Settings, recorder: _Recorder
) -> list[_Job]:
    """Process videos from ``queue`` on a fresh pool of ``workers`` spawned processes until the
    queue is empty or a worker dies; return the videos that were in flight when one died.

    Only a few videos per worker are handed out at a time. If anything is raised here -- an
    interrupt, here or in a worker, or the stop after too many errors in a row -- the videos
    already finished are recorded, every queued one is cancelled, and the pool is shut down
    before it propagates. A video whose worker was interrupted is not recorded at all, so the
    next run does it again.
    """
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_start_worker,
        initargs=(settings,),
    )
    pending: dict[Future[_Outcome], _Job] = {}
    in_flight: list[_Job] = []
    broken = False
    try:
        while True:
            while not broken and queue and len(pending) < workers * _QUEUED_PER_WORKER:
                job = queue.popleft()
                try:
                    pending[pool.submit(_work, job)] = job
                except BrokenProcessPool:
                    queue.appendleft(job)  # never handed out, so never in flight
                    broken = True
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                if _interrupted(future):
                    raise KeyboardInterrupt  # nothing is recorded for it, so it is done again
                outcome = _outcome_of(future, job)
                if outcome is None:
                    in_flight.append(job)
                    broken = True
                else:
                    recorder.add(outcome)
            recorder.check()
    except BaseException:
        for future, job in pending.items():
            if future.done() and not future.cancelled() and not _interrupted(future):
                outcome = _outcome_of(future, job)
                if outcome is not None:
                    recorder.add(outcome)
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return in_flight


def _run_pool(
    jobs: Sequence[_Job], *, workers: int, settings: _Settings, recorder: _Recorder
) -> None:
    """Process ``jobs`` on pools of ``workers`` spawned processes, surviving worker crashes.

    When a worker dies, the pool is lost with every video it had in flight, and which of them
    killed it is unknown. Each of those is run again on its own, in a pool of one; a video that
    kills that worker too is recorded as ``decode_error`` (``"error: worker crashed"``), and the
    rest of the videos go on in a new pool.
    """
    queue = deque(jobs)
    while queue:
        for job in _pool_round(queue, workers, settings, recorder):
            if _pool_round(deque([job]), 1, settings, recorder):
                recorder.add(_Outcome(_failed(job.record, _CRASHED), _CRASHED))
                recorder.check()


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
      videos whose ``md5("<key>|<compression or ''>")`` is ``i`` modulo ``n``.

    Of those, a video the store already has a row for is skipped unless that row's status is in
    ``redo``: a rerun picks up only what never finished, and ``redo={"no_face"}`` retries just
    the videos that found no face. A sharded run appends to its own
    ``index.shard-<i>-of-<n>.jsonl`` instead of ``index.jsonl``, so two machines can each process
    their shard without one overwriting the other's rows; its skip decision reads the union of
    that file and ``index.jsonl``, the latest row winning. :func:`~dfwb.preprocess.face.shard.merge`
    combines every shard file (and any ``index.jsonl``) back into one ``index.jsonl`` afterwards.

    A failing video is recorded and the run goes on. A video whose file is not found in any copy
    of the dataset folder is recorded as ``decode_error`` with reason ``"source not found"``; one
    whose processing raises, with reason ``"error: <Type>: <message>"``; one that kills its
    worker process (after being re-run on its own to be sure), with ``"error: worker crashed"``.
    Every row is appended as its video finishes, so rows written before an interrupt stay.

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
        roots: Resolved roots, for the work root and the datasets roots (default:
            :func:`~dfwb.core.paths.resolve_roots`). Backends find their model files under the
            cache root themselves, resolved from the environment and config files, not from these.

    Returns:
        What was processed and skipped, and where the store is.

    Raises:
        ConfigError: An argument is out of range, a ``where`` field, split or redo status is
            unknown, the protocol is another dataset's, the work root is unset, the inventory is
            missing, the dataset folder is not found, the store would land in a datasets root, or
            the profile needs something the backend cannot do. Raised before any video is tried.
        DFWBError: Ten videos in a row failed with an error; the message names the last one.
            Every row recorded before that stays in the index.
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
        work_root / dataset_id / "processed" / processing.profile_id(),
        processing,
        roots=resolved,
        shard=shard,
    )
    copies = _dataset_copies(dataset_id, resolved)

    to_do, n_skipped = _to_do(in_scope, store, redo_statuses)

    backend_name = processing.backend.name
    backend_params = _backend_params(processing)
    if accept_license:
        _accept_licence(backend_name)
    backend = _build_backend(backend_name, backend_params, device)
    unclosed: FaceBackend | None = backend
    try:
        check_backend(processing, backend)
        prepare = getattr(backend, "prepare", None)
        if prepare is not None:
            prepare()
        store.cleanup_partial()
        store.write_profile(_backend_record(backend))
        if workers:
            _close(backend)  # each worker builds its own
            unclosed = None

        with _progress(len(to_do), f"{dataset_id} {processing.profile_id()}") as advance:
            recorder = _Recorder(store, advance)
            jobs: list[_Job] = []
            missing: list[InventoryRecord] = []
            for record in to_do:
                try:
                    source = resolve_video_path(record, copies, datasets_roots=resolved)
                except ConfigError as exc:
                    _log.debug("%s", exc.message)
                    missing.append(record)
                    recorder.add(_Outcome(_failed(record, _NOT_FOUND)))
                    continue
                out_dir = store.root / video_relpath(record.key, record.compression)
                jobs.append(_Job(source, record, out_dir))
            if missing:
                _log.warning(
                    "%s: %d video(s) were not found in any copy of the dataset folder, the first "
                    "%s; recorded as decode_error (%s)",
                    dataset_id,
                    len(missing),
                    _video(missing[0]),
                    _NOT_FOUND,
                )

            if workers:
                settings = _Settings(processing, backend_name, backend_params, device)
                _run_pool(jobs, workers=workers, settings=settings, recorder=recorder)
            else:
                for job in jobs:
                    recorder.add(_attempt(job, processing, lambda: backend))
                    recorder.check()
    finally:
        if unclosed is not None:
            _close(unclosed)
    counts = recorder.counts
    return RunSummary(
        counts_by_status={status: counts[status] for status in STATUSES if counts[status]},
        n_skipped=n_skipped,
        store=store.root,
    )
