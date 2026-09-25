"""Build, write and read local inventories (contract C3b).

:func:`build_inventory` finds a dataset's folder, runs its builder, checks every record, and
writes ``<work root>/<dataset id>/inventory.jsonl`` plus ``inventory.meta.json``. Rows are sorted
by ``(key, compression)``, so the same files on disk always give the same bytes, and neither file
holds an absolute path: an inventory can be shared or compared across machines.

Builders are looked up in the ``inventory_builders`` registry. Each registration carries the
dataset's expected folder as metadata (``folder``), so :func:`folder_status` can say where a
dataset is without importing its builder.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dfwb._version import __version__
from dfwb.core.errors import ConfigError, ContractError, InstallationError, PluginError
from dfwb.core.paths import (
    DatasetLocation,
    ResolvedRoot,
    RootName,
    absolute,
    dataset_overrides,
    locate_dataset,
    require_root,
    resolve_roots,
)
from dfwb.core.plugins import get_registry
from dfwb.core.records import InventoryRecord, assert_no_absolute_paths, read_jsonl, write_jsonl
from dfwb.preprocess.inventory.base import BaseBuilder, validate_compressions

__all__ = [
    "INVENTORY_FILE",
    "META_FILE",
    "DatasetCopy",
    "FolderStatus",
    "InventoryResult",
    "build_inventory",
    "collect_records",
    "dataset_copies",
    "describe_location",
    "folder_copies",
    "folder_status",
    "get_builder",
    "inventory_path",
    "metadata_copy",
    "read_inventory",
    "resolve_video_path",
]

_log = logging.getLogger(__name__)

INVENTORY_FILE = "inventory.jsonl"
META_FILE = "inventory.meta.json"

_NOT_FOUND = "not found"

# The location_source key for a compression-less dataset's one version (a video_dir without {cX}).
SINGLE_VERSION = "single"


@dataclass(frozen=True)
class InventoryResult:
    """What :func:`build_inventory` wrote."""

    dataset_id: str
    path: Path
    count: int
    by_task: dict[str, int]  # every task of the builder, in task-table order (0 if absent)
    dataset_dir: Path  # the metadata copy: see metadata_copy()
    location_source: Mapping[str, str]  # {"<compression or SINGLE_VERSION>": "root N"/"--root"/
    # "override (...)" (or several, comma-joined, in root order, when a compression's tasks
    # resolve to different copies), ..., "metadata": <same>} -- never an absolute path
    copies: tuple[Path, ...]  # every copy searched (or the one override/--root copy)


@dataclass(frozen=True)
class FolderStatus:
    """Where a dataset's folder is, or why it was not found. Never an error."""

    dataset_id: str
    folder: str | None
    location: DatasetLocation | None
    source: str  # "root N", "override (...)" or "not found"
    problem: str | None = None


def get_builder(dataset_id: str) -> BaseBuilder:
    """The inventory builder registered as ``dataset_id``, instantiated.

    Raises:
        UnknownKeyError: no builder is registered under that id (with a did-you-mean).
        PluginError: the registered object is not a :class:`BaseBuilder`.
    """
    builder = get_registry("inventory_builders").build(dataset_id)
    if not isinstance(builder, BaseBuilder):
        raise PluginError(
            f"inventory_builders/{dataset_id}: {type(builder).__name__} is not a BaseBuilder",
            hint="an inventory builder subclasses dfwb.preprocess.inventory.base.BaseBuilder",
        )
    return builder


def inventory_path(dataset_id: str, work_root: Path) -> Path:
    """``<work_root>/<dataset_id>/inventory.jsonl``."""
    return work_root / dataset_id / INVENTORY_FILE


def _short_source(source: str) -> str:
    """An override's origin without its directory: ``env: DFWB_DATASET_X`` or a file name."""
    if source.startswith("env: "):
        return source
    file, sep, rest = source.partition(" [")
    return Path(file).name + (f" [{rest}" if sep else "")


def describe_location(location: DatasetLocation, roots: Mapping[RootName, ResolvedRoot]) -> str:
    """How a folder was found, without any path: ``root N`` (1-based) or ``override (...)``."""
    if location.root is None:
        return f"override ({_short_source(location.source)})"
    datasets = roots.get("datasets")
    for index, path in enumerate(datasets.paths if datasets is not None else (), start=1):
        if path == location.root:
            return f"root {index}"
    return "a datasets root"


@dataclass(frozen=True)
class DatasetCopy:
    """One folder found for a dataset: whether it holds the builder's layout, and where."""

    path: Path
    source: str  # "root N", "override (...)" or "--root"
    has_layout: bool  # from builder.layout_present(path) -- the one hook a plugin overrides
    video_dirs: tuple[str, ...]  # from builder.videos_present(path) -- display detail only


def _root_source(path: Path, expected_folder: str, roots: Mapping[RootName, ResolvedRoot]) -> str:
    """``root N`` (1-based) for the datasets root that ``path`` (``root / expected_folder``) came
    from, or ``"a datasets root"`` if none matches (unreachable in practice: every candidate comes
    from :func:`~dfwb.core.paths.locate_dataset`'s own search)."""
    datasets = roots.get("datasets")
    for index, root_path in enumerate(datasets.paths if datasets is not None else (), start=1):
        if root_path / expected_folder == path:
            return f"root {index}"
    return "a datasets root"


def dataset_copies(
    builder: BaseBuilder, location: DatasetLocation, roots: Mapping[RootName, ResolvedRoot]
) -> tuple[DatasetCopy, ...]:
    """Every copy ``location`` names, in root order, each annotated with its populated video dirs.

    ``location`` must come from :func:`~dfwb.core.paths.locate_dataset` for this builder's
    dataset id and expected folder. An override (``location.root is None``) names exactly one
    copy, used as given -- never compared against other roots. Whether a copy "has the layout" is
    always :meth:`~dfwb.preprocess.inventory.base.BaseBuilder.layout_present`, so a builder that
    overrides it changes the answer here too.
    """
    if location.root is None:
        source = describe_location(location, roots)
        return (
            DatasetCopy(
                location.path,
                source,
                builder.layout_present(location.path),
                builder.videos_present(location.path),
            ),
        )
    paths = (location.path, *location.also_found)
    return tuple(
        DatasetCopy(
            path,
            _root_source(path, builder.expected_folder, roots),
            builder.layout_present(path),
            builder.videos_present(path),
        )
        for path in paths
    )


def _warn_if_no_videos_anywhere(builder: BaseBuilder, copies: Sequence[DatasetCopy]) -> None:
    """A warning when no copy holds a single video: the missing directories, or every copy."""
    if any(copy.has_layout for copy in copies):
        return
    if len(copies) == 1:
        copy = copies[0]
        missing = [d for d in builder.layout_dirs() if d not in copy.video_dirs]
        _log.warning(
            "%s: %s (%s) does not have the expected raw layout; missing videos in %s",
            builder.dataset_id,
            copy.path,
            copy.source,
            ", ".join(missing) if missing else "its task directories",
        )
    else:
        _log.warning(
            "%s: no copy of the dataset folder has any video (searched %s)",
            builder.dataset_id,
            ", ".join(f"{copy.source} ({copy.path})" for copy in copies),
        )


def metadata_copy(builder: BaseBuilder, copies: Sequence[DatasetCopy]) -> DatasetCopy:
    """The copy :meth:`~BaseBuilder.prepare` and ``official_splits`` read from.

    When the builder declares :attr:`~BaseBuilder.metadata_files`, the first copy holding all of
    them; otherwise (nothing declared, or none holds them all) the first copy holding any video;
    otherwise the first copy found. Public so a later pack-building step can pick the same copy
    without repeating this rule.
    """
    if builder.metadata_files:
        for copy in copies:
            if all((copy.path / name).is_file() for name in builder.metadata_files):
                return copy
    for copy in copies:
        if copy.has_layout:
            return copy
    return copies[0]


def _compression_sources(
    copies: Sequence[DatasetCopy],
    chosen: Mapping[tuple[str, str | None], Path],
    known_compressions: Sequence[str],
) -> dict[str, str]:
    """Which copy(ies) back each compression actually built, from a pre-computed choice.

    ``chosen`` is :meth:`~dfwb.preprocess.inventory.base.BaseBuilder.choose_copies`'s result, so
    this never re-probes the filesystem. Keys are in ``known_compressions`` order (or
    :data:`SINGLE_VERSION` for a compression-less dataset). When two tasks resolve one
    compression to two different copies, every distinct one is listed, comma-joined in root
    order (e.g. ``"root 1, root 2"``) -- naming a split the way ``dfwb datasets info`` shows it,
    rather than picking one arbitrarily.
    """
    per_compression: dict[str | None, set[Path]] = {}
    for (_, compression), copy_path in chosen.items():
        per_compression.setdefault(compression, set()).add(copy_path)

    sources: dict[str, str] = {}
    for compression in (*known_compressions, None):
        path_set = per_compression.get(compression)
        if not path_set:
            continue
        label = compression or SINGLE_VERSION
        ordered = [copy.source for copy in copies if copy.path in path_set]
        sources[label] = ", ".join(ordered)
    return sources


def _warn_shadowed_copies(
    builder: BaseBuilder,
    copies: Sequence[DatasetCopy],
    chosen: Mapping[tuple[str, str | None], Path],
) -> None:
    """A warning for every ``(task, compression)`` where a later copy's video is never used."""
    if len(copies) < 2:
        return
    by_path = {copy.path: copy.source for copy in copies}
    paths = tuple(copy.path for copy in copies)
    tasks = {task.abbr: task for task in builder.tasks}
    for (task_abbr, compression), chosen_path in chosen.items():
        task = tasks[task_abbr]
        shadowed = builder.copies_after(paths, chosen_path, task, compression)
        if not shadowed:
            continue
        label = compression or SINGLE_VERSION
        _log.warning(
            "%s: %s/%s: using %s; %s also has a video there, never scanned",
            builder.dataset_id,
            task_abbr,
            label,
            by_path[chosen_path],
            ", ".join(by_path[p] for p in shadowed),
        )


def _warn_requested_compressions_found_nowhere(
    builder: BaseBuilder, wanted: Sequence[str] | None, location_source: Mapping[str, str]
) -> None:
    """A warning for a requested compression that no copy provided (0 records for it)."""
    if not wanted:
        return
    missing = [c for c in wanted if c not in location_source]
    if missing:
        _log.warning(
            "%s: requested compression(s) %s found in no copy; 0 records for %s",
            builder.dataset_id,
            ", ".join(missing),
            "them" if len(missing) > 1 else "it",
        )


def folder_copies(folder: str, roots: Mapping[RootName, ResolvedRoot]) -> list[Path]:
    """Every datasets root holding ``folder`` as a directory, in root order.

    For a record whose ``folder`` names a sibling dataset's folder (see
    :meth:`~dfwb.preprocess.inventory.base.BaseBuilder.record`): its copies are independent of
    where the record's own dataset folder resolved, so every datasets root is searched again,
    from scratch, for this folder name specifically.
    """
    datasets = roots.get("datasets")
    search_paths = datasets.paths if datasets is not None else ()
    return [root / folder for root in search_paths if (root / folder).is_dir()]


def resolve_video_path(
    record: InventoryRecord,
    copies: Sequence[Path],
    *,
    datasets_roots: Mapping[RootName, ResolvedRoot] | None = None,
) -> Path:
    """The absolute video or frame directory backing ``record``: the first copy holding it.

    A record's ``relpath`` names a video file, or, for a dataset that ships pre-cropped frames
    instead of videos, a directory of frames; either is found. When ``record.folder`` is unset,
    the first of ``copies`` -- the same copies
    :func:`build_inventory` bound the record's own builder to. When it is set (the record lives
    in a sibling dataset's folder, e.g. a TalkingHeadBench real that sits in FaceForensics++),
    ``copies`` is ignored and :func:`folder_copies` resolves that sibling folder across every
    datasets root instead, independent of where the record's own dataset folder resolved -- that
    sibling folder can just as well be split across roots by compression.

    Raises:
        ConfigError: ``record.folder`` is set but ``datasets_roots`` is not given, or none of the
            resolved copies holds the video or frame directory.
    """
    if record.folder is not None:
        if datasets_roots is None:
            raise ConfigError(
                f"{record.key}: its folder {record.folder!r} needs the datasets roots to resolve",
                hint="pass datasets_roots=resolve_roots() (or the roots already in hand)",
            )
        search_copies = folder_copies(record.folder, datasets_roots)
    else:
        search_copies = list(copies)
    for copy in search_copies:
        candidate = copy / record.relpath
        if candidate.is_file() or candidate.is_dir():
            return candidate
    searched = ", ".join(str(copy) for copy in search_copies) or "no copies given"
    raise ConfigError(
        f"{record.key}: {record.relpath!r} was not found in any copy of the dataset folder "
        f"(searched {searched})",
        hint="rebuild the inventory if the dataset's layout on disk changed",
    )


def folder_status(
    dataset_id: str,
    folder: str | None,
    roots: Mapping[RootName, ResolvedRoot],
    overrides: Mapping[str, tuple[Path, str]],
) -> FolderStatus:
    """Locate ``dataset_id``'s folder like :func:`~dfwb.core.paths.locate_dataset`, never raising.

    ``folder`` is the expected folder name (``None`` when the registration names none, in which
    case only an override can locate the dataset).
    """
    if not folder and dataset_id not in overrides:
        return FolderStatus(
            dataset_id,
            folder,
            None,
            _NOT_FOUND,
            f"{dataset_id}: its builder names no folder; set a dataset override to locate it",
        )
    try:
        location = locate_dataset(dataset_id, folder or "", roots, overrides=overrides)
    except ConfigError as exc:
        return FolderStatus(dataset_id, folder, None, _NOT_FOUND, exc.message)
    return FolderStatus(dataset_id, folder, location, describe_location(location, roots))


def _check_record(builder: BaseBuilder, record: object, label_keys: Mapping[str, str]) -> None:
    """The key is ``<task>/<id>`` with a task from the table, and the label key is that task's."""
    if not isinstance(record, InventoryRecord):
        raise ContractError(
            f"{builder.dataset_id}: the builder yielded a {type(record).__name__}",
            hint="a builder yields InventoryRecord objects (see BaseBuilder.record)",
        )
    task, sep, legacy = record.key.partition("/")
    if not sep or not task or not legacy:
        raise ContractError(
            f"{builder.dataset_id}: key {record.key!r} is not <task>/<id>",
            hint="build records with BaseBuilder.record, which adds the task prefix",
        )
    if task not in label_keys:
        raise ContractError(
            f"{builder.dataset_id}: key {record.key!r} has task {task!r}, which is not in the "
            f"task table ({', '.join(label_keys)})",
            hint="every key starts with the abbr of one of the builder's tasks",
        )
    if record.label_key != label_keys[task]:
        raise ContractError(
            f"{builder.dataset_id}: {record.key!r} has label key {record.label_key!r}, but its "
            f"task's label key is {label_keys[task]!r}",
            hint="a record's label key is always its task's (BaseBuilder.record sets it)",
        )


def collect_records(
    builder: BaseBuilder, dataset_dir: Path, *, compressions: Sequence[str] | None = None
) -> list[InventoryRecord]:
    """Run ``builder`` over ``dataset_dir`` and return its records, checked and sorted.

    ``compressions`` is checked against the builder's known compressions here, before the
    builder runs, so a builder that overrides ``discover`` cannot silently ignore a typo. The
    builder's :meth:`~BaseBuilder.prepare` runs once, before :meth:`~BaseBuilder.discover`.

    Raises:
        ConfigError: ``compressions`` names a compression the dataset does not have.
        ContractError: a key is not ``<task>/<id>`` with a task from the builder's table, a label
            key is not its task's, or two records share a ``(key, compression)``.
    """
    wanted = validate_compressions(compressions, builder.known_compressions)
    label_keys = {task.abbr: builder.label_key(task) for task in builder.tasks}
    seen: set[tuple[str, str | None]] = set()
    records: list[InventoryRecord] = []
    builder.prepare(dataset_dir)
    for record in builder.discover(dataset_dir, compressions=wanted):
        _check_record(builder, record, label_keys)
        identity = (record.key, record.compression)
        if identity in seen:
            raise ContractError(
                f"{builder.dataset_id}: duplicate record {record.key!r} "
                f"(compression {record.compression!r})",
                hint="each (key, compression) must be unique; the builder yields it twice",
            )
        seen.add(identity)
        records.append(record)
    records.sort(key=lambda r: (r.key, r.compression or ""))
    return records


def _is_inside(path: Path, parent: Path) -> bool:
    return path.is_relative_to(parent) or path.resolve().is_relative_to(parent.resolve())


def _check_outside(
    out_dir: Path, dataset_dirs: Sequence[Path], datasets_roots: Sequence[Path]
) -> None:
    """Raw data is never written to: refuse an output folder in a copy or a datasets root."""
    label = "the dataset folder" if len(dataset_dirs) == 1 else "a copy of the dataset folder"
    places = [(label, dataset_dir) for dataset_dir in dataset_dirs]
    places += [("the datasets root", root) for root in datasets_roots]
    for place_label, place in places:
        if _is_inside(out_dir, place):
            raise ConfigError(
                f"the work root puts the inventory inside {place_label} {place}",
                hint="set DFWB_WORK_ROOT to a folder outside every datasets root",
            )


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def build_inventory(
    dataset_id: str,
    *,
    root: Path | None = None,
    compressions: Sequence[str] | None = None,
    probe: bool = False,
    jobs: int = 1,
    roots: Mapping[RootName, ResolvedRoot] | None = None,
) -> InventoryResult:
    """Build ``dataset_id``'s inventory and write it under the work root.

    Args:
        dataset_id: The builder's registry key.
        root: The dataset folder; by default it is located through the datasets roots and the
            dataset overrides.
        compressions: Only these compressions (default: every known one).
        probe: Probe each video's media properties (not available in this version).
        jobs: Parallel probes, when probing.
        roots: Resolved roots (default: :func:`~dfwb.core.paths.resolve_roots`).

    Raises:
        ConfigError: the work root is unset, ``root`` is not a directory, the dataset folder is
            not found, the output would land inside the dataset folder, or ``jobs < 1``.
        ContractError: the builder yields a bad or duplicate key.
        InstallationError: ``probe`` is set.
    """
    if jobs < 1:
        raise ConfigError(f"jobs must be at least 1, got {jobs}", hint="use --jobs 1 or more")
    builder = get_builder(dataset_id)
    if probe:
        raise InstallationError(
            "media probing (--probe) is not available in this version of dfwb",
            hint='build without --probe; probing needs the "deepfake-workbench[preprocess]" '
            "extra of a dfwb version that provides it",
        )
    resolved = resolve_roots() if roots is None else roots
    work_root = require_root("work", resolved)
    copies: tuple[DatasetCopy, ...]
    if root is not None:
        dataset_dir = absolute(root)
        if not dataset_dir.is_dir():
            raise ConfigError(
                f"{dataset_id}: --root {dataset_dir} is not a directory",
                hint=f"point --root at the dataset's {builder.expected_folder!r} folder",
            )
        copies = (
            DatasetCopy(
                dataset_dir,
                "--root",
                builder.layout_present(dataset_dir),
                builder.videos_present(dataset_dir),
            ),
        )
    else:
        location = locate_dataset(
            dataset_id, builder.expected_folder, resolved, overrides=dataset_overrides()
        )
        copies = dataset_copies(builder, location, resolved)
    _warn_if_no_videos_anywhere(builder, copies)

    paths = tuple(copy.path for copy in copies)
    wanted = validate_compressions(compressions, builder.known_compressions)
    chosen = builder.choose_copies(paths, compressions=wanted)
    builder.bind_copies(paths, chosen)
    _warn_shadowed_copies(builder, copies, chosen)
    meta_copy = metadata_copy(builder, copies)

    path = inventory_path(dataset_id, work_root)
    datasets = resolved.get("datasets")
    _check_outside(path.parent, paths, datasets.paths if datasets is not None else ())
    records = collect_records(builder, meta_copy.path, compressions=compressions)

    by_task = {task.abbr: 0 for task in builder.tasks}
    for record in records:
        by_task[record.key.partition("/")[0]] += 1
    present = {record.compression for record in records}
    location_source = _compression_sources(copies, chosen, builder.known_compressions)
    _warn_requested_compressions_found_nowhere(builder, wanted, location_source)
    location_source["metadata"] = meta_copy.source
    meta = {
        "builder": {"id": builder.dataset_id, "version": builder.version},
        "dfwb": __version__,
        "count": len(records),
        "compressions": sorted(present, key=lambda c: (c is not None, c or "")),
        "by_task": by_task,
        "location_source": location_source,
    }
    assert_no_absolute_paths(meta, where=META_FILE)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, records)
    _write_text(path.parent / META_FILE, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return InventoryResult(
        dataset_id, path, len(records), by_task, meta_copy.path, location_source, paths
    )


def read_inventory(dataset_id: str, work_root: Path) -> list[InventoryRecord]:
    """The records of ``<work_root>/<dataset_id>/inventory.jsonl``.

    Raises:
        ConfigError: there is no inventory yet (the hint says how to build it).
        ContractError: the file is corrupt.
    """
    path = inventory_path(dataset_id, work_root)
    if not path.is_file():
        raise ConfigError(
            f"{dataset_id}: no inventory at {path}",
            hint=f"run: dfwb inventory build {dataset_id}",
        )
    return read_jsonl(path, InventoryRecord)
