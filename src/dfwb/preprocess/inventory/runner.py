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
    "LayoutChoice",
    "build_inventory",
    "choose_layout",
    "collect_records",
    "describe_location",
    "folder_status",
    "get_builder",
    "inventory_path",
    "read_inventory",
]

_log = logging.getLogger(__name__)

INVENTORY_FILE = "inventory.jsonl"
META_FILE = "inventory.meta.json"

_NOT_FOUND = "not found"


@dataclass(frozen=True)
class InventoryResult:
    """What :func:`build_inventory` wrote."""

    dataset_id: str
    path: Path
    count: int
    by_task: dict[str, int]  # every task of the builder, in task-table order (0 if absent)
    dataset_dir: Path
    location_source: str  # "--root", "root N", "root N (raw layout; ...)" or "override (...)"


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
    """One folder found for a dataset under a datasets root, and whether it has the raw layout."""

    path: Path
    root_index: int  # 1-based, matching describe_location's "root N"
    has_layout: bool


@dataclass(frozen=True)
class LayoutChoice:
    """The dataset folder :func:`choose_layout` picked, and every copy it considered.

    ``copies`` is empty for an override: an override is used as given, never compared against
    other roots.
    """

    path: Path
    source: str
    copies: tuple[DatasetCopy, ...]


def _root_index(path: Path, expected_folder: str, roots: Mapping[RootName, ResolvedRoot]) -> int:
    """The 1-based datasets-root index that ``path`` (``root / expected_folder``) came from."""
    datasets = roots.get("datasets")
    for index, root_path in enumerate(datasets.paths if datasets is not None else (), start=1):
        if root_path / expected_folder == path:
            return index
    return 0  # unreachable: every candidate comes from locate_dataset's own search


def _earlier_copies_text(copies: Sequence[DatasetCopy]) -> str:
    names = [f"root {copy.root_index}" for copy in copies]
    if len(names) == 1:
        return f"{names[0]} has the folder without it"
    return f"{', '.join(names[:-1])} and {names[-1]} have the folder without it"


def _warn_missing_layout(builder: BaseBuilder, folder: Path, source: str) -> None:
    missing = [d for d in builder.layout_dirs() if not (folder / d).is_dir()]
    _log.warning(
        "%s: %s (%s) does not have the expected raw layout; missing %s",
        builder.dataset_id,
        folder,
        source,
        ", ".join(missing) if missing else "its task directories",
    )


def choose_layout(
    builder: BaseBuilder, location: DatasetLocation, roots: Mapping[RootName, ResolvedRoot]
) -> LayoutChoice:
    """Pick, among the copies ``location`` names, the one holding ``builder``'s raw layout.

    ``location`` must come from :func:`~dfwb.core.paths.locate_dataset` for this builder's
    dataset id and expected folder. An override (``location.root is None``) is used as given; if
    its layout is absent, a warning names the missing directories. Otherwise, among
    ``location.path`` and ``location.also_found``, in root order, the first whose folder passes
    :meth:`~dfwb.preprocess.inventory.base.BaseBuilder.layout_present` wins. When none does, the
    first folder found is used, with a warning naming every copy that was searched.
    """
    if location.root is None:
        source = describe_location(location, roots)
        if not builder.layout_present(location.path):
            _warn_missing_layout(builder, location.path, source)
        return LayoutChoice(location.path, source, ())

    candidates = (location.path, *location.also_found)
    copies = tuple(
        DatasetCopy(
            path, _root_index(path, builder.expected_folder, roots), builder.layout_present(path)
        )
        for path in candidates
    )
    with_layout = [copy for copy in copies if copy.has_layout]
    if with_layout:
        chosen = with_layout[0]
        earlier = [copy for copy in copies if copy.root_index < chosen.root_index]
        source = f"root {chosen.root_index}"
        if earlier:
            source += f" (raw layout; {_earlier_copies_text(earlier)})"
    else:
        chosen = copies[0]
        source = f"root {chosen.root_index}"
        _log.warning(
            "%s: no copy of the dataset folder has the expected raw layout (searched %s); "
            "using root %d",
            builder.dataset_id,
            ", ".join(f"root {copy.root_index} ({copy.path})" for copy in copies),
            chosen.root_index,
        )
    return LayoutChoice(chosen.path, source, copies)


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


def _check_outside(out_dir: Path, dataset_dir: Path, datasets_roots: Sequence[Path]) -> None:
    """Raw data is never written to: refuse an output folder in the dataset or a datasets root."""
    places = [("the dataset folder", dataset_dir)]
    places += [("the datasets root", root) for root in datasets_roots]
    for label, place in places:
        if _is_inside(out_dir, place):
            raise ConfigError(
                f"the work root puts the inventory inside {label} {place}",
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
    if root is not None:
        dataset_dir = absolute(root)
        if not dataset_dir.is_dir():
            raise ConfigError(
                f"{dataset_id}: --root {dataset_dir} is not a directory",
                hint=f"point --root at the dataset's {builder.expected_folder!r} folder",
            )
        if not builder.layout_present(dataset_dir):
            _warn_missing_layout(builder, dataset_dir, "--root")
        source = "--root"
    else:
        location = locate_dataset(
            dataset_id, builder.expected_folder, resolved, overrides=dataset_overrides()
        )
        choice = choose_layout(builder, location, resolved)
        dataset_dir, source = choice.path, choice.source

    path = inventory_path(dataset_id, work_root)
    datasets = resolved.get("datasets")
    _check_outside(path.parent, dataset_dir, datasets.paths if datasets is not None else ())
    records = collect_records(builder, dataset_dir, compressions=compressions)

    by_task = {task.abbr: 0 for task in builder.tasks}
    for record in records:
        by_task[record.key.partition("/")[0]] += 1
    present = {record.compression for record in records}
    meta = {
        "builder": {"id": builder.dataset_id, "version": builder.version},
        "dfwb": __version__,
        "count": len(records),
        "compressions": sorted(present, key=lambda c: (c is not None, c or "")),
        "by_task": by_task,
        "location_source": source,
    }
    assert_no_absolute_paths(meta, where=META_FILE)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, records)
    _write_text(path.parent / META_FILE, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return InventoryResult(dataset_id, path, len(records), by_task, dataset_dir, source)


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
