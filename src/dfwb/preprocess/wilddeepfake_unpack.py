"""Safely unpack WildDeepfake's released tar shards into the layout its builder reads.

The release (see :mod:`dfwb.preprocess.inventory.builders.wilddeepfake`) ships four category
folders, ``real_train``, ``real_test``, ``fake_train`` and ``fake_test``, each holding tar shards
named ``<shard>.tar.gz`` -- plain tar archives despite the name, each holding
``<shard>/<label>/<sequence>/<frame>.png``. :func:`unpack_wilddeepfake` turns those into the
frame-directory tree the builder discovers: one folder per sequence, named
``<label>_<split>_<shard>_<sequence>``, under the real or the fake task's folder (read from the
builder, never hard-coded here), holding that sequence's frames renamed ``<nnnnnn>.png``.

Every archive member is checked before anything is written: an absolute path, a ``..`` segment, a
symlink or hard link (of any kind, not only one that would land outside the target -- this
dataset's shards need none, so the simplest safe rule is to allow none), or a device/FIFO file
refuses the whole archive with the offending member named, before a single byte of it is written.
A frame's destination path is never built from the archive's own member name -- only from the
label, sequence and frame components validated out of it -- so even a member that slipped past
those checks could not otherwise steer where its bytes land.

The label comes from the category folder: a frame is keyed by the ``real``/``fake`` of the
``real_train``/``real_test``/``fake_train``/``fake_test`` folder its shard sits in, never by the
``<label>`` path component inside the shard. An inner label folder that disagrees with its
category is warned about (it changes nothing about where the frame ends up, so it is not a
safety problem and not refused), both on :class:`UnpackResult` and, summarised, in the CLI
output.

Two archives with the same shard id (the name up to its first ``.``) in one category would write
the same sequence folders, so they are refused, both named, before anything is written. A folder
next to the categories that is not one of them, and an archive below a category rather than
directly in it, are not unpacked, and each is warned about.

Real shards' members are commonly named with a leading ``./`` (e.g. ``./6/real/54/0.png``) and
carry an explicit entry for every directory they hold; both are handled the same as their
without-prefix, without-directory-entries equivalents (``pathlib.PurePosixPath`` already discards
a leading ``./`` and a trailing ``/``, and a directory entry never matches the four-part
``<shard>/<label>/<sequence>/<frame>`` shape a frame needs, so it is skipped like any other
non-frame member).

The destination is a datasets root: the dataset lands in ``<root>/<builder.expected_folder>``
(``WildDeepfake``), the folder ``dfwb inventory build`` looks for under a datasets root.
Afterwards, discovery runs exactly where ``dfwb inventory build`` will look (the same root and
override lookup, and the same choice of copy), which is not this folder when the configured
datasets roots do not include ``<root>`` or an override points elsewhere.

Unpacking is resumable and idempotent, driven entirely by what is already on disk: a sequence
whose target folder already holds exactly the frame files the archive says it should is left
alone (an already-complete shard costs only a re-read of its tar index, no bytes rewritten); any
other sequence folder is removed and rewritten from scratch, so an interrupted run never leaves a
stale mix of old and new frames. Each frame is written to a private temporary file and renamed
into place, so a crash mid-write never leaves a truncated frame under its final name.
"""

from __future__ import annotations

import os
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, Final

from dfwb.core.errors import ConfigError, ContractError
from dfwb.preprocess.inventory.runner import _discover_located

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dfwb.core.paths import ResolvedRoot, RootName
    from dfwb.preprocess.inventory.base import BaseBuilder, TaskSpec

__all__ = ["UnpackResult", "unpack_wilddeepfake"]

# The release's category folders: <label>_<split>, each holding <shard>.tar.gz shards. Named
# verbatim from the builder's own description of the release.
_CATEGORIES: Final[tuple[str, ...]] = ("real_train", "real_test", "fake_train", "fake_test")

_FRAME_SUFFIX: Final = ".png"


@dataclass(frozen=True)
class UnpackResult:
    """What :func:`unpack_wilddeepfake` did, and what discovery found afterwards."""

    dataset_id: str
    from_dir: Path
    datasets_root: Path
    to: Path  # the dataset folder written: datasets_root / builder.expected_folder
    categories: tuple[str, ...]  # category folders found under from_dir and processed
    shards_found: int
    sequences_total: int
    sequences_written: int  # created or rewritten (missing, or incomplete on disk)
    sequences_skipped: int  # already held exactly the expected frames
    frames_written: int
    # The folder `dfwb inventory build` reads, and what discovery finds there by task abbr;
    # both None when it would not find the dataset at all.
    inventory_folder: Path | None
    by_task: dict[str, int] | None
    warnings: tuple[str, ...]  # one per frame whose inner label disagreed with its category
    layout_warnings: tuple[str, ...]  # folders and archives under from_dir left alone


@dataclass
class _ShardStats:
    sequences_total: int = 0
    sequences_written: int = 0
    sequences_skipped: int = 0
    frames_written: int = 0
    warnings: list[str] = field(default_factory=list)

    def add(self, other: _ShardStats) -> None:
        self.sequences_total += other.sequences_total
        self.sequences_written += other.sequences_written
        self.sequences_skipped += other.sequences_skipped
        self.frames_written += other.frames_written
        self.warnings.extend(other.warnings)


@dataclass
class _Totals(_ShardStats):
    shards_found: int = 0


def _unsafe_reason(member: tarfile.TarInfo) -> str | None:
    """Why ``member`` is refused, or ``None`` if it is safe to consider extracting."""
    path = PurePosixPath(member.name)
    if not member.name or path.is_absolute():
        return "an absolute path"
    if ".." in path.parts:
        return "a '..' path segment"
    if member.issym() or member.islnk():
        return "a symlink or a hard link"
    if member.isdev():
        return "a device or FIFO special file"
    if not member.isfile() and not member.isdir():
        return "an unsupported member type"
    return None


def _output_frame_name(raw_name: str) -> str | None:
    """``raw_name`` (e.g. ``1919.PNG``) as ``<nnnnnn>.png``, or ``None`` if it is not a frame.

    A numeric stem is zero-padded to six digits; any other stem is kept as written. Only a name
    whose suffix is ``.png`` (any case) is a frame; anything else in the archive is ignored.
    """
    stem, dot, suffix = raw_name.rpartition(".")
    if not dot or not stem or suffix.lower() != "png":
        return None
    if stem.isdigit():
        stem = stem.zfill(6)
    return f"{stem}{_FRAME_SUFFIX}"


def _parse_frame_member(name: str) -> tuple[str, str, str] | None:
    """``(label, sequence, output frame name)`` for a member shaped ``<shard>/<label>/<sequence>/
    <frame>``, or ``None`` if it is not shaped like a frame (skipped, not refused: an archive may
    hold other files dfwb does not need)."""
    parts = PurePosixPath(name).parts
    if len(parts) != 4:
        return None
    _shard_dir, label, sequence, frame_name = parts
    if not label or not sequence:
        return None
    out_name = _output_frame_name(frame_name)
    if out_name is None:
        return None
    return label, sequence, out_name


def _visible_files(directory: Path) -> set[str]:
    return {
        entry.name
        for entry in directory.iterdir()
        if entry.is_file() and not entry.name.startswith(".")
    }


def _write_frame(source: IO[bytes], target_dir: Path, out_name: str) -> None:
    tmp = target_dir / f".{out_name}.tmp-{os.getpid()}"
    try:
        with source, tmp.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        tmp.replace(target_dir / out_name)
    finally:
        tmp.unlink(missing_ok=True)


def _grouped_frames(
    tar: tarfile.TarFile, archive_path: Path, label: str, split: str
) -> tuple[dict[str, list[tuple[tarfile.TarInfo, str]]], list[str]]:
    """Every safe member of ``tar``, grouped by sequence, mapped to its output name; and one
    warning per member whose inner ``<label>`` path component disagreed with ``label``.

    Every member is checked before any is used: the first unsafe one (an absolute path, a ``..``
    segment, a symlink, a hard link, or a device or FIFO file) refuses the whole archive. A member
    that is safe but not shaped like a frame (``<shard>/<label>/<sequence>/<frame>``) is skipped,
    not refused -- an archive may hold other files dfwb does not need, and a real shard commonly
    holds an explicit entry for every directory it has, none of which is ever four parts deep.

    The label comes from the category folder: a frame is always grouped (and later keyed) by
    ``label``, the category the shard itself sits in, never by its own inner ``<label>`` path
    component. An inner label folder that disagrees is warned about, not treated as unsafe or
    refused.

    Raises:
        ContractError: an unsafe member (see :func:`_unsafe_reason`).
    """
    members = tar.getmembers()
    for member in members:
        reason = _unsafe_reason(member)
        if reason is not None:
            raise ContractError(
                f"{archive_path}: refusing to unpack {member.name!r}: {reason}",
                hint="the archive is untrusted or corrupt; verify its source and re-download it",
            )

    frames: dict[str, list[tuple[tarfile.TarInfo, str]]] = {}
    warnings: list[str] = []
    for member in members:
        if not member.isfile():
            continue
        parsed = _parse_frame_member(member.name)
        if parsed is None:
            continue
        inner_label, sequence, out_name = parsed
        if inner_label != label:
            warnings.append(
                f"{archive_path.name}: {member.name!r} is filed under {inner_label!r} inside "
                f"the {label!r} category ({label}_{split}); kept under {label!r}"
            )
        frames.setdefault(sequence, []).append((member, out_name))
    return frames, warnings


def _unpack_sequence(
    tar: tarfile.TarFile,
    archive_path: Path,
    entries: list[tuple[tarfile.TarInfo, str]],
    label: str,
    sequence: str,
    target_dir: Path,
) -> tuple[bool, int]:
    """Write ``entries`` into ``target_dir``; return ``(written, n_frames)``.

    Skips (``written=False``) when ``target_dir`` already holds exactly the expected frame names.

    Raises:
        ContractError: two entries normalise to the same output name, or a member cannot be read.
    """
    out_names = [name for _, name in entries]
    if len(set(out_names)) != len(out_names):
        dupes = sorted({n for n in out_names if out_names.count(n) > 1})
        raise ContractError(
            f"{archive_path}: sequence {label}/{sequence}: two frames both normalise to "
            f"{dupes[0]!r}",
            hint="the archive's frame names collide after normalising; re-check the release",
        )
    expected = set(out_names)
    if target_dir.is_dir() and _visible_files(target_dir) == expected:
        return False, 0

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    for member, out_name in entries:
        source = tar.extractfile(member)
        if source is None:
            raise ContractError(
                f"{archive_path}: {member.name!r} could not be read from the archive",
                hint="re-download the shard",
            )
        _write_frame(source, target_dir, out_name)
    return True, len(entries)


def _unpack_shard(
    archive_path: Path, label: str, split: str, task_video_dir: str, to: Path
) -> _ShardStats:
    """Unpack one ``<shard>.tar.gz`` into ``to / task_video_dir``, one folder per sequence.

    Raises:
        ConfigError: never (a missing ``to`` is created by the caller).
        ContractError: the archive cannot be opened, holds an unsafe member (named in the
            message), or two frames of one sequence normalise to the same output name (see
            :func:`_grouped_frames` and :func:`_unpack_sequence`).
    """
    shard_id = _shard_id(archive_path)
    stats = _ShardStats()
    try:
        with tarfile.open(archive_path, mode="r:*") as tar:
            frames, warnings = _grouped_frames(tar, archive_path, label, split)
            stats.warnings.extend(warnings)
            for sequence, entries in sorted(frames.items()):
                key = f"{label}_{split}_{shard_id}_{sequence}"
                target_dir = to / task_video_dir / key
                stats.sequences_total += 1
                written, n_frames = _unpack_sequence(
                    tar, archive_path, entries, label, sequence, target_dir
                )
                if written:
                    stats.sequences_written += 1
                    stats.frames_written += n_frames
                else:
                    stats.sequences_skipped += 1
    except tarfile.TarError as exc:
        raise ContractError(
            f"{archive_path}: not a readable tar archive ({exc})",
            hint="re-download the shard; WildDeepfake's shards are plain tar archives despite "
            "their .tar.gz name",
        ) from None
    return stats


def _shard_id(archive_path: Path) -> str:
    return archive_path.name.partition(".")[0]


def _shards_of(from_dir: Path, category: str) -> tuple[list[Path], list[str]]:
    """The ``*.tar.gz`` shards directly in ``from_dir / category``, and one warning per archive
    below it instead, which is not unpacked.

    Raises:
        ConfigError: two shards share a shard id (see the module docstring).
    """
    folder = from_dir / category
    shards = sorted(folder.glob("*.tar.gz"))
    by_id: dict[str, list[Path]] = {}
    for shard in shards:
        by_id.setdefault(_shard_id(shard), []).append(shard)
    for shard_id, paths in sorted(by_id.items()):
        if len(paths) > 1:
            names = " and ".join(path.name for path in paths)
            raise ConfigError(
                f"{folder}: {names} are both shard {shard_id} of {category}, and would write the "
                "same sequence folders",
                hint="keep one archive per shard in each category folder (a shard's id is its "
                "name up to the first '.'), and move the other out of --from",
            )
    nested = sorted(path for path in folder.rglob("*.tar.gz") if path.parent != folder)
    warnings = [
        f"{path.relative_to(from_dir).as_posix()} is below {category}, not directly in it, so it "
        f"was not unpacked; move it up into {category}/ if it is a shard of the release"
        for path in nested
    ]
    return shards, warnings


def _unrecognised_folders(from_dir: Path) -> list[str]:
    """One warning per folder in ``from_dir`` that is not one of the release's categories."""
    return [
        f"{entry.name}/ in {from_dir} is not one of {', '.join(_CATEGORIES)}, so it was not "
        "unpacked"
        for entry in sorted(from_dir.iterdir())
        if entry.is_dir() and entry.name not in _CATEGORIES and not entry.name.startswith(".")
    ]


def _check_directory(path: Path, what: str) -> None:
    if path.exists() and not path.is_dir():
        raise ConfigError(
            f"{path}: not a directory", hint=f"point --to at {what}, not a file, or remove it"
        )


def unpack_wilddeepfake(
    builder: BaseBuilder,
    from_dir: Path,
    datasets_root: Path,
    *,
    roots: Mapping[RootName, ResolvedRoot] | None = None,
) -> UnpackResult:
    """Unpack every shard under ``from_dir`` into ``datasets_root / builder.expected_folder``,
    then run discovery where ``dfwb inventory build`` will look.

    ``from_dir`` holds one or more of the release's category folders (``real_train``,
    ``real_test``, ``fake_train``, ``fake_test``), each holding ``<shard>.tar.gz`` files.
    ``datasets_root`` is a datasets root: the dataset folder is created inside it if missing.
    ``roots`` are the resolved roots discovery locates the dataset with, exactly as
    ``dfwb inventory build`` does (default: :func:`~dfwb.core.paths.resolve_roots`).

    Raises:
        ConfigError: ``from_dir`` is not a directory, holds none of the release's category
            folders, holds no ``*.tar.gz`` shard directly in one, or holds two shards with the
            same id in one category; or ``datasets_root`` or the dataset folder is a file.
        ContractError: a shard is unreadable or holds an unsafe or inconsistent member (see
            :func:`_unpack_shard`).
    """
    if not from_dir.is_dir():
        raise ConfigError(
            f"{from_dir}: not a directory",
            hint="point --from at the directory the release's tar shards were downloaded into",
        )
    tasks_by_kind: dict[str, TaskSpec] = {task.kind: task for task in builder.tasks}
    categories_present = [name for name in _CATEGORIES if (from_dir / name).is_dir()]
    if not categories_present:
        raise ConfigError(
            f"{from_dir}: none of {', '.join(_CATEGORIES)} was found",
            hint="point --from at the directory holding those category folders of tar shards",
        )
    to = datasets_root / builder.expected_folder
    _check_directory(datasets_root, "a datasets root")
    _check_directory(to, f"a datasets root whose {builder.expected_folder} is a folder")

    layout_warnings = _unrecognised_folders(from_dir)
    shards: list[tuple[str, str, str, Path]] = []  # (label, split, task_video_dir, shard_path)
    for category in categories_present:
        label, _, split = category.partition("_")
        task = tasks_by_kind.get(label)
        if task is None:
            continue
        category_shards, nested = _shards_of(from_dir, category)
        layout_warnings.extend(nested)
        shards.extend((label, split, task.video_dir, path) for path in category_shards)

    if not shards:
        raise ConfigError(
            f"{from_dir}: {', '.join(categories_present)} hold no *.tar.gz shard files",
            hint="check the release download completed",
        )

    # Only created once there is at least one shard to unpack, so a refusal above never leaves an
    # empty folder behind.
    to.mkdir(parents=True, exist_ok=True)
    totals = _Totals()
    for label, split, task_video_dir, shard_path in shards:
        totals.shards_found += 1
        totals.add(_unpack_shard(shard_path, label, split, task_video_dir, to))

    inventory_folder: Path | None
    by_task: dict[str, int] | None
    try:
        inventory_folder, records = _discover_located(builder, roots)
    except ConfigError:
        inventory_folder, by_task = None, None
    else:
        by_task = {task.abbr: 0 for task in builder.tasks}
        for record in records:
            by_task[record.key.partition("/")[0]] += 1

    return UnpackResult(
        dataset_id=builder.dataset_id,
        from_dir=from_dir,
        datasets_root=datasets_root,
        to=to,
        categories=tuple(categories_present),
        shards_found=totals.shards_found,
        sequences_total=totals.sequences_total,
        sequences_written=totals.sequences_written,
        sequences_skipped=totals.sequences_skipped,
        frames_written=totals.frames_written,
        inventory_folder=inventory_folder,
        by_task=by_task,
        warnings=tuple(totals.warnings),
        layout_warnings=tuple(layout_warnings),
    )
