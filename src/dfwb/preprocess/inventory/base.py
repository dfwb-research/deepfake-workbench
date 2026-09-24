"""The inventory builder contract (C3b), a table-driven base class, and file-system helpers.

An inventory builder knows one dataset's raw layout: where each task's videos live, how a file
name becomes a key, and which identities, sources and targets a video carries. It turns the raw
release into :class:`~dfwb.core.records.InventoryRecord` rows. It also carries the rest of what is
specific to that dataset -- the official split reader, the pairing rule, the benchmark spec, the
label table and the dataset card defaults -- so the generic rules in :mod:`dfwb.protocols.rules`
never need to know about any one dataset.

Keys are ``<task>/<legacy key>``. The same video id can appear under several tasks (every fake
method of a dataset may reuse the target's id), so the task prefix keeps keys unique and says
which method a key belongs to, while the part after the first ``/`` stays the dataset's own id.

This module imports only the standard library, pydantic (for the dataset card and label
vocabulary models), :mod:`dfwb.core` and :mod:`dfwb.protocols.rules`: never torch, numpy or a
media library.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final, Literal, Protocol, get_args, runtime_checkable

from pydantic import ValidationError

from dfwb.core.errors import ConfigError, ContractError, did_you_mean, validation_messages
from dfwb.core.records import (
    BuilderRef,
    DatasetCard,
    InventoryRecord,
    LabelVocab,
    SchemeCard,
    VideoRecord,
)
from dfwb.protocols.rules import BenchmarkSpec, Split, task_of

__all__ = [
    "COMPRESSION_TOKEN",
    "SCHEME_KINDS",
    "SCHEME_RULES",
    "VIDEO_SUFFIXES",
    "BaseBuilder",
    "InventoryBuilder",
    "LabelSpec",
    "SchemeKind",
    "SchemeRule",
    "SchemeSpec",
    "TaskSpec",
    "expand_compressions",
    "scan_videos",
    "validate_compressions",
]

VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})

# Stands for the compression level in a task's video directory, e.g. "videos/{cX}".
COMPRESSION_TOKEN: Final = "{cX}"

SchemeRule = Literal["official", "official+ident-80-20", "ident-72-14-14", "all-test", "benchmark"]
SchemeKind = Literal["official", "derived", "subset"]
SCHEME_RULES: Final[tuple[str, ...]] = get_args(SchemeRule)
SCHEME_KINDS: Final[tuple[str, ...]] = get_args(SchemeKind)

# DatasetCard fields a builder never sets in ``card_info``: they come from the builder's scheme
# table, from the pack being built, or from a later terms review.
_CARD_FIELDS_NOT_IN_INFO: Final = frozenset(
    {"id", "schemes", "default_scheme", "distribution", "terms"}
)


# ---------------------------------------------------------------------------------------------
# Table entries
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """One task of a dataset: a real source or one generation method.

    Attributes:
        abbr: The task abbreviation, e.g. ``"FS_DF"``: the key prefix and the label-key suffix.
        name: The task's readable name, e.g. ``"Deepfakes"``; stored as ``attrs["task_name"]``.
        kind: ``"real"`` or ``"fake"``.
        video_dir: Where the task's videos are, relative to the dataset folder; ``{cX}`` stands
            for the compression level.
        method: The record's ``method`` (``"original"`` for reals); a builder may override it
            per record.
        extras: Other per-task locations a builder needs, e.g. ``{"audio_dir": ...}``.
        recursive: Whether the videos sit in sub-folders of ``video_dir`` (e.g. one folder per
            actor); the default is a flat folder.
    """

    abbr: str
    name: str
    kind: Literal["real", "fake"]
    video_dir: str
    method: str
    extras: Mapping[str, str] = field(default_factory=dict)
    recursive: bool = False

    def __post_init__(self) -> None:
        if not self.abbr or "/" in self.abbr or any(c.isspace() for c in self.abbr):
            raise ContractError(
                f"task abbr {self.abbr!r} is empty or holds '/' or a space",
                hint="the abbr is the key prefix, e.g. 'FS_DF'",
            )
        if self.kind not in ("real", "fake"):
            raise ContractError(
                f"task {self.abbr}: kind {self.kind!r} is not 'real' or 'fake'",
                hint="set kind='real' or kind='fake'",
            )


@dataclass(frozen=True)
class LabelSpec:
    """A task's labels, one per label mapping a pack publishes.

    Attributes:
        binary: The visual real/fake label (1 = fake). A fake whose picture is untouched (an
            audio-only fake) is 0 here, because a visual detector cannot see it.
        binary_av: The audiovisual real/fake label: any manipulated stream is 1.
        multiclass: The class index of the task in the dataset's multi-class table.
        family: The manipulation family: ``face-swap``, ``face-reenactment``, ``lip-sync``,
            ``talking-face``, ``talking-head``, ``real``, or another name the builder sets.
    """

    binary: int
    binary_av: int
    multiclass: int
    family: str

    def __post_init__(self) -> None:
        for name in ("binary", "binary_av"):
            if getattr(self, name) not in (0, 1):
                raise ContractError(
                    f"label {name}={getattr(self, name)!r} is not 0 or 1",
                    hint="binary labels are 0 (real) or 1 (fake)",
                )


@dataclass(frozen=True)
class SchemeSpec:
    """One split scheme a dataset supports, and the rule that produces it.

    Attributes:
        rule: The generic rule: ``official`` (the publisher's split),
            ``official+ident-80-20`` (the official test plus an identity-disjoint 80/20
            train/val carve), ``ident-72-14-14`` (an identity-disjoint carve),
            ``all-test`` (every video is test) or ``benchmark`` (a seeded, balanced test subset).
        kind: ``official``, ``derived`` or ``subset`` (the scheme card's kind).
        params: Rule parameters, e.g. ``{"policy": "test-only-official"}``.
        source: Where the official split comes from, when there is one.
        rationale: Why the scheme exists, in a sentence.
    """

    rule: SchemeRule
    kind: SchemeKind
    params: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if self.rule not in SCHEME_RULES:
            raise ContractError(
                f"unknown scheme rule {self.rule!r}{did_you_mean(self.rule, SCHEME_RULES)}",
                hint="rules: " + ", ".join(SCHEME_RULES),
            )
        if self.kind not in SCHEME_KINDS:
            raise ContractError(
                f"unknown scheme kind {self.kind!r}", hint="kinds: " + ", ".join(SCHEME_KINDS)
            )


# ---------------------------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------------------------


@runtime_checkable
class InventoryBuilder(Protocol):
    """The minimal shape of an inventory builder (contract C3b).

    This is the smallest interface an inventory builder has. The ``dfwb`` commands and pack
    building need more (a task table, labels, schemes and the dataset hooks), so a builder
    registered in ``inventory_builders`` subclasses :class:`BaseBuilder`. The three attributes are
    read-only here, so a class attribute, an attribute set in ``__init__`` or a property all
    satisfy the contract.
    """

    @property
    def dataset_id(self) -> str:
        """The registry key, e.g. ``"ffpp"``."""
        ...

    @property
    def version(self) -> str:
        """Bumped whenever discovery changes what it yields."""
        ...

    @property
    def expected_folder(self) -> str:
        """The dataset's folder name under a datasets root."""
        ...

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        """Yield one record per video (and compression) found under the dataset folder."""
        ...

    def describe_layout(self) -> str:
        """Human text describing the expected folder layout (``dfwb datasets info``)."""
        ...


# ---------------------------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------------------------


def _is_video(path: Path) -> bool:
    return (
        not path.name.startswith(".")
        and path.suffix.lower() in VIDEO_SUFFIXES
        and path.is_file()  # follows symlinks; a dangling link is not a video
    )


def _identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino)


def _walk(directory: Path, ancestors: frozenset[tuple[int, int]]) -> Iterator[Path]:
    for path in directory.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_dir():
            ident = _identity(path)
            if ident not in ancestors:  # a link back up the tree would loop forever
                yield from _walk(path, ancestors | {ident})
        elif _is_video(path):
            yield path


def scan_videos(directory: Path, *, recursive: bool = False) -> list[Path]:
    """The video files in ``directory``, sorted.

    A name starting with ``.`` (``.DS_Store``, editor and sync leftovers) is skipped, as is any
    file whose suffix is not in :data:`VIDEO_SUFFIXES` (matched ignoring case, so ``.MP4``
    counts). Symlinks are followed, so a symlinked video is found; a dangling one is not. With
    ``recursive=True``, sub-folders (including symlinked ones, but never one that links back up
    the tree) are searched too. A missing directory gives ``[]``: a task may be absent from a
    partial download, which ``dfwb protocols verify`` then reports.
    """
    if not directory.is_dir():
        return []
    if recursive:
        return sorted(_walk(directory, frozenset({_identity(directory)})))
    return sorted(path for path in directory.iterdir() if _is_video(path))


def validate_compressions(
    requested: Sequence[str] | None, known: Sequence[str]
) -> list[str] | None:
    """``requested`` checked against ``known``: in ``known`` order without repeats, or ``None``.

    Raises:
        ConfigError: a requested compression is not known (with a did-you-mean).
    """
    if requested is None:
        return None
    for value in requested:
        if value not in known:
            raise ConfigError(
                f"unknown compression {value!r}{did_you_mean(value, known)}",
                hint="known compressions: "
                + (", ".join(known) or "none (this dataset has a single version)"),
            )
    wanted = set(requested)
    return [c for c in known if c in wanted]


def expand_compressions(
    video_dir: str, known: Sequence[str], requested: Sequence[str] | None
) -> list[str | None]:
    """The compressions to scan a task's ``video_dir`` with.

    A ``video_dir`` without ``{cX}`` has a single version: ``[None]``. With ``{cX}``, the result
    is every ``known`` compression, or, when ``requested`` is given, the requested ones, always in
    ``known`` order.

    Raises:
        ConfigError: a requested compression is not known (with a did-you-mean).
    """
    wanted = validate_compressions(requested, known)
    if COMPRESSION_TOKEN not in video_dir:
        return [None]
    return list(known) if wanted is None else list(wanted)


def _concrete_dir(video_dir: str, compression: str | None) -> str:
    text = video_dir if compression is None else video_dir.replace(COMPRESSION_TOKEN, compression)
    return PurePosixPath(text).as_posix()


# ---------------------------------------------------------------------------------------------
# The base class
# ---------------------------------------------------------------------------------------------


class BaseBuilder:
    """A table-driven builder: subclasses set the class attributes and override hooks as needed.

    The default :meth:`discover` scans every task's ``video_dir`` (once per compression, and
    into sub-folders for a ``recursive`` task) and turns each video into a record with
    :meth:`record_for_video`, whose default keys a video by its file stem. Override
    ``record_for_video`` to parse identities from the file name, or ``discover`` itself for a
    layout that is not one folder per task.

    A builder that needs a metadata file (a CSV or JSON listing labels, identities or sources)
    reads it once per build in :meth:`prepare`, which the runner calls with the dataset folder
    before :meth:`discover`, and keeps what it needs on ``self`` for ``discover`` and
    ``record_for_video`` to use. A missing file should leave the builder with nothing to yield
    rather than raise: an empty or partial download is reported by ``dfwb protocols verify``.

    Class attributes:
        dataset_id: The registry key, e.g. ``"ffpp"``.
        version: Bumped whenever discovery changes what it yields.
        expected_folder: The dataset's folder name under a datasets root.
        label_prefix: The label-key prefix: ``"FF"`` gives ``"FF-FS_DF"``.
        tasks: The task table; its order is the task rank used to break ties.
        known_compressions: The values ``{cX}`` can take, in order.
        labels: Each task's :class:`LabelSpec`, by abbr; it covers every task.
        schemes: The split schemes the dataset supports, by name.
        default_scheme: The scheme used when a protocol names none.
        benchmark: The benchmark subset's spec (set exactly when a ``benchmark`` scheme exists).
        pairing_rule: The name recorded on each fake/real pair; ``None`` if nothing pairs.
        pairing_fanout: The most reals one fake pairs with through an identity (``None``: all).
        card_info: The dataset card's descriptive fields: name, aliases, release, homepage,
            paper, license, access, modalities, compressions and key_rule.
        layout_notes: Extra text appended to :meth:`describe_layout`.
    """

    dataset_id: ClassVar[str]
    version: ClassVar[str] = "1"
    expected_folder: ClassVar[str]
    label_prefix: ClassVar[str]
    tasks: ClassVar[tuple[TaskSpec, ...]]
    known_compressions: ClassVar[tuple[str, ...]] = ()
    labels: ClassVar[Mapping[str, LabelSpec]]
    schemes: ClassVar[Mapping[str, SchemeSpec]]
    default_scheme: ClassVar[str]
    benchmark: ClassVar[BenchmarkSpec | None] = None
    pairing_rule: ClassVar[str | None] = None
    pairing_fanout: ClassVar[int | None] = None
    card_info: ClassVar[Mapping[str, Any]]
    layout_notes: ClassVar[str] = ""

    # ----------------------------------------------------------------------------- discovery

    def prepare(self, root: Path) -> None:
        """Load whatever the build needs once (e.g. a metadata file); the default does nothing.

        Called by the runner with the dataset folder, once per build, before :meth:`discover`.
        """

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        """Yield a record for every video of every task, once per compression.

        A ``recursive`` task's videos may sit in sub-folders; their relpath keeps that path.

        Raises:
            ConfigError: ``compressions`` names a compression this dataset does not have.
        """
        for task in self.tasks:
            for compression in expand_compressions(
                task.video_dir, self.known_compressions, compressions
            ):
                video_dir = _concrete_dir(task.video_dir, compression)
                base = root / video_dir
                for path in scan_videos(base, recursive=task.recursive):
                    relpath = f"{video_dir}/{path.relative_to(base).as_posix()}"
                    record = self.record_for_video(task, path, relpath, compression)
                    if record is not None:
                        yield record

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """The record for one scanned video, or ``None`` to skip it (a stray file).

        The default keys the video by its file stem and sets nothing else.
        """
        return self.record(task, path.stem, relpath, compression)

    def record(
        self,
        task: TaskSpec,
        legacy_key: str,
        relpath: str,
        compression: str | None,
        *,
        identity: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        pair_key: str | None = None,
        attrs: Mapping[str, Any] | None = None,
        folder: str | None = None,
        method: str | None = None,
    ) -> InventoryRecord:
        """Build one record: key ``<abbr>/<legacy_key>``, label key ``<prefix>-<abbr>``.

        ``attrs`` always gets ``task_name``. ``method`` defaults to the task's method.
        ``relpath`` is relative to the dataset folder, or to ``folder`` -- the name of another
        dataset's folder under the same datasets root -- when the video lives there.

        Raises:
            ContractError: ``legacy_key`` is empty, ``folder`` is not a single folder name, or
                ``relpath`` is not a relative POSIX path.
        """
        if not legacy_key:
            raise ContractError(
                f"{self.dataset_id}: empty key for {relpath!r} in task {task.abbr}",
                hint="every video needs a non-empty key",
            )
        if folder is not None and (folder in ("", ".", "..") or "/" in folder or "\\" in folder):
            raise ContractError(
                f"{self.dataset_id}: folder {folder!r} of {task.abbr}/{legacy_key} is not a "
                "single folder name",
                hint="folder names another dataset's folder, e.g. 'FaceForensics++'",
            )
        return InventoryRecord(
            key=f"{task.abbr}/{legacy_key}",
            compression=compression,
            label_key=self.label_key(task),
            method=task.method if method is None else method,
            relpath=relpath,
            builder=BuilderRef(self.dataset_id, self.version),
            identity=identity,
            source_id=source_id,
            target_id=target_id,
            pair_key=pair_key,
            attrs={"task_name": task.name, **(attrs or {})},
            folder=folder,
        )

    def label_key(self, task: TaskSpec) -> str:
        """``<label_prefix>-<abbr>``, e.g. ``FF-FS_DF``."""
        return f"{self.label_prefix}-{task.abbr}"

    # ----------------------------------------------------------------------------- hooks

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """The publisher's split: record key -> split, read from files under ``root``.

        Raises:
            ContractError: the dataset has no official split (the default).
        """
        raise ContractError(
            f"{self.dataset_id} has no official split",
            hint="use a derived scheme, e.g. ident-72-14-14 or all-test",
        )

    def pair_candidates(self, fake: InventoryRecord) -> str | Sequence[str] | None:
        """What a fake pairs with: real local keys or identities (default: nothing)."""
        return None

    def _label_spec(self, task: str, where: str) -> LabelSpec:
        try:
            return self.labels[task]
        except KeyError:
            raise ContractError(
                f"{self.dataset_id}: {where} has task {task!r}, which has no label",
                hint="labels: " + ", ".join(self.labels),
            ) from None

    def is_real(self, record: InventoryRecord | VideoRecord) -> bool:
        """Whether ``record`` is real under the visual binary label.

        Raises:
            ContractError: the record's task has no label.
        """
        return self._label_spec(task_of(record.key), repr(record.key)).binary == 0

    def task_rank(self) -> dict[str, int]:
        """Each task's position in the task table (used to break ties between equal keys)."""
        return {task.abbr: index for index, task in enumerate(self.tasks)}

    # ----------------------------------------------------------------------------- pack cards

    def label_vocab(self) -> LabelVocab:
        """The label vocabulary: one entry per task, and the four label mappings.

        Raises:
            ContractError: a task has no label.
        """
        vocab: dict[str, dict[str, str | int | float | bool | None]] = {}
        for task in self.tasks:
            spec = self._label_spec(task.abbr, "the task table")
            vocab[self.label_key(task)] = {
                "binary": spec.binary,
                "binary_av": spec.binary_av,
                "multiclass": spec.multiclass,
                "family": spec.family,
                "method": task.method,
                "task": task.abbr,
            }
        mappings = {
            "binary": {"from": "binary"},
            "audiovisual-binary": {"from": "binary_av"},
            "multiclass": {"from": "multiclass"},
            "family": {"from": "family"},
        }
        return LabelVocab.model_validate({"vocab": vocab, "mappings": mappings})

    def dataset_card(self, schemes: Mapping[str, SchemeCard]) -> DatasetCard:
        """The dataset card for a pack holding ``schemes``; its distribution is undecided.

        Raises:
            ContractError: ``schemes`` holds a scheme this builder does not define or lacks the
                default scheme, or ``card_info`` is not a valid card.
        """
        unknown = sorted(set(schemes) - set(self.schemes))
        if unknown:
            raise ContractError(
                f"{self.dataset_id}: unknown scheme(s) {unknown}",
                hint="schemes: " + ", ".join(self.schemes),
            )
        if self.default_scheme not in schemes:
            raise ContractError(
                f"{self.dataset_id}: the default scheme {self.default_scheme!r} is missing",
                hint="a dataset card always holds its default scheme",
            )
        reserved = sorted(_CARD_FIELDS_NOT_IN_INFO & set(self.card_info))
        if reserved:
            raise ContractError(
                f"{self.dataset_id}: card_info sets {reserved}",
                hint="card_info holds only the descriptive fields of the card",
            )
        data = {
            **self.card_info,
            "id": self.dataset_id,
            "schemes": dict(schemes),
            "default_scheme": self.default_scheme,
            "distribution": "undecided",
        }
        try:
            return DatasetCard.model_validate(data)
        except ValidationError as exc:
            raise ContractError(
                f"{self.dataset_id}: invalid dataset card: " + "; ".join(validation_messages(exc)),
                hint="fix the builder's card_info",
            ) from None

    def describe_layout(self) -> str:
        """The expected folder layout, generated from the task table (plus ``layout_notes``)."""
        name = str(self.card_info.get("name", self.dataset_id))
        lines = [
            f"{name} ({self.dataset_id}): folder {self.expected_folder!r} under a datasets root.",
            "Videos, relative to that folder:",
        ]
        width = max((len(task.abbr) for task in self.tasks), default=0)
        for task in self.tasks:
            where = (
                f"{task.video_dir}/**/<video>" if task.recursive else f"{task.video_dir}/<video>"
            )
            note = "; searched recursively" if task.recursive else ""
            lines.append(
                f"  {task.abbr.ljust(width)}  {task.kind:<4}  {where}  ({task.name}{note})"
            )
        if any(COMPRESSION_TOKEN in task.video_dir for task in self.tasks):
            known = ", ".join(self.known_compressions) or "none declared"
            lines.append(f"{COMPRESSION_TOKEN} is the compression: {known}.")
        key_rule = self.card_info.get("key_rule")
        if key_rule:
            lines.append(f"Keys: {key_rule}")
        if self.layout_notes.strip():
            lines.append(self.layout_notes.strip())
        return "\n".join(lines)
