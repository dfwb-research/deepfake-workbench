"""AV-Deepfake1M++: the labelled validation split of a large audio-visual lip-sync deepfake set.

Layout, relative to the ``AV-Deepfake1M++`` folder (a single version, no compression levels). The
release's validation archive unpacks to three source-corpus folders, and every task's videos sit
side by side in them, in one folder per utterance:

* ``vox_celeb_2/<speaker>/<YouTube id>/<utterance>/<name>.mp4``;
* ``lrs3/<clip id>/<utterance>/<name>.mp4``;
* ``silent_videos/subject_<n>_<hash>_vid_<n>_<n>/<name>.mp4``.

The corpus folders may sit in a content folder: the first of ``raw_content/``, the dataset folder
itself and ``val/`` that holds a ``vox_celeb_2`` folder, in any copy of the dataset folder, is
used, and the dataset folder itself, with a warning, when none does. Moving the corpus folders
below ``raw_content/`` keeps later steps that write into the dataset folder from ever mixing their
output with the raw videos.

A folder cannot say which task a video belongs to, so nothing is scanned: the records come from
the release's metadata, ``val_metadata.json`` (kept as ``.official_files/val_metadata.json``), one
per row, whether or not the video is on disk yet. A row's ``file`` is its video's path below the
content folder, and its key is that path without the suffix: every part has each run of
characters other than ASCII letters and digits replaced by ``_`` and is trimmed of ``_``, and the
parts are joined by ``__`` (``vox_celeb_2/id01358/_1nATum8x78/00030/real.mp4`` is
``vox_celeb_2__id01358__1nATum8x78__00030__real``). File names repeat across utterances, so the
whole path is needed.

The task is the row's ``(modify_type, video_model)``: ``real`` is RVRA, ``audio_modified`` RVFA,
``visual_modified`` with Talklip LS_TALKLIP and with diff2lip LS_D2L, and ``both_modified`` with
Talklip LS_TALKLIP_TTS. Any other combination is an error naming it, so a generator a later
release adds is never folded into an existing task by default.

The identity is the person on screen, named after the source corpus so that ids from two corpora
never collide: ``vox2:<speaker folder>``, ``lrs3:<clip id>``, and ``silent:subject_<n>`` (every
clip of a subject shares it). The target is the identity's part after ``:``; there is no source,
because lip sync changes the mouth only. ``vox2_id`` is the speaker folder when it is a VoxCeleb2
id (``id`` and four or five digits), else none. Any other corpus is an error.

A row's ``original`` means different things by row. A path inside a corpus folder names, for a
fake, the real it was made from (its ``pair_key``, the key of that path) and, for a visually real
row (RVRA or RVFA), the real it derives from, such as the whole clip a ``_p1`` part was cut from
(its ``parent_key``). Any other value is the upstream video (``upstream_source``, e.g.
``VoxCeleb2/dev/mp4/...``). The other attributes come from the row as they are: the generators,
the fake segments (as start and end seconds) and their counts, and the frame counts.
``audio_relpath`` is the video's own relpath when it has audio frames, since the audio track is
inside the video; it is absent otherwise.

The release labels its validation split only: the official scheme is val. Real video with fake
audio is real under the visual label and fake under the audiovisual one.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import (
    BaseBuilder,
    LabelSpec,
    SchemeSpec,
    TaskSpec,
    has_videos,
    validate_compressions,
)
from dfwb.protocols.rules import BenchmarkSpec, Split, local_key

__all__ = ["AVDeepfake1MPPBuilder"]

_log = logging.getLogger(__name__)

# The release's metadata file, relative to the dataset folder.
_METADATA: Final = ".official_files/val_metadata.json"
_HINT: Final = (
    "use the release's val_metadata.json as downloaded: a JSON list of objects, each with its "
    "'file'"
)
_VOX2: Final = "vox_celeb_2"
_LRS3: Final = "lrs3"
_SILENT: Final = "silent_videos"
_CORPORA: Final = (_VOX2, _LRS3, _SILENT)
# Where the corpus folders may sit, in the order they are looked for ("" is the dataset folder).
_PREFIXES: Final = ("raw_content", "", "val")
_VOX2_ID: Final = re.compile(r"^id\d{4,5}$")
_SUBJECT: Final = re.compile(r"^(subject_\d+)")
_NOT_ALPHANUMERIC: Final = re.compile(r"[^A-Za-z0-9]+")
# The metadata's split names; a key listed under several takes the first of train, val and test.
_SPLITS: Final[dict[str, Split]] = {"train": "train", "dev": "val", "val": "val", "test": "test"}
_LOOKUP_ORDER: Final[tuple[Split, ...]] = ("train", "val", "test")
# The release labels only its validation split, so that is the only one published.
_PUBLISHED: Final[Split] = "val"
# The tasks whose picture is untouched: an in-dataset original is their parent, never a pair.
_VISUALLY_REAL: Final = frozenset({"RVRA", "RVFA"})

# (modify_type, video_model) -> task abbr.
_TASK_OF: Final[dict[tuple[str | None, str | None], str]] = {
    ("real", None): "RVRA",
    ("audio_modified", None): "RVFA",
    ("visual_modified", "Talklip"): "LS_TALKLIP",
    ("visual_modified", "diff2lip"): "LS_D2L",
    ("both_modified", "Talklip"): "LS_TALKLIP_TTS",
}


def _task(abbr: str, name: str, kind: Literal["real", "fake"], method: str) -> TaskSpec:
    """A task: its videos share the content folder with every other task's."""
    return TaskSpec(abbr, name, kind, "", method, recursive=True)


def _sanitise(text: str) -> str:
    """Every run of characters other than ASCII letters and digits becomes ``_``; trimmed."""
    return _NOT_ALPHANUMERIC.sub("_", text).strip("_")


def _video_key(file: str) -> str:
    """A metadata ``file`` as a key: its parts without the suffix, sanitised, joined by ``__``."""
    return "__".join(_sanitise(part) for part in PurePosixPath(file).with_suffix("").parts)


def _identity(file: str) -> tuple[str, str | None, str]:
    """``(identity, VoxCeleb2 id or None, source corpus)`` of a metadata ``file``.

    Raises:
        ContractError: the file is in none of the three source corpora.
    """
    parts = PurePosixPath(file).parts
    corpus = parts[0] if parts else ""
    if corpus not in _CORPORA:
        raise ContractError(
            f"av-deepfake1m-pp: unknown source corpus {corpus!r} in {file!r}",
            hint="the release's files are under " + ", ".join(_CORPORA) + "; a new corpus changes "
            "what an identity means, so it needs a builder update",
        )
    folder = parts[1] if len(parts) > 1 else ""
    if corpus == _VOX2:
        return f"vox2:{folder}", (folder if _VOX2_ID.match(folder) else None), corpus
    if corpus == _LRS3:
        return f"lrs3:{folder}", None, corpus
    match = _SUBJECT.match(folder)
    return f"silent:{match.group(1) if match else folder}", None, corpus


def _task_abbr(modify_type: Any, video_model: Any) -> str:
    """The task of a row's ``(modify_type, video_model)``.

    Raises:
        ContractError: the combination is not a known task.
    """
    abbr = _TASK_OF.get((modify_type, video_model))
    if abbr is None:
        known = ", ".join(f"({m}, {v})" for m, v in _TASK_OF)
        raise ContractError(
            f"av-deepfake1m-pp: unknown task (modify_type={modify_type!r}, "
            f"video_model={video_model!r})",
            hint=f"known combinations: {known}; a new generator needs a builder update",
        )
    return abbr


def _original_roles(
    original: Any, visually_real: bool
) -> tuple[str | None, str | None, str | None]:
    """``(pair_key, upstream_source, parent_key)``: what a row's ``original`` means for it."""
    if not isinstance(original, str) or not original:
        return None, None, None
    parts = PurePosixPath(original).parts
    if (parts[0] if parts else "") not in _CORPORA:
        return None, original, None  # the upstream video, spelt as upstream spells it
    if visually_real:
        return None, None, _video_key(original)  # the real this row derives from
    return _video_key(original), None, None  # the real this fake was made from


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Every row of the metadata file.

    Raises:
        ContractError: the file cannot be read as JSON, is not a list, or holds an entry that is
            not an object with a non-empty text ``file``.
    """
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ContractError(
            f"av-deepfake1m-pp: cannot read the metadata file {path}: {exc}", hint=_HINT
        ) from None
    if not isinstance(rows, list):
        raise ContractError(
            f"av-deepfake1m-pp: {path} holds a {type(rows).__name__}, not a list", hint=_HINT
        )
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("file"), str) or not row["file"]:
            raise ContractError(
                f"av-deepfake1m-pp: {path}: entry {index} is not an object with a text 'file'",
                hint=_HINT,
            )
    return rows


def _prefixed(prefix: str, path: str) -> str:
    return f"{prefix}/{path}" if prefix else path


def _content_prefix(copies: Sequence[Path]) -> str:
    """The content folder: the first of ``_PREFIXES`` whose ``vox_celeb_2`` exists in any copy."""
    for prefix in _PREFIXES:
        for copy in copies:
            if (copy / _prefixed(prefix, _VOX2)).exists():
                _log.debug("av-deepfake1m-pp: the corpus folders are under %r", prefix or ".")
                return prefix
    _log.warning(
        "av-deepfake1m-pp: no %s folder under %s (the archive not unpacked yet?); the relpaths "
        "assume the corpus folders sit in the dataset folder itself, so rebuild the inventory "
        "once they are unpacked",
        _VOX2,
        ", ".join(str(copy) for copy in copies),
    )
    return ""


class AVDeepfake1MPPBuilder(BaseBuilder):
    """AV-Deepfake1M++ (``av-deepfake1m-pp``): real clips and lip-sync fakes, from the metadata."""

    dataset_id = "av-deepfake1m-pp"
    expected_folder = "AV-Deepfake1M++"
    label_prefix = "AVDF1MPP"
    tasks = (
        _task("RVRA", "RealVideo-RealAudio", "real", "original"),
        _task("LS_TALKLIP", "FakeVideo-RealAudio-Talklip", "fake", "Talklip"),
        _task("LS_TALKLIP_TTS", "FakeVideo-FakeAudio-Talklip", "fake", "Talklip"),
        _task("LS_D2L", "FakeVideo-RealAudio-Diff2Lip", "fake", "diff2lip"),
        _task("RVFA", "RealVideo-FakeAudio", "fake", "tts_audio_only"),
    )
    metadata_files = (_METADATA,)
    labels = {
        "RVRA": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "LS_TALKLIP": LabelSpec(binary=1, binary_av=1, multiclass=2, family="lip-sync"),
        "LS_TALKLIP_TTS": LabelSpec(binary=1, binary_av=1, multiclass=3, family="lip-sync"),
        "LS_D2L": LabelSpec(binary=1, binary_av=1, multiclass=4, family="lip-sync"),
        # The picture is the real one: real to a visual detector, fake once audio counts.
        "RVFA": LabelSpec(binary=0, binary_av=1, multiclass=1, family="audio-only"),
    }
    schemes = {
        "official": SchemeSpec(
            "official",
            "official",
            source=f"the split of each row of the release's metadata ({_METADATA})",
            rationale="the publisher's only labelled split, val, as published: the dataset is "
            "for evaluation, so no train or test is carved from it",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set drawn from every video (there is no "
            "official test): up to 100 fakes per task, so the few diff2lip fakes are never "
            "crowded out by Talklip, balanced with as many reals (real video with fake audio "
            "counting as real)",
        ),
    }
    default_scheme = "official"
    benchmark = BenchmarkSpec(k_fake=100, strata=("task",))
    pairing_rule = "original-video"
    card_info = {
        "name": "AV-Deepfake1M++",
        "aliases": ["AVDF1MPP", "AV-Deepfake1M-PlusPlus"],
        "release": "the validation split of the 2025 1M-Deepfakes Detection Challenge dataset: "
        "77,326 labelled videos (20,220 real, 19,099 with Talklip or diff2lip lip sync, 18,938 "
        "with synthesised audio only and 19,069 with both) from VoxCeleb2, LRS3 and a silent-video "
        "set, with the release's val_metadata.json giving each video's manipulation, generators "
        "and fake segments",
        "homepage": "https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus",
        "license": {
            "spdx": None,
            "summary": "the AV-Deepfake1M++ EULA, signed when registering for the 2025 "
            "1M-Deepfakes Detection Challenge; the terms need review",
            "url": None,
        },
        "access": "register for the 2025 1M-Deepfakes Detection Challenge and sign its EULA, "
        "then download the val archive and val_metadata.json from the gated Hugging Face "
        "dataset; dfwb never distributes media",
        "modalities": ["video", "audio"],
        "compressions": None,
        "key_rule": "<task>/<path>: the video's path in the release without its suffix, each part "
        "sanitised to letters, digits and '_' and the parts joined by '__', e.g. "
        "RVRA/vox_celeb_2__id01358__1nATum8x78__00030__real",
    }
    layout_notes = (
        "Nothing is scanned: every row is a record, whether or not its video is on disk yet, and "
        "a row whose modify_type and video_model are not one of the tasks above is an error.\n"
        "The release's val archive unpacks to vox_celeb_2/, lrs3/ and silent_videos/, which may "
        "stay in the dataset folder or move together under val/ or raw_content/ (keeping them "
        "apart from anything later written into the dataset folder); the content folder is "
        "looked for in every copy of the dataset folder, and every copy must use the same one."
    )

    _rows: list[dict[str, Any]] | None = None

    def prepare(self, root: Path) -> None:
        """Read the release's metadata under ``root`` once per build.

        Raises:
            ContractError: the metadata file is malformed.
        """
        super().prepare(root)
        path = root / _METADATA
        if not path.is_file():
            _log.warning(
                "av-deepfake1m-pp: %s is missing from %s; without the release's val_metadata.json "
                "there are no records",
                _METADATA,
                root,
            )
            self._rows = []
            return
        self._rows = _read_rows(path)
        _log.debug("av-deepfake1m-pp: the metadata lists %d videos", len(self._rows))

    # ----------------------------------------------------------------------------- layout

    def layout_dirs(self) -> tuple[str, ...]:
        """Each corpus folder, in each place the content folder may be."""
        return tuple(_prefixed(prefix, corpus) for prefix in _PREFIXES for corpus in _CORPORA)

    def videos_present(self, folder: Path) -> tuple[str, ...]:
        """Which of :meth:`layout_dirs` hold at least one video under ``folder``."""
        return tuple(d for d in self.layout_dirs() if has_videos(folder / d, recursive=True))

    def _holds_videos(self, folder: Path) -> bool:
        return any(has_videos(folder / d, recursive=True) for d in self.layout_dirs())

    def video_copy_for(
        self, copies: Sequence[Path], task: TaskSpec, compression: str | None
    ) -> Path | None:
        """The first of ``copies`` holding a video in a corpus folder: every task shares them."""
        return next((copy for copy in copies if self._holds_videos(copy)), None)

    def copies_after(
        self, copies: Sequence[Path], chosen: Path, task: TaskSpec, compression: str | None
    ) -> list[Path]:
        """Every one of ``copies`` after ``chosen`` that also holds a video in a corpus folder."""
        index = list(copies).index(chosen)
        return [copy for copy in copies[index + 1 :] if self._holds_videos(copy)]

    # ----------------------------------------------------------------------------- discovery

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        """Yield one record per metadata row, in the metadata's order.

        Raises:
            ConfigError: ``compressions`` names a compression (this dataset has none).
            ContractError: a row's file is in no known source corpus, or its
                ``(modify_type, video_model)`` is not a known task.
        """
        validate_compressions(compressions, self.known_compressions)
        rows = self._rows or []
        if not rows:
            return
        prefix = _content_prefix(self._copies if self._copies is not None else (root,))
        tasks = {task.abbr: task for task in self.tasks}
        for row in rows:
            yield self._record(tasks, row, prefix)

    def _record(
        self, tasks: Mapping[str, TaskSpec], row: Mapping[str, Any], prefix: str
    ) -> InventoryRecord:
        file = str(row["file"])
        identity, vox2_id, corpus = _identity(file)
        abbr = _task_abbr(row.get("modify_type"), row.get("video_model"))
        pair_key, upstream_source, parent_key = _original_roles(
            row.get("original"), abbr in _VISUALLY_REAL
        )
        relpath = _prefixed(prefix, file)
        fake_segments = row.get("fake_segments") or []
        visual = row.get("visual_fake_segments") or []
        audio = row.get("audio_fake_segments") or []
        attrs: dict[str, Any] = {
            "vox2_id": vox2_id,
            "source_corpus": corpus,
            "original_raw": row.get("original"),
            "upstream_source": upstream_source,
            "parent_key": parent_key,
            "modify_type": row.get("modify_type"),
            "video_model": row.get("video_model"),
            "audio_model": row.get("audio_model"),
            "modify_video": abbr not in _VISUALLY_REAL,
            "modify_audio": bool(audio) or row.get("audio_model") is not None,
            "fake_segments": fake_segments,
            "visual_fake_segments": visual,
            "audio_fake_segments": audio,
            "n_fakes": len(fake_segments),
            "n_visual_fakes": len(visual),
            "n_audio_fakes": len(audio),
            "video_frames": row.get("video_frames"),
            "audio_frames": row.get("audio_frames"),
        }
        if row.get("audio_frames"):
            attrs["audio_relpath"] = relpath  # the audio track is inside the video
        return self.record(
            tasks[abbr],
            _video_key(file),
            relpath,
            None,
            identity=identity,
            target_id=identity.split(":", 1)[1],
            pair_key=pair_key,
            attrs=attrs,
        )

    # ----------------------------------------------------------------------------- hooks

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record whose key the metadata lists first under val (train, val, test order).

        The release labels its validation split only, so val is the one split published; a key
        listed under train first (none is, in the release) gets no split.

        Raises:
            ConfigError: the metadata file is missing.
            ContractError: the metadata file is malformed.
        """
        path = root / _METADATA
        if not path.is_file():
            raise ConfigError(
                f"av-deepfake1m-pp: the metadata file {_METADATA} is missing from {root}",
                hint="copy the release's val_metadata.json into .official_files/ in the dataset "
                "folder",
            )
        listed: dict[Split, set[str]] = {split: set() for split in _LOOKUP_ORDER}
        for row in _read_rows(path):
            split = _SPLITS.get(str(row.get("split", "")).strip().lower())
            if split is not None:
                listed[split].add(_video_key(row["file"]))
        assignment: dict[str, Split] = {}
        for record in records:
            key = local_key(record.key)
            first = next((split for split in _LOOKUP_ORDER if key in listed[split]), None)
            if first == _PUBLISHED:
                assignment[record.key] = first
        return assignment

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The real the fake was made from (its row's ``original``)."""
        return fake.pair_key

    # ----------------------------------------------------------------------------- text

    def describe_layout(self) -> str:
        """The expected layout: the metadata, the content folder and the task of each row."""
        name = str(self.card_info["name"])
        by_task = {abbr: combination for combination, abbr in _TASK_OF.items()}
        width = max(len(task.abbr) for task in self.tasks)
        lines = [
            f"{name} ({self.dataset_id}): folder {self.expected_folder!r} under a datasets root.",
            f"Records come from the release's metadata, {_METADATA}; each row's file is a path "
            "below the content folder:",
            f"  {_VOX2}/<speaker>/<YouTube id>/<utterance>/<name>.mp4",
            f"  {_LRS3}/<clip id>/<utterance>/<name>.mp4",
            f"  {_SILENT}/subject_<n>_<hash>_vid_<n>_<n>/<name>.mp4",
            "The content folder is the first of raw_content/, the dataset folder itself and val/ "
            f"that holds {_VOX2}/.",
            "Tasks, by the row's modify_type and video_model:",
        ]
        for task in self.tasks:
            modify_type, video_model = by_task[task.abbr]
            model = f" + {video_model}" if video_model else ""
            lines.append(
                f"  {task.abbr.ljust(width)}  {task.kind:<4}  {modify_type}{model}  ({task.name})"
            )
        lines.append(f"Keys: {self.card_info['key_rule']}")
        lines.append(self.layout_notes)
        return "\n".join(lines)
