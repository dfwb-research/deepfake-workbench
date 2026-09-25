"""LAV-DF: VoxCeleb2 clips and fakes of them with a few words changed in the audio, lips or both.

Layout, relative to the ``LAV-DF`` folder (a single version, no compression levels). The tasks are
named, as in FakeAVCeleb, by which streams were manipulated, and each task folder is searched
recursively:

* reals: ``original_content/RealVideo-RealAudio/videos/<id>.mp4``;
* fakes: ``manipulated_content/<category>/videos/<id>.mp4`` for ``FakeVideo-FakeAudio``,
  ``FakeVideo-RealAudio`` and ``RealVideo-FakeAudio``.

The release does not ship these folders: it unpacks to a ``LAV-DF`` folder holding flat
``train/``, ``dev/`` and ``test/`` folders, real and fake videos side by side, beside
``metadata.json`` and ``metadata.min.json``. That folder is kept as ``.official_files/LAV-DF``,
and its videos may stay where they are or be moved into the task folders above. A video left in
``.official_files/LAV-DF/{train,dev,test}/`` takes the task its metadata row's ``modify_video``
and ``modify_audio`` name, which is the folder moving it would put it in: neither is
RealVideo-RealAudio, both FakeVideo-FakeAudio, the video alone FakeVideo-RealAudio and the audio
alone RealVideo-FakeAudio. One without a row, or whose row does not give both flags as true or
false, has no known task and is skipped: it is never labelled by default.

Every video is keyed by its file stem, a six-digit id unique across the release. Its attributes
come from its row of the release's ``metadata.json`` (``metadata.min.json`` when the full file is
absent), looked up by stem: ``n_fakes`` and ``fake_periods`` (the fake segments, as start and end
seconds), ``modify_video``, ``modify_audio``, ``duration``, ``video_frames``, ``audio_channels``
and ``audio_frames``. The transcript and its word timestamps are not copied. A video without a row
has ``n_fakes`` 0, no fake periods and none of the rest. No row names a speaker, so identity,
target and source are none. A fake's ``pair_key`` is the stem of its row's ``original``, the real
video it was made from, and it pairs with that real.

The official split is each row's ``split``: ``train``, ``dev`` (published here as val) or
``test``. Real video with fake audio is real under the visual label and fake under the audiovisual
one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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
    scan_videos,
)
from dfwb.protocols.rules import BenchmarkSpec, Split, local_key

__all__ = ["LAVDFBuilder"]

_log = logging.getLogger(__name__)

# The release's folder, as it unpacks, relative to the dataset folder, and its metadata files.
_RELEASE: Final = ".official_files/LAV-DF"
_METADATA: Final = f"{_RELEASE}/metadata.json"
_METADATA_MIN: Final = f"{_RELEASE}/metadata.min.json"
# The release's own video folders, one per split, in the order they are scanned.
_RELEASE_DIRS: Final = tuple(f"{_RELEASE}/{split}" for split in ("train", "dev", "test"))
_HINT: Final = (
    "use the release's metadata.json (or metadata.min.json) as downloaded: a JSON list of "
    "objects, each with its 'file'"
)
# The metadata's split names, and the order a stem listed under two is looked up in.
_SPLITS: Final[dict[str, Split]] = {"train": "train", "dev": "val", "val": "val", "test": "test"}
_LOOKUP_ORDER: Final[tuple[Split, ...]] = ("train", "val", "test")
# (modify_video, modify_audio) of a row: the task its video belongs to.
_TASK_OF_FLAGS: Final = {
    (False, False): "RVRA",
    (True, True): "FVFA",
    (True, False): "FVRA",
    (False, True): "RVFA",
}
# The row fields copied into a record's attributes as they are (None when absent).
_COPIED: Final = (
    "modify_video",
    "modify_audio",
    "duration",
    "video_frames",
    "audio_channels",
    "audio_frames",
)


@dataclass(frozen=True, slots=True)
class _Row:
    """What a video takes from its metadata row; ``task`` is None when a flag is missing."""

    task: str | None
    pair_key: str | None
    attrs: Mapping[str, Any]


def _task(abbr: str, name: str, kind: Literal["real", "fake"]) -> TaskSpec:
    """A category folder: its name is the task's, and a fake's method."""
    if kind == "real":
        return TaskSpec(
            abbr, name, kind, f"original_content/{name}/videos", "original", recursive=True
        )
    return TaskSpec(abbr, name, kind, f"manipulated_content/{name}/videos", name, recursive=True)


def _metadata_path(root: Path) -> Path | None:
    """The release's full metadata file under ``root``, else its minified one, else None."""
    for name in (_METADATA, _METADATA_MIN):
        path = root / name
        if path.is_file():
            return path
    return None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Every row of a metadata file.

    Raises:
        ContractError: the file cannot be read as JSON, is not a list, or holds an entry that is
            not an object with a text ``file``.
    """
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ContractError(
            f"lav-df: cannot read the metadata file {path}: {exc}", hint=_HINT
        ) from None
    if not isinstance(rows, list):
        raise ContractError(f"lav-df: {path} holds a {type(rows).__name__}, not a list", hint=_HINT)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("file"), str):
            raise ContractError(
                f"lav-df: {path}: entry {index} is not an object with a text 'file'", hint=_HINT
            )
    return rows


def _stem(file: str) -> str:
    """A metadata ``file`` (``<split>/<id>.mp4``) as a key: its stem."""
    return PurePosixPath(file).stem


def _row(row: Mapping[str, Any]) -> _Row:
    """What a metadata row gives its video: its task, ``pair_key`` and attributes."""
    original = row.get("original")
    attrs = {
        "n_fakes": row.get("n_fakes", 0),
        "fake_periods": row.get("fake_periods", []),
        **{name: row.get(name) for name in _COPIED},
    }
    video, audio = row.get("modify_video"), row.get("modify_audio")
    task = (
        _TASK_OF_FLAGS[(video, audio)]
        if isinstance(video, bool) and isinstance(audio, bool)
        else None
    )
    return _Row(
        task=task,
        pair_key=_stem(str(original)) if original else None,
        attrs=attrs,
    )


def _no_row_attrs() -> dict[str, Any]:
    """The attributes of a video without a metadata row."""
    return {"n_fakes": 0, "fake_periods": [], **dict.fromkeys(_COPIED)}


class LAVDFBuilder(BaseBuilder):
    """LAV-DF (``lav-df``): real clips and fakes with changed words, by manipulated stream."""

    dataset_id = "lav-df"
    expected_folder = "LAV-DF"
    label_prefix = "LAVDF"
    tasks = (
        _task("RVRA", "RealVideo-RealAudio", "real"),
        _task("FVFA", "FakeVideo-FakeAudio", "fake"),
        _task("FVRA", "FakeVideo-RealAudio", "fake"),
        _task("RVFA", "RealVideo-FakeAudio", "fake"),
    )
    metadata_files = (_METADATA,)
    labels = {
        "RVRA": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FVFA": LabelSpec(binary=1, binary_av=1, multiclass=2, family="video-and-audio"),
        "FVRA": LabelSpec(binary=1, binary_av=1, multiclass=3, family="video-only"),
        # The picture is the real one: real to a visual detector, fake once audio counts.
        "RVFA": LabelSpec(binary=0, binary_av=1, multiclass=1, family="audio-only"),
    }
    schemes = {
        "official": SchemeSpec(
            "official",
            "official",
            source=f"the split of each video's row in the release's metadata ({_METADATA})",
            rationale="the publisher's train/dev/test split, with dev published as val",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set drawn from the official test: up to 100 "
            "fakes per category, balanced with as many reals (real video with fake audio "
            "counting as real)",
        ),
    }
    default_scheme = "official"
    benchmark = BenchmarkSpec(k_fake=100, strata=("task",))
    pairing_rule = "original-video"
    card_info = {
        "name": "LAV-DF",
        "aliases": ["LAVDF", "Localized Audio Visual DeepFake"],
        "release": "136,304 videos of 153 VoxCeleb2 speakers: 36,431 real and 99,873 fakes in "
        "which a few words were changed with synthesised speech, re-synced lips or both "
        "(33,160 with fake video and fake audio, 33,543 with fake video only and 33,170 with "
        "fake audio only), in train, dev and test folders of 78,703, 31,501 and 26,100 videos, "
        "with metadata giving each fake's source video and fake segments",
        "homepage": "https://github.com/ControlNet/LAV-DF",
        "paper": {
            "title": "Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and "
            "Multimodal Method for Temporal Forgery Localization",
            "venue": "DICTA",
            "year": 2022,
            "doi": "10.1109/DICTA56598.2022.10034605",
        },
        "license": {
            "spdx": None,
            "summary": "the LAV-DF terms and conditions: non-commercial research and education "
            "only, and VoxCeleb2's own terms apply too; the terms need review",
            "url": None,
        },
        "access": "download from the links in the LAV-DF repository after agreeing to its terms "
        "and conditions; dfwb never distributes media",
        "modalities": ["video", "audio"],
        "compressions": None,
        "key_rule": "<task>/<id>: the file stem, a six-digit id unique across the release, e.g. "
        "RVRA/000002",
    }
    layout_notes = (
        "The release unpacks to a LAV-DF folder with train/, dev/ and test/ folders of videos "
        "beside metadata.json and metadata.min.json: keep it as .official_files/LAV-DF. Its "
        "videos may stay in .official_files/LAV-DF/{train,dev,test}/, where each takes the "
        "category its metadata row's modify_video and modify_audio name (a video without a row, "
        "or whose row lacks either flag, is skipped), or be moved by those two flags into the "
        "folders above: neither is "
        "RealVideo-RealAudio, both FakeVideo-FakeAudio, the video alone FakeVideo-RealAudio and "
        "the audio alone RealVideo-FakeAudio.\n"
        f"Attributes and pairs come from {_METADATA} ({_METADATA_MIN} when it is absent), by "
        "file stem; the official split is each row's split, dev published as val."
    )

    _rows: Mapping[str, _Row] | None = None

    def prepare(self, root: Path) -> None:
        """Read the release's metadata under ``root`` once per build.

        Raises:
            ContractError: the metadata file is malformed.
        """
        super().prepare(root)
        path = _metadata_path(root)
        if path is None:
            _log.warning(
                "lav-df: neither %s nor %s is in %s; the videos get no attributes or pair_key, "
                "and those left in the release's own folders no task",
                _METADATA,
                _METADATA_MIN,
                root,
            )
            self._rows = {}
            return
        self._rows = {_stem(row["file"]): _row(row) for row in _read_rows(path)}
        _log.debug("lav-df: %s lists %d videos", path.name, len(self._rows))

    # ----------------------------------------------------------------------------- layout

    def layout_dirs(self) -> tuple[str, ...]:
        """The task folders, then the release's own train, dev and test folders."""
        return (*super().layout_dirs(), *_RELEASE_DIRS)

    def videos_present(self, folder: Path) -> tuple[str, ...]:
        """Which of :meth:`layout_dirs` hold at least one video under ``folder``."""
        release = tuple(dir_ for dir_ in _RELEASE_DIRS if has_videos(folder / dir_))
        return (*super().videos_present(folder), *release)

    # ----------------------------------------------------------------------------- discovery

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        """Yield the videos of the task folders, then those left in the release's own folders.

        Each release folder is scanned in the first copy holding a video there (unbound, just
        ``root``); a video there takes the task its metadata row names.

        Raises:
            ConfigError: ``compressions`` names a compression (this dataset has none).
        """
        yield from super().discover(root, compressions=compressions)
        copies = self._copies if self._copies is not None else (root,)
        tasks = {task.abbr: task for task in self.tasks}
        rows = self._rows or {}
        skipped = 0
        for folder in _RELEASE_DIRS:
            copy = next((c for c in copies if has_videos(c / folder)), None)
            if copy is None:
                continue
            for path in scan_videos(copy / folder):
                row = rows.get(path.stem)
                if row is None or row.task is None:
                    skipped += 1
                    continue
                relpath = f"{folder}/{path.name}"
                yield self._record(tasks[row.task], path.stem, relpath, None, row)
        if skipped:
            _log.warning(
                "lav-df: %d video(s) in the release's own folders have no metadata row, or a "
                "row without both modify flags, so no known task; they are skipped",
                skipped,
            )

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; its metadata row adds its fields."""
        return self._record(
            task, path.stem, relpath, compression, (self._rows or {}).get(path.stem)
        )

    def _record(
        self, task: TaskSpec, stem: str, relpath: str, compression: str | None, row: _Row | None
    ) -> InventoryRecord:
        if row is None:
            return self.record(task, stem, relpath, compression, attrs=_no_row_attrs())
        return self.record(task, stem, relpath, compression, pair_key=row.pair_key, attrs=row.attrs)

    # ----------------------------------------------------------------------------- hooks

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's split: the first of train, val and test that its stem's rows name.

        Raises:
            ConfigError: neither metadata file is present.
            ContractError: the metadata file is malformed.
        """
        path = _metadata_path(root)
        if path is None:
            raise ConfigError(
                f"lav-df: the metadata file {_METADATA} is missing from {root}",
                hint=f"copy the release's metadata.json (or metadata.min.json) into {_RELEASE}/ "
                "in the dataset folder",
            )
        listed: dict[Split, set[str]] = {split: set() for split in _LOOKUP_ORDER}
        for row in _read_rows(path):
            split = _SPLITS.get(str(row.get("split", "")).strip().lower())
            if split is not None:
                listed[split].add(_stem(row["file"]))
        assignment: dict[str, Split] = {}
        for record in records:
            stem = local_key(record.key)
            for split in _LOOKUP_ORDER:
                if stem in listed[split]:
                    assignment[record.key] = split
                    break
        return assignment

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The real video the fake was made from (its row's ``original``)."""
        return fake.pair_key
