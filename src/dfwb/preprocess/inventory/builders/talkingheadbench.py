"""TalkingHeadBench: talking-head fakes of FFHQ portraits driven by CelebV-HQ clips.

Layout, relative to the ``TalkingHeadBench`` folder (a single version, no compression levels):

* fakes: ``manipulated_content/<generator>/videos/<stem>.mp4``, flat, for the six core generators
  (LivePortrait, AniPortraitAudio, AniPortraitVideo, Hallo, Hallo2, EmoPortrait) and the two held
  out for testing generalisation (MAGI-1, Hallo3);
* reals: the release ships none of them, only the list splitting them,
  ``real_dataset_split_official_ff++.json`` (kept as ``.official_files/``). The reals are the
  videos that list names, never a scan: a FaceForensics++ real is
  ``original_content/YouTube/raw/videos/<id>.mp4`` in the sibling ``FaceForensics++`` folder, a
  CelebV-HQ real is ``original_content/YouTube/videos/<stem>.mp4`` in the sibling ``CelebV-HQ``
  folder, and each record names that folder (its relpath is relative to it). A real is a record
  whether or not its video is on disk.

Every video is keyed by its file stem (a real by the stem the list gives it). A real's identity
and target are its stem. A fake animates an FFHQ portrait its name does not give, so it has no
identity or target; its source is the CelebV-HQ clip driving it, read from its name:
``<n>--<driver>--<generator>`` for the six core generators, ``<n>-<driver>`` for Hallo3, and none
for MAGI-1, whose name is just ``<n>``. Splitting on ``-`` loses a driver's own leading ``-``, so
a driver the list names only with a leading ``-`` gets it back; any other driver is kept as
written.

A video's ``audio_relpath`` is ``<audio folder>/<stem>.wav`` when that file exists in any copy of
the dataset folder, and absent otherwise; it is relative to this dataset's folder even for a real,
because the release ships the reals' audio itself. The audio folders are
``manipulated_content/<generator>/audio`` for a fake, ``original_content/real/audio`` for a
FaceForensics++ real and ``original_content/fake_celebvhq`` for a CelebV-HQ real.

The official split is the list's split for the reals (FaceForensics++'s own 720/140/140 split, and
CelebV-HQ's); a stem it lists twice takes the last of train, val and test. A fake follows its
driver: test when the list puts the driver in test, train otherwise, so fakes get no val. The two
held-out generators, MAGI-1 and Hallo3, are always test. A fake pairs with its driving video.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

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
from dfwb.protocols.rules import BenchmarkSpec, Split, local_key, task_of

__all__ = ["TalkingHeadBenchBuilder"]

_log = logging.getLogger(__name__)

# The release's list splitting the real videos, relative to the dataset folder.
_SPLIT_LIST: Final = ".official_files/real_dataset_split_official_ff++.json"
_HINT: Final = (
    "use the release's real_dataset_split_official_ff++.json as downloaded: an object whose "
    "Train, Val and Test each map FaceForensics++ and CelebV-HQ to lists of file names"
)
_FFPP: Final = "FaceForensics++"
_CELEBVHQ: Final = "CelebV-HQ"
# Where each sibling folder keeps the real videos, relative to that folder.
_FFPP_VIDEOS: Final = "original_content/YouTube/raw/videos"
_CELEBVHQ_VIDEOS: Final = "original_content/YouTube/videos"
# The list's blocks in the order they are read (a stem listed twice keeps the last) and the
# split each one is.
_BLOCKS: Final[tuple[tuple[str, Split], ...]] = (
    ("Train", "train"),
    ("Val", "val"),
    ("Test", "test"),
)
# Test-only generators: they test generalisation to a generator never trained on.
_HELD_OUT: Final = frozenset({"TH_M1", "TH_H3"})
# Each real task's sibling folder (also the split list's name for its stems) and where that
# folder keeps its videos.
_REAL_FOLDER: Final = {"REAL_FF": (_FFPP, _FFPP_VIDEOS), "REAL_CVHQ": (_CELEBVHQ, _CELEBVHQ_VIDEOS)}


def _real(abbr: str, name: str, folder: str, videos: str, audio_dir: str) -> TaskSpec:
    """A real task: its videos sit in a sibling dataset's folder, beside this dataset's."""
    return TaskSpec(
        abbr, name, "real", f"../{folder}/{videos}", "original", extras={"audio_dir": audio_dir}
    )


def _fake(abbr: str, name: str) -> TaskSpec:
    """A generator: its name is the task's name, the method and its folder."""
    folder = f"manipulated_content/{name}"
    return TaskSpec(
        abbr, name, "fake", f"{folder}/videos", name, extras={"audio_dir": f"{folder}/audio"}
    )


def _read_split_list(path: Path) -> dict[str, dict[str, Split]]:
    """``{"FaceForensics++": {stem: split}, "CelebV-HQ": {stem: split}}`` from the split list.

    Raises:
        ContractError: the file cannot be read as JSON, or is not an object of blocks mapping
            each source to a list of file names.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ContractError(
            f"talkingheadbench: cannot read the split list {path}: {exc}", hint=_HINT
        ) from None
    if not isinstance(data, dict):
        raise ContractError(
            f"talkingheadbench: {path} holds a {type(data).__name__}, not an object", hint=_HINT
        )
    splits: dict[str, dict[str, Split]] = {_FFPP: {}, _CELEBVHQ: {}}
    for block_name, split in _BLOCKS:
        block = data.get(block_name) or {}
        if not isinstance(block, dict):
            raise ContractError(
                f"talkingheadbench: {path}: {block_name} is not an object", hint=_HINT
            )
        for source, stems in splits.items():
            names = block.get(source) or []
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ContractError(
                    f"talkingheadbench: {path}: {block_name}/{source} is not a list of file names",
                    hint=_HINT,
                )
            for name in names:
                stems[PurePosixPath(name).stem] = split
    return splits


def _restore_dash(driver: str, listed: Mapping[str, object]) -> str:
    """The driver as the list names it: with a leading ``-`` restored if only that is listed."""
    if driver in listed:
        return driver
    if f"-{driver}" in listed:
        return f"-{driver}"
    return driver


def _driver(stem: str, abbr: str, listed: Mapping[str, object]) -> str | None:
    """The CelebV-HQ clip driving a fake, from its file stem; None when the name gives none."""
    if abbr == "TH_M1":
        return None
    if abbr == "TH_H3":
        return _restore_dash(stem.split("-", 1)[1], listed) if "-" in stem else None
    parts = stem.split("--")
    if len(parts) == 3:
        return _restore_dash(parts[1], listed)
    _log.debug("talkingheadbench: %s: %r is not <n>--<driver>--<generator>", abbr, stem)
    return None


class TalkingHeadBenchBuilder(BaseBuilder):
    """TalkingHeadBench (``talkingheadbench``): talking-head fakes and the reals its list names."""

    dataset_id = "talkingheadbench"
    expected_folder = "TalkingHeadBench"
    label_prefix = "THB"
    tasks = (
        _real("REAL_FF", "Real-FFpp", _FFPP, _FFPP_VIDEOS, "original_content/real/audio"),
        _real(
            "REAL_CVHQ",
            "Real-CelebVHQ",
            _CELEBVHQ,
            _CELEBVHQ_VIDEOS,
            "original_content/fake_celebvhq",
        ),
        _fake("TH_LP", "LivePortrait"),
        _fake("TH_APA", "AniPortraitAudio"),
        _fake("TH_APV", "AniPortraitVideo"),
        _fake("TH_H1", "Hallo"),
        _fake("TH_H2", "Hallo2"),
        _fake("TH_EP", "EmoPortrait"),
        _fake("TH_M1", "MAGI-1"),
        _fake("TH_H3", "Hallo3"),
    )
    metadata_files = (_SPLIT_LIST,)
    labels = {
        "REAL_FF": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "REAL_CVHQ": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "TH_LP": LabelSpec(binary=1, binary_av=1, multiclass=2, family="talking-head"),
        "TH_APA": LabelSpec(binary=1, binary_av=1, multiclass=3, family="talking-head"),
        "TH_APV": LabelSpec(binary=1, binary_av=1, multiclass=4, family="talking-head"),
        "TH_H1": LabelSpec(binary=1, binary_av=1, multiclass=5, family="talking-head"),
        "TH_H2": LabelSpec(binary=1, binary_av=1, multiclass=6, family="talking-head"),
        "TH_EP": LabelSpec(binary=1, binary_av=1, multiclass=7, family="talking-head"),
        "TH_M1": LabelSpec(binary=1, binary_av=1, multiclass=8, family="talking-head"),
        "TH_H3": LabelSpec(binary=1, binary_av=1, multiclass=9, family="talking-head"),
    }
    schemes = {
        "official": SchemeSpec(
            "official",
            "official",
            source=f"the release's split list ({_SPLIT_LIST}) for the reals; each fake's "
            "CelebV-HQ driver's split in that list for the fakes",
            rationale="the reals as the release's list splits them, FaceForensics++ by its own "
            "split; each fake by its driving CelebV-HQ clip, test when the clip is and train "
            "otherwise (fakes get no val), so no driver's motion is in both train and test; the "
            "two held-out generators, MAGI-1 and Hallo3, are always test",
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
            "fakes per generator (the fakes' portraits are not recorded, so the generator is the "
            "only stratum), balanced with as many reals",
        ),
    }
    default_scheme = "official"
    benchmark = BenchmarkSpec(k_fake=100, strata=("task",))
    pairing_rule = "driving-video"
    card_info = {
        "name": "TalkingHeadBench",
        "aliases": ["THB"],
        "release": "the Hugging Face release: 2,994 talking-head fakes, each an FFHQ portrait "
        "animated by a CelebV-HQ driving clip, from six generators (LivePortrait 529, "
        "AniPortraitAudio 542, AniPortraitVideo 422, Hallo 420, Hallo2 432, EmoPortrait 573) and "
        "two held out for testing (MAGI-1 66, Hallo3 10), with the official split list covering "
        "1,000 FaceForensics++ and 1,608 CelebV-HQ real videos, which the release does not include",
        "homepage": "https://huggingface.co/datasets/luchaoqi/TalkingHeadBench",
        "paper": {
            "title": "TalkingHeadBench: A Multi-Modal Benchmark & Analysis of Talking-Head "
            "DeepFake Detection",
            "venue": "arXiv",
            "year": 2025,
        },
        "license": {
            "spdx": None,
            "summary": "the release's Hugging Face card declares MIT; the FFHQ, CelebV-HQ and "
            "FaceForensics++ terms apply to the source material and to the real videos; the "
            "terms need review",
            "url": None,
        },
        "access": "download the fakes, audio and split lists from the TalkingHeadBench Hugging "
        "Face dataset; obtain the FaceForensics++ and CelebV-HQ real videos separately under "
        "their own terms; dfwb never distributes media",
        "modalities": ["video", "audio"],
        "compressions": None,
        "key_rule": "<task>/<stem>: the file stem, e.g. TH_LP/00198--abcDEF_0--LivePortrait; a "
        "real's stem as the split list names it, e.g. REAL_CVHQ/-5be_UPkLRw_4",
    }
    layout_notes = (
        f"The reals are never scanned: the release's split list, {_SPLIT_LIST}, names them, and "
        f"their videos sit in the sibling {_FFPP} folder (the raw YouTube originals) and "
        f"{_CELEBVHQ} folder, each looked for under every datasets root. A real's relpath is "
        "relative to that folder.\n"
        "The release unpacks to fake/<generator>/{train,val,test}/<stem>.mp4 for the six core "
        "generators, fake/additional_dataset/{MAGI-1,Hallo3}/<stem>.mp4 for the held-out two, "
        "real/ for the official real split list and audio/: move every fake of a generator into "
        "manipulated_content/<generator>/videos/ (the folder names stay), and "
        "real/real_dataset_split_official_ff++.json into .official_files/. The release's "
        "train/val/test folders of fakes are not read: a fake's split follows its driver.\n"
        "The audio is optional: a fake's <stem>.wav goes in manipulated_content/<generator>/"
        "audio/, the FaceForensics++ reals' (audio/ff++) in original_content/real/audio/ and the "
        "CelebV-HQ reals' (audio/fake_celebvhq) in original_content/fake_celebvhq/. A video's "
        "audio_relpath is set when its .wav is in any copy of this dataset's folder, and is "
        "relative to it, for a real too."
    )

    _splits: dict[str, dict[str, Split]] | None = None
    _search: tuple[Path, ...] = ()

    def prepare(self, root: Path) -> None:
        """Read the release's split list under ``root`` once per build.

        Raises:
            ContractError: the split list is malformed.
        """
        super().prepare(root)
        path = root / _SPLIT_LIST
        if not path.is_file():
            _log.warning(
                "talkingheadbench: %s is missing from %s; without the release's split list there "
                "are no reals, and no driver gets its leading '-' back",
                _SPLIT_LIST,
                root,
            )
            self._splits = {_FFPP: {}, _CELEBVHQ: {}}
            return
        self._splits = _read_split_list(path)

    # ----------------------------------------------------------------------------- layout

    def _fakes(self) -> tuple[TaskSpec, ...]:
        return tuple(task for task in self.tasks if task.kind == "fake")

    def layout_dirs(self) -> tuple[str, ...]:
        """The generators' video folders: the reals are listed, not looked for."""
        return tuple(task.video_dir for task in self._fakes())

    def videos_present(self, folder: Path) -> tuple[str, ...]:
        """Which generators' video folders hold at least one video under ``folder``."""
        return tuple(d for d in self.layout_dirs() if has_videos(folder / d))

    def video_copy_for(
        self, copies: Sequence[Path], task: TaskSpec, compression: str | None
    ) -> Path | None:
        """A generator's first copy holding a video; none for a real (the reals are listed)."""
        if task.kind == "real":
            return None
        return super().video_copy_for(copies, task, compression)

    # ----------------------------------------------------------------------------- discovery

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        """Yield the reals the split list names, then every generator's videos.

        Raises:
            ConfigError: ``compressions`` names a compression (this dataset has none).
        """
        validate_compressions(compressions, self.known_compressions)
        self._search = self._copies if self._copies is not None else (root,)
        splits = self._splits or {}
        for task in self.tasks:
            if task.kind != "real":
                continue
            folder, videos = _REAL_FOLDER[task.abbr]
            for stem in sorted(splits.get(folder, {})):
                yield self.record(
                    task,
                    stem,
                    f"{videos}/{stem}.mp4",
                    None,
                    identity=stem,
                    target_id=stem,
                    attrs=self._audio(task, stem),
                    folder=folder,
                )
        yield from super().discover(root, compressions=compressions)

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """A fake, keyed by its file stem; its source is the clip driving it."""
        stem = path.stem
        listed = (self._splits or {}).get(_CELEBVHQ, {})
        return self.record(
            task,
            stem,
            relpath,
            compression,
            source_id=_driver(stem, task.abbr, listed),
            attrs=self._audio(task, stem),
        )

    def _audio(self, task: TaskSpec, stem: str) -> dict[str, Any]:
        """``{"audio_relpath": ...}`` when the video's .wav is in a copy, else nothing."""
        relpath = f"{task.extras['audio_dir']}/{stem}.wav"
        if any((copy / relpath).exists() for copy in self._search):
            return {"audio_relpath": relpath}
        return {}

    # ----------------------------------------------------------------------------- hooks

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's split: a real's from the split list, a fake's from its driver's.

        Raises:
            ConfigError: the split list is missing.
            ContractError: the split list is malformed.
        """
        path = root / _SPLIT_LIST
        if not path.is_file():
            raise ConfigError(
                f"talkingheadbench: the split list {_SPLIT_LIST} is missing from {root}",
                hint="copy the release's real/real_dataset_split_official_ff++.json into "
                ".official_files/ in the dataset folder",
            )
        splits = _read_split_list(path)
        celeb = splits[_CELEBVHQ]
        assignment: dict[str, Split] = {}
        for record in records:
            abbr = task_of(record.key)
            split: Split | None
            if abbr in _REAL_FOLDER:
                split = splits[_REAL_FOLDER[abbr][0]].get(local_key(record.key))
            elif abbr in _HELD_OUT:
                split = "test"
            else:
                driver = record.source_id
                split = "test" if driver is not None and celeb.get(driver) == "test" else "train"
            if split is not None:
                assignment[record.key] = split
        return assignment

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The CelebV-HQ clip driving the fake (none for MAGI-1)."""
        return fake.source_id
