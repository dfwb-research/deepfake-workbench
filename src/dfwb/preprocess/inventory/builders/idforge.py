"""IDForge-v1: pristine videos of 54 people and fakes whose face, lips or speech is manipulated.

Layout, relative to the ``IDForge-v1`` folder (a single version, no compression levels). Every
task nests its videos two levels deep, ``<identity>/<session>/<video>.mp4`` (e.g.
``id00/id00_03/id00_scene_0015_0_infoswap.mp4``), as in the release, so each is searched
recursively:

* reals: ``original_content/pristine/videos/...``;
* fakes: ``manipulated_content/<task>/videos/...`` for ten tasks: face swaps
  (``face_audiomismatch_textmismatch``, ``face_rvc_textmismatch``, ``face_tts``,
  ``face_tts_textgen``), lip-syncs (``lip_audiomismatch_textmismatch``,
  ``lip_rvc_textmismatch``, ``lip_tts_textgen``) and real video with cloned or synthesised speech
  (``rvc_textmismatch``, ``tts_textgen``, ``tts_textmismatch``).

A video's key is ``<session>__<stem>``: the name of the folder it sits in and its file name
without the last suffix, so a ``.mp3.mp4`` file keeps ``.mp3`` in its key. The same key often
appears in several tasks, whose videos are named alike; the task prefix keeps them apart. Its
identity and target are the ``id<digits>`` its file name starts with (none if it does not);
nothing records a source. A fake's ``pair_key`` is its identity: IDForge does not say which
recording a fake was made from, so a fake pairs with the reals of its person, and only the first
of them by key is kept.

The official split is the release's ``train.csv``, ``val.csv`` and ``test.csv`` (kept under
``.official_files/splits/``), one ``<task>/<identity>/<session>/<file>`` path per line. A line
names the key ``<session>__<stem>`` from its last two parts (the bare stem for a line without a
folder), whatever its task, so a key shared by several tasks goes, in every one of them, to the
first of train, val and test that lists it.

Real video with cloned or synthesised speech is real under the visual label and fake under the
audiovisual one, so it counts as real when fakes are paired and is left out of the benchmark.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BENCHMARK_REALS, BenchmarkSpec, Split, local_key

__all__ = ["IDForgeV1Builder"]

_log = logging.getLogger(__name__)

# The release's split lists, relative to the dataset folder, in the order a key is looked up.
_SPLIT_DIR: Final = ".official_files/splits"
_OFFICIAL_SPLITS: Final[tuple[Split, ...]] = ("train", "val", "test")
_HINT: Final = "each list is UTF-8 text with one <task>/<identity>/<session>/<file> path per line"
_IDENTITY: Final = re.compile(r"^(id\d+)")
# Real video with cloned or synthesised speech: no visual manipulation.
_AUDIO_ONLY: Final = ("RVC_TM", "TTS_TG", "TTS_TM")


def _fake(abbr: str, name: str) -> TaskSpec:
    """A fake task: its folder and method are its name, and its videos nest."""
    return TaskSpec(abbr, name, "fake", f"manipulated_content/{name}/videos", name, recursive=True)


def _list_key(line: str) -> str:
    """The key a split-list line names: ``<session>__<stem>`` of its last two parts."""
    parts = line.split("/")
    stem = PurePosixPath(parts[-1]).stem
    return f"{parts[-2]}__{stem}" if len(parts) >= 2 and parts[-2] else stem


def _read_split_lists(root: Path) -> dict[Split, frozenset[str]]:
    """The keys each release split list names, by split in train, val, test order.

    Each non-blank line, stripped of surrounding space, is a path.

    Raises:
        ConfigError: a list is missing.
        ContractError: a list cannot be read as UTF-8 text.
    """
    folder = root / _SPLIT_DIR
    missing = [
        f"{split}.csv" for split in _OFFICIAL_SPLITS if not (folder / f"{split}.csv").is_file()
    ]
    if missing:
        raise ConfigError(
            f"idforge-v1: the official split list(s) {', '.join(missing)} are missing from "
            f"{folder}",
            hint="copy train.csv, val.csv and test.csv from the release's IDForge_v1 folder into "
            f"{_SPLIT_DIR}/ in the dataset folder",
        )
    keys: dict[Split, frozenset[str]] = {}
    for split in _OFFICIAL_SPLITS:
        path = folder / f"{split}.csv"
        try:
            with path.open(encoding="utf-8") as handle:
                lines = [line.strip() for line in handle]
        except (OSError, UnicodeDecodeError) as exc:
            raise ContractError(
                f"idforge-v1: cannot read the official split list {path}: {exc}", hint=_HINT
            ) from None
        keys[split] = frozenset(_list_key(line) for line in lines if line)
    _log.debug("idforge-v1: the split lists name %s", {s: len(v) for s, v in keys.items()})
    return keys


class IDForgeV1Builder(BaseBuilder):
    """IDForge-v1 (``idforge-v1``): pristine videos and ten face, lip and speech manipulations."""

    dataset_id = "idforge-v1"
    expected_folder = "IDForge-v1"
    label_prefix = "IDFv1"
    tasks = (
        TaskSpec(
            "REAL",
            "pristine",
            "real",
            "original_content/pristine/videos",
            "original",
            recursive=True,
        ),
        _fake("FS_AM_TM", "face_audiomismatch_textmismatch"),
        _fake("FS_RVC_TM", "face_rvc_textmismatch"),
        _fake("FS_TTS", "face_tts"),
        _fake("FS_TTS_TG", "face_tts_textgen"),
        _fake("LS_AM_TM", "lip_audiomismatch_textmismatch"),
        _fake("LS_RVC_TM", "lip_rvc_textmismatch"),
        _fake("LS_TTS_TG", "lip_tts_textgen"),
        _fake("RVC_TM", "rvc_textmismatch"),
        _fake("TTS_TG", "tts_textgen"),
        _fake("TTS_TM", "tts_textmismatch"),
    )
    metadata_files = tuple(f"{_SPLIT_DIR}/{split}.csv" for split in _OFFICIAL_SPLITS)
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FS_AM_TM": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
        "FS_RVC_TM": LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-swap"),
        "FS_TTS": LabelSpec(binary=1, binary_av=1, multiclass=4, family="face-swap"),
        "FS_TTS_TG": LabelSpec(binary=1, binary_av=1, multiclass=5, family="face-swap"),
        "LS_AM_TM": LabelSpec(binary=1, binary_av=1, multiclass=6, family="lip-sync"),
        "LS_RVC_TM": LabelSpec(binary=1, binary_av=1, multiclass=7, family="lip-sync"),
        "LS_TTS_TG": LabelSpec(binary=1, binary_av=1, multiclass=8, family="lip-sync"),
        # The picture is the real one: real to a visual detector, fake once audio counts.
        "RVC_TM": LabelSpec(binary=0, binary_av=1, multiclass=1, family="audio-only"),
        "TTS_TG": LabelSpec(binary=0, binary_av=1, multiclass=1, family="audio-only"),
        "TTS_TM": LabelSpec(binary=0, binary_av=1, multiclass=1, family="audio-only"),
    }
    schemes = {
        "official": SchemeSpec(
            "official",
            "official",
            source=f"the release's split lists ({_SPLIT_DIR}/{{train,val,test}}.csv)",
            rationale="the publisher's train/val/test split, matched on each video's "
            "<session>__<stem>",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set drawn from the official test: up to 5 "
            f"fakes per person and task, {BENCHMARK_REALS}; real video with cloned or "
            "synthesised speech is left out, since a visual detector cannot see it",
        ),
    }
    default_scheme = "official"
    benchmark = BenchmarkSpec(k_fake=5, strata=("identity", "task"), exclude_tasks=_AUDIO_ONLY)
    pairing_rule = "identity-fanout"
    pairing_fanout = 1
    card_info = {
        "name": "IDForge",
        "aliases": ["IDForge-v1", "IDFv1", "IDForge v1"],
        "release": "IDForge v1: 80,000 pristine videos of 54 people and 170,000 fakes in ten "
        "tasks: four face swap and three lip-sync tasks paired with mismatched, cloned or "
        "synthesised speech, and three of real video with cloned or synthesised speech, with "
        "the release's train/val/test split lists",
        "homepage": "https://github.com/xyyandxyy/IDForge",
        "paper": {
            "title": "Identity-Driven Multimedia Forgery Detection via Reference Assistance",
            "venue": "ACM MM",
            "year": 2024,
            "doi": "10.1145/3664647.3680622",
        },
        "license": {
            "spdx": None,
            "summary": "no licence statement found; the forged videos are released on request "
            "through the authors' access site; the terms need review",
            "url": None,
        },
        "access": "the pristine videos download from links in the IDForge repository; the "
        "forged videos are released on request through the authors' access site; dfwb never "
        "distributes media",
        "modalities": ["video", "audio"],
        "compressions": None,
        "key_rule": "<task>/<session>__<stem>: the video's folder name and its file name "
        "without the last suffix, e.g. REAL/id00_03__id00_scene_0010-0 or "
        "RVC_TM/id00_03__id00_scene_0014.mp3",
    }
    layout_notes = (
        "Every task folder is searched recursively: videos nest as <identity>/<session>/<video>, "
        "as in the release.\n"
        "The release's pristine/ folder goes in original_content/pristine/videos/ and each of "
        "its ten fake task folders in manipulated_content/<task>/videos/, each keeping its "
        "<identity>/<session>/ sub-folders.\n"
        f"The official split reads the release's IDForge_v1/{{train,val,test}}.csv, kept in "
        f"{_SPLIT_DIR}/: one <task>/<identity>/<session>/<file> path per line. A video goes to "
        "the first of train, val and test that lists its <session>__<stem>, whatever the task."
    )

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its session folder and stem; its person from its name."""
        stem = path.stem
        match = _IDENTITY.match(stem)
        identity = match.group(1) if match else None
        if identity is None:
            _log.debug("idforge-v1: %s does not start with its person's id", relpath)
        return self.record(
            task,
            f"{path.parent.name}__{stem}",
            relpath,
            compression,
            identity=identity,
            target_id=identity,
            pair_key=identity if task.kind == "fake" else None,
        )

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's split: the first of train, val and test whose list names its key.

        Raises:
            ConfigError: a split list is missing.
            ContractError: a split list cannot be read.
        """
        lists = _read_split_lists(root)
        assignment: dict[str, Split] = {}
        for record in records:
            key = local_key(record.key)
            for split, keys in lists.items():
                if key in keys:
                    assignment[record.key] = split
                    break
        return assignment

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The fake's person: the first real of that person."""
        return fake.identity
