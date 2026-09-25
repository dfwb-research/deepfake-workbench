"""DeeperForensics-1.0: source videos of paid actors and face swaps made from them.

Layout, relative to the ``DeeperForensics-1.0`` folder (a single version, no compression levels):

* reals: ``original_content/source_videos/videos/<actor>/<lighting>/<expression>/<camera>/``
  ``<video>.mp4``, nested one folder per actor, lighting, expression and camera (some actors add
  ``BlendShape`` recordings with no expression folder), so the task is searched recursively. The
  file name already spells out the whole path, e.g. ``M101_light_down_contempt_camera_front``;
* fakes: ``manipulated_content/<version>/videos/<n>_<actor>.mp4``, flat, for eleven versions of
  the same 1,000 face swaps: ``end_to_end``, ``end_to_end_level_1`` to ``_5``,
  ``end_to_end_random_level``, ``end_to_end_mix_{2,3,4}_distortions`` and
  ``reenact_postprocess``.

Every video is keyed by its file stem, so a fake id repeats in every version (the task prefix
keeps the keys apart). A real's identity and target are the actor, the part of the stem before its
first ``_``. A fake's identity and target are the part after its first ``_`` (a stem without
``_`` has neither). Neither has a source or a ``pair_key``: a fake pairs with the reals of its
actor, and only the first of them by key is kept.

The official split comes from the release's own split lists, ``lists/splits/{train,val,test}.txt``
(kept under ``.official_files/`` in the dataset folder), which name the file of each of the 1,000
face swaps (703/96/201). Every ``.mp4`` video is given a split first:

* a fake takes the split of the list naming its file name (test before val before train, if
  several do); every version of a face swap shares its file name, so all eleven are listed;
* a real takes the split of its actor's listed fakes, test before val before train, where a
  listed name's actor is its second ``_``-separated part (``001_W101.mp4`` names ``W101``) and a
  real's actor is its file name up to the first ``_``. A real whose actor has no listed fake gets
  no split.

Each video's split then goes to every record with the same file stem, the first of train, val and
test winning if a stem was given two.
"""

from __future__ import annotations

import logging
import posixpath
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BenchmarkSpec, Split, local_key, task_of

__all__ = ["DeeperForensicsBuilder"]

_log = logging.getLogger(__name__)

# The release's split lists, relative to the dataset folder, in the order they are read.
_SPLIT_DIR: Final = ".official_files/lists/splits"
_OFFICIAL_SPLITS: Final[tuple[Split, ...]] = ("train", "val", "test")
# When a video is given more than one split, the first of these wins.
_PRIORITY: Final[tuple[Split, ...]] = ("test", "val", "train")
# Only these files are given a split (the suffix matched exactly, as the release names them).
_LISTED_SUFFIX: Final = ".mp4"
_HINT: Final = "each list is UTF-8 text with one face-swap file name per line"

# Only a fake named "<digits>_<M or W><digits>" (e.g. 000_M101) names an actor to pair with.
_PAIRABLE_FAKE: Final = re.compile(r"^\d+_[MW]\d+$")


def _fake(abbr: str, name: str, folder: str) -> TaskSpec:
    """A fake version, whose method is its name."""
    return TaskSpec(abbr, name, "fake", f"manipulated_content/{folder}/videos", name)


def _read_split_lists(root: Path) -> dict[Split, frozenset[str]]:
    """The file names each release split list holds, keyed by split in train, val, test order.

    Each non-blank line, stripped of surrounding space, is a file name.

    Raises:
        ConfigError: a list is missing.
        ContractError: a list cannot be read as UTF-8 text.
    """
    folder = root / _SPLIT_DIR
    missing = [
        f"{split}.txt" for split in _OFFICIAL_SPLITS if not (folder / f"{split}.txt").is_file()
    ]
    if missing:
        raise ConfigError(
            f"deeperforensics: the official split list(s) {', '.join(missing)} are missing from "
            f"{folder}",
            hint="copy train.txt, val.txt and test.txt from the release's lists/splits/ folder "
            f"into {_SPLIT_DIR}/ in the dataset folder",
        )
    names: dict[Split, frozenset[str]] = {}
    for split in _OFFICIAL_SPLITS:
        path = folder / f"{split}.txt"
        try:
            with path.open(encoding="utf-8") as handle:
                lines = list(handle)
        except (OSError, UnicodeDecodeError) as exc:
            raise ContractError(
                f"deeperforensics: cannot read the official split list {path}: {exc}", hint=_HINT
            ) from None
        names[split] = frozenset(line.strip() for line in lines if line.strip())
    _log.debug("deeperforensics: the split lists name %s", {s: len(v) for s, v in names.items()})
    return names


def _listed_actor(name: str) -> str | None:
    """The actor a listed file name gives: the second ``_``-part of the name without its suffix."""
    parts = posixpath.splitext(name)[0].split("_")
    return parts[1] if len(parts) >= 2 else None


def _first(splits: set[Split]) -> Split | None:
    """The first of test, val and train in ``splits``."""
    return next((split for split in _PRIORITY if split in splits), None)


class DeeperForensicsBuilder(BaseBuilder):
    """DeeperForensics-1.0 (``deeperforensics``): actor recordings and eleven face-swap versions."""

    dataset_id = "deeperforensics"
    expected_folder = "DeeperForensics-1.0"
    label_prefix = "DeFo"
    tasks = (
        TaskSpec(
            "SR",
            "Source-Real",
            "real",
            "original_content/source_videos/videos",
            "original",
            recursive=True,
        ),
        _fake("FS_E2E", "End-to-End", "end_to_end"),
        _fake("FS_L1", "E2E-Level-1", "end_to_end_level_1"),
        _fake("FS_L2", "E2E-Level-2", "end_to_end_level_2"),
        _fake("FS_L3", "E2E-Level-3", "end_to_end_level_3"),
        _fake("FS_L4", "E2E-Level-4", "end_to_end_level_4"),
        _fake("FS_L5", "E2E-Level-5", "end_to_end_level_5"),
        _fake("FS_RND", "E2E-Random", "end_to_end_random_level"),
        _fake("FS_M2", "E2E-Mix-2", "end_to_end_mix_2_distortions"),
        _fake("FS_M3", "E2E-Mix-3", "end_to_end_mix_3_distortions"),
        _fake("FS_M4", "E2E-Mix-4", "end_to_end_mix_4_distortions"),
        _fake("FR_RP", "Reenact-Post", "reenact_postprocess"),
    )
    metadata_files = tuple(f"{_SPLIT_DIR}/{split}.txt" for split in _OFFICIAL_SPLITS)
    labels = {
        "SR": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FS_E2E": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
        "FS_L1": LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-swap"),
        "FS_L2": LabelSpec(binary=1, binary_av=1, multiclass=4, family="face-swap"),
        "FS_L3": LabelSpec(binary=1, binary_av=1, multiclass=5, family="face-swap"),
        "FS_L4": LabelSpec(binary=1, binary_av=1, multiclass=6, family="face-swap"),
        "FS_L5": LabelSpec(binary=1, binary_av=1, multiclass=7, family="face-swap"),
        "FS_RND": LabelSpec(binary=1, binary_av=1, multiclass=8, family="face-swap"),
        "FS_M2": LabelSpec(binary=1, binary_av=1, multiclass=9, family="face-swap"),
        "FS_M3": LabelSpec(binary=1, binary_av=1, multiclass=10, family="face-swap"),
        "FS_M4": LabelSpec(binary=1, binary_av=1, multiclass=11, family="face-swap"),
        "FR_RP": LabelSpec(binary=1, binary_av=1, multiclass=12, family="face-reenactment"),
    }
    schemes = {
        "official": SchemeSpec(
            "official",
            "official",
            source=f"the release's split lists ({_SPLIT_DIR}/{{train,val,test}}.txt)",
            rationale="the publisher's train/val/test split of the 1,000 face swaps, with each "
            "actor's source videos in the split of that actor's fakes (test before val before "
            "train)",
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
            "fakes per version, balanced with as many reals",
        ),
    }
    default_scheme = "official"
    benchmark = BenchmarkSpec(k_fake=100, strata=("task",))
    pairing_rule = "identity-fanout"
    pairing_fanout = 1
    card_info = {
        "name": "DeeperForensics-1.0",
        "aliases": ["DeeperForensics", "DeFo"],
        "release": "source videos of paid actors under varied lighting, expressions and camera "
        "angles, and 1,000 face swaps in eleven versions: the end-to-end swap, nine perturbed "
        "copies of it and a reenactment post-process",
        "homepage": "https://github.com/EndlessSora/DeeperForensics-1.0",
        "paper": {
            "title": "DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery "
            "Detection",
            "venue": "CVPR",
            "year": 2020,
        },
        "license": {
            "spdx": None,
            "summary": "the DeeperForensics-1.0 terms of use: non-commercial research only, "
            "no redistribution",
            "url": None,
        },
        "access": "request the download from the authors through the DeeperForensics-1.0 "
        "repository after agreeing to its terms of use; dfwb never distributes media",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": "real: SR/<actor>_<lighting>_<expression>_<camera> or "
        "SR/<actor>_BlendShape_<camera>; fake: <task>/<n>_<actor> (the file stem)",
    }
    layout_notes = (
        "Reals nest as <actor>/<lighting>/<expression>/<camera>/<video>; fakes are flat.\n"
        f"The official split reads the release's lists/splits/ files, kept in "
        f"{_SPLIT_DIR}/{{train,val,test}}.txt: one face-swap file name per line. A fake takes the "
        "split of its file name; a real takes the split of its actor's listed fakes, test before "
        "val before train, and a real whose actor has no listed fake has none."
    )

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; its actor comes from the name."""
        stem = path.stem
        if task.kind == "real":
            actor = stem.split("_", 1)[0]
            return self.record(task, stem, relpath, compression, identity=actor, target_id=actor)
        parts = stem.split("_", 1)
        if len(parts) < 2:
            _log.debug("deeperforensics: %s is not named <n>_<actor>", relpath)
            return self.record(task, stem, relpath, compression)
        return self.record(task, stem, relpath, compression, identity=parts[1], target_id=parts[1])

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's split, derived from the release's split lists (see the module docstring).

        Raises:
            ConfigError: a split list is missing.
            ContractError: a split list cannot be read.
        """
        lists = _read_split_lists(root)
        actor_splits: dict[str, set[Split]] = {}
        for split, names in lists.items():
            for name in names:
                actor = _listed_actor(name)
                if actor:
                    actor_splits.setdefault(actor, set()).add(split)
        real_tasks = {task.abbr for task in self.tasks if task.kind == "real"}

        # First, the split of every listed video, by its file stem.
        stems: dict[Split, set[str]] = {split: set() for split in _OFFICIAL_SPLITS}
        for record in records:
            name = PurePosixPath(record.relpath).name
            if not name.endswith(_LISTED_SUFFIX):
                continue
            if task_of(record.key) in real_tasks:
                given = _first(actor_splits.get(name.split("_")[0], set()))
            else:
                given = _first({split for split, names in lists.items() if name in names})
            if given is not None:
                stems[given].add(PurePosixPath(name).stem)

        # Then every record whose stem was given a split; train, val, then test.
        assignment: dict[str, Split] = {}
        for record in records:
            stem = PurePosixPath(record.relpath).stem
            for split in _OFFICIAL_SPLITS:
                if stem in stems[split]:
                    assignment[record.key] = split
                    break
        return assignment

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The actor of a fake named ``<digits>_<M|W><digits>``: every real of that actor."""
        stem = local_key(fake.key)
        return stem.split("_", 1)[1] if _PAIRABLE_FAKE.match(stem) else None
