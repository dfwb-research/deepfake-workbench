"""FFIW-10K (Face Forensics in the Wild): source videos and face-swapped target videos.

Layout, relative to the ``FFIW10K`` folder (a single version, no compression levels), both flat:

* reals: ``original_content/source/videos/<split>_<index>.mp4``;
* fakes: ``manipulated_content/target/videos/<split>_<index>.mp4``,

where ``<split>`` is ``train``, ``val`` or ``test`` and ``<index>`` has eight digits
(e.g. ``train_00000000``). The same names appear in both tasks.

Every video is keyed by its file stem. No actor ids ship with the dataset, so a video's identity
is its own stem, and it has no target or source. A fake's ``pair_key`` comes from the optional
pair lists ``.official_files/splits/{train,val,test}.json``: each is a JSON list of
``[source, target]`` integer ids, and both ids are read as ``test_<8 digits>`` stems, whichever
list they are in. Each list maps both ways and a later list (train, val, test order) overrides an
earlier one; without the lists, ``pair_key`` is None. The pair key is informational: nothing is
paired from it.

The official split is the prefix of each name (``train_``, ``val_`` or ``test_``, ignoring case).
A video counts when a ``.mp4`` of its stem exists in either task.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from dfwb.core.errors import ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BenchmarkSpec, Split, local_key

__all__ = ["FFIW10KBuilder"]

_log = logging.getLogger(__name__)

# The optional pair lists, relative to the dataset folder, in the order they are read.
_PAIR_DIR: Final = ".official_files/splits"
_PAIR_LISTS: Final = ("train", "val", "test")
_PAIR_HINT: Final = "each pair list is a JSON list of [source, target] integer video ids"
# The split prefixes a video's name can start with (compared in lower case).
_OFFICIAL_SPLITS: Final[dict[str, Split]] = {"train": "train", "val": "val", "test": "test"}
# Only these names carry the split: the split is read from the release's .mp4 files.
_SPLIT_SUFFIX: Final = ".mp4"


def _stem(video_id: Any) -> str:
    """``test_<8 digits>`` for an integer id (or anything ``int()`` accepts)."""
    # This assumes the older file naming, where every video was named test_<8 digits>.
    return f"test_{int(video_id):08d}"


def _read_pair_map(root: Path) -> dict[str, str]:
    """``{stem: paired stem}`` from the pair lists under ``root``, both ways; ``{}`` if none.

    A missing list is skipped; a list that cannot be read as JSON is skipped with a warning.

    Raises:
        ContractError: a list is not a JSON list of ``[source, target]`` integer ids.
    """
    folder = root / _PAIR_DIR
    if not folder.exists():
        return {}
    pairs: dict[str, str] = {}
    for name in _PAIR_LISTS:
        path = folder / f"{name}.json"
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            _log.warning("ffiw10k: skipping the pair list %s: %s", path, exc)
            continue
        if not isinstance(rows, list):
            raise ContractError(
                f"ffiw10k: the pair list {path} holds a {type(rows).__name__}, not a list",
                hint=_PAIR_HINT,
            )
        for row in rows:
            try:
                source, target = _stem(row[0]), _stem(row[1])
            except (TypeError, ValueError, IndexError, KeyError):
                raise ContractError(
                    f"ffiw10k: the pair list {path} has an entry {row!r} that is not a "
                    "[source, target] pair of ids",
                    hint=_PAIR_HINT,
                ) from None
            pairs[source] = target
            pairs[target] = source
    _log.debug("ffiw10k: the pair lists name %d videos", len(pairs))
    return pairs


class FFIW10KBuilder(BaseBuilder):
    """FFIW-10K (``ffiw10k``): in-the-wild source videos and their face-swapped targets."""

    dataset_id = "ffiw10k"
    expected_folder = "FFIW10K"
    label_prefix = "FFIW"
    tasks = (
        TaskSpec("REAL", "source", "real", "original_content/source/videos", "original"),
        TaskSpec("FS_FAKE", "target", "fake", "manipulated_content/target/videos", "unknown"),
    )
    metadata_files = tuple(f"{_PAIR_DIR}/{name}.json" for name in _PAIR_LISTS)
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FS_FAKE": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
    }
    schemes = {
        "official": SchemeSpec(
            "official",
            "official",
            source="the split prefix of each file name (train_, val_ or test_)",
            rationale="the publisher's train/val/test split, carried in each video's name",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set: up to 1,000 fakes drawn at random from "
            "the official test, balanced with as many reals",
        ),
    }
    default_scheme = "official"
    benchmark = BenchmarkSpec(k_fake=1000)
    card_info = {
        "name": "FFIW-10K",
        "aliases": ["FFIW10K", "FFIW", "Face Forensics in the Wild"],
        "release": "9,988 in-the-wild source videos, often with several faces, and 9,988 "
        "face-swapped target videos, named by split: 8,000 train, 250 val and 1,738 test of each",
        "homepage": "https://github.com/tfzhou/FFIW",
        "paper": {"title": "Face Forensics in the Wild", "venue": "CVPR", "year": 2021},
        "license": {
            "spdx": None,
            "summary": "the FFIW-10K terms to use: the download is granted by the authors on "
            "request",
            "url": None,
        },
        "access": "request the download through the Terms to Use form linked from the FFIW "
        "repository; dfwb never distributes media",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": "<task>/<split>_<8-digit index> (the file stem, e.g. REAL/train_00000000); "
        "the same stem appears in both tasks",
    }
    layout_notes = (
        "The official split is the prefix of each name: train_, val_ or test_ (only .mp4 names "
        "count).\n"
        f"Optional pair lists {_PAIR_DIR}/{{train,val,test}}.json ([[source, target], ...] "
        "integer ids) set each fake's pair_key to the stem test_<8 digits> of its pair; nothing "
        "is paired from them."
    )

    _pair_map: dict[str, str] | None = None

    def prepare(self, root: Path) -> None:
        """Read the optional pair lists under ``root`` once per build.

        Raises:
            ContractError: a pair list is not a list of ``[source, target]`` ids.
        """
        super().prepare(root)
        self._pair_map = _read_pair_map(root)

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem, which is also its identity."""
        stem = path.stem
        pair_key = None if task.kind == "real" else (self._pair_map or {}).get(stem)
        return self.record(task, stem, relpath, compression, identity=stem, pair_key=pair_key)

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's split, from the prefix of its name.

        A record counts when a ``.mp4`` of its stem is among the records (in either task).
        """
        named = {
            local_key(record.key)
            for record in records
            if PurePosixPath(record.relpath).suffix.lower() == _SPLIT_SUFFIX
        }
        assignment: dict[str, Split] = {}
        for record in records:
            stem = local_key(record.key)
            split = _OFFICIAL_SPLITS.get(stem.split("_", 1)[0].lower())
            if stem in named and split is not None:
                assignment[record.key] = split
        return assignment
