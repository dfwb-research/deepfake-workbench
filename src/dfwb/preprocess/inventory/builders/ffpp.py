"""FaceForensics++: 1,000 YouTube sequences and five manipulation methods applied to them.

Layout, relative to the ``FaceForensics++`` folder (``{cX}`` is ``raw``, ``c23`` or ``c40``):

* reals: ``original_content/YouTube/{cX}/videos/<seq>.mp4``, where ``<seq>`` is a zero-padded
  three-digit sequence id such as ``000``;
* fakes: ``manipulated_content/<Method>/{cX}/videos/<target>_<source>.mp4``, for the methods
  Deepfakes, Face2Face, FaceShifter, FaceSwap and NeuralTextures. The target sequence keeps its
  background; the source sequence gives the face or the expression.

A real's identity and target are its sequence id. A fake's identity and target are its target
sequence, its source is the source sequence, and its ``pair_key`` is ``<target>_<source>``. A file
in a method folder whose name has no ``_`` is not a fake of this release and is skipped.

The official split is three JSON files, ``train.json``, ``val.json`` and ``test.json``, each a
list of ``[target, source]`` sequence pairs (720/140/140 sequences). A video goes to the first
list, in train, val, test order, that names the sequence before the first ``_`` of its key: its
own sequence for a real, its target sequence for a fake.

The DeepFakeDetection release (``dfd``) is distributed through the same download and lives in the
same folder; its builder reads only its own sub-folders, and this one only these.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BENCHMARK_REALS, BenchmarkSpec, Split, local_key

__all__ = ["FaceForensicsBuilder"]

_log = logging.getLogger(__name__)

# The official split files, relative to the dataset folder, in the order they are consulted.
_OFFICIAL_DIR: Final = ".official_files/splits"
_OFFICIAL_SPLITS: Final[tuple[Split, ...]] = ("train", "val", "test")


def _method(abbr: str, name: str) -> TaskSpec:
    return TaskSpec(abbr, name, "fake", f"manipulated_content/{name}/{{cX}}/videos", name)


def _read_official_ids(root: Path) -> dict[Split, frozenset[str]]:
    """The sequence ids each official split file names, keyed by split in train, val, test order.

    Each file is a JSON list whose entries are ``[target, source]`` pairs; a bare id is accepted
    too. Every id of a pair belongs to that split.

    Raises:
        ConfigError: a split file is missing.
        ContractError: a split file is not a JSON list.
    """
    folder = root / _OFFICIAL_DIR
    missing = [
        f"{split}.json" for split in _OFFICIAL_SPLITS if not (folder / f"{split}.json").is_file()
    ]
    if missing:
        raise ConfigError(
            f"ffpp: the official split file(s) {', '.join(missing)} are missing from {folder}",
            hint="copy train.json, val.json and test.json from the FaceForensics repository's "
            f"split lists into {_OFFICIAL_DIR}/ in the dataset folder",
        )
    ids: dict[Split, frozenset[str]] = {}
    for split in _OFFICIAL_SPLITS:
        path = folder / f"{split}.json"
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ContractError(
                f"ffpp: cannot read the official split file {path}: {exc}",
                hint="the file is a JSON list of [target, source] sequence-id pairs",
            ) from None
        if not isinstance(rows, list):
            raise ContractError(
                f"ffpp: the official split file {path} holds a {type(rows).__name__}, not a list",
                hint="the file is a JSON list of [target, source] sequence-id pairs",
            )
        found: set[str] = set()
        for row in rows:
            if isinstance(row, list):
                found.update(str(value) for value in row)
            else:
                found.add(str(row))
        ids[split] = frozenset(found)
        _log.debug("ffpp: %s lists %d sequence ids", path.name, len(found))
    return ids


class FaceForensicsBuilder(BaseBuilder):
    """FaceForensics++ (``ffpp``): the YouTube originals and five manipulation methods."""

    dataset_id = "ffpp"
    expected_folder = "FaceForensics++"
    label_prefix = "FF"
    tasks = (
        TaskSpec("REAL", "YouTube", "real", "original_content/YouTube/{cX}/videos", "original"),
        _method("FS_DF", "Deepfakes"),
        _method("FR_F2F", "Face2Face"),
        _method("FS_FSH", "FaceShifter"),
        _method("FS_FS", "FaceSwap"),
        _method("FR_NT", "NeuralTextures"),
    )
    known_compressions = ("raw", "c23", "c40")
    metadata_files = tuple(f"{_OFFICIAL_DIR}/{split}.json" for split in _OFFICIAL_SPLITS)
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FS_DF": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
        "FR_F2F": LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-reenactment"),
        "FS_FSH": LabelSpec(binary=1, binary_av=1, multiclass=4, family="face-swap"),
        "FS_FS": LabelSpec(binary=1, binary_av=1, multiclass=5, family="face-swap"),
        "FR_NT": LabelSpec(binary=1, binary_av=1, multiclass=6, family="face-reenactment"),
    }
    schemes = {
        "official": SchemeSpec(
            "official",
            "official",
            source=f"the FaceForensics++ train/val/test split lists ({_OFFICIAL_DIR}/*.json)",
            rationale="the publisher's split of the 1,000 sequences (720/140/140); every fake "
            "follows its target sequence",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set at c23: 500 fakes drawn at random from the "
            f"official test split's c23 videos, {BENCHMARK_REALS}",
        ),
    }
    default_scheme = "official"
    # Defined at c23, so the subset is the same whichever other compressions a copy holds.
    benchmark = BenchmarkSpec(k_fake=500, compressions=("c23",))
    pairing_rule = "target-id"
    card_info = {
        "name": "FaceForensics++",
        "aliases": ["FF++", "FaceForensics++"],
        "release": "1,000 YouTube sequences, each manipulated with Deepfakes, Face2Face, "
        "FaceShifter, FaceSwap and NeuralTextures",
        "homepage": "https://github.com/ondyari/FaceForensics",
        "license": {
            "spdx": None,
            "summary": "the FaceForensics terms of use: non-commercial research only",
            "url": None,
        },
        "access": "request the download script through the form linked from the FaceForensics "
        "repository; dfwb never distributes media",
        "modalities": ["video"],
        "compressions": ["raw", "c23", "c40"],
        "key_rule": "real: REAL/<seq>; fake: <task>/<target>_<source> (the file stem)",
    }
    layout_notes = (
        "Fakes are named <target>_<source>; a file in a method folder without '_' is skipped.\n"
        f"The official split lists train.json, val.json and test.json go in {_OFFICIAL_DIR}/.\n"
        "DeepFakeDetection (dfd) shares this folder."
    )

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """A real is keyed by its sequence id; a fake by ``<target>_<source>``."""
        stem = path.stem
        if task.kind == "real":
            return self.record(task, stem, relpath, compression, identity=stem, target_id=stem)
        if "_" not in stem:
            _log.debug("ffpp: skipping %s: a fake is named <target>_<source>", relpath)
            return None
        target, source = stem.split("_")[:2]
        return self.record(
            task,
            stem,
            relpath,
            compression,
            identity=target,
            target_id=target,
            source_id=source,
            pair_key=f"{target}_{source}",
        )

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's official split, by the sequence before the first ``_`` of its key.

        Raises:
            ConfigError: a split file is missing.
            ContractError: a split file is not a JSON list.
        """
        official = _read_official_ids(root)
        assignment: dict[str, Split] = {}
        for record in records:
            sequence = local_key(record.key).split("_", 1)[0]
            for split, ids in official.items():
                if sequence in ids:
                    assignment[record.key] = split
                    break
        return assignment

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The fake's target sequence: the real of the same id."""
        return fake.target_id or local_key(fake.key).split("_")[0]
