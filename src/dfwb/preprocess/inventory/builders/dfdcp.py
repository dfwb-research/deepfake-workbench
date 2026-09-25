"""DFDC Preview: the preview release of the Deepfake Detection Challenge dataset.

Layout, relative to the ``DFDC-P`` folder (a single version, no compression levels). Every task
nests its videos in sub-folders, so each task directory is searched recursively:

* reals: ``original_content/original/videos/<id>/<id>_<suffix>_<counter>.mp4``, one folder per
  actor: clip ``<counter>`` of recording ``<suffix>`` of the actor ``<id>``;
* fakes: ``manipulated_content/method_A/videos/<target>/<target>_<suffix>/<video>.mp4`` (and the
  same under ``method_B``), a folder per target actor and one per recording, each named
  ``<swapped>_<target>_<suffix>_<counter>``: the face of actor ``<swapped>`` put onto clip
  ``<target>_<suffix>_<counter>``.

Every video is keyed by its file stem; the relpath keeps the nested folders. A real's identity and
target are its first part (a name needs at least three parts). A fake's identity and target are
``<target>``, its source is ``<swapped>`` (both must be numeric), and its ``pair_key`` is
``<target>_<suffix>_<counter>``: the key of the real clip it was made from. Any parts past the
fourth stay in the counter. A name that does not parse is kept, with none of these fields set.

The official split is ``.official_files/dataset.json``, keyed by video path, where each video's
``set`` is ``train`` or ``test``; a video goes to the split that lists its file stem, train
first. The default scheme keeps the official test and carves the official train 80/20 into train
and val by identity.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Final

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BenchmarkSpec, Split, local_key

__all__ = ["DFDCPreviewBuilder"]

_log = logging.getLogger(__name__)

# The preview's metadata, relative to the dataset folder.
_METADATA: Final = ".official_files/dataset.json"
# The sets the metadata can name, in the order a video's stem is looked up.
_OFFICIAL_SPLITS: Final[tuple[Split, ...]] = ("train", "test")
_HINT: Final = "the file is a JSON object keyed by video path; each value holds its 'set'"


def _parse_fake(stem: str) -> tuple[str, str, str] | None:
    """``(target, swapped, pair_key)`` of ``<swapped>_<target>_<suffix>_<counter>``, else None."""
    parts = stem.split("_")
    if len(parts) < 4 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return parts[1], parts[0], "_".join(parts[1:])


def _read_official_stems(root: Path) -> dict[Split, frozenset[str]]:
    """The file stems of each official set (train, then test).

    A video whose ``set`` is neither ``train`` nor ``test`` (ignoring case and surrounding space)
    is in neither.

    Raises:
        ConfigError: the metadata file is missing.
        ContractError: the file is not a JSON object of objects, or a ``set`` is not text.
    """
    path = root / _METADATA
    if not path.is_file():
        raise ConfigError(
            f"dfdc-p: the official split file {_METADATA} is missing from {root}",
            hint="copy the preview release's dataset.json into .official_files/ in the "
            "dataset folder",
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ContractError(
            f"dfdc-p: cannot read the official split file {path}: {exc}", hint=_HINT
        ) from None
    if not isinstance(metadata, dict):
        raise ContractError(
            f"dfdc-p: {path} holds a {type(metadata).__name__}, not an object", hint=_HINT
        )
    found: dict[Split, set[str]] = {split: set() for split in _OFFICIAL_SPLITS}
    for video, entry in metadata.items():
        if not isinstance(entry, dict):
            raise ContractError(
                f"dfdc-p: {path}: the entry for {video!r} is not an object", hint=_HINT
            )
        value = entry.get("set") or ""
        if not isinstance(value, str):
            raise ContractError(
                f"dfdc-p: {path}: the 'set' of {video!r} is {value!r}, not text", hint=_HINT
            )
        name = value.strip().lower()
        for split in _OFFICIAL_SPLITS:
            if name == split:
                found[split].add(PurePosixPath(video).stem)
    _log.debug("dfdc-p: %s lists %s", path.name, {s: len(v) for s, v in found.items()})
    return {split: frozenset(stems) for split, stems in found.items()}


class DFDCPreviewBuilder(BaseBuilder):
    """DFDC Preview (``dfdc-p``): actor clips and face swaps made with two methods."""

    dataset_id = "dfdc-p"
    expected_folder = "DFDC-P"
    label_prefix = "DFDCP"
    tasks = (
        TaskSpec(
            "REAL",
            "Original",
            "real",
            "original_content/original/videos",
            "original",
            recursive=True,
        ),
        TaskSpec(
            "FS_METHOD_A",
            "method_A",
            "fake",
            "manipulated_content/method_A/videos",
            "method_a",
            recursive=True,
        ),
        TaskSpec(
            "FS_METHOD_B",
            "method_B",
            "fake",
            "manipulated_content/method_B/videos",
            "method_b",
            recursive=True,
        ),
    )
    metadata_files = (_METADATA,)
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FS_METHOD_A": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
        "FS_METHOD_B": LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-swap"),
    }
    schemes = {
        "official+ident-80-20": SchemeSpec(
            "official+ident-80-20",
            "derived",
            params={"policy": "official-train-test"},
            source=f"the preview's train/test sets ({_METADATA})",
            rationale="the publisher's split has train and test but no val: the official test "
            "is kept, and the official train is carved 80/20 into train and val by an md5 of "
            "the target identity, so an actor's clips and their fakes stay on one side",
        ),
        "official": SchemeSpec(
            "official",
            "official",
            source=f"the preview's train/test sets ({_METADATA})",
            rationale="the publisher's split as released: train and test, no val",
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
            "fakes per target identity and method, balanced with as many reals",
        ),
    }
    default_scheme = "official+ident-80-20"
    benchmark = BenchmarkSpec(k_fake=5, strata=("identity", "task"))
    pairing_rule = "target-clip"
    card_info = {
        "name": "DFDC Preview",
        "aliases": ["DFDC-P", "DFDCP", "Deepfake Detection Challenge Preview"],
        "release": "the preview release of the Deepfake Detection Challenge dataset: 1,131 "
        "clips of paid actors and 4,119 face swaps made with two methods",
        "homepage": None,
        "license": {
            "spdx": None,
            "summary": "the DFDC preview terms of use: non-commercial research only",
            "url": None,
        },
        "access": "the preview was distributed through deepfakedetectionchallenge.ai after "
        "accepting its terms; dfwb never distributes media",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": "real: REAL/<id>_<suffix>_<counter>; fake: "
        "<task>/<swapped>_<target>_<suffix>_<counter> (the file stem)",
    }
    layout_notes = (
        "Videos are nested: reals as <id>/<video>, fakes as <target>/<target>_<suffix>/<video>.\n"
        f"The official split reads {_METADATA} (the preview's train/test sets)."
    )

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; its fields come from the name."""
        stem = path.stem
        if task.kind == "real":
            parts = stem.split("_")
            identity = parts[0] if len(parts) >= 3 else None
            return self.record(
                task, stem, relpath, compression, identity=identity, target_id=identity
            )
        parsed = _parse_fake(stem)
        if parsed is None:
            _log.debug("dfdc-p: %s is not named <swapped>_<target>_<suffix>_<counter>", relpath)
            return self.record(task, stem, relpath, compression)
        target, swapped, pair_key = parsed
        return self.record(
            task,
            stem,
            relpath,
            compression,
            identity=target,
            target_id=target,
            source_id=swapped,
            pair_key=pair_key,
        )

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's official set, by its file stem; train is looked up before test.

        Raises:
            ConfigError: the metadata file is missing.
            ContractError: the metadata file is malformed.
        """
        official = _read_official_stems(root)
        assignment: dict[str, Split] = {}
        for record in records:
            stem = local_key(record.key)
            for split, stems in official.items():
                if stem in stems:
                    assignment[record.key] = split
                    break
        return assignment

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """``<target>_<suffix>_<counter>``: the key of the real clip the fake was made from."""
        parsed = _parse_fake(local_key(fake.key))
        return None if parsed is None else parsed[2]
