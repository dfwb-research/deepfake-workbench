"""DeepFakeDetection: recordings of paid actors and face swaps made between them.

The release is distributed through the FaceForensics++ download and lives in the same
``FaceForensics++`` folder. This builder reads only its own two sub-folders (``{cX}`` is ``raw``,
``c23`` or ``c40``):

* reals: ``original_content/Actors/{cX}/videos/<actor>__<scene>.mp4``;
* fakes: ``manipulated_content/DeepFakeDetection/{cX}/videos/<name>.mp4``, where ``<name>`` is
  ``<target>_<source>__<scene>__<code>``.

A real's identity and target are its actor id (the part before the first ``__``). A fake's
identity and target are its target actor, its source is the source actor, and its ``pair_key`` is
``<target>__<scene>``: the key of the real recording it was made from. A file whose name does not
parse is kept, with none of these fields set.

There is no official split: the default scheme is an identity-disjoint carve on the actor, so an
actor's recordings and every fake of that actor land on the same side.
"""

from __future__ import annotations

from pathlib import Path

from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BenchmarkSpec, local_key

__all__ = ["DeepFakeDetectionBuilder"]


def _parse_fake(stem: str) -> tuple[str, str, str] | None:
    """``(target, source, scene)`` of ``<target>_<source>__<scene>__<code>``, else ``None``."""
    parts = stem.split("__")
    if len(parts) < 3:
        return None
    ids = parts[0].split("_")
    if len(ids) < 2:
        return None
    return ids[0], ids[1], parts[1]


class DeepFakeDetectionBuilder(BaseBuilder):
    """DeepFakeDetection (``dfd``): actor recordings and their face swaps."""

    dataset_id = "dfd"
    expected_folder = "FaceForensics++"
    label_prefix = "DFD"
    tasks = (
        TaskSpec("AR", "Actors-real", "real", "original_content/Actors/{cX}/videos", "original"),
        TaskSpec(
            "FS_DFD",
            "DeepFakeDetection",
            "fake",
            "manipulated_content/DeepFakeDetection/{cX}/videos",
            "deepfakedetection",
        ),
    )
    known_compressions = ("raw", "c23", "c40")
    labels = {
        "AR": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FS_DFD": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
    }
    schemes = {
        "ident-72-14-14": SchemeSpec(
            "ident-72-14-14",
            "derived",
            rationale="no official split; identity-disjoint md5 carve (72/14/14) on the target "
            "actor, so an actor's recordings and their fakes stay on one side",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set: 100 fakes drawn at random from the whole "
            "dataset, balanced with as many reals",
        ),
    }
    default_scheme = "ident-72-14-14"
    benchmark = BenchmarkSpec(k_fake=100)
    pairing_rule = "dfd_target_scene"
    card_info = {
        "name": "DeepFakeDetection",
        "aliases": ["DFD"],
        "release": "363 actor recordings and 3,068 face swaps, as distributed with FaceForensics++",
        "homepage": "https://github.com/ondyari/FaceForensics",
        "license": {
            "spdx": None,
            "summary": "distributed under the FaceForensics terms of use: non-commercial "
            "research only",
            "url": None,
        },
        "access": "downloaded with the FaceForensics++ download script, requested through the "
        "form linked from the FaceForensics repository; dfwb never distributes media",
        "modalities": ["video"],
        "compressions": ["raw", "c23", "c40"],
        "key_rule": "real: AR/<actor>__<scene>; fake: FS_DFD/<target>_<source>__<scene>__<code> "
        "(the file stem)",
    }
    layout_notes = (
        "The folder is shared with FaceForensics++ (ffpp); each builder reads only its own "
        "sub-folders.\n"
        "Reals are named <actor>__<scene>; fakes <target>_<source>__<scene>__<code>."
    )

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; its fields come from the name."""
        stem = path.stem
        if task.kind == "real":
            actor = stem.split("__", 1)[0]
            return self.record(
                task,
                stem,
                relpath,
                compression,
                identity=actor or None,
                target_id=actor or None,
            )
        parsed = _parse_fake(stem)
        if parsed is None:
            return self.record(task, stem, relpath, compression)
        target, source, scene = parsed
        return self.record(
            task,
            stem,
            relpath,
            compression,
            identity=target,
            target_id=target,
            source_id=source,
            pair_key=f"{target}__{scene}",
        )

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """``<target>__<scene>``: the key of the real recording the fake was made from."""
        parsed = _parse_fake(local_key(fake.key))
        return None if parsed is None else f"{parsed[0]}__{parsed[2]}"
