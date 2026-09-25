"""DFDM (DeepFakes from Different Models): face swaps made with five autoencoder models.

Layout, relative to the ``DFDM`` folder (``{cX}`` is ``c0``, ``c10`` or ``c23``):

* reals: ``original_content/Celeb-real/videos/<id>_<rec>.mp4``, a single version with no
  compression level;
* fakes: ``manipulated_content/<model>/{cX}/videos/<name>.mp4``, flat, for the models
  ``DFaker``, ``DFL-H128``, ``FaceSwap``, ``IAE`` and ``LightWeight``, where ``<name>`` is
  ``<target>_<source>_<rec>_<tag>`` (e.g. ``id28_id20_0008_4dfaker``).

The release ships only the fakes: 430 videos per model and quality level, so 2,150 per quality
level and 6,450 in all, made from 415 of Celeb-DF v2's 590 Celeb-real videos. This builder
expects all 590 Celeb-real videos in its own ``original_content/Celeb-real/videos``. The
release's ``DFDM_crf0``, ``DFDM_crf10`` and ``DFDM_crf23`` folders are the ``c0``, ``c10`` and
``c23`` compressions.

Every video is keyed by its file stem, so a fake's key repeats once per compression. A fake whose
stem has at least four ``_``-separated parts, the first two starting with ``id``, has the first
as its identity and target, the second as its source, and ``<target>_<rec>`` as its
``pair_key``: the key of the real it was made from. A real whose stem has at least two parts, the
first starting with ``id``, has that first part as its identity and target. A name that does not
parse is kept, with none of these fields set. A fake's method is its model's name.

There is no official split: the default scheme is an identity-disjoint carve on the target, so a
person's real videos and every fake of that person land on the same side, in every compression.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BenchmarkSpec, local_key

__all__ = ["DFDMBuilder"]

_log = logging.getLogger(__name__)


def _model(abbr: str, name: str) -> TaskSpec:
    """A face-swap model: its folder, task name and method share one name."""
    return TaskSpec(abbr, name, "fake", f"manipulated_content/{name}/{{cX}}/videos", name)


def _parse_fake(stem: str) -> tuple[str, str, str] | None:
    """``(target, source, rec)`` of ``id<t>_id<s>_<rec>_<tag>...``, else ``None``."""
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0].startswith("id") and parts[1].startswith("id"):
        return parts[0], parts[1], parts[2]
    return None


class DFDMBuilder(BaseBuilder):
    """DFDM (``dfdm``): Celeb-DF v2 reals and face swaps from five models at three qualities."""

    dataset_id = "dfdm"
    expected_folder = "DFDM"
    label_prefix = "DFDM"
    tasks = (
        TaskSpec("REAL", "Celeb-real", "real", "original_content/Celeb-real/videos", "original"),
        _model("FS_DF", "DFaker"),
        _model("FS_DFL", "DFL-H128"),
        _model("FS_FS", "FaceSwap"),
        _model("FS_IAE", "IAE"),
        _model("FS_LW", "LightWeight"),
    )
    known_compressions = ("c0", "c10", "c23")
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FS_DF": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
        "FS_DFL": LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-swap"),
        "FS_FS": LabelSpec(binary=1, binary_av=1, multiclass=4, family="face-swap"),
        "FS_IAE": LabelSpec(binary=1, binary_av=1, multiclass=5, family="face-swap"),
        "FS_LW": LabelSpec(binary=1, binary_av=1, multiclass=6, family="face-swap"),
    }
    schemes = {
        "ident-72-14-14": SchemeSpec(
            "ident-72-14-14",
            "derived",
            rationale="no official split; identity-disjoint md5 carve (72/14/14) on the target, "
            "so a person's real videos and every fake of that person stay on one side",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set: up to 50 fakes per model, drawn from the "
            "whole dataset across its compressions and balanced with as many reals",
        ),
    }
    default_scheme = "ident-72-14-14"
    benchmark = BenchmarkSpec(k_fake=50, strata=("task",))
    pairing_rule = "target-recording"
    card_info = {
        "name": "DFDM",
        "aliases": ["DeepFakes from Different Models"],
        "release": "6,450 face swaps: 430 videos per model and quality level from five "
        "autoencoder models (Faceswap, Lightweight, IAE, Dfaker, DFL-H128) at three H.264 "
        "quality levels (lossless, high and low), so 2,150 per quality level, made from 415 of "
        "Celeb-DF v2's 590 Celeb-real videos; the reals are Celeb-DF v2's, not part of this "
        "release, and all 590 go in the real folder",
        "homepage": "https://github.com/shanface33/Deepfake_Model_Attribution",
        "paper": {
            "title": "Model Attribution of Face-Swap Deepfake Videos",
            "venue": "ICIP",
            "year": 2022,
        },
        "license": {
            "spdx": None,
            "summary": "released for academic research only: researchers at educational "
            "institutes may use it free of charge for non-commercial purposes",
            "url": None,
        },
        "access": "download from the link in the DFDM repository; the reals come from "
        "Celeb-DF v2, which is requested from its own authors; dfwb never distributes media",
        "modalities": ["video"],
        "compressions": ["c0", "c10", "c23"],
        "key_rule": "real: REAL/<id>_<rec>; fake: <task>/<target>_<source>_<rec>_<tag> "
        "(the file stem)",
    }
    layout_notes = (
        "The release's DFDM_crf0, DFDM_crf10 and DFDM_crf23 folders are the c0, c10 and c23 "
        "compressions: each model's videos go in manipulated_content/<model>/{cX}/videos/.\n"
        "The release has no reals: its fakes were made from 415 of Celeb-DF v2's 590 "
        "Celeb-real videos. All 590 go in original_content/Celeb-real/videos/ here (the same "
        "files as Celeb-DF v2's own Celeb-real folder).\n"
        "Fakes are named <target>_<source>_<rec>_<tag>; each pairs with the real <target>_<rec>."
    )

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; its fields come from the name."""
        stem = path.stem
        if task.kind == "real":
            parts = stem.split("_")
            if len(parts) >= 2 and parts[0].startswith("id"):
                return self.record(
                    task, stem, relpath, compression, identity=parts[0], target_id=parts[0]
                )
            _log.debug("dfdm: real %s is not named id<n>_<rec>", relpath)
            return self.record(task, stem, relpath, compression)
        parsed = _parse_fake(stem)
        if parsed is None:
            _log.debug("dfdm: fake %s is not named id<t>_id<s>_<rec>_<tag>", relpath)
            return self.record(task, stem, relpath, compression)
        target, source, rec = parsed
        return self.record(
            task,
            stem,
            relpath,
            compression,
            identity=target,
            target_id=target,
            source_id=source,
            pair_key=f"{target}_{rec}",
        )

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """``<target>_<rec>``: the key of the real the fake was made from."""
        parsed = _parse_fake(local_key(fake.key))
        return None if parsed is None else f"{parsed[0]}_{parsed[2]}"
