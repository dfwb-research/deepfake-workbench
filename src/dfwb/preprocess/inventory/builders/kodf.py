"""KoDF: real clips of Korean actors and clips synthesized from them with five methods.

Layout, relative to the ``KoDF`` folder (a single version, no compression levels). The videos
nest, so every task is searched recursively:

* reals: ``original_content/actors/videos/<actor>/<actor>_<NNN>.mp4``, one folder per actor; an
  actor id is either 20 hexadecimal characters or six digits;
* fakes: ``manipulated_content/<method>/videos/<date>/<target>/<name>.mp4`` for the methods
  ``audio-driven``, ``dffs``, ``dfl``, ``fo`` and ``fsgan``, where ``<name>`` is
  ``<target>_<source>_<code>_<seq>``.

Any nesting below a task's ``videos`` folder is found: only the file names matter.

Every video is keyed by its file stem. A real's identity and target are the stem up to its last
``_`` (a stem without ``_`` has neither). A fake whose stem has exactly four ``_``-separated
parts takes the first as its identity and target and the second as its source, and its method is
the one its code names: 1 ``dfl``, 2 ``dffs``, 3 ``fsgan``, 4 ``fo``, 5 ``audio-driven``. An
unknown code keeps the task's method, and a stem of any other shape keeps the task's method and
has none of these fields. No record has a ``pair_key``: a fake pairs with the reals of its target
actor, and only the first of them by key is kept.

There is no official split: the default scheme is an identity-disjoint carve on the actor, so an
actor's clips and every fake of that actor land on the same side.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BENCHMARK_REALS, BenchmarkSpec, local_key

__all__ = ["KoDFBuilder"]

_log = logging.getLogger(__name__)

# The method each code in a fake's name stands for.
_METHOD_CODES: Final[dict[str, str]] = {
    "1": "dfl",
    "2": "dffs",
    "3": "fsgan",
    "4": "fo",
    "5": "audio-driven",
}


def _fake(abbr: str, method: str) -> TaskSpec:
    """A synthesis method: its folder, task name and method share one name."""
    return TaskSpec(
        abbr, method, "fake", f"manipulated_content/{method}/videos", method, recursive=True
    )


def _parse_fake(stem: str) -> list[str] | None:
    """``[target, source, code, seq]`` of a four-part stem, else ``None``."""
    parts = stem.split("_")
    return parts if len(parts) == 4 else None


class KoDFBuilder(BaseBuilder):
    """KoDF (``kodf``): Korean actors' clips and five synthesis methods applied to them."""

    dataset_id = "kodf"
    expected_folder = "KoDF"
    label_prefix = "KoDF"
    tasks = (
        TaskSpec(
            "REAL", "Actors", "real", "original_content/actors/videos", "original", recursive=True
        ),
        _fake("LS_AD", "audio-driven"),
        _fake("FS_DFFS", "dffs"),
        _fake("FS_DFL", "dfl"),
        _fake("FR_FO", "fo"),
        _fake("FS_FSG", "fsgan"),
    )
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "LS_AD": LabelSpec(binary=1, binary_av=1, multiclass=2, family="lip-sync"),
        "FS_DFFS": LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-swap"),
        "FS_DFL": LabelSpec(binary=1, binary_av=1, multiclass=4, family="face-swap"),
        "FR_FO": LabelSpec(binary=1, binary_av=1, multiclass=5, family="face-reenactment"),
        "FS_FSG": LabelSpec(binary=1, binary_av=1, multiclass=6, family="face-swap"),
    }
    schemes = {
        "ident-72-14-14": SchemeSpec(
            "ident-72-14-14",
            "derived",
            rationale="no official split; identity-disjoint md5 carve (72/14/14) on the actor, "
            "so an actor's clips and every fake of that actor stay on one side",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set: up to 2 fakes per target actor and method, "
            f"drawn from the whole dataset, {BENCHMARK_REALS}",
        ),
    }
    default_scheme = "ident-72-14-14"
    benchmark = BenchmarkSpec(k_fake=2, strata=("target_id", "task"))
    pairing_rule = "identity-fanout"
    pairing_fanout = 1
    card_info = {
        "name": "KoDF",
        "aliases": ["Korean DeepFake Detection Dataset"],
        "release": "62,166 real clips of 403 Korean subjects and 175,776 clips synthesized from "
        "them with six synthesis models, in five method folders: FaceSwap, DeepFaceLab and "
        "FSGAN (face swapping), First Order Motion Model (FOMM; video-driven face "
        "reenactment), and Audio-driven Talking Face Head Pose (ATFHP) and Wav2Lip (audio-driven "
        "face reenactment), which share the audio-driven folder",
        "homepage": "https://github.com/deepbrainai-research/kodf",
        "paper": {
            "title": "KoDF: A Large-Scale Korean DeepFake Detection Dataset",
            "venue": "ICCV",
            "year": 2021,
        },
        "license": {
            "spdx": None,
            "summary": "the KoDF terms of use, agreed when the download is requested; the terms "
            "need review",
            "url": None,
        },
        "access": "request the download through the form linked from the KoDF repository "
        "(Korean users download it from AI-Hub); dfwb never distributes media",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": "real: REAL/<actor>_<NNN>; fake: <task>/<target>_<source>_<code>_<seq> "
        "(the file stem)",
    }
    layout_notes = (
        "Every task folder is searched recursively, so any nesting below each videos/ folder "
        "works and only the file names matter; reals usually nest one folder per actor "
        "(<actor>/<video>) and fakes as <date>/<target>/<video>.\n"
        "Reals are named <actor>_<NNN>; fakes <target>_<source>_<code>_<seq>, where <code> is 1 "
        "dfl, 2 dffs, 3 fsgan, 4 fo or 5 audio-driven and sets the record's method."
    )

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; its fields come from the name."""
        stem = path.stem
        if task.kind == "real":
            cut = stem.rfind("_")
            if cut == -1:
                _log.debug("kodf: real %s is not named <actor>_<NNN>", relpath)
                return self.record(task, stem, relpath, compression)
            actor = stem[:cut]
            return self.record(task, stem, relpath, compression, identity=actor, target_id=actor)
        parts = _parse_fake(stem)
        if parts is None:
            _log.debug("kodf: fake %s is not named <target>_<source>_<code>_<seq>", relpath)
            return self.record(task, stem, relpath, compression)
        target, source, code, _ = parts
        return self.record(
            task,
            stem,
            relpath,
            compression,
            identity=target,
            target_id=target,
            source_id=source,
            method=_METHOD_CODES.get(code, task.method),
        )

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The target actor of a four-part fake: every real of that actor."""
        parts = _parse_fake(local_key(fake.key))
        return None if parts is None else parts[0]
