"""UADFV: a small set of real videos, each with one face-swapped copy.

Layout, relative to the ``UADFV`` folder (a single version, no compression levels):

* reals: ``original_content/real/videos/<n>.mp4``;
* fakes: ``manipulated_content/fake/videos/<n>_fake.mp4``, the face-swapped copy of real ``<n>``.

File names carry no identity, target or source, only the one-to-one link between a fake and its
real. A real's identity is its own id; a fake's identity, target and ``pair_key`` are the id of
its real (the name without ``_fake``), and it records no source. A fake not named ``<n>_fake`` is
kept, with none of these fields set.

The dataset is too small to split, so every video is test and there is no benchmark subset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import local_key

__all__ = ["UADFVBuilder"]

_FAKE_SUFFIX: Final = "_fake"
# UADFV has no homepage: the owners release it through this agreement form.
_AGREEMENT_FORM: Final = (
    "https://docs.google.com/forms/d/e/1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/"
    "viewform"
)


def _real_of(stem: str) -> str | None:
    """The real's id a fake named ``<n>_fake`` was made from, else ``None``."""
    return stem[: -len(_FAKE_SUFFIX)] if stem.endswith(_FAKE_SUFFIX) else None


class UADFVBuilder(BaseBuilder):
    """UADFV (``uadfv``): real videos and one face swap of each."""

    dataset_id = "uadfv"
    expected_folder = "UADFV"
    label_prefix = "UADFV"
    tasks = (
        TaskSpec("REAL", "real", "real", "original_content/real/videos", "original"),
        TaskSpec("FS_FAKE", "fake", "fake", "manipulated_content/fake/videos", "faceswap"),
    )
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FS_FAKE": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
    }
    schemes = {
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="too small to split (49 reals and 49 fakes): every video is test",
        ),
    }
    default_scheme = "all-test"
    pairing_rule = "strip-fake-suffix"
    card_info = {
        "name": "UADFV",
        "aliases": [],
        "release": "49 real videos and their 49 face swaps",
        "homepage": None,
        "paper": {
            "title": "Exposing Deep Fakes Using Inconsistent Head Poses",
            "venue": "ICASSP",
            "year": 2019,
            "doi": "10.1109/ICASSP.2019.8683164",
        },
        "license": {
            "spdx": None,
            "summary": "the Terms to use UADFV, agreed in the owners' agreement form; the terms "
            "need review",
            "url": None,
        },
        "access": f"request the download through the owners' agreement form ({_AGREEMENT_FORM}); "
        "dfwb never distributes media",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": "real: REAL/<n>; fake: FS_FAKE/<n>_fake (the file stem)",
    }
    layout_notes = "Fakes are named <n>_fake and pair with the real <n>."

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; a fake's fields name its real."""
        stem = path.stem
        if task.kind == "real":
            return self.record(task, stem, relpath, compression, identity=stem)
        real = _real_of(stem)
        return self.record(
            task, stem, relpath, compression, identity=real, target_id=real, pair_key=real
        )

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The fake's name without ``_fake``: the key of its real."""
        return _real_of(local_key(fake.key))
