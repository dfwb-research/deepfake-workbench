"""DFDC: the public test set of the Deepfake Detection Challenge.

Layout, relative to the ``DFDC`` folder (a single version, no compression levels):

* reals: ``original_content/testing_real/videos/<id>.mp4``;
* fakes: ``manipulated_content/testing_fake/videos/<id>.mp4``.

Each video is named by an opaque ten-letter id, which is its key. The names carry no identity,
source or target, and nothing links a fake to the real it was made from, so none of those fields
is set and nothing pairs.

Only the challenge's test set is on disk, so the official split is test only: a video whose file
stem is a key of ``.official_files/testing_metadata.json`` (the challenge's test metadata, keyed
by file name) is test. There is no train or val carve.
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
from dfwb.protocols.rules import BENCHMARK_REALS, BenchmarkSpec, Split, local_key

__all__ = ["DFDCBuilder"]

_log = logging.getLogger(__name__)

# The challenge's test metadata, relative to the dataset folder.
_METADATA: Final = ".official_files/testing_metadata.json"


def _read_test_stems(root: Path) -> frozenset[str]:
    """The file stems of the videos the test metadata lists (its keys are file names).

    Raises:
        ConfigError: the metadata file is missing.
        ContractError: the file is not a JSON object.
    """
    path = root / _METADATA
    if not path.is_file():
        raise ConfigError(
            f"dfdc: the test metadata {_METADATA} is missing from {root}",
            hint="copy the challenge's testing_metadata.json into .official_files/ in the "
            "dataset folder",
        )
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ContractError(
            f"dfdc: cannot read the test metadata {path}: {exc}",
            hint="the file is a JSON object keyed by video file name",
        ) from None
    if not isinstance(metadata, dict):
        raise ContractError(
            f"dfdc: the test metadata {path} holds a {type(metadata).__name__}, not an object",
            hint="the file is a JSON object keyed by video file name",
        )
    stems = frozenset(PurePosixPath(name).stem for name in metadata)
    _log.debug("dfdc: %s lists %d videos", path.name, len(stems))
    return stems


class DFDCBuilder(BaseBuilder):
    """DFDC (``dfdc``): the Deepfake Detection Challenge's public test set."""

    dataset_id = "dfdc"
    expected_folder = "DFDC"
    label_prefix = "DFDC"
    tasks = (
        TaskSpec(
            "REAL", "testing_real", "real", "original_content/testing_real/videos", "original"
        ),
        TaskSpec(
            "FS_FAKE", "testing_fake", "fake", "manipulated_content/testing_fake/videos", "deepfake"
        ),
    )
    metadata_files = (_METADATA,)
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FS_FAKE": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
    }
    schemes = {
        "official": SchemeSpec(
            "official",
            "official",
            source=f"the challenge's test metadata ({_METADATA})",
            rationale="only the challenge's public test set is released, so the publisher's "
            "split is test only; there is no train or val",
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
            f"the official test, {BENCHMARK_REALS}",
        ),
    }
    default_scheme = "official"
    benchmark = BenchmarkSpec(k_fake=1000)
    card_info = {
        "name": "DFDC",
        "aliases": ["Deepfake Detection Challenge"],
        "release": "the public test set of the Deepfake Detection Challenge: 5,000 videos of "
        "paid actors, half of them manipulated",
        "homepage": "https://ai.meta.com/datasets/dfdc/",
        "license": {
            "spdx": None,
            "summary": "the DFDC dataset terms of use: non-commercial research only",
            "url": None,
        },
        "access": "download the test set from the challenge's data page after accepting its "
        "terms; dfwb never distributes media",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": "real: REAL/<id>; fake: FS_FAKE/<id> (the file stem, an opaque ten-letter id)",
    }
    layout_notes = (
        f"The official split reads {_METADATA} (the challenge's test metadata).\n"
        "Names carry no identity and nothing pairs a fake with a real."
    )

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """``test`` for every record whose file stem the test metadata lists.

        Raises:
            ConfigError: the metadata file is missing.
            ContractError: the metadata file is not a JSON object.
        """
        stems = _read_test_stems(root)
        return {record.key: "test" for record in records if local_key(record.key) in stems}
