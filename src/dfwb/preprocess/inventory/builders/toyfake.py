"""toyfake: the synthetic dataset ``dfwb datasets synth toyfake`` generates on this machine.

Nobody can try a deepfake framework without first obtaining licensed data, so dfwb generates its
own: short videos of smooth, moving, textured blobs (the reals), and copies of them with a patch
blended in that carries a known high-frequency artefact (the fakes). The generator lives in
:mod:`dfwb.preprocess.toyfake`; this builder reads what it wrote, exactly as any other builder
reads a downloaded release. Layout, relative to the ``toyfake`` folder:

* reals: ``original/<id>.mkv``, where ``<id>`` is an identity ``p000``, ``p001``, ...;
* fakes: ``blend-a/<target>_<source>.mkv`` (a period-2 checkerboard artefact) and
  ``blend-b/<target>_<source>.mkv`` (a period-3 diagonal stripe): real ``<target>`` with a patch
  of real ``<source>`` blended in;
* ``official_splits.json``: ``{"train": [identity, ...], "val": [...], "test": [...]}``, an
  identity-disjoint split drawn from the seed.

A real's identity and target are its own id. A fake's identity, target and ``pair_key`` are its
target and its source is its source, so every fake pairs with the real it was made from and
follows that identity into its split. A fake whose name does not parse is kept, with none of these
fields set.

This module holds the layout the generator writes, so the two never disagree; it imports neither
numpy nor a media library.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, get_args

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import DatasetCard, InventoryRecord, SchemeCard
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BENCHMARK_REALS, BenchmarkSpec, Split

__all__ = [
    "FOLDER",
    "OFFICIAL_SPLITS_FILE",
    "SPLIT_NAMES",
    "TASK_DIRS",
    "VIDEO_SUFFIX",
    "ToyfakeBuilder",
]

_log = logging.getLogger(__name__)

FOLDER: Final = "toyfake"
OFFICIAL_SPLITS_FILE: Final = "official_splits.json"
VIDEO_SUFFIX: Final = ".mkv"
# The splits official_splits.json holds, in the order the generator writes them.
SPLIT_NAMES: Final[tuple[Split, ...]] = ("train", "val", "test")
# Each task's folder, which is also each fake's method.
TASK_DIRS: Final[Mapping[str, str]] = {
    "REAL": "original",
    "BLEND_A": "blend-a",
    "BLEND_B": "blend-b",
}

_KNOWN_SPLITS: Final = frozenset(get_args(Split))
_TERMS_NOTES: Final = (
    "toyfake is synthetic: dfwb generates every video from a seed, and no real person appears in "
    "any of them. These lists are released under the MIT licence with dfwb and may be "
    "redistributed."
)


def _parse_fake(stem: str) -> tuple[str, str] | None:
    """``(target, source)`` of ``<target>_<source>``, else ``None``."""
    target, sep, source = stem.partition("_")
    if not sep or not target or not source or "_" in source:
        return None
    return target, source


def _read_official_splits(root: Path) -> dict[str, Split]:
    """Identity -> split, from ``official_splits.json``.

    Raises:
        ConfigError: the file is missing.
        ContractError: the file is not a JSON object of identity lists keyed by split, or an
            identity is listed in more than one split.
    """
    path = root / OFFICIAL_SPLITS_FILE
    shape = 'the file is a JSON object {"train": [identity, ...], "val": [...], "test": [...]}'
    if not path.is_file():
        raise ConfigError(
            f"toyfake: {OFFICIAL_SPLITS_FILE} is missing from {root}",
            hint="generate the dataset again with dfwb datasets synth toyfake",
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ContractError(f"toyfake: cannot read {path}: {exc}", hint=shape) from None
    if not isinstance(data, dict):
        raise ContractError(
            f"toyfake: {path} holds a {type(data).__name__}, not an object", hint=shape
        )
    split_of: dict[str, Split] = {}
    for split, identities in data.items():
        if split not in _KNOWN_SPLITS:
            raise ContractError(f"toyfake: {path} has an unknown split {split!r}", hint=shape)
        if not isinstance(identities, list):
            raise ContractError(
                f"toyfake: {path}: {split!r} is not a list of identities", hint=shape
            )
        for identity in map(str, identities):
            if identity in split_of:
                raise ContractError(
                    f"toyfake: {path} lists {identity} in both {split_of[identity]!r} and "
                    f"{split!r}",
                    hint="the official split is identity-disjoint: list each identity once",
                )
            split_of[identity] = split
    _log.debug("toyfake: %s lists %d identities", path.name, len(split_of))
    return split_of


class ToyfakeBuilder(BaseBuilder):
    """toyfake (``toyfake``): dfwb's own synthetic reals and two blended-patch fake methods."""

    dataset_id = "toyfake"
    expected_folder = FOLDER
    label_prefix = "TOY"
    tasks = (
        TaskSpec("REAL", "original", "real", TASK_DIRS["REAL"], "original"),
        TaskSpec("BLEND_A", "blend-a", "fake", TASK_DIRS["BLEND_A"], TASK_DIRS["BLEND_A"]),
        TaskSpec("BLEND_B", "blend-b", "fake", TASK_DIRS["BLEND_B"], TASK_DIRS["BLEND_B"]),
    )
    metadata_files = (OFFICIAL_SPLITS_FILE,)
    labels = {
        "REAL": LabelSpec(binary=0, binary_av=0, multiclass=0, family="real"),
        "BLEND_A": LabelSpec(binary=1, binary_av=1, multiclass=1, family="blend"),
        "BLEND_B": LabelSpec(binary=1, binary_av=1, multiclass=2, family="blend"),
    }
    schemes = {
        "official": SchemeSpec(
            "official",
            "official",
            source=f"{OFFICIAL_SPLITS_FILE}, written by dfwb datasets synth toyfake",
            rationale="the generator's identity-disjoint split (60/20/20 of the identities, "
            "drawn from the seed); every fake follows its target identity",
        ),
        "ident-72-14-14": SchemeSpec(
            "ident-72-14-14",
            "derived",
            rationale="an identity-disjoint md5 carve (72/14/14) on the target identity, to "
            "exercise a split that does not come from the publisher",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set: 20 fakes drawn at random from the "
            f"official test split, {BENCHMARK_REALS}",
        ),
    }
    default_scheme = "official"
    benchmark = BenchmarkSpec(k_fake=20)
    pairing_rule = "target-id"
    card_info = {
        "name": "toyfake",
        "aliases": [],
        "release": "synthetic: generated locally by dfwb from a seed; this pack lists the tree "
        "made with --videos 200 --seed 0 (80 reals, 60 fakes of each method)",
        "homepage": "https://github.com/dfwb-research/deepfake-workbench",
        "license": {
            "spdx": "MIT",
            "summary": "MIT, with dfwb: synthetic videos generated from a seed, with no real "
            "person in them",
            "url": None,
        },
        "access": "generated locally with `dfwb datasets synth toyfake`; there is nothing to "
        "download",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": "real: REAL/<id>; fake: BLEND_A/<target>_<source> or "
        "BLEND_B/<target>_<source> (the file stem)",
    }
    layout_notes = (
        "Generate it with: dfwb datasets synth toyfake --out <datasets root>\n"
        "Fakes are named <target>_<source> and pair with the real <target>.\n"
        f"The official split, {OFFICIAL_SPLITS_FILE}, lists the identities of each split."
    )

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; a fake's fields come from its name."""
        stem = path.stem
        if task.kind == "real":
            return self.record(task, stem, relpath, compression, identity=stem, target_id=stem)
        parsed = _parse_fake(stem)
        if parsed is None:
            return self.record(task, stem, relpath, compression)
        target, source = parsed
        return self.record(
            task,
            stem,
            relpath,
            compression,
            identity=target,
            target_id=target,
            source_id=source,
            pair_key=target,
        )

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's split: the one ``official_splits.json`` lists its identity in.

        Raises:
            ConfigError: the file is missing.
            ContractError: the file is malformed, or lists an identity twice.
        """
        split_of = _read_official_splits(root)
        return {
            record.key: split_of[record.identity]
            for record in records
            if record.identity is not None and record.identity in split_of
        }

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The fake's target, the real it was made from; ``None`` if its name did not parse."""
        return fake.target_id

    def dataset_card(self, schemes: Mapping[str, SchemeCard]) -> DatasetCard:
        """The card, with its terms settled: the lists are dfwb's own, so they are published.

        Raises:
            ContractError: as :meth:`BaseBuilder.dataset_card`.
        """
        card = super().dataset_card(schemes)
        data: dict[str, Any] = card.model_dump(mode="json", by_alias=True)
        data["distribution"] = "list"
        data["terms"] = {"source": "dfwb", "reviewed": None, "notes": _TERMS_NOTES}
        return DatasetCard.model_validate(data)
