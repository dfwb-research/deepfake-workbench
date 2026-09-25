"""DeepSpeak v1 and v2: recordings of actors talking and fakes of them made by several engines.

The two releases share one layout and one split file and differ in their engines, their fake
file names and where a fake's identities come from. Relative to the dataset folder
(``DeepSpeak-v1`` or ``DeepSpeak-v2``; each release has a single version, with no compression
levels), every task folder is flat:

* reals: ``original_content/Actors/videos/<identity>-<recording>.mp4``, recording ``<recording>``
  of the actor ``<identity>`` (e.g. ``87-0``);
* fakes: ``manipulated_content/<Engine>/videos/<video>.mp4``, one folder per engine.

Every video is keyed by its file stem. A real's identity and target are the part of its name
before the first ``-`` (a name without ``-`` has neither). A fake's identity and target are the
actor on screen and its source the actor whose face or voice drove it; its ``pair_key`` is
``<target>-<target recording>``, the key of the real recording it was made from:

* v1 fakes are named ``<engine>--<source>-<target>--<source recording>-<target recording>--<n>``
  (e.g. ``facefusion--10-11--4100-4897--0``), so the name gives all of this: the first two
  ``-`` parts of the second and third ``--`` groups (a name with fewer has none of it);
* v2 fakes are named ``<engine>-<source recording>-<target recording>-<audio>`` (e.g.
  ``diff2lip-6868-6860-playht``) and name no actor, so each is looked up in the release's
  ``annotations-fake.csv`` (kept as ``.official_files/annotations-fake.csv``) by its file name:
  ``identity-target``, ``identity-source`` and ``recording-target`` give the fields, and
  ``engine`` and ``audio-config`` become the ``engine`` and ``audio_config`` attributes. A v2 fake
  without a row has no identity, source or ``pair_key``; its ``engine`` is the first ``-`` part
  of its name and its ``audio_config`` the fourth (neither for a name of fewer than three parts,
  no ``audio_config`` for one of three).

A fake pairs with the real its ``pair_key`` names, else (its target recording unknown) with every
real of its target.

The official split is the release's ``annotations-split-def.csv`` (kept under
``.official_files/``), which assigns each actor to ``train`` or ``test``. A video goes to the
split of its identity; the first of train, val and test that lists the identity decides it, and
since DeepSpeak publishes no val, an identity that falls to val is left out, as is a video without
an identity. The default scheme keeps that test and carves the official train 80/20 into
train and val by identity.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BenchmarkSpec, Split

__all__ = ["DeepSpeakV1Builder", "DeepSpeakV2Builder"]

_log = logging.getLogger(__name__)

# The release's files, relative to the dataset folder.
_SPLIT_DEF: Final = ".official_files/annotations-split-def.csv"
_FAKE_ANNOTATIONS: Final = ".official_files/annotations-fake.csv"
# The splits the split file can name, in the order an identity is looked up.
_LOOKUP_ORDER: Final[tuple[str, ...]] = ("train", "val", "test")
_SPLIT_HINT: Final = (
    "the file is the release's UTF-8 CSV with an 'identity' and a 'split' column, one row per actor"
)
_ACTORS: Final = TaskSpec("AR", "Actors", "real", "original_content/Actors/videos", "original")
_REAL_LABEL: Final = LabelSpec(binary=0, binary_av=0, multiclass=1, family="real")


def _fake(abbr: str, name: str) -> TaskSpec:
    """An engine's folder: its name is the task's, and its method the name in lower case."""
    return TaskSpec(abbr, name, "fake", f"manipulated_content/{name}/videos", name.lower())


def _real_identity(stem: str) -> str | None:
    """The actor of ``<identity>-<recording>``: the part before the first ``-``, else None."""
    parts = stem.split("-")
    return parts[0] if len(parts) >= 2 else None


def _value(text: str | None) -> str | None:
    """A CSV field stripped of space; empty is no value."""
    return (text or "").strip() or None


def _read_split_def(dataset_id: str, root: Path) -> dict[str, frozenset[str]]:
    """The identities the release's split file lists under each split, in train/val/test order.

    Each row's ``identity`` is stripped of space and its ``split`` stripped and lower-cased; a
    split other than train, val or test is ignored, and a split nobody is listed under is left
    out.

    Raises:
        ConfigError: the file is missing.
        ContractError: the file cannot be read as UTF-8 CSV or has no ``identity`` or ``split``
            column.
    """
    path = root / _SPLIT_DEF
    if not path.is_file():
        raise ConfigError(
            f"{dataset_id}: the official split file {_SPLIT_DEF} is missing from {root}",
            hint="copy the release's annotations-split-def.csv into .official_files/ in the "
            "dataset folder",
        )
    found: dict[str, set[str]] = {split: set() for split in _LOOKUP_ORDER}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            if not {"identity", "split"} <= columns:
                raise ContractError(
                    f"{dataset_id}: the official split file {path} has no 'identity' and "
                    "'split' columns",
                    hint=_SPLIT_HINT,
                )
            for row in reader:
                split = (row["split"] or "").strip().lower()
                if split in found:
                    found[split].add((row["identity"] or "").strip())
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ContractError(
            f"{dataset_id}: cannot read the official split file {path}: {exc}", hint=_SPLIT_HINT
        ) from None
    _log.debug("%s: %s lists %s", dataset_id, path.name, {s: len(v) for s, v in found.items()})
    return {split: frozenset(ids) for split, ids in found.items() if ids}


class _DeepSpeakBuilder(BaseBuilder):
    """What the two DeepSpeak releases share: the real naming, the split file and the pairing.

    A subclass sets the release's id, folder, label prefix, engines and labels, and parses a fake.
    """

    metadata_files: ClassVar[tuple[str, ...]] = (_SPLIT_DEF,)
    default_scheme = "official+ident-80-20"
    pairing_rule = "target-recording"
    schemes: ClassVar[Mapping[str, SchemeSpec]] = {
        "official+ident-80-20": SchemeSpec(
            "official+ident-80-20",
            "derived",
            params={"policy": "official-train-test"},
            source=f"the release's actor split ({_SPLIT_DEF})",
            rationale="the publisher's split has train and test but no val: the official test "
            "is kept, and the official train is carved 80/20 into train and val by an md5 of "
            "the identity, so an actor's recordings and the fakes of them stay on one side",
        ),
        "official": SchemeSpec(
            "official",
            "official",
            source=f"the release's actor split ({_SPLIT_DEF})",
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
            rationale="a small, seeded evaluation set drawn from the official test: up to 2 "
            "fakes per target actor and engine, balanced with as many reals",
        ),
    }

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; a real's actor comes from its name."""
        stem = path.stem
        if task.kind == "real":
            identity = _real_identity(stem)
            if identity is None:
                _log.debug("%s: %s is not named <identity>-<recording>", self.dataset_id, relpath)
            return self.record(
                task,
                stem,
                relpath,
                compression,
                identity=identity,
                target_id=identity,
                attrs=self.real_attrs(),
            )
        return self.fake_record(task, stem, relpath, compression)

    def real_attrs(self) -> dict[str, str | None]:
        """The attributes of a real, besides ``task_name`` (none by default)."""
        return {}

    def fake_record(
        self, task: TaskSpec, stem: str, relpath: str, compression: str | None
    ) -> InventoryRecord:
        """The record of a fake video."""
        raise NotImplementedError

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """Each record's split: the first of train, val and test listing its identity.

        An identity whose first listed split is val is left out: DeepSpeak publishes train and
        test only.

        Raises:
            ConfigError: the split file is missing.
            ContractError: the split file cannot be read.
        """
        listed = _read_split_def(self.dataset_id, root)
        assignment: dict[str, Split] = {}
        for record in records:
            identity = str(record.identity or "")
            if not identity:
                continue
            split = next((s for s, ids in listed.items() if identity in ids), None)
            if split == "train":
                assignment[record.key] = "train"
            elif split == "test":
                assignment[record.key] = "test"
        return assignment

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The target recording the fake was made from, else (unknown) its target actor."""
        return fake.pair_key or fake.target_id


_CARD_LICENSE: Final = {
    "spdx": None,
    "summary": "the DeepSpeak licence on Hugging Face: free to qualifying academic institutions, "
    "for a fee to others, with attribution; the terms need review",
    "url": None,
}
_CARD_ACCESS: Final = (
    "accept the conditions on the dataset's Hugging Face page to download it; dfwb never "
    "distributes media"
)


class DeepSpeakV1Builder(_DeepSpeakBuilder):
    """DeepSpeak v1 (``deepspeak-v1``): actor recordings and fakes by five engines."""

    dataset_id = "deepspeak-v1"
    expected_folder = "DeepSpeak-v1"
    label_prefix = "DSv1"
    tasks = (
        _ACTORS,
        _fake("FS_FF", "FaceFusion"),
        _fake("FS_FFGAN", "FaceFusion_GAN"),
        _fake("FS_FFLIVE", "FaceFusion_Live"),
        _fake("LS_RT", "ReTalking"),
        _fake("LS_W2L", "Wav2Lip"),
    )
    labels = {
        "AR": _REAL_LABEL,
        "FS_FF": LabelSpec(binary=1, binary_av=1, multiclass=2, family="face-swap"),
        "FS_FFGAN": LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-swap"),
        "FS_FFLIVE": LabelSpec(binary=1, binary_av=1, multiclass=4, family="face-swap"),
        "LS_RT": LabelSpec(binary=1, binary_av=1, multiclass=5, family="lip-sync"),
        "LS_W2L": LabelSpec(binary=1, binary_av=1, multiclass=6, family="lip-sync"),
    }
    benchmark = BenchmarkSpec(k_fake=2, strata=("identity", "task"))
    card_info = {
        "name": "DeepSpeak v1",
        "aliases": ["DeepSpeak-v1", "DSv1", "DeepSpeak v1.1"],
        "release": "DeepSpeak v1.1 as published on Hugging Face: 6,667 real recordings of "
        "actors and 6,796 fakes from five engines (FaceFusion, FaceFusion GAN, FaceFusion Live, "
        "ReTalking and Wav2Lip), with the release's annotation and split-definition files",
        "homepage": "https://huggingface.co/datasets/faridlab/deepspeak_v1_1",
        "license": _CARD_LICENSE,
        "access": _CARD_ACCESS,
        "modalities": ["video", "audio"],
        "compressions": None,
        "key_rule": "real: AR/<identity>-<recording>; fake: "
        "<task>/<engine>--<source>-<target>--<source recording>-<target recording>--<n> "
        "(the file stem)",
    }
    layout_notes = (
        "The release's real videos go in original_content/Actors/videos/ and each fake in the "
        "folder of the engine its file name starts with: facefusion in manipulated_content/"
        "FaceFusion/videos/, facefusion_gan in FaceFusion_GAN, facefusion_live in "
        "FaceFusion_Live, retalking in ReTalking and wav2lip in Wav2Lip.\n"
        f"The official split reads the release's annotations-split-def.csv, kept in {_SPLIT_DEF}: "
        "each actor's split, train or test."
    )

    def fake_record(
        self, task: TaskSpec, stem: str, relpath: str, compression: str | None
    ) -> InventoryRecord:
        """The actors and target recording come from the fake's ``--``-separated name."""
        groups = stem.split("--")
        if len(groups) >= 3:
            actors = groups[1].split("-")
            recordings = groups[2].split("-")
            if len(actors) >= 2 and len(recordings) >= 2:
                source, target = actors[0], actors[1]
                return self.record(
                    task,
                    stem,
                    relpath,
                    compression,
                    identity=target,
                    target_id=target,
                    source_id=source,
                    pair_key=f"{target}-{recordings[1]}",
                )
        _log.debug("deepspeak-v1: %s does not name its actors and recordings", relpath)
        return self.record(task, stem, relpath, compression)


@dataclass(frozen=True, slots=True)
class _FakeRow:
    """What a v2 fake takes from its row of the release's ``annotations-fake.csv``."""

    source: str | None
    target: str | None
    target_recording: str | None
    engine: str | None
    audio_config: str | None


def _read_fake_annotations(root: Path) -> dict[str, _FakeRow]:
    """The rows of the release's fake annotations, by file stem; ``{}`` if the file is missing.

    A row's ``video-file`` is stripped of space and of a trailing ``.mp4``; a row without one is
    skipped, and a later row for the same file replaces an earlier one.

    Raises:
        ContractError: the file cannot be read as UTF-8 CSV.
    """
    path = root / _FAKE_ANNOTATIONS
    if not path.is_file():
        _log.warning(
            "deepspeak-v2: %s is missing; the fakes get no identity, source or pair_key", path
        )
        return {}
    rows: dict[str, _FakeRow] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = (row.get("video-file") or "").strip()
                if not name:
                    continue
                rows[name.removesuffix(".mp4")] = _FakeRow(
                    source=_value(row.get("identity-source")),
                    target=_value(row.get("identity-target")),
                    target_recording=_value(row.get("recording-target")),
                    engine=_value(row.get("engine")),
                    audio_config=_value(row.get("audio-config")),
                )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ContractError(
            f"deepspeak-v2: cannot read the fake annotations {path}: {exc}",
            hint="use the release's annotations-fake.csv as it was downloaded (UTF-8 CSV)",
        ) from None
    _log.debug("deepspeak-v2: the fake annotations list %d videos", len(rows))
    return rows


class DeepSpeakV2Builder(_DeepSpeakBuilder):
    """DeepSpeak v2 (``deepspeak-v2``): actor recordings and fakes by six engines."""

    dataset_id = "deepspeak-v2"
    expected_folder = "DeepSpeak-v2"
    label_prefix = "DSv2"
    tasks = (
        _ACTORS,
        _fake("LS_D2L", "Diff2Lip"),
        _fake("FS_FF", "FaceFusion"),
        _fake("TF_HM", "HelloMeme"),
        _fake("LS_LS", "LatentSync"),
        _fake("TF_LP", "LivePortrait"),
        _fake("TF_MEMO", "Memo"),
    )
    metadata_files = (_SPLIT_DEF, _FAKE_ANNOTATIONS)
    labels = {
        "AR": _REAL_LABEL,
        "LS_D2L": LabelSpec(binary=1, binary_av=1, multiclass=2, family="lip-sync"),
        "FS_FF": LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-swap"),
        "TF_HM": LabelSpec(binary=1, binary_av=1, multiclass=4, family="talking-face"),
        "LS_LS": LabelSpec(binary=1, binary_av=1, multiclass=5, family="lip-sync"),
        "TF_LP": LabelSpec(binary=1, binary_av=1, multiclass=6, family="talking-face"),
        "TF_MEMO": LabelSpec(binary=1, binary_av=1, multiclass=7, family="talking-face"),
    }
    benchmark = BenchmarkSpec(k_fake=2, strata=("target_id", "task"))
    card_info = {
        "name": "DeepSpeak v2",
        "aliases": ["DeepSpeak-v2", "DSv2", "DeepSpeak v2.0"],
        "release": "DeepSpeak v2.0 as published on Hugging Face: 9,376 real recordings of "
        "actors and 7,209 fakes from six engines (Diff2Lip, FaceFusion, HelloMeme, LatentSync, "
        "LivePortrait and Memo), with the release's annotation and split-definition files",
        "homepage": "https://huggingface.co/datasets/faridlab/deepspeak_v2",
        "license": _CARD_LICENSE,
        "access": _CARD_ACCESS,
        "modalities": ["video", "audio"],
        "compressions": None,
        "key_rule": "real: AR/<identity>-<recording>; fake: "
        "<task>/<engine>-<source recording>-<target recording>-<audio> (the file stem)",
    }
    layout_notes = (
        "The release's real videos go in original_content/Actors/videos/ and each fake in the "
        "folder of the engine its file name starts with: diff2lip in manipulated_content/"
        "Diff2Lip/videos/, facefusion in FaceFusion, hellomeme in HelloMeme, latentsync in "
        "LatentSync, liveportrait in LivePortrait and memo in Memo.\n"
        f"The release's annotations-fake.csv, kept in {_FAKE_ANNOTATIONS}, gives each fake its "
        "target and source actors, its target recording, its engine and its audio; a fake "
        "missing from it gets no identity, so no official split and no pair.\n"
        f"The official split reads the release's annotations-split-def.csv, kept in {_SPLIT_DEF}: "
        "each actor's split, train or test."
    )

    _fakes: Mapping[str, _FakeRow] | None = None

    def prepare(self, root: Path) -> None:
        """Read the release's fake annotations under ``root`` once per build.

        Raises:
            ContractError: the annotation file cannot be read as UTF-8 CSV.
        """
        super().prepare(root)
        self._fakes = _read_fake_annotations(root)

    def real_attrs(self) -> dict[str, str | None]:
        """A real has no engine and no synthetic audio."""
        return {"engine": None, "audio_config": None}

    def fake_record(
        self, task: TaskSpec, stem: str, relpath: str, compression: str | None
    ) -> InventoryRecord:
        """The actors and target recording come from the fake's annotation row."""
        row = (self._fakes or {}).get(stem)
        if row is None:
            _log.debug("deepspeak-v2: %s has no annotation row", relpath)
            parts = stem.split("-")
            engine = parts[0] if len(parts) >= 3 else None
            audio_config = parts[3] if len(parts) >= 4 else None
            return self.record(
                task,
                stem,
                relpath,
                compression,
                attrs={"engine": engine, "audio_config": audio_config},
            )
        pair_key = (
            None
            if row.target is None or row.target_recording is None
            else f"{row.target}-{row.target_recording}"
        )
        return self.record(
            task,
            stem,
            relpath,
            compression,
            identity=row.target,
            target_id=row.target,
            source_id=row.source,
            pair_key=pair_key,
            attrs={"engine": row.engine, "audio_config": row.audio_config},
        )
