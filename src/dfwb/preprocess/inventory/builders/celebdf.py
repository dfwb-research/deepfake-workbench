"""Celeb-DF v1, v2 and v3: celebrity interview videos and the deepfakes made from them.

The three releases share one layout and one naming scheme and differ only in their tasks and
their test list. Relative to the dataset folder (``Celeb-DF-v1``, ``Celeb-DF-v2`` or
``Celeb-DF-v3``; each release has a single version, with no compression levels):

* reals: ``original_content/Celeb-real/videos/<id>_<rec>.mp4``, recording ``<rec>`` of the
  celebrity ``<id>`` (e.g. ``id1_0007``), and ``original_content/YouTube-real/videos/<n>.mp4``;
* fakes in v1 and v2: ``manipulated_content/Celeb-synthesis/videos/<target>_<source>_<rec>.mp4``,
  the face of celebrity ``<source>`` swapped into recording ``<rec>`` of celebrity ``<target>``;
* fakes in v3: one folder per method, under ``manipulated_content/FaceSwap/``,
  ``manipulated_content/FaceReenact/`` and ``manipulated_content/TalkingFace/``. Face swaps and
  reenactments are named ``<target>_<source>_<rec>``. A talking face is named
  ``<target>_<rec>_test_<ref>``: it is driven by a reference clip, not by another celebrity.

Every video is keyed by its file stem. A Celeb-real video's identity and target are its celebrity
id; a YouTube-real video's are its own number. A fake's identity and target are the celebrity in
the background and its source is the donor celebrity (a talking face has none). Its ``pair_key``
is ``<target>_<rec>``: the key of the real recording it was made from. A name that follows none of
these patterns is kept, with none of these fields set.

Celeb-DF publishes a test list and nothing else: one ``<label> <path>`` line per video. A video
whose file stem is on the list is test, in every task it appears in. The default scheme keeps that
test and carves every other video 80/20 into train and val by identity.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import ClassVar, Final

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BENCHMARK_REALS, BenchmarkSpec, Split, local_key

__all__ = ["CelebDFv1Builder", "CelebDFv2Builder", "CelebDFv3Builder"]

_log = logging.getLogger(__name__)

# A Celeb-real name is "id<n>_<rec>"; a YouTube-real name is a number.
_CELEB_REAL: Final = re.compile(r"^id\d+_\d+$")
_YOUTUBE_REAL: Final = re.compile(r"^\d+$")

# The name the Celeb-DF download gives its test list; each release's copy is renamed (below).
_UPSTREAM_TESTING_LIST: Final = "List_of_testing_videos.txt"

_REALS: Final = (
    TaskSpec("CR", "Celeb-real", "real", "original_content/Celeb-real/videos", "original"),
    TaskSpec("YTR", "YouTube-real", "real", "original_content/YouTube-real/videos", "original"),
)
_REAL_LABELS: Final = {
    "CR": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
    "YTR": LabelSpec(binary=0, binary_av=0, multiclass=2, family="real"),
}


def _testing_list(version: str) -> str:
    """Where a release's test list lives, relative to its dataset folder."""
    return f".official_files/CDF{version}_testing_videos.txt"


def _fake(abbr: str, name: str, video_dir: str) -> TaskSpec:
    """A fake task, whose method is its name."""
    return TaskSpec(abbr, name, "fake", video_dir, name)


def _v3_fake(abbr: str, family: str, name: str) -> TaskSpec:
    return _fake(abbr, name, f"manipulated_content/{family}/{name}/videos")


def _parse_fake(stem: str) -> tuple[str, str | None, str] | None:
    """``(target, source, pair_key)`` of a fake's file stem, else ``None``.

    The name splits on ``_`` and its first part must start with ``id``. If there are at least
    three parts and the second also starts with ``id``, the name is ``<target>_<source>_<rec>``;
    otherwise (a talking face, ``<target>_<rec>_test_<ref>``) there is no source and the second
    part is the recording. Either way the pair key is ``<target>_<rec>``.
    """
    parts = stem.split("_")
    if len(parts) < 2 or not parts[0].startswith("id"):
        return None
    target = parts[0]
    if len(parts) >= 3 and parts[1].startswith("id"):
        return target, parts[1], f"{target}_{parts[2]}"
    return target, None, f"{target}_{parts[1]}"


def _real_identity(stem: str) -> str | None:
    """A real's identity: the celebrity of ``id<n>_<rec>``, a YouTube number itself, else None."""
    if _CELEB_REAL.match(stem):
        return stem.split("_")[0]
    if _YOUTUBE_REAL.match(stem):
        return stem
    return None


def _read_testing_list(dataset_id: str, root: Path, relpath: str) -> frozenset[str]:
    """The file stems a Celeb-DF test list names.

    Each line is ``<label> <path>``; only the stem of the path counts (its folder and the label
    are ignored). Blank lines and lines without a space are skipped.

    Raises:
        ConfigError: the list is missing.
        ContractError: the list cannot be read as UTF-8 text.
    """
    path = root / relpath
    if not path.is_file():
        raise ConfigError(
            f"{dataset_id}: the official test list {relpath} is missing from {root}",
            hint=f"copy the release's {_UPSTREAM_TESTING_LIST} to {relpath} in the dataset folder",
        )
    try:
        with path.open(encoding="utf-8") as handle:
            lines = list(handle)
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError(
            f"{dataset_id}: cannot read the official test list {path}: {exc}",
            hint="the list is UTF-8 text with one '<label> <path>' line per test video",
        ) from None
    stems: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        stems.add(PurePosixPath(parts[1].strip()).stem)
    _log.debug("%s: %s names %d video stems", dataset_id, path.name, len(stems))
    return frozenset(stems)


def _schemes(testing_list: str, benchmark_rationale: str) -> dict[str, SchemeSpec]:
    return {
        "official+ident-80-20": SchemeSpec(
            "official+ident-80-20",
            "derived",
            params={"policy": "test-only-official"},
            source=f"the Celeb-DF test list ({testing_list})",
            rationale="Celeb-DF publishes only a test list: its videos are test, and every "
            "other video is carved 80/20 into train and val by an md5 of its identity (its key "
            "when it has none), so a celebrity's recordings and their fakes stay on one side",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec("benchmark", "subset", rationale=benchmark_rationale),
    }


def _card(
    name: str, aliases: list[str], release: str, homepage: str, key_rule: str
) -> dict[str, object]:
    return {
        "name": name,
        "aliases": aliases,
        "release": release,
        "homepage": homepage,
        "license": {
            "spdx": None,
            "summary": "the Celeb-DF terms of use: non-commercial research only",
            "url": None,
        },
        "access": "request the download from the authors through the form linked from the "
        "Celeb-DF repository; dfwb never distributes media",
        "modalities": ["video"],
        "compressions": None,
        "key_rule": key_rule,
    }


# The v1/v2 fake task and its label. Each release builds its own task tuple and label dict from
# them, so the two classes never share a table object.
_CELEB_SYNTHESIS: Final = _fake(
    "FS_CS", "Celeb-synthesis", "manipulated_content/Celeb-synthesis/videos"
)
_CELEB_SYNTHESIS_LABEL: Final = LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-swap")
_RANDOM_BENCHMARK: Final = (
    "a small, seeded evaluation set: up to 100 fakes drawn at random from the official test "
    f"list, {BENCHMARK_REALS}"
)
_V1_V2_KEY_RULE: Final = (
    "real: CR/<id>_<rec> or YTR/<n>; fake: FS_CS/<target>_<source>_<rec> (the file stem)"
)


class _CelebDFBuilder(BaseBuilder):
    """What the three Celeb-DF releases share: the naming, the test list and the pairing.

    A subclass sets the release's id, folder, label prefix, test list and task table (with its
    labels, benchmark and card).
    """

    testing_list: ClassVar[str]
    default_scheme = "official+ident-80-20"
    pairing_rule = "target-recording"

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; its fields come from the name."""
        stem = path.stem
        if task.kind == "real":
            identity = _real_identity(stem)
            return self.record(
                task, stem, relpath, compression, identity=identity, target_id=identity
            )
        parsed = _parse_fake(stem)
        if parsed is None:
            _log.debug("%s: %s does not name its target celebrity", self.dataset_id, relpath)
            return self.record(task, stem, relpath, compression)
        target, source, pair_key = parsed
        return self.record(
            task,
            stem,
            relpath,
            compression,
            identity=target,
            target_id=target,
            source_id=source,
            pair_key=pair_key,
        )

    def official_splits(self, root: Path, records: Sequence[InventoryRecord]) -> dict[str, Split]:
        """``test`` for every record whose file stem is on the release's test list.

        Raises:
            ConfigError: the test list is missing.
            ContractError: the test list cannot be read.
        """
        stems = _read_testing_list(self.dataset_id, root, self.testing_list)
        return {record.key: "test" for record in records if local_key(record.key) in stems}

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """``<target>_<rec>``: the key of the real recording the fake was made from."""
        parsed = _parse_fake(local_key(fake.key))
        return None if parsed is None else parsed[2]


class CelebDFv1Builder(_CelebDFBuilder):
    """Celeb-DF v1 (``celebdf-v1``): Celeb-real and YouTube-real videos and their face swaps."""

    dataset_id = "celebdf-v1"
    expected_folder = "Celeb-DF-v1"
    label_prefix = "CDFv1"
    testing_list = _testing_list("v1")
    metadata_files = (testing_list,)
    tasks = (*_REALS, _CELEB_SYNTHESIS)
    labels = {**_REAL_LABELS, "FS_CS": _CELEB_SYNTHESIS_LABEL}
    schemes = _schemes(testing_list, _RANDOM_BENCHMARK)
    benchmark = BenchmarkSpec(k_fake=100)
    card_info = _card(
        "Celeb-DF v1",
        ["Celeb-DF-v1", "CDFv1"],
        "the first Celeb-DF release: 158 Celeb-real and 250 YouTube-real videos and 795 "
        "Celeb-synthesis face swaps",
        "https://github.com/yuezunli/celeb-deepfakeforensics",
        _V1_V2_KEY_RULE,
    )
    layout_notes = (
        f"The official test list goes in {testing_list} (the release's {_UPSTREAM_TESTING_LIST})."
    )


class CelebDFv2Builder(_CelebDFBuilder):
    """Celeb-DF v2 (``celebdf-v2``): Celeb-real and YouTube-real videos and their face swaps."""

    dataset_id = "celebdf-v2"
    expected_folder = "Celeb-DF-v2"
    label_prefix = "CDFv2"
    testing_list = _testing_list("v2")
    metadata_files = (testing_list,)
    tasks = (*_REALS, _CELEB_SYNTHESIS)
    labels = {**_REAL_LABELS, "FS_CS": _CELEB_SYNTHESIS_LABEL}
    schemes = _schemes(testing_list, _RANDOM_BENCHMARK)
    benchmark = BenchmarkSpec(k_fake=100)
    card_info = _card(
        "Celeb-DF v2",
        ["Celeb-DF-v2", "CDFv2", "Celeb-DF"],
        "590 Celeb-real and 300 YouTube-real videos and 5,639 Celeb-synthesis face swaps",
        "https://github.com/yuezunli/celeb-deepfakeforensics",
        _V1_V2_KEY_RULE,
    )
    layout_notes = (
        f"The official test list goes in {testing_list} (the release's {_UPSTREAM_TESTING_LIST})."
    )


class CelebDFv3Builder(_CelebDFBuilder):
    """Celeb-DF v3 (``celebdf-v3``): the v2 reals and fakes from 22 methods of three families."""

    dataset_id = "celebdf-v3"
    expected_folder = "Celeb-DF-v3"
    label_prefix = "CDFv3"
    testing_list = _testing_list("v3")
    metadata_files = (testing_list,)
    tasks = (
        *_REALS,
        _v3_fake("FS_CDFV2", "FaceSwap", "Celeb-DF-v2"),
        _v3_fake("FS_BLEND", "FaceSwap", "BlendFace"),
        _v3_fake("FS_GHOST", "FaceSwap", "GHOST"),
        _v3_fake("FS_HIFI", "FaceSwap", "HifiFace"),
        _v3_fake("FS_INSW", "FaceSwap", "InSwapper"),
        _v3_fake("FS_MOBILE", "FaceSwap", "MobileFaceSwap"),
        _v3_fake("FS_SIM", "FaceSwap", "SimSwap"),
        _v3_fake("FS_UNI", "FaceSwap", "UniFace"),
        _v3_fake("FR_DAGAN", "FaceReenact", "DaGAN"),
        _v3_fake("FR_FSRT", "FaceReenact", "FSRT"),
        _v3_fake("FR_HYPER", "FaceReenact", "HyperReenact"),
        _v3_fake("FR_LIA", "FaceReenact", "LIA"),
        _v3_fake("FR_LP", "FaceReenact", "LivePortrait"),
        _v3_fake("FR_MCNET", "FaceReenact", "MCNET"),
        _v3_fake("FR_TPSMM", "FaceReenact", "TPSMM"),
        _v3_fake("TF_ANI", "TalkingFace", "AniTalker"),
        _v3_fake("TF_ECHO", "TalkingFace", "EchoMimic"),
        _v3_fake("TF_EDTALK", "TalkingFace", "EDTalk"),
        _v3_fake("TF_FLOAT", "TalkingFace", "FLOAT"),
        _v3_fake("TF_IPLAP", "TalkingFace", "IP_LAP"),
        _v3_fake("TF_REAL3D", "TalkingFace", "Real3DPortrait"),
        _v3_fake("TF_SAD", "TalkingFace", "SadTalker"),
    )
    labels = {
        **_REAL_LABELS,
        "FR_DAGAN": LabelSpec(binary=1, binary_av=1, multiclass=3, family="face-reenactment"),
        "FR_FSRT": LabelSpec(binary=1, binary_av=1, multiclass=4, family="face-reenactment"),
        "FR_HYPER": LabelSpec(binary=1, binary_av=1, multiclass=5, family="face-reenactment"),
        "FR_LIA": LabelSpec(binary=1, binary_av=1, multiclass=6, family="face-reenactment"),
        "FR_LP": LabelSpec(binary=1, binary_av=1, multiclass=7, family="face-reenactment"),
        "FR_MCNET": LabelSpec(binary=1, binary_av=1, multiclass=8, family="face-reenactment"),
        "FR_TPSMM": LabelSpec(binary=1, binary_av=1, multiclass=9, family="face-reenactment"),
        "FS_BLEND": LabelSpec(binary=1, binary_av=1, multiclass=10, family="face-swap"),
        "FS_CDFV2": LabelSpec(binary=1, binary_av=1, multiclass=11, family="face-swap"),
        "FS_GHOST": LabelSpec(binary=1, binary_av=1, multiclass=12, family="face-swap"),
        "FS_HIFI": LabelSpec(binary=1, binary_av=1, multiclass=13, family="face-swap"),
        "FS_INSW": LabelSpec(binary=1, binary_av=1, multiclass=14, family="face-swap"),
        "FS_MOBILE": LabelSpec(binary=1, binary_av=1, multiclass=15, family="face-swap"),
        "FS_SIM": LabelSpec(binary=1, binary_av=1, multiclass=16, family="face-swap"),
        "FS_UNI": LabelSpec(binary=1, binary_av=1, multiclass=17, family="face-swap"),
        "TF_ANI": LabelSpec(binary=1, binary_av=1, multiclass=18, family="talking-face"),
        "TF_ECHO": LabelSpec(binary=1, binary_av=1, multiclass=19, family="talking-face"),
        "TF_EDTALK": LabelSpec(binary=1, binary_av=1, multiclass=20, family="talking-face"),
        "TF_FLOAT": LabelSpec(binary=1, binary_av=1, multiclass=21, family="talking-face"),
        "TF_IPLAP": LabelSpec(binary=1, binary_av=1, multiclass=22, family="talking-face"),
        "TF_REAL3D": LabelSpec(binary=1, binary_av=1, multiclass=23, family="talking-face"),
        "TF_SAD": LabelSpec(binary=1, binary_av=1, multiclass=24, family="talking-face"),
    }
    schemes = _schemes(
        testing_list,
        "a small, seeded evaluation set drawn from the official test list: up to 2 fakes per "
        f"identity and method, {BENCHMARK_REALS}",
    )
    benchmark = BenchmarkSpec(k_fake=2, strata=("identity", "task"))
    card_info = _card(
        "Celeb-DF v3",
        ["Celeb-DF-v3", "CDFv3", "Celeb-DF++"],
        "the Celeb-DF v2 reals (590 Celeb-real and 300 YouTube-real videos) and 53,196 fakes "
        "from 22 methods: 8 face swap (including the v2 Celeb-synthesis set), 7 face "
        "reenactment and 7 talking face",
        "https://github.com/OUC-VAS/Celeb-DF-PP",
        "real: CR/<id>_<rec> or YTR/<n>; face swap and reenactment: "
        "<task>/<target>_<source>_<rec>; talking face: <task>/<target>_<rec>_test_<ref> "
        "(the file stem)",
    )
    layout_notes = (
        "Fakes sit in one folder per method under manipulated_content/{FaceSwap,FaceReenact,"
        "TalkingFace}/.\n"
        f"The official test list goes in {testing_list} (the release's "
        f"{_UPSTREAM_TESTING_LIST})."
    )
