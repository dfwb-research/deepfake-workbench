"""FakeAVCeleb v1.2: real VoxCeleb2 interviews and fakes whose video, audio or both are synthetic.

Layout, relative to the ``FakeAVCeleb-v1_2`` folder (a single version, no compression levels).
Every task nests its videos as ``<race>/<gender>/<id>/<clip>.mp4``, as in the release, so each is
searched recursively:

* reals: ``original_content/RealVideo-RealAudio/videos/...``;
* fakes: ``manipulated_content/<category>/videos/...`` for ``FakeVideo-FakeAudio``,
  ``FakeVideo-RealAudio`` and ``RealVideo-FakeAudio``.

A video's key is its path below its task folder, without the suffix: each part has every run of
characters other than ASCII letters and digits replaced by ``_`` and is trimmed of ``_``, and the
parts are joined by ``__`` (``Asian (East)/women/id10006/00905.mp4`` is
``Asian_East__women__id10006__00905``). Its identity is the first three parts of the key (the
whole key if it has fewer).

The release's metadata file, ``meta_data.csv`` (kept as ``.official_files/meta_data.csv``), lists
every video with its ``source`` (the person on screen), ``target1``, ``method``, ``category``,
``type``, ``race``, ``gender`` and ``path`` (the file name). A row is used when none of ``type``,
``race``, ``gender``, ``source`` and ``path`` is empty or ``-``; it is keyed by race, gender,
source and the file name's stem, sanitised the same way, so a later row with the same key
replaces an earlier one whatever its type. A video whose key has a row takes from it:

* its target: ``source``; its source: ``target1`` when that starts with ``id``, else none;
* its ``pair_key`` and ``raw_video_stem``: for a row that is not ``RealVideo-RealAudio``, the key
  of the real clip it was made from (race, gender, source, and the file name up to its first
  ``_``); none otherwise;
* its method: ``method``, and its ``category``.

A video without a row has its identity as its target, no source, the task's method and no
category; if it is a fake whose key ends in ``__fake``, its ``pair_key`` and ``raw_video_stem``
are the key without that ending. A fake pairs with the real its ``raw_video_stem`` names, else
with the real its key names without a trailing ``_fake``.

There is no official split: the default scheme is an identity-disjoint carve on the person, so
their real clips and every fake of them land on the same side. Real video with fake audio is real
under the visual label and fake under the audiovisual one.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from dfwb.core.errors import ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BENCHMARK_REALS, BenchmarkSpec, local_key

__all__ = ["FakeAVCelebBuilder"]

_log = logging.getLogger(__name__)

# The release's metadata file, relative to the dataset folder.
_META: Final = ".official_files/meta_data.csv"
_REAL_TYPE: Final = "RealVideo-RealAudio"
_NOT_ALPHANUMERIC: Final = re.compile(r"[^A-Za-z0-9]+")
# A fake with no metadata row whose key ends in this names its real by the rest of the key.
_FAKE_ENDING: Final = "__fake"
# A fake without a raw_video_stem pairs with its key minus this ending, if it has it.
_FAKE_SUFFIX: Final = "_fake"


@dataclass(frozen=True, slots=True)
class _Meta:
    """What a metadata row gives the video with its key."""

    source: str
    target1: str | None
    method: str | None
    category: str | None
    raw_video_stem: str | None


def _sanitise(text: str) -> str:
    """Every run of characters other than ASCII letters and digits becomes ``_``; trimmed."""
    return _NOT_ALPHANUMERIC.sub("_", text).strip("_")


def _meta_key(race: str, gender: str, source: str, file_name: str) -> str:
    """The key of the video a metadata row lists: its four fields, sanitised, joined by ``__``."""
    return "__".join(_sanitise(s) for s in (race, gender, source, PurePosixPath(file_name).stem))


def _value(text: str | None) -> str | None:
    """A metadata field stripped of space; empty or ``-`` is no value."""
    text = (text or "").strip()
    return text if text and text != "-" else None


def _read_meta(root: Path) -> dict[str, _Meta]:
    """The metadata rows under ``root``, by the key of the video each lists; ``{}`` if none.

    Raises:
        ContractError: the file cannot be read as UTF-8 CSV.
    """
    path = root / _META
    if not path.is_file():
        _log.warning(
            "fakeavceleb: %s is missing; source, method, category and pair_key come from the "
            "file names only",
            path,
        )
        return {}
    rows: dict[str, _Meta] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                kind = _value(row.get("type"))
                race = _value(row.get("race"))
                gender = _value(row.get("gender"))
                source = _value(row.get("source"))
                file_name = (row.get("path") or "").strip()
                if not (kind and race and gender and source and file_name):
                    continue
                raw_video_stem = None
                if kind != _REAL_TYPE:
                    clip = PurePosixPath(file_name).stem.split("_", 1)[0]
                    raw_video_stem = _meta_key(race, gender, source, f"{clip}.mp4")
                rows[_meta_key(race, gender, source, file_name)] = _Meta(
                    source=source,
                    target1=_value(row.get("target1")),
                    method=_value(row.get("method")),
                    category=_value(row.get("category")),
                    raw_video_stem=raw_video_stem,
                )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ContractError(
            f"fakeavceleb: cannot read the metadata file {path}: {exc}",
            hint="use the release's meta_data.csv as it was downloaded (UTF-8 CSV)",
        ) from None
    _log.debug("fakeavceleb: the metadata lists %d videos", len(rows))
    return rows


def _task(abbr: str, name: str, kind: Literal["real", "fake"], folder: str) -> TaskSpec:
    """A category folder: its name is the task's name, and a fake's method until a row says."""
    method = "original" if kind == "real" else name
    return TaskSpec(abbr, name, kind, f"{folder}/{name}/videos", method, recursive=True)


class FakeAVCelebBuilder(BaseBuilder):
    """FakeAVCeleb v1.2 (``fakeavceleb``): real interviews and three kinds of audio-video fake."""

    dataset_id = "fakeavceleb"
    expected_folder = "FakeAVCeleb-v1_2"
    label_prefix = "FAVC"
    tasks = (
        _task("RVRA", "RealVideo-RealAudio", "real", "original_content"),
        _task("FVFA", "FakeVideo-FakeAudio", "fake", "manipulated_content"),
        _task("FVRA", "FakeVideo-RealAudio", "fake", "manipulated_content"),
        _task("RVFA", "RealVideo-FakeAudio", "fake", "manipulated_content"),
    )
    metadata_files = (_META,)
    labels = {
        "RVRA": LabelSpec(binary=0, binary_av=0, multiclass=1, family="real"),
        "FVFA": LabelSpec(binary=1, binary_av=1, multiclass=2, family="video-and-audio"),
        "FVRA": LabelSpec(binary=1, binary_av=1, multiclass=3, family="video-only"),
        # The picture is the real one: real to a visual detector, fake once audio counts.
        "RVFA": LabelSpec(binary=0, binary_av=1, multiclass=1, family="audio-only"),
    }
    schemes = {
        "ident-72-14-14": SchemeSpec(
            "ident-72-14-14",
            "derived",
            rationale="no official split; identity-disjoint md5 carve (72/14/14) on the person "
            "(race, gender and id), so their real clips and every fake of them stay on one side",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set: up to 2 fakes per face donor and "
            f"category, drawn from the whole dataset, {BENCHMARK_REALS} (real video with fake "
            "audio counting as real)",
        ),
    }
    default_scheme = "ident-72-14-14"
    benchmark = BenchmarkSpec(k_fake=2, strata=("source_id", "task"))
    pairing_rule = "target-recording"
    card_info = {
        "name": "FakeAVCeleb",
        "aliases": ["FakeAVCeleb v1.2", "FakeAVCeleb-v1_2", "FAVC"],
        "release": "version 1.2: 500 real VoxCeleb2 videos, one per person, and fakes in three "
        "categories: 10,835 with fake video and fake audio, 9,709 with fake video and real "
        "audio, and 500 with real video and cloned audio (video by Faceswap, FSGAN and Wav2Lip; "
        "audio by SV2TTS voice cloning), with the release's meta_data.csv",
        "homepage": "https://github.com/DASH-Lab/FakeAVCeleb",
        "paper": {
            "title": "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset",
            "venue": "arXiv",
            "year": 2021,
        },
        "license": {
            "spdx": None,
            "summary": "the terms of the FakeAVCeleb request form: research use; the terms need "
            "review",
            "url": None,
        },
        "access": "request access through the form linked from the FakeAVCeleb repository; the "
        "authors send a download script; dfwb never distributes media",
        "modalities": ["video", "audio"],
        "compressions": None,
        "key_rule": "<task>/<race>__<gender>__<id>__<clip>: the video's path below its task "
        "folder, each part sanitised to letters, digits and '_', e.g. "
        "RVRA/African__men__id10001__00901",
    }
    layout_notes = (
        "Every task folder is searched recursively: videos nest as <race>/<gender>/<id>/<clip>, "
        "as in the release.\n"
        "The release's four category folders go under original_content/RealVideo-RealAudio/"
        "videos/ (the reals) and manipulated_content/<category>/videos/ (the other three), each "
        "keeping its <race>/<gender>/<id>/ sub-folders; the release's meta_data.csv goes in "
        f"{_META}.\n"
        "meta_data.csv gives each video its target, source, method and category, and each fake "
        "the real clip it was made from. Without it, a video's target is its identity and it has "
        "no source; a fake is then paired only when its key ends in _fake."
    )

    _meta: Mapping[str, _Meta] | None = None

    def prepare(self, root: Path) -> None:
        """Read the release's metadata file under ``root`` once per build.

        Raises:
            ContractError: the metadata file cannot be read as UTF-8 CSV.
        """
        super().prepare(root)
        self._meta = _read_meta(root)

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its sanitised path; the metadata adds its fields."""
        below = PurePosixPath(relpath).relative_to(task.video_dir).with_suffix("")
        key = "__".join(_sanitise(part) for part in below.parts)
        parts = key.split("__")
        identity = ("__".join(parts[:3]) if len(parts) >= 3 else key) or None
        meta = (self._meta or {}).get(key)
        if meta is not None:
            donor = meta.target1 or ""
            return self.record(
                task,
                key,
                relpath,
                compression,
                identity=identity,
                target_id=meta.source,
                source_id=donor if donor.startswith("id") else None,
                pair_key=meta.raw_video_stem,
                attrs={"raw_video_stem": meta.raw_video_stem, "category": meta.category},
                method=meta.method or task.method,
            )
        pair_key = None
        if task.kind == "fake" and key.endswith(_FAKE_ENDING):
            pair_key = key[: -len(_FAKE_ENDING)]
        return self.record(
            task,
            key,
            relpath,
            compression,
            identity=identity,
            target_id=identity,
            pair_key=pair_key,
            attrs={"raw_video_stem": pair_key, "category": None},
        )

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The real clip the fake was made from, else its key without a trailing ``_fake``."""
        raw = fake.attrs.get("raw_video_stem")
        if raw:
            return str(raw)
        key = local_key(fake.key)
        return key[: -len(_FAKE_SUFFIX)] if key.endswith(_FAKE_SUFFIX) else None
