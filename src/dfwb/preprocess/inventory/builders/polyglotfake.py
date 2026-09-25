"""PolyGlotFake: real clips in seven languages and fakes dubbed from them into the others.

Layout, relative to the ``PolyGlotFake`` folder (a single version, no compression levels), one
flat folder per language for the reals and per target language for the fakes:

* reals: ``original_content/<lang>/videos/<lang>_<n>.mp4``;
* fakes: ``manipulated_content/to_<lang>/videos/<src>_<n>_to_<lang>_<tts>.mp4``,

for the languages ``ar``, ``en``, ``es``, ``fr``, ``ja``, ``ru`` and ``zh``, where ``<src>_<n>``
names the real clip the fake was dubbed from and ``<tts>`` the speech synthesis used.

Every video is keyed by its file stem. A real's target is its folder's language. A fake whose
stem holds ``_to_`` is split at the first one: the part before is its ``pair_key`` and the real it
pairs with, whose language (up to the first ``_``) is its source; the part after starts with its
target language (up to the first ``_``). A fake without ``_to_`` has its folder's language as its
target and nothing else.

The release names no speakers, so a speaker identity is built from its per-language metadata of
the real clips, ``real_json_files/<lang>.json`` (kept under ``.official_files/.json/``). Each
lists its videos, each with a ``filename``, a ``source`` and ``characters``; a video with a
source is given ``<source>__age<age>_<sex>`` from its first character's age and sex, or the
source alone when either is missing (files read in ``ar``, ``en``, ``es``, ``fr``, ``ja``,
``ru``, ``zh`` order; a later entry for a file name replaces an earlier one). A real takes the
speaker of its file name; a fake the speaker of ``<pair_key>.mp4``. Without a speaker, a real's
identity is its language and a fake's its source language.

There is no official split: the default scheme is an identity-disjoint carve on the speaker, so
their clips and every dub of them land on the same side.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from dfwb.core.errors import ContractError
from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import BaseBuilder, LabelSpec, SchemeSpec, TaskSpec
from dfwb.protocols.rules import BENCHMARK_REALS, BenchmarkSpec, local_key

__all__ = ["PolyGlotFakeBuilder"]

_log = logging.getLogger(__name__)

# The languages, in the order their tasks are listed and their metadata files are read.
_LANGUAGES: Final = ("ar", "en", "es", "fr", "ja", "ru", "zh")
# The release's per-language metadata of the real clips, relative to the dataset folder.
_SPEAKER_DIR: Final = ".official_files/.json/real_json_files"
_DUB: Final = "_to_"
_HINT: Final = (
    "use the release's real_json_files/<lang>.json as downloaded: a JSON object whose 'videos' "
    "list holds one object per clip"
)


def _text(value: Any, path: Path) -> str:
    """A metadata field as stripped text; missing or empty is ``""``.

    Raises:
        ContractError: the field is not text.
    """
    if not value:
        return ""
    if not isinstance(value, str):
        raise ContractError(
            f"polyglotfake: {path} has a field {value!r} that is not text", hint=_HINT
        )
    return value.strip()


def _speaker(entry: Mapping[str, Any], path: Path) -> str | None:
    """``<source>__age<age>_<sex>`` of a metadata entry, the source alone, or ``None``."""
    source = _text(entry.get("source"), path)
    if not source:
        return None
    characters = entry.get("characters") or []
    age, sex = None, ""
    if isinstance(characters, list) and characters and isinstance(characters[0], Mapping):
        age = characters[0].get("age")
        sex = _text(characters[0].get("sex"), path)
    return f"{source}__age{age}_{sex}" if age is not None and sex else source


def _read_speakers(root: Path) -> dict[str, str]:
    """``{file name: speaker}`` from the per-language metadata under ``root``; ``{}`` if none.

    A missing language file is skipped; one that is not JSON is skipped with a warning.

    Raises:
        ContractError: a file cannot be read, or is JSON of another shape.
    """
    folder = root / _SPEAKER_DIR
    if not folder.is_dir():
        _log.warning(
            "polyglotfake: %s is missing; every identity falls back to a language code", folder
        )
        return {}
    speakers: dict[str, str] = {}
    for lang in _LANGUAGES:
        path = folder / f"{lang}.json"
        if not path.is_file():
            _log.debug("polyglotfake: %s is missing; its clips keep a language identity", path)
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _log.warning("polyglotfake: %s is not JSON (%s); skipped", path, exc)
            continue
        except (OSError, UnicodeDecodeError) as exc:
            raise ContractError(f"polyglotfake: cannot read {path}: {exc}", hint=_HINT) from None
        videos = document.get("videos") if isinstance(document, dict) else None
        if not isinstance(document, dict) or not isinstance(videos or [], list):
            raise ContractError(f"polyglotfake: {path} has no list of videos", hint=_HINT)
        for entry in videos or []:
            if not isinstance(entry, Mapping):
                raise ContractError(
                    f"polyglotfake: {path} lists {entry!r}, not a video", hint=_HINT
                )
            file_name = _text(entry.get("filename"), path)
            speaker = _speaker(entry, path) if file_name else None
            if speaker:
                speakers[file_name] = speaker
    _log.debug("polyglotfake: the metadata names the speaker of %d clips", len(speakers))
    return speakers


def _real(lang: str) -> TaskSpec:
    return TaskSpec(lang.upper(), lang, "real", f"original_content/{lang}/videos", "original")


def _dub(lang: str) -> TaskSpec:
    return TaskSpec(
        f"LS_TO_{lang.upper()}",
        f"to_{lang}",
        "fake",
        f"manipulated_content/to_{lang}/videos",
        "lip_sync",
    )


class PolyGlotFakeBuilder(BaseBuilder):
    """PolyGlotFake (``polyglotfake``): seven languages of real clips and their dubbed fakes."""

    dataset_id = "polyglotfake"
    expected_folder = "PolyGlotFake"
    label_prefix = "PGF"
    tasks = (*(_real(lang) for lang in _LANGUAGES), *(_dub(lang) for lang in _LANGUAGES))
    metadata_files = tuple(f"{_SPEAKER_DIR}/{lang}.json" for lang in _LANGUAGES)
    labels = {
        **{
            lang.upper(): LabelSpec(binary=0, binary_av=0, multiclass=1, family="real")
            for lang in _LANGUAGES
        },
        **{
            f"LS_TO_{lang.upper()}": LabelSpec(
                binary=1, binary_av=1, multiclass=2 + index, family="lip-sync"
            )
            for index, lang in enumerate(_LANGUAGES)
        },
    }
    schemes = {
        "ident-72-14-14": SchemeSpec(
            "ident-72-14-14",
            "derived",
            rationale="no official split; identity-disjoint md5 carve (72/14/14) on the speaker "
            "built from the release's metadata, so a speaker's clips and every dub of them stay "
            "on one side",
        ),
        "all-test": SchemeSpec(
            "all-test",
            "subset",
            rationale="every video is test, for evaluating a model trained on another dataset",
        ),
        "benchmark": SchemeSpec(
            "benchmark",
            "subset",
            rationale="a small, seeded evaluation set: up to 15 fakes per target language, "
            f"source language and task, drawn from the whole dataset, {BENCHMARK_REALS}",
        ),
    }
    default_scheme = "ident-72-14-14"
    benchmark = BenchmarkSpec(k_fake=15, strata=("target_id", "source_id", "task"))
    pairing_rule = "strip-to-suffix"
    card_info = {
        "name": "PolyGlotFake",
        "aliases": ["PGF"],
        "release": "766 real clips in Arabic, English, Spanish, French, Japanese, Russian and "
        "Chinese, and 14,472 fakes dubbed from them into the other languages with speech "
        "synthesis or voice cloning and lip-sync (VideoRetalking, Wav2Lip), with the release's "
        "per-language metadata",
        "homepage": "https://github.com/tobuta/PolyGlotFake",
        "paper": {
            "title": "PolyGlotFake: A Novel Multilingual and Multimodal DeepFake Dataset",
            "venue": "arXiv",
            "year": 2024,
        },
        "license": {
            "spdx": None,
            "summary": "access is restricted to academic institutions, for research use only; "
            "the terms need review",
            "url": None,
        },
        "access": "download from the link in the PolyGlotFake repository; dfwb never "
        "distributes media",
        "modalities": ["video", "audio"],
        "compressions": None,
        "key_rule": "real: <LANG>/<lang>_<n>; fake: LS_TO_<LANG>/<src>_<n>_to_<lang>_<tts> "
        "(the file stem)",
    }
    layout_notes = (
        "Reals are named <lang>_<n>; fakes <src>_<n>_to_<lang>_<tts>, each pairing with the real "
        "<src>_<n>.\n"
        f"Speaker identities come from the release's real_json_files/<lang>.json, kept in "
        f"{_SPEAKER_DIR}/<lang>.json: '<source>__age<age>_<sex>' of each listed real clip (the "
        "source alone when the age or sex is missing); a fake takes the speaker of the real it "
        "was dubbed from. Without them, a real's identity is its language and a fake's is its "
        "source language."
    )

    _speakers: Mapping[str, str] | None = None

    def prepare(self, root: Path) -> None:
        """Read the per-language metadata under ``root`` once per build.

        Raises:
            ContractError: a metadata file cannot be read, or is JSON of another shape.
        """
        super().prepare(root)
        self._speakers = _read_speakers(root)

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        """Every video is kept, keyed by its file stem; its fields come from the name."""
        stem = path.stem
        identity: str | None
        source: str | None = None
        pair_key: str | None = None
        if task.kind == "real":
            identity = target = task.name
            lookup: str | None = path.name
        elif _DUB in stem:
            pair_key, dubbed = stem.split(_DUB, 1)
            identity = source = pair_key.split("_", 1)[0]
            target = dubbed.split("_", 1)[0]
            lookup = f"{pair_key}.mp4" if pair_key else None
        else:
            _log.debug("polyglotfake: fake %s is not named <src>_<n>_to_<lang>_<tts>", relpath)
            identity, target, lookup = None, task.name.replace("to_", ""), None
        speaker = (self._speakers or {}).get(lookup) if lookup else None
        return self.record(
            task,
            stem,
            relpath,
            compression,
            identity=speaker or identity,
            target_id=target,
            source_id=source,
            pair_key=pair_key,
        )

    def pair_candidates(self, fake: InventoryRecord) -> str | None:
        """The part of the fake's name before its first ``_to_``: the real it was dubbed from."""
        key = local_key(fake.key)
        return (key.split(_DUB, 1)[0] or None) if _DUB in key else None
