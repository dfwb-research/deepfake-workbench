"""The PolyGlotFake inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``, and the metadata files
are small hand-written JSON documents in the release's per-language format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.polyglotfake import PolyGlotFakeBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_72_14_14,
    assign_benchmark,
    local_key,
    resolve_pairs,
    task_of,
)

REAL_ES = ("original_content", "es", "videos")
FAKE_RU = ("manipulated_content", "to_ru", "videos")
JSON_DIR = ".official_files/.json/real_json_files"
SOURCE = "URL:https://example.org/v/1"


def _make_synthetic_pgf_root(tmp_path: Path) -> Path:
    """A tiny PolyGlotFake tree without its metadata: two Spanish reals, three fakes into Russian.

    The first fake is dubbed from the first real (``es_1``); the third from an English clip
    that is not in the tree.
    """
    root = tmp_path / "PolyGlotFake"
    _touch(root.joinpath(*REAL_ES), "es_1.mp4", "es_2.mp4")
    _touch(
        root.joinpath(*FAKE_RU),
        "es_1_to_ru_Xtts.mp4",
        "es_2_to_ru_Bark.mp4",
        "en_5_to_ru_Micro.mp4",
    )
    return root


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()


def _video(filename: str, source: str | None, *characters: object) -> dict:
    entry: dict = {"filename": filename, "duration": "00:00:10", "characters": list(characters)}
    if source is not None:
        entry["source"] = source
    return entry


def _write_lang(root: Path, lang: str, *videos: dict) -> None:
    folder = root / JSON_DIR
    folder.mkdir(parents=True, exist_ok=True)
    document = {"language": lang, "version": "1.0", "videos": list(videos)}
    (folder / f"{lang}.json").write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture
def pgf_root(tmp_path: Path) -> Path:
    return _make_synthetic_pgf_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(PolyGlotFakeBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(pgf_root):
    # 2 reals + 3 fakes.
    assert len(_discover(pgf_root)) == 5


def test_entry_schema(pgf_root):
    for rec in _discover(pgf_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"ES", "LS_TO_RU"}
        assert rec.builder == BuilderRef("polyglotfake", PolyGlotFakeBuilder.version)
        assert rec.attrs == {"task_name": {"ES": "es", "LS_TO_RU": "to_ru"}[task]}
        # PolyGlotFake has a single version: no compression level.
        assert rec.compression is None
        assert not rec.relpath.startswith("/")
        assert rec.folder is None


def test_no_media_field_and_no_label_class(pgf_root):
    for rec in _discover(pgf_root):
        assert rec.label_key in {"PGF-ES", "PGF-LS_TO_RU"}
        assert rec.probe is None
        assert not hasattr(rec, "media")
        assert not hasattr(rec, "label")


def test_pairing_attrs_present(pgf_root):
    fakes = [r for r in _discover(pgf_root) if r.attrs["task_name"].startswith("to_")]
    assert fakes
    paired = [r for r in fakes if r.pair_key]
    assert paired
    sample = paired[0]
    assert sample.target_id
    assert sample.source_id


def test_fake_pair_key_matches_real_prefix(pgf_root):
    fakes = [r for r in _discover(pgf_root) if r.attrs["task_name"].startswith("to_")]
    assert fakes
    for rec in fakes:
        stem = local_key(rec.key)
        expected = stem.split("_to_", 1)[0]
        assert rec.pair_key == expected
        # The source is the language at the start of the pair key; the target the one after
        # "_to_".
        assert rec.source_id == expected.split("_", 1)[0]
        assert rec.target_id == "ru"
        assert rec.method == "lip_sync"


def test_without_metadata_the_identity_is_the_language(pgf_root):
    records = {rec.key: rec for rec in _discover(pgf_root)}
    real = records["ES/es_1"]
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "es",
        "es",
        None,
        None,
    )
    assert real.method == "original"
    assert real.relpath == "original_content/es/videos/es_1.mp4"
    fake = records["LS_TO_RU/en_5_to_ru_Micro"]
    assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
        "en",
        "ru",
        "en",
        "en_5",
    )


def test_a_fake_without_to_takes_its_folders_language(pgf_root):
    _touch(pgf_root.joinpath(*FAKE_RU), "es_9.mp4")
    rec = {r.key: r for r in _discover(pgf_root)}["LS_TO_RU/es_9"]
    assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (None, "ru", None, None)


def test_every_language_is_a_task(pgf_root):
    for lang in ("ar", "en", "fr", "ja", "zh"):
        _touch(pgf_root / "original_content" / lang / "videos", f"{lang}_1.mp4")
        _touch(pgf_root / "manipulated_content" / f"to_{lang}" / "videos", f"es_1_to_{lang}_X.mp4")
    _touch(pgf_root / "original_content" / "ru" / "videos", "ru_1.mp4")
    _touch(pgf_root / "manipulated_content" / "to_es" / "videos", "en_1_to_es_X.mp4")
    tasks = {task_of(r.key) for r in _discover(pgf_root)}
    langs = ("AR", "EN", "ES", "FR", "JA", "RU", "ZH")
    assert tasks == {*langs, *(f"LS_TO_{lang}" for lang in langs)}


def test_the_language_folders_are_flat_and_no_compression_can_be_requested(pgf_root):
    _touch(pgf_root.joinpath(*REAL_ES, "nested"), "es_3.mp4")
    assert "ES/es_3" not in {rec.key for rec in _discover(pgf_root)}
    with pytest.raises(ConfigError, match="c0") as caught:
        _discover(pgf_root, compressions=["c0"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ metadata


def test_the_metadata_gives_reals_and_their_fakes_a_speaker_identity(pgf_root):
    _write_lang(
        pgf_root,
        "es",
        _video("es_1.mp4", SOURCE, {"age": 30, "sex": "male"}),
        # Without an age or a sex, the source alone.
        _video("es_2.mp4", SOURCE, {"sex": "female"}),
    )
    records = {rec.key: rec for rec in _discover(pgf_root)}
    speaker = f"{SOURCE}__age30_male"
    assert records["ES/es_1"].identity == speaker
    assert records["ES/es_1"].target_id == "es"
    assert records["ES/es_2"].identity == SOURCE
    # A fake takes the identity of the real it was dubbed from.
    assert records["LS_TO_RU/es_1_to_ru_Xtts"].identity == speaker
    assert records["LS_TO_RU/es_1_to_ru_Xtts"].source_id == "es"
    assert records["LS_TO_RU/es_2_to_ru_Bark"].identity == SOURCE
    # A fake whose real is not listed keeps its source language.
    assert records["LS_TO_RU/en_5_to_ru_Micro"].identity == "en"


def test_metadata_entries_that_give_no_speaker_are_skipped(pgf_root):
    _write_lang(
        pgf_root,
        "es",
        _video("es_1.mp4", None, {"age": 30, "sex": "male"}),
        _video("", SOURCE, {"age": 30, "sex": "male"}),
        _video("es_2.mp4", "  ", {"age": 30, "sex": "male"}),
    )
    records = {rec.key: rec for rec in _discover(pgf_root)}
    assert records["ES/es_1"].identity == "es"
    assert records["ES/es_2"].identity == "es"


def test_the_first_character_counts_and_a_later_language_file_wins(pgf_root):
    _write_lang(
        pgf_root,
        "es",
        _video("es_1.mp4", SOURCE, {"age": 50, "sex": " female "}, {"age": 20, "sex": "male"}),
    )
    _write_lang(pgf_root, "zh", _video("es_2.mp4", "B", {"age": 0, "sex": "male"}))
    _write_lang(pgf_root, "ar", _video("es_2.mp4", "A", {"age": 1, "sex": "male"}))
    records = {rec.key: rec for rec in _discover(pgf_root)}
    assert records["ES/es_1"].identity == f"{SOURCE}__age50_female"
    # Files are read in the order ar, en, es, fr, ja, ru, zh; an age of 0 still counts.
    assert records["ES/es_2"].identity == "B__age0_male"


def test_a_real_is_looked_up_by_its_file_name(pgf_root):
    # The lookup is by the file name as it is on disk, suffix included.
    _touch(pgf_root.joinpath(*REAL_ES), "es_3.MP4")
    _write_lang(pgf_root, "es", _video("es_3.mp4", SOURCE, {"age": 30, "sex": "male"}))
    rec = {r.key: r for r in _discover(pgf_root)}["ES/es_3"]
    assert rec.identity == "es"


def test_missing_metadata_falls_back_with_a_warning(pgf_root, caplog):
    with caplog.at_level(logging.WARNING):
        records = _discover(pgf_root)
    assert {r.identity for r in records} == {"es", "en"}
    assert "real_json_files" in caplog.text


def test_a_metadata_file_that_is_not_json_is_skipped_with_a_warning(pgf_root, caplog):
    folder = pgf_root / JSON_DIR
    folder.mkdir(parents=True)
    (folder / "es.json").write_text("{not json", encoding="utf-8")
    _write_lang(pgf_root, "ru", _video("es_2.mp4", SOURCE, {"age": 30, "sex": "male"}))
    with caplog.at_level(logging.WARNING):
        records = {rec.key: rec for rec in _discover(pgf_root)}
    assert "es.json" in caplog.text
    assert records["ES/es_1"].identity == "es"
    assert records["ES/es_2"].identity == f"{SOURCE}__age30_male"


@pytest.mark.parametrize(
    "content",
    [
        [],
        {"videos": {"a": 1}},
        {"videos": ["es_1.mp4"]},
        {"videos": [{"filename": 3, "source": "A"}]},
        {"videos": [{"filename": "es_1.mp4", "source": "A", "characters": [{"sex": 1}]}]},
    ],
)
def test_a_malformed_metadata_file_is_a_contract_error(pgf_root, content):
    folder = pgf_root / JSON_DIR
    folder.mkdir(parents=True)
    (folder / "es.json").write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ContractError, match=r"es\.json"):
        _discover(pgf_root)


# ------------------------------------------------------------------------------ splits, pairs


def test_there_is_no_official_split(pgf_root):
    builder = PolyGlotFakeBuilder()
    with pytest.raises(ContractError, match="no official split"):
        builder.official_splits(pgf_root, collect_records(builder, pgf_root))


def test_the_carve_keeps_a_speaker_on_one_side(pgf_root):
    _write_lang(pgf_root, "es", _video("es_1.mp4", SOURCE, {"age": 30, "sex": "male"}))
    records = _discover(pgf_root)
    assignment = assign_72_14_14(records)
    speaker = [r for r in records if r.identity == f"{SOURCE}__age30_male"]
    assert {r.key for r in speaker} == {"ES/es_1", "LS_TO_RU/es_1_to_ru_Xtts"}
    assert len({assignment[(r.key, r.compression)] for r in speaker}) == 1


def test_pair_candidates_are_the_name_before_to(pgf_root):
    builder = PolyGlotFakeBuilder()
    _touch(pgf_root.joinpath(*FAKE_RU), "es_9.mp4", "_to_ru_X.mp4")
    records = collect_records(builder, pgf_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "LS_TO_RU/es_1_to_ru_Xtts": "es_1",
        "LS_TO_RU/es_2_to_ru_Bark": "es_2",
        "LS_TO_RU/en_5_to_ru_Micro": "en_5",
        "LS_TO_RU/es_9": None,
        "LS_TO_RU/_to_ru_X": None,
    }
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    # en_5 is not in the tree, so that fake has no pair.
    assert pairs == [
        PairRecord("LS_TO_RU/es_1_to_ru_Xtts", "ES/es_1", "strip-to-suffix"),
        PairRecord("LS_TO_RU/es_2_to_ru_Bark", "ES/es_2", "strip-to-suffix"),
    ]


def test_label_vocab_covers_every_task():
    builder = PolyGlotFakeBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"PGF-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    langs = ("AR", "EN", "ES", "FR", "JA", "RU", "ZH")
    assert table == {
        **{f"PGF-{lang}": (0, 0, 1, "real") for lang in langs},
        **{f"PGF-LS_TO_{lang}": (1, 1, 2 + i, "lip-sync") for i, lang in enumerate(langs)},
    }


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = PolyGlotFakeBuilder()
    assert builder.default_scheme == "ident-72-14-14"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "ident-72-14-14": "ident-72-14-14",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=15, strata=("target_id", "source_id", "task"))
    assert builder.pairing_rule == "strip-to-suffix"
    assert builder.pairing_fanout is None
    assert builder.metadata_files == tuple(
        f"{JSON_DIR}/{lang}.json" for lang in ("ar", "en", "es", "fr", "ja", "ru", "zh")
    )


def test_the_benchmark_draws_per_language_pair(pgf_root):
    _touch(pgf_root.joinpath(*FAKE_RU), *(f"es_{n}_to_ru_X.mp4" for n in range(3, 23)))
    _touch(pgf_root / "original_content" / "ru" / "videos", "ru_1.mp4")
    builder = PolyGlotFakeBuilder()
    records = collect_records(builder, pgf_root)
    chosen = [
        key
        for key, _ in assign_benchmark(
            records,
            spec=builder.benchmark,
            is_real=builder.is_real,
            task_rank=builder.task_rank(),
            pool_keys=None,
        )
    ]
    fakes = [key for key in chosen if task_of(key) == "LS_TO_RU"]
    # es -> ru has 22 fakes and keeps 15; en -> ru keeps its one; all three reals are kept.
    assert len([k for k in fakes if local_key(k).startswith("es_")]) == 15
    assert "LS_TO_RU/en_5_to_ru_Micro" in fakes
    assert sorted(k for k in chosen if k not in fakes) == ["ES/es_1", "ES/es_2", "RU/ru_1"]


def test_dataset_card():
    builder = PolyGlotFakeBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "polyglotfake"
    assert card.name == "PolyGlotFake"
    assert "PGF" in card.aliases
    assert card.compressions is None
    assert card.modalities == ["video", "audio"]
    assert card.default_scheme == "ident-72-14-14"
    assert card.homepage == "https://github.com/tobuta/PolyGlotFake"
    assert card.paper is not None
    assert card.paper.title == "PolyGlotFake: A Novel Multilingual and Multimodal DeepFake Dataset"
    assert card.paper.doi is None
    assert card.license.spdx is None


def test_the_layout_names_the_language_folders_and_the_metadata():
    text = PolyGlotFakeBuilder().describe_layout()
    assert "'PolyGlotFake'" in text
    assert "original_content/ar/videos/<video>" in text
    assert "manipulated_content/to_zh/videos/<video>" in text
    assert f"{JSON_DIR}/<lang>.json" in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("polyglotfake"), PolyGlotFakeBuilder)
    assert PolyGlotFakeBuilder.expected_folder == "PolyGlotFake"
    assert PolyGlotFakeBuilder.label_prefix == "PGF"


def test_an_unreadable_metadata_file_is_a_contract_error(pgf_root):
    folder = pgf_root / JSON_DIR
    folder.mkdir(parents=True)
    (folder / "es.json").write_bytes(b'{"videos": ["\xff"]}')
    with pytest.raises(ContractError, match=r"cannot read .*es\.json"):
        _discover(pgf_root)


def test_a_video_without_characters_is_its_source(pgf_root):
    _write_lang(pgf_root, "es", _video("es_1.mp4", SOURCE), _video("es_2.mp4", "B", "a text"))
    records = {rec.key: rec for rec in _discover(pgf_root)}
    assert records["ES/es_1"].identity == SOURCE
    assert records["ES/es_2"].identity == "B"
