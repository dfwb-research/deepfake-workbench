"""The TalkingHeadBench inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``, and the split list is
a hand-written file in the release's ``real_dataset_split_official_ff++.json`` format. The fakes
are scanned; the reals are listed by the split list and live in two sibling dataset folders,
FaceForensics++ and CelebV-HQ, which the release does not ship.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.paths import resolve_roots
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.talkingheadbench import TalkingHeadBenchBuilder
from dfwb.preprocess.inventory.runner import (
    build_inventory,
    collect_records,
    get_builder,
    read_inventory,
    resolve_video_path,
)
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    resolve_pairs,
    task_of,
)

SPLIT_LIST = ".official_files/real_dataset_split_official_ff++.json"
FFPP_VIDEOS = "original_content/YouTube/raw/videos"
CELEBVHQ_VIDEOS = "original_content/YouTube/videos"
LP = "manipulated_content/LivePortrait"
H3 = "manipulated_content/Hallo3"
MAGI = "manipulated_content/MAGI-1"
FAKE_TASKS = ("TH_LP", "TH_APA", "TH_APV", "TH_H1", "TH_H2", "TH_EP", "TH_M1", "TH_H3")

SPLITS = {
    "Train": {"FaceForensics++": ["001.mp4", "002.mp4"], "CelebV-HQ": ["abcDEF_0.mp4"]},
    "Val": {"FaceForensics++": [], "CelebV-HQ": []},
    "Test": {"FaceForensics++": [], "CelebV-HQ": ["-5be_UPkLRw_4.mp4"]},
}


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"\x00")


def _write_splits(root: Path, splits: Any) -> None:
    path = root / SPLIT_LIST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits), encoding="utf-8")


def _make_synthetic_thb_root(tmp_path: Path) -> Path:
    """A tiny release: the split list, four fakes (one with its audio) and one real's audio."""
    root = tmp_path / "TalkingHeadBench"
    _write_splits(root, SPLITS)
    _touch(
        root / LP / "videos",
        "00198--abcDEF_0--LivePortrait.mp4",  # a driver named as listed
        "47608--5be_UPkLRw_4--LivePortrait.mp4",  # the listed driver's leading '-' was lost
    )
    _touch(root / LP / "audio", "00198--abcDEF_0--LivePortrait.wav")  # only one has audio
    _touch(root / H3 / "videos", "65594-abcDEF_0.mp4")  # Hallo3: a single '-'
    _touch(root / MAGI / "videos", "10452.mp4")  # MAGI-1: no driver
    _touch(root / "original_content" / "real" / "audio", "001.wav")
    return root


@pytest.fixture
def thb_root(tmp_path: Path) -> Path:
    return _make_synthetic_thb_root(tmp_path)


def _records(root: Path, builder: TalkingHeadBenchBuilder | None = None) -> dict[str, Any]:
    return {rec.key: rec for rec in collect_records(builder or TalkingHeadBenchBuilder(), root)}


# ------------------------------------------------------------------------------ discovery


def test_discover_lists_the_reals_and_scans_the_fakes(thb_root):
    # 2 FaceForensics++ and 2 CelebV-HQ reals from the split list, 2 LivePortrait, 1 Hallo3 and
    # 1 MAGI-1 fakes from disk.
    assert sorted(_records(thb_root)) == [
        "REAL_CVHQ/-5be_UPkLRw_4",
        "REAL_CVHQ/abcDEF_0",
        "REAL_FF/001",
        "REAL_FF/002",
        "TH_H3/65594-abcDEF_0",
        "TH_LP/00198--abcDEF_0--LivePortrait",
        "TH_LP/47608--5be_UPkLRw_4--LivePortrait",
        "TH_M1/10452",
    ]


def test_entry_schema(thb_root):
    for rec in collect_records(TalkingHeadBenchBuilder(), thb_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert rec.builder == BuilderRef("talkingheadbench", TalkingHeadBenchBuilder.version)
        assert rec.compression is None  # a single version, no compression levels
        assert rec.label_key == f"THB-{task}"
        assert set(rec.attrs) - {"audio_relpath"} == {"task_name"}
        assert rec.pair_key is None
        assert not rec.relpath.startswith("/")
        assert ".." not in rec.relpath.split("/")
        assert rec.probe is None


def test_reals_live_in_their_sibling_folders(thb_root):
    records = _records(thb_root)
    ff = records["REAL_FF/002"]
    assert (ff.folder, ff.relpath) == ("FaceForensics++", f"{FFPP_VIDEOS}/002.mp4")
    cv = records["REAL_CVHQ/-5be_UPkLRw_4"]
    assert (cv.folder, cv.relpath) == ("CelebV-HQ", f"{CELEBVHQ_VIDEOS}/-5be_UPkLRw_4.mp4")
    for rec in (ff, cv):
        assert rec.method == "original"
        assert rec.identity == rec.target_id == rec.key.partition("/")[2]
        assert rec.source_id is None
    assert all(rec.folder is None for rec in records.values() if rec.key.startswith("TH_"))


def test_reals_are_listed_not_scanned(thb_root, tmp_path):
    # A sibling folder's other videos are never records, and a listed real needs no video.
    _touch(tmp_path / "FaceForensics++" / FFPP_VIDEOS, "001.mp4", "999.mp4")
    records = _records(thb_root)
    assert {key for key in records if key.startswith("REAL_FF/")} == {"REAL_FF/001", "REAL_FF/002"}


def test_a_stem_listed_twice_is_one_real(thb_root):
    splits = json.loads(json.dumps(SPLITS))
    splits["Test"]["FaceForensics++"] = ["001.mp4"]
    _write_splits(thb_root, splits)
    assert sum(key == "REAL_FF/001" for key in _records(thb_root)) == 1


def test_the_fake_driver_is_parsed_and_its_leading_dash_restored(thb_root):
    records = _records(thb_root)
    assert records["TH_LP/00198--abcDEF_0--LivePortrait"].source_id == "abcDEF_0"
    assert records["TH_LP/47608--5be_UPkLRw_4--LivePortrait"].source_id == "-5be_UPkLRw_4"
    assert records["TH_H3/65594-abcDEF_0"].source_id == "abcDEF_0"
    assert records["TH_M1/10452"].source_id is None
    for key in ("TH_LP/00198--abcDEF_0--LivePortrait", "TH_M1/10452"):
        # The FFHQ portrait a fake animates is not in its name.
        assert records[key].identity is None
        assert records[key].target_id is None


def test_a_hallo3_driver_has_its_leading_dash_restored_too(thb_root):
    _touch(thb_root / H3 / "videos", "00002-5be_UPkLRw_4.mp4")
    assert _records(thb_root)["TH_H3/00002-5be_UPkLRw_4"].source_id == "-5be_UPkLRw_4"


def test_an_unlisted_driver_is_kept_as_written(thb_root):
    _touch(thb_root / LP / "videos", "00001--zzz_1--LivePortrait.mp4")
    assert _records(thb_root)["TH_LP/00001--zzz_1--LivePortrait"].source_id == "zzz_1"


def test_a_name_without_the_expected_parts_has_no_driver(thb_root):
    _touch(thb_root / LP / "videos", "00001--LivePortrait.mp4")
    _touch(thb_root / H3 / "videos", "00002.mp4")
    records = _records(thb_root)
    assert records["TH_LP/00001--LivePortrait"].source_id is None
    assert records["TH_H3/00002"].source_id is None


def test_the_audio_relpath_is_set_only_when_the_wav_exists(thb_root):
    _touch(thb_root / "original_content" / "fake_celebvhq", "abcDEF_0.wav")
    records = _records(thb_root)
    assert records["TH_LP/00198--abcDEF_0--LivePortrait"].attrs["audio_relpath"] == (
        f"{LP}/audio/00198--abcDEF_0--LivePortrait.wav"
    )
    assert "audio_relpath" not in records["TH_LP/47608--5be_UPkLRw_4--LivePortrait"].attrs
    # A real's audio ships with the release, so its path is inside this dataset's folder.
    assert records["REAL_FF/001"].attrs["audio_relpath"] == "original_content/real/audio/001.wav"
    assert "audio_relpath" not in records["REAL_FF/002"].attrs
    assert records["REAL_CVHQ/abcDEF_0"].attrs["audio_relpath"] == (
        "original_content/fake_celebvhq/abcDEF_0.wav"
    )


def test_the_audio_is_found_in_any_copy(thb_root, tmp_path):
    other = tmp_path / "other" / "TalkingHeadBench"
    _touch(other / MAGI / "audio", "10452.wav")
    builder = TalkingHeadBenchBuilder()
    builder.bind_copies([thb_root, other])
    assert _records(thb_root, builder)["TH_M1/10452"].attrs["audio_relpath"] == (
        f"{MAGI}/audio/10452.wav"
    )


def test_every_task_and_method(thb_root):
    for name in ("AniPortraitAudio", "AniPortraitVideo", "Hallo", "Hallo2", "EmoPortrait"):
        _touch(thb_root / "manipulated_content" / name / "videos", f"00001--x_0--{name}.mp4")
    methods = {task_of(key): rec.method for key, rec in _records(thb_root).items()}
    assert methods == {
        "REAL_FF": "original",
        "REAL_CVHQ": "original",
        "TH_LP": "LivePortrait",
        "TH_APA": "AniPortraitAudio",
        "TH_APV": "AniPortraitVideo",
        "TH_H1": "Hallo",
        "TH_H2": "Hallo2",
        "TH_EP": "EmoPortrait",
        "TH_M1": "MAGI-1",
        "TH_H3": "Hallo3",
    }


def test_without_the_split_list_there_are_no_reals_with_a_warning(thb_root, caplog):
    (thb_root / SPLIT_LIST).unlink()
    with caplog.at_level(logging.WARNING, logger="dfwb.preprocess.inventory.builders"):
        records = _records(thb_root)
    assert "real_dataset_split_official_ff++.json" in caplog.text
    assert not any(key.startswith("REAL_") for key in records)
    # Without the list the lost leading '-' cannot be restored.
    assert records["TH_LP/47608--5be_UPkLRw_4--LivePortrait"].source_id == "5be_UPkLRw_4"


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json", "cannot read"),
        ("[]", "not an object"),
        ('{"Train": ["a.mp4"]}', "Train"),
        ('{"Train": {"CelebV-HQ": "a.mp4"}}', "CelebV-HQ"),
        ('{"Test": {"FaceForensics++": [1]}}', "FaceForensics"),
    ],
)
def test_a_malformed_split_list_is_a_contract_error(thb_root, content, match):
    (thb_root / SPLIT_LIST).write_text(content, encoding="utf-8")
    with pytest.raises(ContractError, match=match):
        collect_records(TalkingHeadBenchBuilder(), thb_root)


def test_missing_blocks_are_empty(thb_root):
    _write_splits(thb_root, {"Train": {"FaceForensics++": ["001.mp4"]}, "Val": None, "Test": []})
    assert sorted(k for k in _records(thb_root) if k.startswith("REAL_")) == ["REAL_FF/001"]


def test_no_compression_can_be_requested(thb_root):
    with pytest.raises(ConfigError, match="unknown compression"):
        collect_records(TalkingHeadBenchBuilder(), thb_root, compressions=["c23"])
    with pytest.raises(ConfigError, match="unknown compression"):
        list(TalkingHeadBenchBuilder().discover(thb_root, compressions=["c23"]))


# ------------------------------------------------------------------------------ layout


def test_the_layout_is_the_fake_folders_only(thb_root, tmp_path):
    builder = TalkingHeadBenchBuilder()
    assert builder.layout_dirs() == tuple(
        f"manipulated_content/{name}/videos"
        for name in (
            "LivePortrait",
            "AniPortraitAudio",
            "AniPortraitVideo",
            "Hallo",
            "Hallo2",
            "EmoPortrait",
            "MAGI-1",
            "Hallo3",
        )
    )
    assert builder.videos_present(thb_root) == (f"{LP}/videos", f"{MAGI}/videos", f"{H3}/videos")
    # A sibling folder full of real videos is not this dataset's layout, nor ever scanned.
    _touch(tmp_path / "FaceForensics++" / FFPP_VIDEOS, "001.mp4")
    lonely = tmp_path / "copy" / "TalkingHeadBench"
    _write_splits(lonely, SPLITS)
    _touch(tmp_path / "copy" / "FaceForensics++" / FFPP_VIDEOS, "001.mp4")
    assert not builder.layout_present(lonely)
    chosen = builder.choose_copies([lonely, thb_root])
    assert set(chosen) == {("TH_LP", None), ("TH_M1", None), ("TH_H3", None)}
    assert set(chosen.values()) == {thb_root}


def test_a_build_resolves_each_real_in_its_sibling_folder_under_any_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith("DFWB_"):
            monkeypatch.delenv(name)
    first, second = tmp_path / "first", tmp_path / "second"
    _make_synthetic_thb_root(first)
    _touch(second / "FaceForensics++" / FFPP_VIDEOS, "002.mp4")
    _touch(first / "CelebV-HQ" / CELEBVHQ_VIDEOS, "-5be_UPkLRw_4.mp4")
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{first}:{second}")
    monkeypatch.setenv("DFWB_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    roots = resolve_roots()
    result = build_inventory("talkingheadbench", roots=roots)
    records = {rec.key: rec for rec in read_inventory("talkingheadbench", result.path.parents[1])}
    assert len(records) == 8
    assert resolve_video_path(records["REAL_FF/002"], (), datasets_roots=roots) == (
        second / "FaceForensics++" / FFPP_VIDEOS / "002.mp4"
    )
    assert resolve_video_path(records["REAL_CVHQ/-5be_UPkLRw_4"], (), datasets_roots=roots) == (
        first / "CelebV-HQ" / CELEBVHQ_VIDEOS / "-5be_UPkLRw_4.mp4"
    )
    with pytest.raises(ConfigError, match="not found"):
        resolve_video_path(records["REAL_FF/001"], (), datasets_roots=roots)


# ------------------------------------------------------------------------------ official split


def _official_fixture(root: Path) -> None:
    _write_splits(
        root,
        {
            "Train": {"FaceForensics++": ["001.mp4"], "CelebV-HQ": ["cv_tr.mp4"]},
            "Val": {"FaceForensics++": ["002.mp4"], "CelebV-HQ": ["cv_va.mp4"]},
            "Test": {"FaceForensics++": ["003.mp4"], "CelebV-HQ": ["cv_te.mp4"]},
        },
    )
    _touch(
        root / LP / "videos",
        "00001--cv_te--LivePortrait.mp4",  # driver in CelebV-HQ test: test
        "00002--cv_tr--LivePortrait.mp4",  # driver in CelebV-HQ train: train
        "00003--cv_va--LivePortrait.mp4",  # driver in CelebV-HQ val: train (fakes get no val)
        "00004--unlisted--LivePortrait.mp4",  # driver not listed: train
        "00005--LivePortrait.mp4",  # no driver: train
    )
    _touch(root / H3 / "videos", "00006-cv_tr.mp4")  # held out: test, whatever its driver


def test_the_official_split(thb_root):
    _official_fixture(thb_root)
    builder = TalkingHeadBenchBuilder()
    records = collect_records(builder, thb_root)
    official = builder.official_splits(thb_root, records)
    assert official == {
        "REAL_FF/001": "train",
        "REAL_FF/002": "val",
        "REAL_FF/003": "test",
        "REAL_CVHQ/cv_tr": "train",
        "REAL_CVHQ/cv_va": "val",
        "REAL_CVHQ/cv_te": "test",
        "TH_LP/00001--cv_te--LivePortrait": "test",
        "TH_LP/00002--cv_tr--LivePortrait": "train",
        "TH_LP/00003--cv_va--LivePortrait": "train",
        "TH_LP/00004--unlisted--LivePortrait": "train",
        "TH_LP/00005--LivePortrait": "train",
        "TH_H3/00006-cv_tr": "test",
        "TH_M1/10452": "test",
        # The fixture's other fakes name drivers the new list no longer holds.
        "TH_LP/00198--abcDEF_0--LivePortrait": "train",
        "TH_LP/47608--5be_UPkLRw_4--LivePortrait": "train",
        "TH_H3/65594-abcDEF_0": "test",
    }
    assignment = assign_official(records, official)
    assert len(assignment) == len(records)


def test_a_real_the_list_no_longer_holds_gets_no_split(thb_root):
    builder = TalkingHeadBenchBuilder()
    records = collect_records(builder, thb_root)
    _official_fixture(thb_root)
    official = builder.official_splits(thb_root, records)
    assert "REAL_FF/002" in official  # still listed, now as val
    assert "REAL_CVHQ/abcDEF_0" not in official
    assert "REAL_CVHQ/-5be_UPkLRw_4" not in official


def test_a_stem_listed_twice_takes_the_last_of_train_val_test(thb_root):
    splits = json.loads(json.dumps(SPLITS))
    splits["Val"]["FaceForensics++"] = ["001.mp4"]
    splits["Val"]["CelebV-HQ"] = ["-5be_UPkLRw_4.mp4"]
    _write_splits(thb_root, splits)
    builder = TalkingHeadBenchBuilder()
    records = collect_records(builder, thb_root)
    official = builder.official_splits(thb_root, records)
    assert official["REAL_FF/001"] == "val"
    assert official["REAL_CVHQ/-5be_UPkLRw_4"] == "test"
    assert official["TH_LP/47608--5be_UPkLRw_4--LivePortrait"] == "test"


def test_the_official_split_needs_the_split_list(thb_root):
    builder = TalkingHeadBenchBuilder()
    records = collect_records(builder, thb_root)
    (thb_root / SPLIT_LIST).unlink()
    with pytest.raises(ConfigError, match=r"real_dataset_split_official_ff\+\+\.json") as caught:
        builder.official_splits(thb_root, records)
    assert ".official_files" in caught.value.hint


# ------------------------------------------------------------------------------ pairs, labels


def test_each_fake_pairs_with_its_driving_video(thb_root):
    builder = TalkingHeadBenchBuilder()
    records = collect_records(builder, thb_root)
    fakes = {r.key: builder.pair_candidates(r) for r in records if not builder.is_real(r)}
    assert fakes == {
        "TH_LP/00198--abcDEF_0--LivePortrait": "abcDEF_0",
        "TH_LP/47608--5be_UPkLRw_4--LivePortrait": "-5be_UPkLRw_4",
        "TH_H3/65594-abcDEF_0": "abcDEF_0",
        "TH_M1/10452": None,  # MAGI-1 has no driver, so no pair
    }
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    assert pairs == [
        PairRecord(
            "TH_LP/47608--5be_UPkLRw_4--LivePortrait", "REAL_CVHQ/-5be_UPkLRw_4", "driving-video"
        ),
        PairRecord("TH_H3/65594-abcDEF_0", "REAL_CVHQ/abcDEF_0", "driving-video"),
        PairRecord("TH_LP/00198--abcDEF_0--LivePortrait", "REAL_CVHQ/abcDEF_0", "driving-video"),
    ]


def test_label_vocab_covers_every_task():
    builder = TalkingHeadBenchBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"THB-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {
        "THB-REAL_FF": (0, 0, 1, "real"),
        "THB-REAL_CVHQ": (0, 0, 1, "real"),
        "THB-TH_LP": (1, 1, 2, "talking-head"),
        "THB-TH_APA": (1, 1, 3, "talking-head"),
        "THB-TH_APV": (1, 1, 4, "talking-head"),
        "THB-TH_H1": (1, 1, 5, "talking-head"),
        "THB-TH_H2": (1, 1, 6, "talking-head"),
        "THB-TH_EP": (1, 1, 7, "talking-head"),
        "THB-TH_M1": (1, 1, 8, "talking-head"),
        "THB-TH_H3": (1, 1, 9, "talking-head"),
    }


def test_the_task_table_order():
    assert [task.abbr for task in TalkingHeadBenchBuilder.tasks] == [
        "REAL_FF",
        "REAL_CVHQ",
        *FAKE_TASKS,
    ]


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = TalkingHeadBenchBuilder()
    assert builder.default_scheme == "official"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=100, strata=("task",))
    assert builder.pairing_rule == "driving-video"
    assert builder.pairing_fanout is None
    assert builder.metadata_files == (SPLIT_LIST,)


def test_the_benchmark_draws_from_the_official_test(thb_root):
    _official_fixture(thb_root)
    builder = TalkingHeadBenchBuilder()
    records = collect_records(builder, thb_root)
    official = builder.official_splits(thb_root, records)
    assert builder.benchmark is not None
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=[key for key, split in official.items() if split == "test"],
    )
    assert {key for key, _ in chosen} == {
        "TH_LP/00001--cv_te--LivePortrait",
        "TH_H3/00006-cv_tr",
        "TH_H3/65594-abcDEF_0",
        "TH_M1/10452",
        "REAL_FF/003",
        "REAL_CVHQ/cv_te",
    }


def test_dataset_card():
    builder = TalkingHeadBenchBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "talkingheadbench"
    assert card.name == "TalkingHeadBench"
    assert "THB" in card.aliases
    assert card.compressions is None
    assert card.modalities == ["video", "audio"]
    assert card.default_scheme == "official"
    assert card.homepage == "https://huggingface.co/datasets/luchaoqi/TalkingHeadBench"
    assert card.paper is not None
    assert card.paper.title.startswith("TalkingHeadBench:")
    assert card.paper.doi is None
    assert card.license.spdx is None
    assert "2,994" in card.release


def test_the_layout_names_the_split_list_the_sibling_folders_and_the_release():
    text = TalkingHeadBenchBuilder().describe_layout()
    assert "'TalkingHeadBench'" in text
    assert f"../FaceForensics++/{FFPP_VIDEOS}/<video>" in text
    assert f"../CelebV-HQ/{CELEBVHQ_VIDEOS}/<video>" in text
    assert f"{LP}/videos/<video>" in text
    assert SPLIT_LIST in text
    assert "additional_dataset" in text
    assert "audio_relpath" in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("talkingheadbench"), TalkingHeadBenchBuilder)
    assert TalkingHeadBenchBuilder.expected_folder == "TalkingHeadBench"
    assert TalkingHeadBenchBuilder.label_prefix == "THB"
