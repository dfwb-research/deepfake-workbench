"""The IDForge-v1 inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``, and the split lists
are a few hand-written lines in the release's format.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.idforge import IDForgeV1Builder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    local_key,
    resolve_pairs,
    task_of,
)

PRISTINE = "original_content/pristine/videos"
FACE_AM_TM = "manipulated_content/face_audiomismatch_textmismatch/videos"
LIP_AM_TM = "manipulated_content/lip_audiomismatch_textmismatch/videos"
RVC_TM = "manipulated_content/rvc_textmismatch/videos"
SPLITS = ".official_files/splits"


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")


def _make_synthetic_idfv1_root(tmp_path: Path) -> Path:
    """A tiny IDForge-v1 tree: two pristine videos and two face swaps of ``id00``.

    Every task nests its videos two levels deep, as the release does::

        <task videos>/<identity>/<session>/<file>.mp4

    Pristine stems are ``<identity>_scene_<NNNN>-<n>``; face swap stems are
    ``<identity>_scene_<NNNN>_<clip>_<method>``.
    """
    root = tmp_path / "IDForge-v1"
    _touch(root / PRISTINE / "id00/id00_03", "id00_scene_0012-0.mp4", "id00_scene_0012-1.mp4")
    _touch(
        root / FACE_AM_TM / "id00/id00_03",
        "id00_scene_0015_0_infoswap.mp4",
        "id00_scene_0015_1_roop.mp4",
    )
    return root


def _write_lists(root: Path, train: str = "", val: str = "", test: str = "") -> None:
    folder = root / SPLITS
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in (("train", train), ("val", val), ("test", test)):
        (folder / f"{name}.csv").write_text(text, encoding="utf-8")


@pytest.fixture
def idf_root(tmp_path: Path) -> Path:
    return _make_synthetic_idfv1_root(tmp_path)


def _records(root, **kwargs):
    return {rec.key: rec for rec in collect_records(IDForgeV1Builder(), root, **kwargs)}


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_expected_count(idf_root):
    # 2 real + 2 fake.
    assert len(collect_records(IDForgeV1Builder(), idf_root)) == 4


def test_entry_schema(idf_root):
    for rec in collect_records(IDForgeV1Builder(), idf_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"REAL", "FS_AM_TM"}
        assert rec.builder == BuilderRef("idforge-v1", IDForgeV1Builder.version)
        assert rec.compression is None  # IDForge-v1 has no compression levels
        assert rec.label_key == f"IDFv1-{task}"
        assert set(rec.attrs) == {"task_name"}
        assert not rec.relpath.startswith("/")
        assert rec.folder is None
        assert rec.probe is None
        assert not hasattr(rec, "media")
        assert not hasattr(rec, "label")


def test_key_follows_session_group_prefix_rule(idf_root):
    records = _records(idf_root)
    # The key is <session folder>__<file stem>; the relpath keeps both levels of nesting.
    real = records["REAL/id00_03__id00_scene_0012-0"]
    assert real.relpath == f"{PRISTINE}/id00/id00_03/id00_scene_0012-0.mp4"
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "id00",
        "id00",
        None,
        None,
    )
    assert real.method == "original"
    assert real.attrs == {"task_name": "pristine"}
    assert "FS_AM_TM/id00_03__id00_scene_0015_0_infoswap" in records


def test_pairing_attrs_present_for_fakes(idf_root):
    fakes = [r for r in collect_records(IDForgeV1Builder(), idf_root) if task_of(r.key) != "REAL"]
    assert len(fakes) == 2
    for fake in fakes:
        # A fake's pair_key is its identity: IDForge links a fake to its actor, not a recording.
        assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
            "id00",
            "id00",
            None,
            "id00",
        )
        assert fake.method == "face_audiomismatch_textmismatch"


def test_a_double_extension_keeps_its_inner_suffix(idf_root):
    _touch(idf_root / RVC_TM / "id00/id00_03", "id00_scene_0014.mp3.mp4")
    rec = _records(idf_root)["RVC_TM/id00_03__id00_scene_0014.mp3"]
    assert rec.relpath == f"{RVC_TM}/id00/id00_03/id00_scene_0014.mp3.mp4"
    # Real video with cloned speech is a fake task of the release: it carries a pair_key.
    assert (rec.identity, rec.pair_key) == ("id00", "id00")


def test_names_without_an_identity_and_shallow_videos(idf_root):
    _touch(idf_root / FACE_AM_TM / "id00/id00_03", "scene_0015_0_simswap.mp4")
    _touch(idf_root / PRISTINE, "id07_loose-0.mp4")
    records = _records(idf_root)
    odd = records["FS_AM_TM/id00_03__scene_0015_0_simswap"]
    assert (odd.identity, odd.target_id, odd.pair_key) == (None, None, None)
    # A video straight in the task folder takes that folder's name as its session.
    loose = records["REAL/videos__id07_loose-0"]
    assert loose.identity == "id07"


def test_the_same_key_in_two_tasks_stays_two_records(idf_root):
    _touch(idf_root / LIP_AM_TM / "id00/id00_03", "id00_scene_0012-0.mp4")
    records = _records(idf_root)
    assert "REAL/id00_03__id00_scene_0012-0" in records
    lip = records["LS_AM_TM/id00_03__id00_scene_0012-0"]
    assert lip.pair_key == "id00"
    assert lip.method == "lip_audiomismatch_textmismatch"


def test_every_task_and_method(tmp_path):
    root = tmp_path / "IDForge-v1"
    names = {
        "REAL": "pristine",
        "FS_AM_TM": "face_audiomismatch_textmismatch",
        "FS_RVC_TM": "face_rvc_textmismatch",
        "FS_TTS": "face_tts",
        "FS_TTS_TG": "face_tts_textgen",
        "LS_AM_TM": "lip_audiomismatch_textmismatch",
        "LS_RVC_TM": "lip_rvc_textmismatch",
        "LS_TTS_TG": "lip_tts_textgen",
        "RVC_TM": "rvc_textmismatch",
        "TTS_TG": "tts_textgen",
        "TTS_TM": "tts_textmismatch",
    }
    for abbr, name in names.items():
        branch = "original_content" if abbr == "REAL" else "manipulated_content"
        _touch(root / branch / name / "videos/id01/id01_00", "id01_scene_0001-0.mp4")
    records = collect_records(IDForgeV1Builder(), root)
    assert {task_of(r.key): r.attrs["task_name"] for r in records} == names
    assert {task_of(r.key): r.method for r in records} == {
        abbr: "original" if abbr == "REAL" else name for abbr, name in names.items()
    }


def test_no_compression_can_be_requested(idf_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        collect_records(IDForgeV1Builder(), idf_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ official split


def test_the_official_split_matches_session_and_stem(idf_root):
    _write_lists(
        idf_root,
        train="pristine/id00/id00_03/id00_scene_0012-0.mp4\n\n",
        val="face_audiomismatch_textmismatch/id00/id00_03/id00_scene_0015_0_infoswap.mp4\n",
        test="  pristine/id00/id00_03/id00_scene_0012-1.mp4  \n"
        "pristine/id09/id09_01/id09_scene_0001-0.mp4\n",  # not on disk: ignored
    )
    builder = IDForgeV1Builder()
    official = builder.official_splits(idf_root, collect_records(builder, idf_root))
    assert official == {
        "REAL/id00_03__id00_scene_0012-0": "train",
        "FS_AM_TM/id00_03__id00_scene_0015_0_infoswap": "val",
        "REAL/id00_03__id00_scene_0012-1": "test",
    }


def test_a_listed_key_splits_every_task_that_has_it(idf_root):
    # The lists name each video by task, but the match is on <session>__<stem> alone, so a key
    # shared by two tasks takes the first of train, val and test that lists it, in both tasks.
    _touch(idf_root / LIP_AM_TM / "id00/id00_03", "id00_scene_0012-0.mp4")
    _write_lists(
        idf_root,
        val="pristine/id00/id00_03/id00_scene_0012-0.mp4\n",
        test="lip_audiomismatch_textmismatch/id00/id00_03/id00_scene_0012-0.mp4\n",
    )
    builder = IDForgeV1Builder()
    official = builder.official_splits(idf_root, collect_records(builder, idf_root))
    assert official == {
        "REAL/id00_03__id00_scene_0012-0": "val",
        "LS_AM_TM/id00_03__id00_scene_0012-0": "val",
    }


def test_list_lines_keep_an_inner_suffix_and_may_be_bare_names(idf_root):
    _touch(idf_root / RVC_TM / "id00/id00_03", "id00_scene_0014.mp3.mp4")
    _touch(idf_root / PRISTINE, "loose.mp4")
    _write_lists(
        idf_root,
        train="rvc_textmismatch/id00/id00_03/id00_scene_0014.mp3.mp4\n",
        # A line with no folder is the bare stem, which no nested video's key can equal.
        test="id00_scene_0012-1.mp4\nvideos/loose.mp4\n",
    )
    builder = IDForgeV1Builder()
    official = builder.official_splits(idf_root, collect_records(builder, idf_root))
    assert official == {
        "RVC_TM/id00_03__id00_scene_0014.mp3": "train",
        "REAL/videos__loose": "test",
    }


def test_the_official_scheme_assigns_the_listed_videos(idf_root):
    _write_lists(idf_root, train="p/id00/id00_03/id00_scene_0012-0.mp4\n")
    builder = IDForgeV1Builder()
    records = collect_records(builder, idf_root)
    assert assign_official(records, builder.official_splits(idf_root, records)) == {
        ("REAL/id00_03__id00_scene_0012-0", None): "train"
    }


def test_the_official_split_needs_every_release_list(idf_root):
    _write_lists(idf_root)
    (idf_root / SPLITS / "val.csv").unlink()
    builder = IDForgeV1Builder()
    with pytest.raises(ConfigError, match=r"val\.csv") as caught:
        builder.official_splits(idf_root, collect_records(builder, idf_root))
    assert ".official_files/splits" in caught.value.hint


def test_an_unreadable_list_is_a_contract_error(idf_root):
    _write_lists(idf_root)
    (idf_root / SPLITS / "test.csv").write_bytes(b"\xff\xfe\x00bad\n")
    builder = IDForgeV1Builder()
    with pytest.raises(ContractError, match=r"test\.csv"):
        builder.official_splits(idf_root, collect_records(builder, idf_root))


# ------------------------------------------------------------------------------ pairs, labels


def test_each_fake_pairs_with_the_first_visually_real_video_of_its_identity(idf_root):
    # Real video with cloned speech is real to a visual detector, so it is a candidate real too:
    # its key sorts first, so it is the one pair of each face swap.
    _touch(idf_root / RVC_TM / "id00/id00_03", "id00_scene_0010.mp3.mp4")
    _touch(idf_root / FACE_AM_TM / "id01/id01_00", "id01_scene_0001_0_roop.mp4")
    builder = IDForgeV1Builder()
    records = collect_records(builder, idf_root)
    fakes = {r.key: builder.pair_candidates(r) for r in records if not builder.is_real(r)}
    assert fakes == {
        "FS_AM_TM/id00_03__id00_scene_0015_0_infoswap": "id00",
        "FS_AM_TM/id00_03__id00_scene_0015_1_roop": "id00",
        "FS_AM_TM/id01_00__id01_scene_0001_0_roop": "id01",
    }
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    real = "RVC_TM/id00_03__id00_scene_0010.mp3"
    assert pairs == [
        PairRecord("FS_AM_TM/id00_03__id00_scene_0015_0_infoswap", real, "identity-fanout"),
        PairRecord("FS_AM_TM/id00_03__id00_scene_0015_1_roop", real, "identity-fanout"),
    ]


def test_label_vocab_covers_every_task():
    builder = IDForgeV1Builder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"IDFv1-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {
        "IDFv1-REAL": (0, 0, 1, "real"),
        "IDFv1-FS_AM_TM": (1, 1, 2, "face-swap"),
        "IDFv1-FS_RVC_TM": (1, 1, 3, "face-swap"),
        "IDFv1-FS_TTS": (1, 1, 4, "face-swap"),
        "IDFv1-FS_TTS_TG": (1, 1, 5, "face-swap"),
        "IDFv1-LS_AM_TM": (1, 1, 6, "lip-sync"),
        "IDFv1-LS_RVC_TM": (1, 1, 7, "lip-sync"),
        "IDFv1-LS_TTS_TG": (1, 1, 8, "lip-sync"),
        # Real video with cloned or synthesised speech: real to a visual detector.
        "IDFv1-RVC_TM": (0, 1, 1, "audio-only"),
        "IDFv1-TTS_TG": (0, 1, 1, "audio-only"),
        "IDFv1-TTS_TM": (0, 1, 1, "audio-only"),
    }


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = IDForgeV1Builder()
    assert builder.default_scheme == "official"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(
        k_fake=5, strata=("identity", "task"), exclude_tasks=("RVC_TM", "TTS_TG", "TTS_TM")
    )
    assert builder.pairing_rule == "identity-fanout"
    assert builder.pairing_fanout == 1
    assert builder.metadata_files == tuple(f"{SPLITS}/{s}.csv" for s in ("train", "val", "test"))


def test_the_benchmark_leaves_out_the_audio_only_tasks(idf_root):
    # Five face swaps (one stratum, all kept under the cap of 5) and five visually real videos:
    # the two pristine ones and one of each audio-only task. With as many reals as fakes every
    # real would be kept, so an audio-only video in the draw can only be kept out by the
    # exclusion, which drops those tasks before anything is drawn.
    _touch(
        idf_root / FACE_AM_TM / "id00/id00_03",
        "id00_scene_0016_0_roop.mp4",
        "id00_scene_0016_1_simswap.mp4",
        "id00_scene_0017_0_infoswap.mp4",
    )
    _touch(idf_root / RVC_TM / "id00/id00_03", "id00_scene_0010.mp3.mp4")
    for task in ("tts_textgen", "tts_textmismatch"):
        _touch(
            idf_root / f"manipulated_content/{task}/videos/id00/id00_03", "id00_scene_0011-0.mp4"
        )
    builder = IDForgeV1Builder()
    records = collect_records(builder, idf_root)
    assert {task_of(r.key) for r in records} >= {"RVC_TM", "TTS_TG", "TTS_TM"}
    chosen = {
        key
        for key, _ in assign_benchmark(
            records,
            spec=builder.benchmark,
            is_real=builder.is_real,
            task_rank=builder.task_rank(),
            pool_keys=None,
        )
    }
    assert chosen == {
        "FS_AM_TM/id00_03__id00_scene_0015_0_infoswap",
        "FS_AM_TM/id00_03__id00_scene_0015_1_roop",
        "FS_AM_TM/id00_03__id00_scene_0016_0_roop",
        "FS_AM_TM/id00_03__id00_scene_0016_1_simswap",
        "FS_AM_TM/id00_03__id00_scene_0017_0_infoswap",
        "REAL/id00_03__id00_scene_0012-0",
        "REAL/id00_03__id00_scene_0012-1",
    }


def test_dataset_card():
    builder = IDForgeV1Builder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "idforge-v1"
    assert card.name == "IDForge"
    assert {"IDForge-v1", "IDFv1"} <= set(card.aliases)
    assert card.compressions is None
    assert card.modalities == ["video", "audio"]
    assert card.default_scheme == "official"
    assert card.homepage == "https://github.com/xyyandxyy/IDForge"
    assert card.paper is not None
    assert (card.paper.title, card.paper.venue, card.paper.year, card.paper.doi) == (
        "Identity-Driven Multimedia Forgery Detection via Reference Assistance",
        "ACM MM",
        2024,
        "10.1145/3664647.3680622",
    )
    assert card.license.spdx is None


def test_the_layout_names_the_nesting_and_the_split_lists():
    text = IDForgeV1Builder().describe_layout()
    assert "'IDForge-v1'" in text
    assert f"{PRISTINE}/**/<video>" in text
    assert "<identity>/<session>" in text
    assert SPLITS in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("idforge-v1"), IDForgeV1Builder)
    assert IDForgeV1Builder.expected_folder == "IDForge-v1"
    assert IDForgeV1Builder.label_prefix == "IDFv1"
    assert local_key("REAL/id00_03__id00_scene_0012-0") == "id00_03__id00_scene_0012-0"
