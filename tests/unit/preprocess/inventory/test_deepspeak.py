"""The DeepSpeak v1 and v2 inventory builders, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``, and the release's
annotation files are a few hand-written rows in their CSV formats.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.deepspeak import DeepSpeakV1Builder, DeepSpeakV2Builder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    assign_official_plus_80_20,
    local_key,
    md5_mod_100,
    resolve_pairs,
    task_of,
)

ACTORS = "original_content/Actors/videos"
FACEFUSION = "manipulated_content/FaceFusion/videos"
DIFF2LIP = "manipulated_content/Diff2Lip/videos"
SPLIT_DEF = ".official_files/annotations-split-def.csv"
FAKE_CSV = ".official_files/annotations-fake.csv"
FAKE_HEADER = (
    "video-file,kind,engine,identity-source,identity-target,recording-source,"
    "recording-target,audio-config,gen-config,gesture-type,script-type\n"
)


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"\x00")


def _make_synthetic_dsv1_root(tmp_path: Path) -> Path:
    """A tiny DeepSpeak v1 tree: two reals and two FaceFusion fakes.

    Real stems are ``<identity>-<recording>``; fake stems are
    ``<engine>--<source>-<target>--<source recording>-<target recording>--<n>``, so the first fake
    pairs with the real key ``11-4897``.
    """
    root = tmp_path / "DeepSpeak-v1"
    _touch(root / ACTORS, "87-0.mp4", "89-1.mp4")
    _touch(root / FACEFUSION, "facefusion--10-11--4100-4897--0.mp4")
    _touch(root / FACEFUSION, "facefusion--12-13--4500-5000--0.mp4")
    return root


def _make_synthetic_dsv2_root(tmp_path: Path) -> Path:
    """A tiny DeepSpeak v2 tree: two reals, two Diff2Lip fakes and their annotation rows.

    Fake stems are ``<engine>-<source recording>-<target recording>-<audio>`` and carry no
    identity: the release's ``annotations-fake.csv`` names each fake's identities.
    """
    root = tmp_path / "DeepSpeak-v2"
    _touch(root / ACTORS, "87-0.mp4", "88-1.mp4")
    _touch(root / DIFF2LIP, "diff2lip-3-0-playht.mp4", "diff2lip-4-1-playht.mp4")
    _write_fake_csv(
        root,
        "diff2lip-3-0-playht.mp4,lip-sync,diff2lip,12,87,3,0,playht,,no-gesture,scripted",
        "diff2lip-4-1-playht.mp4,lip-sync,diff2lip,13,88,4,1,playht,,no-gesture,scripted",
    )
    return root


def _write_fake_csv(root: Path, *rows: str) -> None:
    path = root / FAKE_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_HEADER + "".join(f"{row}\n" for row in rows), encoding="utf-8")


def _write_split_def(root: Path, text: str) -> None:
    path = root / SPLIT_DEF
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def dsv1_root(tmp_path: Path) -> Path:
    return _make_synthetic_dsv1_root(tmp_path)


@pytest.fixture
def dsv2_root(tmp_path: Path) -> Path:
    return _make_synthetic_dsv2_root(tmp_path)


def _records(builder, root, **kwargs):
    return {rec.key: rec for rec in collect_records(builder, root, **kwargs)}


# ------------------------------------------------------------------------------ v1 discovery


def test_v1_discover_yields_entries(dsv1_root):
    # 2 reals + 2 fakes.
    assert len(collect_records(DeepSpeakV1Builder(), dsv1_root)) == 4


def test_v1_entry_schema(dsv1_root):
    for rec in collect_records(DeepSpeakV1Builder(), dsv1_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"AR", "FS_FF"}
        assert rec.builder == BuilderRef("deepspeak-v1", DeepSpeakV1Builder.version)
        assert rec.compression is None  # DeepSpeak has no compression levels
        assert rec.label_key == f"DSv1-{task}"
        assert set(rec.attrs) == {"task_name"}
        assert not rec.relpath.startswith("/")
        assert "{cX}" not in rec.relpath
        assert rec.relpath.endswith(".mp4")
        assert rec.folder is None
        assert rec.probe is None
        assert not hasattr(rec, "media")
        assert not hasattr(rec, "label")


def test_v1_real_attrs(dsv1_root):
    records = _records(DeepSpeakV1Builder(), dsv1_root)
    real = records["AR/87-0"]
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "87",
        "87",
        None,
        None,
    )
    assert real.method == "original"
    assert real.relpath == f"{ACTORS}/87-0.mp4"
    assert real.attrs == {"task_name": "Actors"}


def test_v1_fake_pairing_attrs(dsv1_root):
    records = _records(DeepSpeakV1Builder(), dsv1_root)
    fake = records["FS_FF/facefusion--10-11--4100-4897--0"]
    # The target is the second identity and its recording the second recording.
    assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
        "11",
        "11",
        "10",
        "11-4897",
    )
    assert fake.method == "facefusion"
    assert fake.relpath == f"{FACEFUSION}/facefusion--10-11--4100-4897--0.mp4"
    assert fake.attrs == {"task_name": "FaceFusion"}


def test_v1_names_that_do_not_parse_are_kept_without_fields(dsv1_root):
    _touch(dsv1_root / ACTORS, "87.mp4")
    _touch(dsv1_root / FACEFUSION, "facefusion--10-11.mp4", "facefusion--10--4100-4897--0.mp4")
    records = _records(DeepSpeakV1Builder(), dsv1_root)
    for key in ("AR/87", "FS_FF/facefusion--10-11", "FS_FF/facefusion--10--4100-4897--0"):
        rec = records[key]
        assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (
            None,
            None,
            None,
            None,
        ), key


def test_v1_extra_parts_are_tolerated(dsv1_root):
    # A real recording with more "-" parts keeps its first part as the identity; a fake reads
    # the first two parts of its second and third "--" groups.
    _touch(dsv1_root / ACTORS, "87-0-extra.mp4")
    _touch(dsv1_root / FACEFUSION, "facefusion--1-2-3--40-50-60.mp4")
    records = _records(DeepSpeakV1Builder(), dsv1_root)
    assert records["AR/87-0-extra"].identity == "87"
    fake = records["FS_FF/facefusion--1-2-3--40-50-60"]
    assert (fake.target_id, fake.source_id, fake.pair_key) == ("2", "1", "2-50")


def test_v1_every_task_and_method(tmp_path):
    root = tmp_path / "DeepSpeak-v1"
    folders = {
        "AR": ("original_content/Actors/videos", "original"),
        "FS_FF": ("manipulated_content/FaceFusion/videos", "facefusion"),
        "FS_FFGAN": ("manipulated_content/FaceFusion_GAN/videos", "facefusion_gan"),
        "FS_FFLIVE": ("manipulated_content/FaceFusion_Live/videos", "facefusion_live"),
        "LS_RT": ("manipulated_content/ReTalking/videos", "retalking"),
        "LS_W2L": ("manipulated_content/Wav2Lip/videos", "wav2lip"),
    }
    for folder, _ in folders.values():
        _touch(root / folder, "e--1-2--3-4--0.mp4")
    records = collect_records(DeepSpeakV1Builder(), root)
    assert {task_of(r.key): (r.relpath.rsplit("/", 1)[0], r.method) for r in records} == folders


def test_no_compression_can_be_requested(dsv1_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        collect_records(DeepSpeakV1Builder(), dsv1_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ v2 discovery


def test_v2_discover_yields_entries(dsv2_root):
    assert len(collect_records(DeepSpeakV2Builder(), dsv2_root)) == 4


def test_v2_entry_schema(dsv2_root):
    for rec in collect_records(DeepSpeakV2Builder(), dsv2_root):
        task = task_of(rec.key)
        assert task in {"AR", "LS_D2L"}
        assert rec.builder == BuilderRef("deepspeak-v2", DeepSpeakV2Builder.version)
        assert rec.compression is None
        assert rec.label_key == f"DSv2-{task}"
        assert set(rec.attrs) == {"task_name", "engine", "audio_config"}
        assert rec.folder is None


def test_v2_real_attrs(dsv2_root):
    records = _records(DeepSpeakV2Builder(), dsv2_root)
    real = records["AR/87-0"]
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "87",
        "87",
        None,
        None,
    )
    assert real.method == "original"
    assert real.attrs == {"task_name": "Actors", "engine": None, "audio_config": None}


def test_v2_fake_attrs_come_from_the_annotations(dsv2_root):
    records = _records(DeepSpeakV2Builder(), dsv2_root)
    fake = records["LS_D2L/diff2lip-3-0-playht"]
    # The identities are the annotation row's; the pair key is the target's recording.
    assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
        "87",
        "87",
        "12",
        "87-0",
    )
    assert fake.method == "diff2lip"
    assert fake.attrs == {"task_name": "Diff2Lip", "engine": "diff2lip", "audio_config": "playht"}


def test_v2_a_fake_without_an_annotation_row_parses_its_name(dsv2_root):
    _touch(
        dsv2_root / DIFF2LIP,
        "diff2lip-5-2-elevenlabs.mp4",
        "diff2lip-6-2.mp4",
        "diff2lip-7.mp4",
    )
    records = _records(DeepSpeakV2Builder(), dsv2_root)
    expected = {
        "LS_D2L/diff2lip-5-2-elevenlabs": ("diff2lip", "elevenlabs"),
        "LS_D2L/diff2lip-6-2": ("diff2lip", None),
        "LS_D2L/diff2lip-7": (None, None),
    }
    for key, (engine, audio) in expected.items():
        rec = records[key]
        # Without a row the identities are unknown, so nothing pairs it or splits it.
        assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (
            None,
            None,
            None,
            None,
        ), key
        assert (rec.attrs["engine"], rec.attrs["audio_config"]) == (engine, audio), key


def test_v2_annotation_values_are_trimmed_and_blank_is_none(dsv2_root):
    _touch(dsv2_root / DIFF2LIP, "diff2lip-9-8-real.mp4", "diff2lip-9-7-real.mp4")
    _write_fake_csv(
        dsv2_root,
        # The ".mp4" is optional, and an empty file name is skipped.
        " diff2lip-9-8-real , lip-sync , diff2lip , 5 , 6 , 9 , 8 , real ,,,",
        ",lip-sync,diff2lip,1,2,3,4,x,,,",
        # A blank target recording leaves the pair key unset.
        "diff2lip-9-7-real.mp4,lip-sync,,5,6,9, ,,,,",
    )
    records = _records(DeepSpeakV2Builder(), dsv2_root)
    first = records["LS_D2L/diff2lip-9-8-real"]
    assert (first.target_id, first.source_id, first.pair_key) == ("6", "5", "6-8")
    assert first.attrs["engine"] == "diff2lip"
    assert first.attrs["audio_config"] == "real"
    second = records["LS_D2L/diff2lip-9-7-real"]
    assert (second.target_id, second.source_id, second.pair_key) == ("6", "5", None)
    assert (second.attrs["engine"], second.attrs["audio_config"]) == (None, None)


def test_v2_a_later_annotation_row_for_the_same_file_wins(dsv2_root):
    _write_fake_csv(
        dsv2_root,
        "diff2lip-3-0-playht.mp4,lip-sync,diff2lip,12,87,3,0,playht,,,",
        "diff2lip-3-0-playht.mp4,lip-sync,diff2lip,14,88,3,1,playht,,,",
    )
    fake = _records(DeepSpeakV2Builder(), dsv2_root)["LS_D2L/diff2lip-3-0-playht"]
    assert (fake.target_id, fake.source_id, fake.pair_key) == ("88", "14", "88-1")


def test_v2_a_missing_annotation_file_warns_and_leaves_fakes_without_identities(dsv2_root, caplog):
    (dsv2_root / FAKE_CSV).unlink()
    with caplog.at_level(logging.WARNING):
        records = _records(DeepSpeakV2Builder(), dsv2_root)
    assert "annotations-fake.csv" in caplog.text
    assert records["LS_D2L/diff2lip-3-0-playht"].identity is None
    assert records["AR/87-0"].identity == "87"


def test_v2_an_unreadable_annotation_file_is_a_contract_error(dsv2_root):
    (dsv2_root / FAKE_CSV).write_bytes(FAKE_HEADER.encode() + b"\xff\xfe\x00bad\n")
    with pytest.raises(ContractError, match=r"annotations-fake\.csv"):
        collect_records(DeepSpeakV2Builder(), dsv2_root)


def test_v2_every_task_and_method(tmp_path):
    root = tmp_path / "DeepSpeak-v2"
    folders = {
        "AR": ("original_content/Actors/videos", "original"),
        "LS_D2L": ("manipulated_content/Diff2Lip/videos", "diff2lip"),
        "FS_FF": ("manipulated_content/FaceFusion/videos", "facefusion"),
        "TF_HM": ("manipulated_content/HelloMeme/videos", "hellomeme"),
        "LS_LS": ("manipulated_content/LatentSync/videos", "latentsync"),
        "TF_LP": ("manipulated_content/LivePortrait/videos", "liveportrait"),
        "TF_MEMO": ("manipulated_content/Memo/videos", "memo"),
    }
    for folder, _ in folders.values():
        _touch(root / folder, "e-1-2-x.mp4")
    records = collect_records(DeepSpeakV2Builder(), root)
    assert {task_of(r.key): (r.relpath.rsplit("/", 1)[0], r.method) for r in records} == folders


# ------------------------------------------------------------------------------ official split


def test_v1_the_official_split_matches_each_identity(dsv1_root):
    # The v1 file has a leading unnamed index column; the split is matched ignoring case/space.
    _write_split_def(dsv1_root, ",identity,split\n0,87,train\n1, 11 , Test \n2,13,val\n")
    builder = DeepSpeakV1Builder()
    official = builder.official_splits(dsv1_root, collect_records(builder, dsv1_root))
    assert official == {
        "AR/87-0": "train",
        "FS_FF/facefusion--10-11--4100-4897--0": "test",
        # 89 is not listed, and 13 is listed as val, which DeepSpeak does not publish.
    }


def test_v2_the_official_split_matches_the_annotated_identity(dsv2_root):
    _write_split_def(dsv2_root, "identity,split\n87,test\n88,train\n")
    builder = DeepSpeakV2Builder()
    official = builder.official_splits(dsv2_root, collect_records(builder, dsv2_root))
    assert official == {
        "AR/87-0": "test",
        "LS_D2L/diff2lip-3-0-playht": "test",
        "AR/88-1": "train",
        "LS_D2L/diff2lip-4-1-playht": "train",
    }


def test_an_identity_listed_twice_takes_train_before_val_before_test(dsv1_root):
    _write_split_def(
        dsv1_root,
        "identity,split\n87,test\n87,train\n11,test\n11,val\n89,unknown\n89,test\n",
    )
    builder = DeepSpeakV1Builder()
    official = builder.official_splits(dsv1_root, collect_records(builder, dsv1_root))
    # 87: train wins; 11: val wins over test, and val is not published, so it is left out.
    assert official == {"AR/87-0": "train", "AR/89-1": "test"}


def test_a_video_without_an_identity_never_matches_a_blank_row(dsv1_root):
    _touch(dsv1_root / ACTORS, "87.mp4")  # no "-": no identity
    _write_split_def(dsv1_root, "identity,split\n,train\n87,test\n")
    builder = DeepSpeakV1Builder()
    official = builder.official_splits(dsv1_root, collect_records(builder, dsv1_root))
    assert official == {"AR/87-0": "test"}


def test_the_default_scheme_carves_val_from_the_official_train(dsv1_root):
    _write_split_def(dsv1_root, "identity,split\n87,train\n11,train\n89,test\n")
    builder = DeepSpeakV1Builder()
    records = collect_records(builder, dsv1_root)
    official = builder.official_splits(dsv1_root, records)
    scheme = builder.schemes[builder.default_scheme]
    assert scheme.rule == "official+ident-80-20"
    assert scheme.params == {"policy": "official-train-test"}
    assignment = assign_official_plus_80_20(records, official, policy="official-train-test")

    def carve(identity: str) -> str:
        return "val" if md5_mod_100(identity) < 20 else "train"

    assert assignment == {
        ("AR/87-0", None): carve("87"),
        ("FS_FF/facefusion--10-11--4100-4897--0", None): carve("11"),
        ("AR/89-1", None): "test",
        # identity 13 is not listed, so its fake is left out
    }
    assert assign_official(records, official) == {
        ("AR/87-0", None): "train",
        ("FS_FF/facefusion--10-11--4100-4897--0", None): "train",
        ("AR/89-1", None): "test",
    }


def test_the_official_split_needs_its_file(dsv1_root):
    builder = DeepSpeakV1Builder()
    with pytest.raises(ConfigError, match=r"annotations-split-def\.csv") as caught:
        builder.official_splits(dsv1_root, collect_records(builder, dsv1_root))
    assert ".official_files" in caught.value.hint


@pytest.mark.parametrize("text", ["name,split\n87,train\n", "identity,set\n87,train\n", ""])
def test_a_split_file_without_its_columns_is_a_contract_error(dsv1_root, text):
    builder = DeepSpeakV1Builder()
    records = collect_records(builder, dsv1_root)
    _write_split_def(dsv1_root, text)
    with pytest.raises(ContractError, match=r"annotations-split-def\.csv") as caught:
        builder.official_splits(dsv1_root, records)
    assert "identity" in caught.value.hint


def test_a_split_file_that_lists_nobody_assigns_nothing(dsv1_root):
    builder = DeepSpeakV1Builder()
    records = collect_records(builder, dsv1_root)
    _write_split_def(dsv1_root, "identity,split\n")
    assert builder.official_splits(dsv1_root, records) == {}


def test_an_unreadable_split_file_is_a_contract_error(dsv1_root):
    builder = DeepSpeakV1Builder()
    records = collect_records(builder, dsv1_root)
    _write_split_def(dsv1_root, "")
    (dsv1_root / SPLIT_DEF).write_bytes(b"identity,split\n\xff\xfe,train\n")
    with pytest.raises(ContractError, match=r"annotations-split-def\.csv"):
        builder.official_splits(dsv1_root, records)


# ------------------------------------------------------------------------------ pairs, labels


def test_v1_pairs_name_the_target_recording(dsv1_root):
    _touch(dsv1_root / ACTORS, "11-4897.mp4", "13-1.mp4")
    builder = DeepSpeakV1Builder()
    records = collect_records(builder, dsv1_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "FS_FF/facefusion--10-11--4100-4897--0": "11-4897",
        "FS_FF/facefusion--12-13--4500-5000--0": "13-5000",
    }
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    # 13-5000 is not a real recording, and it is not an identity either: no pair.
    assert pairs == [
        PairRecord("FS_FF/facefusion--10-11--4100-4897--0", "AR/11-4897", "target-recording")
    ]


def test_v2_a_fake_without_a_target_recording_pairs_with_its_targets_reals(dsv2_root):
    _touch(dsv2_root / ACTORS, "87-5.mp4")
    _write_fake_csv(
        dsv2_root,
        "diff2lip-3-0-playht.mp4,lip-sync,diff2lip,12,87,3,0,playht,,,",
        "diff2lip-4-1-playht.mp4,lip-sync,diff2lip,13,87,4,,playht,,,",
    )
    builder = DeepSpeakV2Builder()
    records = collect_records(builder, dsv2_root)
    fakes = {r.key: builder.pair_candidates(r) for r in records if not builder.is_real(r)}
    assert fakes == {"LS_D2L/diff2lip-3-0-playht": "87-0", "LS_D2L/diff2lip-4-1-playht": "87"}
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    # The second fake has no target recording: every real of identity 87, no cap.
    assert pairs == [
        PairRecord("LS_D2L/diff2lip-3-0-playht", "AR/87-0", "target-recording"),
        PairRecord("LS_D2L/diff2lip-4-1-playht", "AR/87-0", "target-recording"),
        PairRecord("LS_D2L/diff2lip-4-1-playht", "AR/87-5", "target-recording"),
    ]


def _table(vocab):
    return {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }


def test_label_vocab_covers_every_task():
    v1 = DeepSpeakV1Builder().label_vocab().vocab
    assert _table(v1) == {
        "DSv1-AR": (0, 0, 1, "real"),
        "DSv1-FS_FF": (1, 1, 2, "face-swap"),
        "DSv1-FS_FFGAN": (1, 1, 3, "face-swap"),
        "DSv1-FS_FFLIVE": (1, 1, 4, "face-swap"),
        "DSv1-LS_RT": (1, 1, 5, "lip-sync"),
        "DSv1-LS_W2L": (1, 1, 6, "lip-sync"),
    }
    v2 = DeepSpeakV2Builder().label_vocab().vocab
    assert _table(v2) == {
        "DSv2-AR": (0, 0, 1, "real"),
        "DSv2-LS_D2L": (1, 1, 2, "lip-sync"),
        "DSv2-FS_FF": (1, 1, 3, "face-swap"),
        "DSv2-TF_HM": (1, 1, 4, "talking-face"),
        "DSv2-LS_LS": (1, 1, 5, "lip-sync"),
        "DSv2-TF_LP": (1, 1, 6, "talking-face"),
        "DSv2-TF_MEMO": (1, 1, 7, "talking-face"),
    }
    assert v2["DSv2-TF_MEMO"]["method"] == "memo"


# ------------------------------------------------------------------------------ schemes, card


@pytest.mark.parametrize(
    ("builder", "benchmark"),
    [
        (DeepSpeakV1Builder(), BenchmarkSpec(k_fake=2, strata=("identity", "task"))),
        (DeepSpeakV2Builder(), BenchmarkSpec(k_fake=2, strata=("target_id", "task"))),
    ],
)
def test_schemes_and_benchmark(builder, benchmark):
    assert builder.default_scheme == "official+ident-80-20"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official+ident-80-20": "official+ident-80-20",
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == benchmark
    assert builder.pairing_rule == "target-recording"
    assert builder.pairing_fanout is None


def test_metadata_files():
    assert DeepSpeakV1Builder.metadata_files == (SPLIT_DEF,)
    assert DeepSpeakV2Builder.metadata_files == (SPLIT_DEF, FAKE_CSV)


def test_the_v2_benchmark_stratifies_on_the_annotated_target(dsv2_root):
    _touch(dsv2_root / DIFF2LIP, "diff2lip-5-0-elevenlabs.mp4", "diff2lip-6-0-real.mp4")
    _write_fake_csv(
        dsv2_root,
        "diff2lip-3-0-playht.mp4,lip-sync,diff2lip,12,87,3,0,playht,,,",
        "diff2lip-4-1-playht.mp4,lip-sync,diff2lip,13,87,4,1,playht,,,",
        "diff2lip-5-0-elevenlabs.mp4,lip-sync,diff2lip,14,87,5,0,elevenlabs,,,",
        "diff2lip-6-0-real.mp4,lip-sync,diff2lip,15,88,6,0,real,,,",
    )
    builder = DeepSpeakV2Builder()
    records = collect_records(builder, dsv2_root)
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
    fakes = {key for key in chosen if not key.startswith("AR/")}
    # Three fakes of target 87 give two; the one of target 88 is kept; then two reals of two.
    assert len(fakes) == 3
    assert "LS_D2L/diff2lip-6-0-real" in fakes
    assert chosen - fakes == {"AR/87-0", "AR/88-1"}


def test_dataset_cards():
    for builder, name, alias in (
        (DeepSpeakV1Builder(), "DeepSpeak v1", "DSv1"),
        (DeepSpeakV2Builder(), "DeepSpeak v2", "DSv2"),
    ):
        cards = {
            scheme: SchemeCard(kind=spec.kind, sha256="a" * 64)
            for scheme, spec in builder.schemes.items()
        }
        card = builder.dataset_card(cards)
        assert card.id == builder.dataset_id
        assert card.name == name
        assert alias in card.aliases
        assert card.compressions is None
        assert card.modalities == ["video", "audio"]
        assert card.default_scheme == "official+ident-80-20"
        assert card.homepage is not None
        assert card.homepage.startswith("https://huggingface.co/datasets/faridlab/")
        assert card.paper is None
        assert card.license.spdx is None


def test_the_layout_names_the_engines_and_the_release_files():
    v1 = DeepSpeakV1Builder().describe_layout()
    assert "'DeepSpeak-v1'" in v1
    assert f"{ACTORS}/<video>" in v1
    assert SPLIT_DEF in v1
    assert "facefusion_gan" in v1
    assert "{cX}" not in v1
    v2 = DeepSpeakV2Builder().describe_layout()
    assert "'DeepSpeak-v2'" in v2
    assert FAKE_CSV in v2
    assert "hellomeme" in v2


def test_they_are_registered():
    assert isinstance(get_builder("deepspeak-v1"), DeepSpeakV1Builder)
    assert isinstance(get_builder("deepspeak-v2"), DeepSpeakV2Builder)
    assert DeepSpeakV1Builder.expected_folder == "DeepSpeak-v1"
    assert DeepSpeakV2Builder.expected_folder == "DeepSpeak-v2"
    assert (DeepSpeakV1Builder.label_prefix, DeepSpeakV2Builder.label_prefix) == ("DSv1", "DSv2")
    assert local_key("AR/87-0") == "87-0"
