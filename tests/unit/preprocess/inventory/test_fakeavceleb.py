"""The FakeAVCeleb inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``, and the metadata file
is a few hand-written rows in the release's ``meta_data.csv`` format.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.fakeavceleb import FakeAVCelebBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_72_14_14,
    assign_benchmark,
    local_key,
    resolve_pairs,
    task_of,
)

RVRA_DIR = "original_content/RealVideo-RealAudio/videos"
FVFA_DIR = "manipulated_content/FakeVideo-FakeAudio/videos"
FVRA_DIR = "manipulated_content/FakeVideo-RealAudio/videos"
RVFA_DIR = "manipulated_content/RealVideo-FakeAudio/videos"
META = ".official_files/meta_data.csv"
HEADER = "source,target1,target2,method,category,type,race,gender,path,\n"


def _make_synthetic_favc_root(tmp_path: Path) -> Path:
    """A tiny FakeAVCeleb tree without its metadata file: 2 RVRA, 2 FVFA and 1 RVFA video.

    Layout::

        original_content/RealVideo-RealAudio/videos/African/men/id10001/00901.mp4
        original_content/RealVideo-RealAudio/videos/African/men/id10001/00902.mp4
        manipulated_content/FakeVideo-FakeAudio/videos/African/men/id10001/00901_faceswap.mp4
        manipulated_content/FakeVideo-FakeAudio/videos/Caucasian/women/id10002/00903_wav2lip.mp4
        manipulated_content/RealVideo-FakeAudio/videos/African/men/id10001/00901/fake.mp4

    The extra ``fake.mp4`` leaf makes the RVFA video's key end in ``__fake`` (``__`` joins the
    path's parts), which is what the fallback without metadata keys on.
    """
    root = tmp_path / "FakeAVCeleb-v1_2"
    _touch(root / RVRA_DIR / "African/men/id10001", "00901.mp4", "00902.mp4")
    _touch(root / FVFA_DIR / "African/men/id10001", "00901_faceswap.mp4")
    _touch(root / FVFA_DIR / "Caucasian/women/id10002", "00903_wav2lip.mp4")
    _touch(root / RVFA_DIR / "African/men/id10001/00901", "fake.mp4")
    return root


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()


def _write_meta(root: Path, *rows: str) -> None:
    path = root / META
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "".join(f"{row}\n" for row in rows), encoding="utf-8")


def _release_tree(root: Path) -> None:
    """Videos named as in the release, with metadata rows for them (and one for a stray file)."""
    _touch(root / RVRA_DIR / "Asian (East)/women/id10006", "00905.mp4")
    _touch(root / FVFA_DIR / "Asian (East)/women/id10006", "00905_3_id10004_wavtolip.mp4")
    _touch(root / FVRA_DIR / "Asian (East)/women/id10006", "00905_id10005_faceswap.mp4")
    _touch(root / RVFA_DIR / "Asian (East)/women/id10006", "00905_fake.mp4")
    base = "Asian (East),women"
    where = "FakeAVCeleb/{}/Asian (East)/women/id10006"
    _write_meta(
        root,
        f"id10006,-,-,real,A,RealVideo-RealAudio,{base},00905.mp4,"
        + where.format("RealVideo-RealAudio"),
        f"id10006,id10004,-,fsgan-wav2lip,D,FakeVideo-FakeAudio,{base},"
        f"00905_3_id10004_wavtolip.mp4," + where.format("FakeVideo-FakeAudio"),
        # target1 that is not an id gives no source.
        f"id10006,3,-,faceswap,C,FakeVideo-RealAudio,{base},00905_id10005_faceswap.mp4,"
        + where.format("FakeVideo-RealAudio"),
        f"id10006,fake.mp4,-,rtvc,B,RealVideo-FakeAudio,{base},00905_fake.mp4,"
        + where.format("RealVideo-FakeAudio"),
    )


@pytest.fixture
def favc_root(tmp_path: Path) -> Path:
    return _make_synthetic_favc_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(FakeAVCelebBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_expected_count(favc_root):
    # 2 RVRA + 2 FVFA + 1 RVFA.
    assert len(_discover(favc_root)) == 5


def test_entry_schema(favc_root):
    for rec in _discover(favc_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"RVRA", "FVFA", "RVFA"}
        assert rec.builder == BuilderRef("fakeavceleb", FakeAVCelebBuilder.version)
        # FakeAVCeleb has a single version: no compression level.
        assert rec.compression is None
        assert set(rec.attrs) == {"task_name", "raw_video_stem", "category"}
        assert not rec.relpath.startswith("/")
        assert rec.folder is None


def test_no_media_or_label_class(favc_root):
    for rec in _discover(favc_root):
        assert rec.label_key in {"FAVC-RVRA", "FAVC-FVFA", "FAVC-RVFA"}
        assert rec.probe is None
        assert not hasattr(rec, "media")
        assert not hasattr(rec, "label")


def test_video_id_format_and_identity(favc_root):
    records = {rec.key: rec for rec in _discover(favc_root)}
    keys = {local_key(key) for key in records}
    # The two reals, keyed by their four path parts.
    assert "African__men__id10001__00901" in keys
    assert "African__men__id10001__00902" in keys
    # A fake keeps its clip name, "_" included.
    assert "African__men__id10001__00901_faceswap" in keys
    # The identity is the first three parts; without metadata it is also the target.
    real = records["RVRA/African__men__id10001__00901"]
    assert real.identity == "African__men__id10001"
    assert real.target_id == "African__men__id10001"
    assert real.relpath == f"{RVRA_DIR}/African/men/id10001/00901.mp4"


def test_pairing_attrs_for_rvfa_without_metadata(favc_root):
    rvfa = [r for r in _discover(favc_root) if task_of(r.key) == "RVFA"]
    assert len(rvfa) == 1
    rec = rvfa[0]
    # .../id10001/00901/fake.mp4 gives African__men__id10001__00901__fake, whose pair_key
    # drops the "__fake".
    assert local_key(rec.key) == "African__men__id10001__00901__fake"
    assert rec.pair_key == "African__men__id10001__00901"
    assert rec.attrs["raw_video_stem"] == "African__men__id10001__00901"
    assert rec.attrs["category"] is None
    assert rec.method == "RealVideo-FakeAudio"


def test_real_entries_have_null_pair_key(favc_root):
    reals = [r for r in _discover(favc_root) if task_of(r.key) == "RVRA"]
    assert reals
    for rec in reals:
        assert rec.pair_key is None
        assert rec.source_id is None
        assert rec.label_key == "FAVC-RVRA"
        assert rec.method == "original"


def test_each_path_part_is_sanitised(favc_root):
    _touch(favc_root / FVRA_DIR / "Asian (East)/women/id10006", "00905 (copy).v2.mp4")
    records = {local_key(rec.key): rec for rec in _discover(favc_root)}
    rec = records["Asian_East__women__id10006__00905_copy_v2"]
    assert rec.identity == "Asian_East__women__id10006"


def test_the_videos_nest_and_no_compression_can_be_requested(favc_root):
    _touch(favc_root / FVFA_DIR, "loose.mp4")
    records = {local_key(rec.key): rec for rec in _discover(favc_root)}
    # A key of fewer than three parts is its own identity.
    assert records["loose"].identity == "loose"
    with pytest.raises(ConfigError, match="c23") as caught:
        _discover(favc_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ metadata


def test_the_metadata_gives_target_source_method_and_pair(favc_root):
    _release_tree(favc_root)
    records = {rec.key: rec for rec in _discover(favc_root)}
    real_key = "Asian_East__women__id10006__00905"
    real = records[f"RVRA/{real_key}"]
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "Asian_East__women__id10006",
        "id10006",
        None,
        None,
    )
    assert real.method == "real"
    assert real.attrs == {
        "task_name": "RealVideo-RealAudio",
        "raw_video_stem": None,
        "category": "A",
    }

    fvfa = records[f"FVFA/{real_key}_3_id10004_wavtolip"]
    assert (fvfa.identity, fvfa.target_id, fvfa.source_id, fvfa.pair_key) == (
        "Asian_East__women__id10006",
        "id10006",
        "id10004",
        real_key,
    )
    assert fvfa.method == "fsgan-wav2lip"
    assert fvfa.attrs == {
        "task_name": "FakeVideo-FakeAudio",
        "raw_video_stem": real_key,
        "category": "D",
    }

    fvra = records[f"FVRA/{real_key}_id10005_faceswap"]
    assert (fvra.source_id, fvra.pair_key, fvra.method) == (None, real_key, "faceswap")

    rvfa = records[f"RVFA/{real_key}_fake"]
    assert (rvfa.target_id, rvfa.source_id, rvfa.pair_key, rvfa.method) == (
        "id10006",
        None,
        real_key,
        "rtvc",
    )
    # Videos without a metadata row keep the fallback.
    other = records["FVFA/Caucasian__women__id10002__00903_wav2lip"]
    assert (other.target_id, other.source_id, other.pair_key, other.method) == (
        "Caucasian__women__id10002",
        None,
        None,
        "FakeVideo-FakeAudio",
    )


def test_a_later_metadata_row_with_the_same_key_wins(favc_root):
    # The metadata is keyed by race, gender, source and file name only, so a face swap listed
    # after a real under the same file name gives that real its fields.
    _touch(favc_root / RVRA_DIR / "African/women/id10003", "00904.mp4")
    base = "African,women"
    _write_meta(
        favc_root,
        f"id10003,-,-,real,A,RealVideo-RealAudio,{base},00904.mp4,x",
        f"id10003,-,-,faceswap,C,FakeVideo-RealAudio,{base},00904.mp4,x",
    )
    records = {rec.key: rec for rec in _discover(favc_root)}
    real = records["RVRA/African__women__id10003__00904"]
    assert real.method == "faceswap"
    assert real.pair_key == "African__women__id10003__00904"
    assert real.attrs["category"] == "C"


def test_metadata_rows_missing_a_field_are_ignored(favc_root):
    base = "African,men"
    _write_meta(
        favc_root,
        f"-,-,-,real,A,RealVideo-RealAudio,{base},00901.mp4,x",
        f"id10001,-,-,real,A,-,{base},00902.mp4,x",
        f"id10001,id1,-,faceswap,C,FakeVideo-FakeAudio,{base}, ,x",
    )
    for rec in _discover(favc_root):
        assert rec.attrs["category"] is None, rec.key


def test_a_missing_metadata_file_falls_back_with_a_warning(favc_root, caplog):
    with caplog.at_level(logging.WARNING):
        records = _discover(favc_root)
    assert len(records) == 5
    assert "meta_data.csv" in caplog.text


def test_an_unreadable_metadata_file_is_a_contract_error(favc_root):
    path = favc_root / META
    path.parent.mkdir(parents=True)
    path.write_bytes(HEADER.encode() + b"\xff\xfe\x00bad\n")
    with pytest.raises(ContractError, match=r"meta_data\.csv"):
        _discover(favc_root)


# ------------------------------------------------------------------------------ splits, pairs


def test_there_is_no_official_split(favc_root):
    builder = FakeAVCelebBuilder()
    with pytest.raises(ContractError, match="no official split"):
        builder.official_splits(favc_root, collect_records(builder, favc_root))


def test_the_carve_keeps_a_person_on_one_side(favc_root):
    records = _discover(favc_root)
    assignment = assign_72_14_14(records)
    person = [r for r in records if r.identity == "African__men__id10001"]
    assert {task_of(r.key) for r in person} == {"RVRA", "FVFA", "RVFA"}
    assert len({assignment[(r.key, r.compression)] for r in person}) == 1


def test_real_video_fake_audio_is_visually_real():
    builder = FakeAVCelebBuilder()
    rvfa = builder.labels["RVFA"]
    assert (rvfa.binary, rvfa.binary_av) == (0, 1)


def test_pair_candidates_prefer_the_metadata_then_strip_fake(favc_root):
    _release_tree(favc_root)
    _touch(favc_root / FVRA_DIR / "African/men/id10001", "00902_fake.mp4", "00902_x.mp4")
    builder = FakeAVCelebBuilder()
    records = collect_records(builder, favc_root)
    fakes = [r for r in records if not builder.is_real(r)]
    real_key = "Asian_East__women__id10006__00905"
    # Real video with fake audio is real under the visual label, so it is not a fake here.
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        f"FVFA/{real_key}_3_id10004_wavtolip": real_key,
        f"FVRA/{real_key}_id10005_faceswap": real_key,
        "FVFA/African__men__id10001__00901_faceswap": None,
        "FVFA/Caucasian__women__id10002__00903_wav2lip": None,
        "FVRA/African__men__id10001__00902_fake": "African__men__id10001__00902",
        "FVRA/African__men__id10001__00902_x": None,
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
            "FVRA/African__men__id10001__00902_fake",
            "RVRA/African__men__id10001__00902",
            "target-recording",
        ),
        PairRecord(f"FVFA/{real_key}_3_id10004_wavtolip", f"RVRA/{real_key}", "target-recording"),
        PairRecord(f"FVRA/{real_key}_id10005_faceswap", f"RVRA/{real_key}", "target-recording"),
    ]


def test_label_vocab_covers_every_task():
    builder = FakeAVCelebBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"FAVC-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {
        "FAVC-RVRA": (0, 0, 1, "real"),
        "FAVC-FVFA": (1, 1, 2, "video-and-audio"),
        "FAVC-FVRA": (1, 1, 3, "video-only"),
        "FAVC-RVFA": (0, 1, 1, "audio-only"),
    }


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = FakeAVCelebBuilder()
    assert builder.default_scheme == "ident-72-14-14"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "ident-72-14-14": "ident-72-14-14",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=2, strata=("source_id", "task"))
    assert builder.pairing_rule == "target-recording"
    assert builder.pairing_fanout is None
    assert builder.metadata_files == (META,)


def test_the_benchmark_counts_real_video_fake_audio_as_real(favc_root):
    _release_tree(favc_root)
    builder = FakeAVCelebBuilder()
    records = collect_records(builder, favc_root)
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
    fakes = {key for key in chosen if task_of(key) in {"FVFA", "FVRA"}}
    # Four fakes, no (source, task) stratum holding more than two, so all are kept; the five
    # reals include the RVFA videos and four of them are drawn.
    assert len(fakes) == 4
    assert len(chosen - fakes) == 4
    assert {task_of(key) for key in chosen - fakes} <= {"RVRA", "RVFA"}


def test_dataset_card():
    builder = FakeAVCelebBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "fakeavceleb"
    assert card.name == "FakeAVCeleb"
    assert {"FAVC", "FakeAVCeleb v1.2"} <= set(card.aliases)
    assert card.compressions is None
    assert card.modalities == ["video", "audio"]
    assert card.default_scheme == "ident-72-14-14"
    assert card.homepage == "https://github.com/DASH-Lab/FakeAVCeleb"
    assert card.paper is not None
    assert card.paper.title == "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset"
    assert card.paper.doi is None
    assert card.license.spdx is None


def test_the_layout_names_the_nesting_and_the_metadata_file():
    text = FakeAVCelebBuilder().describe_layout()
    assert "'FakeAVCeleb-v1_2'" in text
    assert f"{RVRA_DIR}/**/<video>" in text
    assert "<race>/<gender>/<id>/<clip>" in text
    assert META in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("fakeavceleb"), FakeAVCelebBuilder)
    assert FakeAVCelebBuilder.expected_folder == "FakeAVCeleb-v1_2"
    assert FakeAVCelebBuilder.label_prefix == "FAVC"
