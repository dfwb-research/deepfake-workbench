"""The Celeb-DF v1, v2 and v3 inventory builders, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.celebdf import (
    CelebDFv1Builder,
    CelebDFv2Builder,
    CelebDFv3Builder,
)
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official_plus_80_20,
    local_key,
    md5_mod_100,
    resolve_pairs,
    task_of,
)

BUILDERS = (CelebDFv1Builder, CelebDFv2Builder, CelebDFv3Builder)

CELEB_REAL = ("original_content", "Celeb-real", "videos")
YOUTUBE_REAL = ("original_content", "YouTube-real", "videos")
SYNTHESIS = ("manipulated_content", "Celeb-synthesis", "videos")


def _make_synthetic_celebdf_root(tmp_path: Path, folder: str = "Celeb-DF-v1") -> Path:
    """A tiny Celeb-DF tree: 2 Celeb-real, 1 YouTube-real and 3 fakes in Celeb-synthesis.

    Stem coverage:
      * Celeb-real   : ``id1_0007``, ``id3_0002``   (``id<n>_<rec>``)
      * YouTube-real : ``00170``                     (all digits)
      * face swap    : ``id1_id0_0007``, ``id3_id4_0002``  (``<target>_<source>_<rec>``)
      * talking face : ``id29_0002_test_id08701_aaa``      (``<target>_<rec>_test_<ref>``)
    """
    root = tmp_path / folder
    celeb_dir = root.joinpath(*CELEB_REAL)
    youtube_dir = root.joinpath(*YOUTUBE_REAL)
    fake_dir = root.joinpath(*SYNTHESIS)
    for folder_path in (celeb_dir, youtube_dir, fake_dir):
        folder_path.mkdir(parents=True)
    (celeb_dir / "id1_0007.mp4").write_bytes(b"\x00")
    (celeb_dir / "id3_0002.mp4").write_bytes(b"\x00")
    (youtube_dir / "00170.mp4").write_bytes(b"\x00")
    # Face swaps (pair_key id1_0007, id3_0002) and a talking face (pair_key id29_0002).
    (fake_dir / "id1_id0_0007.mp4").write_bytes(b"\x00")
    (fake_dir / "id3_id4_0002.mp4").write_bytes(b"\x00")
    (fake_dir / "id29_0002_test_id08701_aaa.mp4").write_bytes(b"\x00")
    return root


def _touch(folder: Path, *stems: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (folder / f"{stem}.mp4").touch()


def _write_testing_list(root: Path, name: str, lines: list[str]) -> None:
    """A Celeb-DF test list: ``<label> <relpath>`` per line (1 real, 0 fake)."""
    folder = root / ".official_files"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def cdf_root(tmp_path: Path) -> Path:
    return _make_synthetic_celebdf_root(tmp_path)


def _discover(root, builder=None, **kwargs):
    return collect_records(builder or CelebDFv1Builder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_all_stubs(cdf_root):
    # 2 Celeb-real + 1 YouTube-real + 3 fakes = 6
    assert len(_discover(cdf_root)) == 6


def test_entry_schema(cdf_root):
    for rec in _discover(cdf_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"CR", "YTR", "FS_CS"}
        assert rec.builder == BuilderRef("celebdf-v1", CelebDFv1Builder.version)
        assert rec.compression is None  # the Celeb-DF releases have no compression levels
        assert rec.attrs["task_name"] in {"Celeb-real", "YouTube-real", "Celeb-synthesis"}
        assert not rec.relpath.startswith("/")
        assert "{cX}" not in rec.relpath
        assert rec.folder is None


def test_no_media_no_label_class(cdf_root):
    for rec in _discover(cdf_root):
        assert rec.label_key == f"CDFv1-{task_of(rec.key)}"
        assert rec.probe is None
        assert not hasattr(rec, "label")
        assert not hasattr(rec, "media")


def test_fake_pairing_attrs(cdf_root):
    records = {rec.key: rec for rec in _discover(cdf_root)}
    fake = records["FS_CS/id1_id0_0007"]
    assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
        "id1",
        "id1",
        "id0",
        "id1_0007",
    )
    assert fake.method == "Celeb-synthesis"
    assert fake.relpath == "manipulated_content/Celeb-synthesis/videos/id1_id0_0007.mp4"
    # The pair_key is the local key of the real recording the fake was made from.
    real = records[f"CR/{fake.pair_key}"]
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "id1",
        "id1",
        None,
        None,
    )
    assert real.method == "original"
    assert real.relpath == "original_content/Celeb-real/videos/id1_0007.mp4"


def test_talking_face_and_youtube_real(cdf_root):
    records = {rec.key: rec for rec in _discover(cdf_root)}
    talking = records["FS_CS/id29_0002_test_id08701_aaa"]
    assert (talking.identity, talking.target_id, talking.source_id, talking.pair_key) == (
        "id29",
        "id29",
        None,
        "id29_0002",
    )
    youtube = records["YTR/00170"]
    # A YouTube-real video's identity is its own number.
    assert (youtube.identity, youtube.target_id, youtube.source_id, youtube.pair_key) == (
        "00170",
        "00170",
        None,
        None,
    )
    assert youtube.attrs == {"task_name": "YouTube-real"}
    assert youtube.relpath == "original_content/YouTube-real/videos/00170.mp4"


def test_names_that_do_not_parse_are_kept_without_fields(cdf_root):
    _touch(cdf_root.joinpath(*SYNTHESIS), "junk", "id5", "xid1_id2_0001")
    _touch(cdf_root.joinpath(*CELEB_REAL), "id7_0001_extra", "celeb_0001")
    _touch(cdf_root.joinpath(*YOUTUBE_REAL), "00170a")
    records = {rec.key: rec for rec in _discover(cdf_root)}
    for key in (
        "FS_CS/junk",
        "FS_CS/id5",
        "FS_CS/xid1_id2_0001",
        "CR/id7_0001_extra",
        "CR/celeb_0001",
        "YTR/00170a",
    ):
        rec = records[key]
        assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (None,) * 4, key


def test_the_name_decides_a_real_identity_whatever_its_task(cdf_root):
    # A real is parsed by its name alone: a celebrity-style name under YouTube-real still gives
    # the celebrity id, and a number under Celeb-real is its own identity.
    _touch(cdf_root.joinpath(*YOUTUBE_REAL), "id9_0003")
    _touch(cdf_root.joinpath(*CELEB_REAL), "00042")
    records = {rec.key: rec for rec in _discover(cdf_root)}
    assert records["YTR/id9_0003"].identity == "id9"
    assert records["CR/00042"].identity == "00042"


def test_a_two_part_fake_name_pairs_with_both_parts(cdf_root):
    # With fewer than three parts the second is taken as the recording, even if it starts
    # with "id": the name has no room for a source.
    _touch(cdf_root.joinpath(*SYNTHESIS), "id1_id0", "id8_0004")
    records = {rec.key: rec for rec in _discover(cdf_root)}
    assert (records["FS_CS/id1_id0"].source_id, records["FS_CS/id1_id0"].pair_key) == (
        None,
        "id1_id0",
    )
    assert (records["FS_CS/id8_0004"].target_id, records["FS_CS/id8_0004"].pair_key) == (
        "id8",
        "id8_0004",
    )


def test_label_keys_follow_each_release(tmp_path):
    for builder_class, prefix in zip(BUILDERS[:2], ("CDFv1", "CDFv2"), strict=True):
        root = _make_synthetic_celebdf_root(tmp_path / prefix, builder_class.expected_folder)
        records = collect_records(builder_class(), root)
        assert len(records) == 6
        for rec in records:
            assert rec.label_key == f"{prefix}-{task_of(rec.key)}"
            assert rec.builder.id == builder_class.dataset_id


def test_no_compression_can_be_requested(cdf_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        _discover(cdf_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ Celeb-DF v3

V3_FAKES = {
    # abbr: (family folder, method folder)
    "FS_CDFV2": ("FaceSwap", "Celeb-DF-v2"),
    "FS_BLEND": ("FaceSwap", "BlendFace"),
    "FS_GHOST": ("FaceSwap", "GHOST"),
    "FS_HIFI": ("FaceSwap", "HifiFace"),
    "FS_INSW": ("FaceSwap", "InSwapper"),
    "FS_MOBILE": ("FaceSwap", "MobileFaceSwap"),
    "FS_SIM": ("FaceSwap", "SimSwap"),
    "FS_UNI": ("FaceSwap", "UniFace"),
    "FR_DAGAN": ("FaceReenact", "DaGAN"),
    "FR_FSRT": ("FaceReenact", "FSRT"),
    "FR_HYPER": ("FaceReenact", "HyperReenact"),
    "FR_LIA": ("FaceReenact", "LIA"),
    "FR_LP": ("FaceReenact", "LivePortrait"),
    "FR_MCNET": ("FaceReenact", "MCNET"),
    "FR_TPSMM": ("FaceReenact", "TPSMM"),
    "TF_ANI": ("TalkingFace", "AniTalker"),
    "TF_ECHO": ("TalkingFace", "EchoMimic"),
    "TF_EDTALK": ("TalkingFace", "EDTalk"),
    "TF_FLOAT": ("TalkingFace", "FLOAT"),
    "TF_IPLAP": ("TalkingFace", "IP_LAP"),
    "TF_REAL3D": ("TalkingFace", "Real3DPortrait"),
    "TF_SAD": ("TalkingFace", "SadTalker"),
}


def _make_v3_root(tmp_path: Path) -> Path:
    """A Celeb-DF v3 tree with every task: one fake per method, all made from ``id1_0007``."""
    root = tmp_path / "Celeb-DF-v3"
    _touch(root.joinpath(*CELEB_REAL), "id1_0007")
    _touch(root.joinpath(*YOUTUBE_REAL), "00170")
    for abbr, (family, method) in V3_FAKES.items():
        stem = "id1_0007_test_id00001_ref" if abbr.startswith("TF_") else "id1_id2_0007"
        _touch(root / "manipulated_content" / family / method / "videos", stem)
    return root


def test_v3_discovers_every_method(tmp_path):
    root = _make_v3_root(tmp_path)
    records = {rec.key: rec for rec in collect_records(CelebDFv3Builder(), root)}
    assert len(records) == 2 + len(V3_FAKES)
    assert [task.abbr for task in CelebDFv3Builder.tasks] == ["CR", "YTR", *V3_FAKES]
    for abbr, (family, method) in V3_FAKES.items():
        stem = "id1_0007_test_id00001_ref" if abbr.startswith("TF_") else "id1_id2_0007"
        rec = records[f"{abbr}/{stem}"]
        assert rec.label_key == f"CDFv3-{abbr}"
        assert rec.method == method
        assert rec.attrs == {"task_name": method}
        assert rec.relpath == f"manipulated_content/{family}/{method}/videos/{stem}.mp4"
        assert (rec.identity, rec.target_id, rec.pair_key) == ("id1", "id1", "id1_0007")
        assert rec.source_id == (None if abbr.startswith("TF_") else "id2")


def test_v3_pairs_every_method_with_the_real_recording(tmp_path):
    root = _make_v3_root(tmp_path)
    builder = CelebDFv3Builder()
    records = collect_records(builder, root)
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    assert {p.real_key for p in pairs} == {"CR/id1_0007"}
    assert len(pairs) == len(V3_FAKES)


# ------------------------------------------------------------------------------ official split


def test_official_split_is_the_testing_list(cdf_root):
    _write_testing_list(
        cdf_root,
        "CDFv1_testing_videos.txt",
        [
            "1 YouTube-real/00170.mp4",
            "0 Celeb-synthesis/id1_id0_0007.mp4",
            "",
            "malformed-line-without-a-space",
            "1 Celeb-real/id9_0001.mp4",  # listed, but not on disk: ignored
        ],
    )
    builder = CelebDFv1Builder()
    official = builder.official_splits(cdf_root, collect_records(builder, cdf_root))
    assert official == {"YTR/00170": "test", "FS_CS/id1_id0_0007": "test"}


def test_the_testing_list_matches_by_file_stem_in_every_task(tmp_path):
    # Only the stem of the listed path counts: its folder (with or without "videos/") and its
    # label are ignored, and a stem shared by several methods is test in every one of them.
    root = _make_v3_root(tmp_path)
    _write_testing_list(
        root,
        "CDFv3_testing_videos.txt",
        [
            "1 YouTube-real/videos/00170.mp4",
            "0 Celeb-synthesis/videos/FaceSwap/BlendFace/id1_id2_0007.mp4",
        ],
    )
    builder = CelebDFv3Builder()
    official = builder.official_splits(root, collect_records(builder, root))
    fs_fr = [a for a in V3_FAKES if not a.startswith("TF_")]
    assert official == {"YTR/00170": "test", **{f"{a}/id1_id2_0007": "test" for a in fs_fr}}


def test_the_default_scheme_carves_everything_off_the_list(cdf_root):
    _write_testing_list(cdf_root, "CDFv1_testing_videos.txt", ["0 x/id3_id4_0002.mp4"])
    builder = CelebDFv1Builder()
    records = collect_records(builder, cdf_root)
    scheme = builder.schemes[builder.default_scheme]
    assert scheme.rule == "official+ident-80-20"
    assert scheme.params == {"policy": "test-only-official"}
    assignment = assign_official_plus_80_20(
        records, builder.official_splits(cdf_root, records), policy="test-only-official"
    )
    assert assignment.pop(("FS_CS/id3_id4_0002", None)) == "test"
    assert len(assignment) == 5
    for rec in records:
        if (rec.key, None) in assignment:
            carve = rec.identity or local_key(rec.key)
            expected = "val" if md5_mod_100(carve) < 20 else "train"
            assert assignment[(rec.key, None)] == expected, rec.key


def test_each_release_reads_its_own_testing_list(tmp_path):
    for builder_class, version in zip(BUILDERS, ("v1", "v2", "v3"), strict=True):
        builder = builder_class()
        name = f"CDF{version}_testing_videos.txt"
        assert builder.metadata_files == (f".official_files/{name}",)
        root = tmp_path / builder.expected_folder
        _touch(root.joinpath(*YOUTUBE_REAL), "00170")
        _write_testing_list(root, name, ["1 YouTube-real/00170.mp4"])
        assert builder.official_splits(root, collect_records(builder, root)) == {
            "YTR/00170": "test"
        }


def test_the_official_split_needs_the_testing_list(cdf_root):
    builder = CelebDFv1Builder()
    with pytest.raises(ConfigError, match=r"CDFv1_testing_videos\.txt") as caught:
        builder.official_splits(cdf_root, collect_records(builder, cdf_root))
    assert ".official_files" in caught.value.hint
    assert "List_of_testing_videos.txt" in caught.value.hint


def test_an_unreadable_testing_list_is_a_contract_error(cdf_root):
    folder = cdf_root / ".official_files"
    folder.mkdir()
    (folder / "CDFv1_testing_videos.txt").write_bytes(b"\xff\xfe\x00 not utf-8 \xff")
    builder = CelebDFv1Builder()
    with pytest.raises(ContractError, match=r"CDFv1_testing_videos\.txt"):
        builder.official_splits(cdf_root, collect_records(builder, cdf_root))


# ------------------------------------------------------------------------------ pairs, labels


def test_pair_candidates_name_the_target_recording(cdf_root):
    builder = CelebDFv1Builder()
    _touch(cdf_root.joinpath(*SYNTHESIS), "junk", "id3_id9_0005")
    records = collect_records(builder, cdf_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "FS_CS/id1_id0_0007": "id1_0007",
        "FS_CS/id3_id4_0002": "id3_0002",
        "FS_CS/id29_0002_test_id08701_aaa": "id29_0002",
        "FS_CS/id3_id9_0005": "id3_0005",
        "FS_CS/junk": None,
    }
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    # id29_0002 and id3_0005 have no real on disk, so those fakes have no pair.
    assert pairs == [
        PairRecord("FS_CS/id1_id0_0007", "CR/id1_0007", "target-recording"),
        PairRecord("FS_CS/id3_id4_0002", "CR/id3_0002", "target-recording"),
    ]


def test_label_vocab_covers_every_task_v1_v2():
    for builder_class, prefix in zip(BUILDERS[:2], ("CDFv1", "CDFv2"), strict=True):
        builder = builder_class()
        vocab = builder.label_vocab().vocab
        assert set(vocab) == {f"{prefix}-{task.abbr}" for task in builder.tasks}
        table = {
            k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
        }
        assert table == {
            f"{prefix}-CR": (0, 0, 1, "real"),
            f"{prefix}-YTR": (0, 0, 2, "real"),
            f"{prefix}-FS_CS": (1, 1, 3, "face-swap"),
        }
        assert vocab[f"{prefix}-FS_CS"]["method"] == "Celeb-synthesis"


def test_label_vocab_covers_every_task_v3():
    builder = CelebDFv3Builder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"CDFv3-{task.abbr}" for task in builder.tasks}
    multiclass = {k.removeprefix("CDFv3-"): v["multiclass"] for k, v in vocab.items()}
    # Reals first, then the fakes in the order of their abbreviations.
    assert multiclass == {
        "CR": 1,
        "YTR": 2,
        **{abbr: 3 + i for i, abbr in enumerate(sorted(V3_FAKES))},
    }
    family = {"FS": "face-swap", "FR": "face-reenactment", "TF": "talking-face"}
    for key, entry in vocab.items():
        abbr = key.removeprefix("CDFv3-")
        if abbr in ("CR", "YTR"):
            assert (entry["binary"], entry["binary_av"], entry["family"]) == (0, 0, "real")
        else:
            assert (entry["binary"], entry["binary_av"]) == (1, 1)
            assert entry["family"] == family[abbr[:2]]
            assert entry["method"] == V3_FAKES[abbr][1]


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    for builder_class in BUILDERS:
        builder = builder_class()
        assert builder.default_scheme == "official+ident-80-20"
        assert {name: s.rule for name, s in builder.schemes.items()} == {
            "official+ident-80-20": "official+ident-80-20",
            "all-test": "all-test",
            "benchmark": "benchmark",
        }
        assert builder.pairing_rule == "target-recording"
        assert builder.pairing_fanout is None
    assert CelebDFv1Builder.benchmark == BenchmarkSpec(k_fake=100)
    assert CelebDFv2Builder.benchmark == BenchmarkSpec(k_fake=100)
    assert CelebDFv3Builder.benchmark == BenchmarkSpec(k_fake=2, strata=("identity", "task"))


def test_the_benchmark_draws_from_the_testing_list(cdf_root):
    _write_testing_list(
        cdf_root,
        "CDFv1_testing_videos.txt",
        ["0 x/id1_id0_0007.mp4", "1 x/id1_0007.mp4", "1 x/00170.mp4"],
    )
    builder = CelebDFv1Builder()
    records = collect_records(builder, cdf_root)
    official = builder.official_splits(cdf_root, records)
    assert builder.benchmark is not None
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=[key for key, split in official.items() if split == "test"],
    )
    # One fake in the pool, so one of the two listed reals balances it.
    assert ("FS_CS/id1_id0_0007", None) in chosen
    assert len(chosen) == 2
    assert {key for key, _ in chosen} <= {"FS_CS/id1_id0_0007", "CR/id1_0007", "YTR/00170"}


def test_dataset_cards():
    names = {
        CelebDFv1Builder: ("celebdf-v1", "Celeb-DF v1", "Celeb-DF-v1"),
        CelebDFv2Builder: ("celebdf-v2", "Celeb-DF v2", "Celeb-DF-v2"),
        CelebDFv3Builder: ("celebdf-v3", "Celeb-DF v3", "Celeb-DF-v3"),
    }
    for builder_class, (dataset_id, name, folder) in names.items():
        builder = builder_class()
        cards = {
            scheme: SchemeCard(kind=spec.kind, sha256="a" * 64)
            for scheme, spec in builder.schemes.items()
        }
        card = builder.dataset_card(cards)
        assert card.id == dataset_id
        assert card.name == name
        assert builder.expected_folder == folder
        assert card.compressions is None
        assert builder.known_compressions == ()
        assert card.default_scheme == "official+ident-80-20"
        assert card.license.spdx is None
        assert card.paper is None


def test_the_layout_names_the_testing_list():
    for builder_class, version in zip(BUILDERS, ("v1", "v2", "v3"), strict=True):
        text = builder_class().describe_layout()
        assert "original_content/Celeb-real/videos" in text
        assert f".official_files/CDF{version}_testing_videos.txt" in text
        assert "List_of_testing_videos.txt" in text
        assert "{cX}" not in text


def test_they_are_registered():
    for builder_class in BUILDERS:
        assert isinstance(get_builder(builder_class.dataset_id), builder_class)
