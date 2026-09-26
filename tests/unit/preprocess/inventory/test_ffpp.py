"""The FaceForensics++ inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.ffpp import FaceForensicsBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    local_key,
    resolve_pairs,
    task_of,
)

# task abbr -> (method folder, method)
METHODS = {
    "FS_DF": "Deepfakes",
    "FR_F2F": "Face2Face",
    "FS_FSH": "FaceShifter",
    "FS_FS": "FaceSwap",
    "FR_NT": "NeuralTextures",
}


def _make_synthetic_ffpp_root(tmp_path: Path) -> Path:
    """A tiny FaceForensics++ tree: two reals and two Deepfakes fakes, at c23 only."""
    root = tmp_path / "FaceForensics++"
    real_dir = root / "original_content" / "YouTube" / "c23" / "videos"
    fake_dir = root / "manipulated_content" / "Deepfakes" / "c23" / "videos"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    # Reals are zero-padded 3-digit sequence ids; fakes are <target>_<source>.
    (real_dir / "000.mp4").write_bytes(b"\x00")
    (real_dir / "001.mp4").write_bytes(b"\x00")
    (fake_dir / "000_870.mp4").write_bytes(b"\x00")
    (fake_dir / "001_872.mp4").write_bytes(b"\x00")
    return root


def _touch(folder: Path, *stems: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (folder / f"{stem}.mp4").touch()


def _write_official(root: Path, splits: dict[str, list]) -> None:
    """The official split files: each a JSON list of ``[target, source]`` pairs."""
    folder = root / ".official_files" / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    for split, pairs in splits.items():
        (folder / f"{split}.json").write_text(json.dumps(pairs, indent=4), encoding="utf-8")


@pytest.fixture
def ffpp_root(tmp_path: Path) -> Path:
    return _make_synthetic_ffpp_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(FaceForensicsBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(ffpp_root):
    # 2 reals + 2 fakes, one compression on disk -> 4 records.
    assert len(_discover(ffpp_root)) == 4


def test_entry_schema(ffpp_root):
    for rec in _discover(ffpp_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"REAL", "FS_DF"}
        assert rec.builder == BuilderRef("ffpp", FaceForensicsBuilder.version)
        assert rec.attrs["task_name"] in {"YouTube", "Deepfakes"}
        assert not rec.relpath.startswith("/")
        assert rec.folder is None


def test_records_carry_a_label_key_but_no_label_or_media(ffpp_root):
    for rec in _discover(ffpp_root):
        assert rec.label_key.startswith("FF-")
        assert rec.probe is None
        assert not hasattr(rec, "label")
        assert not hasattr(rec, "media")


def test_pairing_attrs_present(ffpp_root):
    fakes = [rec for rec in _discover(ffpp_root) if task_of(rec.key) != "REAL"]
    assert fakes
    paired = [rec for rec in fakes if rec.pair_key]
    assert paired
    sample = paired[0]
    assert sample.target_id
    assert sample.source_id
    # pair_key is "<target_id>_<source_id>", which is the file stem for a regular fake
    assert sample.pair_key == f"{sample.target_id}_{sample.source_id}"
    assert sample.pair_key == local_key(sample.key)
    assert sample.identity == sample.target_id


def test_relpath_is_the_concrete_path_of_its_compression(ffpp_root):
    records = {rec.key: rec for rec in _discover(ffpp_root)}
    assert records["REAL/000"].relpath == "original_content/YouTube/c23/videos/000.mp4"
    assert (
        records["FS_DF/000_870"].relpath == "manipulated_content/Deepfakes/c23/videos/000_870.mp4"
    )
    for rec in records.values():
        assert "{cX}" not in rec.relpath
        assert rec.compression == "c23"


def test_real_vs_fake_label_keys(ffpp_root):
    records = _discover(ffpp_root)
    assert {r.label_key for r in records if task_of(r.key) == "REAL"} == {"FF-REAL"}
    assert {r.label_key for r in records if task_of(r.key) != "REAL"} == {"FF-FS_DF"}
    for rec in records:
        if task_of(rec.key) == "REAL":
            assert rec.pair_key is None
            assert rec.source_id is None
            assert rec.identity == local_key(rec.key)
            assert rec.target_id == local_key(rec.key)
            assert rec.method == "original"


def test_every_task_and_compression_is_discovered(tmp_path):
    root = tmp_path / "FaceForensics++"
    for compression in ("raw", "c23", "c40"):
        _touch(root / "original_content" / "YouTube" / compression / "videos", "000", "003")
        for folder in METHODS.values():
            _touch(root / "manipulated_content" / folder / compression / "videos", "000_003")
    records = _discover(root)
    assert len(records) == 3 * (2 + len(METHODS))
    by_task = {task_of(r.key): r for r in records if r.compression == "c40"}
    assert list(by_task) == ["FR_F2F", "FR_NT", "FS_DF", "FS_FS", "FS_FSH", "REAL"]
    for abbr, method in METHODS.items():
        assert by_task[abbr].method == method
        assert by_task[abbr].attrs == {"task_name": method}
        assert by_task[abbr].key == f"{abbr}/000_003"
        assert by_task[abbr].relpath == f"manipulated_content/{method}/c40/videos/000_003.mp4"
    assert by_task["REAL"].attrs == {"task_name": "YouTube"}

    only_c23 = _discover(root, compressions=["c23"])
    assert {r.compression for r in only_c23} == {"c23"}
    assert len(only_c23) == 2 + len(METHODS)


def test_an_unknown_compression_is_a_config_error(ffpp_root):
    with pytest.raises(ConfigError, match="c0"):
        _discover(ffpp_root, compressions=["c0"])


def test_a_fake_without_an_underscore_is_skipped(ffpp_root):
    _touch(ffpp_root / "manipulated_content" / "Deepfakes" / "c23" / "videos", "junk")
    keys = {rec.key for rec in _discover(ffpp_root)}
    assert "FS_DF/junk" not in keys
    assert len(keys) == 4


def test_a_fake_stem_with_more_parts_keeps_its_first_two_ids(ffpp_root):
    _touch(ffpp_root / "manipulated_content" / "Deepfakes" / "c23" / "videos", "002_004_extra")
    rec = {r.key: r for r in _discover(ffpp_root)}["FS_DF/002_004_extra"]
    assert (rec.identity, rec.target_id, rec.source_id) == ("002", "002", "004")
    assert rec.pair_key == "002_004"


# ------------------------------------------------------------------------------ official split


def test_official_splits_follow_the_target_sequence(tmp_path):
    root = tmp_path / "FaceForensics++"
    videos = root / "original_content" / "YouTube" / "c23" / "videos"
    _touch(videos, "000", "001", "002", "003", "004", "005", "006")
    _touch(
        root / "manipulated_content" / "Deepfakes" / "c23" / "videos",
        "000_001",
        "001_000",
        "002_003",
        "004_005",
        "006_000",
    )
    _write_official(
        root, {"train": [["000", "001"]], "val": [["002", "003"]], "test": [["004", "005"]]}
    )
    builder = FaceForensicsBuilder()
    records = collect_records(builder, root)
    official = builder.official_splits(root, records)
    assert official == {
        "REAL/000": "train",
        "REAL/001": "train",
        "REAL/002": "val",
        "REAL/003": "val",
        "REAL/004": "test",
        "REAL/005": "test",
        "FS_DF/000_001": "train",
        "FS_DF/001_000": "train",
        "FS_DF/002_003": "val",
        "FS_DF/004_005": "test",
        # 006 is in no official list, so neither the real nor the fake it is the target of
        # is assigned; the fake's source (000) does not matter.
    }


def test_official_splits_take_the_first_list_that_names_the_target(tmp_path):
    root = _make_synthetic_ffpp_root(tmp_path)
    # "000" is listed in train and test: train is read first and wins, as a source id in
    # val ("001") places its own real sequence there.
    _write_official(
        root, {"train": [["000", "900"]], "val": [["901", "001"]], "test": [["000", "902"]]}
    )
    builder = FaceForensicsBuilder()
    official = builder.official_splits(root, collect_records(builder, root))
    assert official == {
        "REAL/000": "train",
        "FS_DF/000_870": "train",
        "REAL/001": "val",
        "FS_DF/001_872": "val",
    }


def test_official_split_entries_may_be_bare_ids(tmp_path):
    root = _make_synthetic_ffpp_root(tmp_path)
    _write_official(root, {"train": ["000"], "val": [], "test": [["001", "002"]]})
    builder = FaceForensicsBuilder()
    official = builder.official_splits(root, collect_records(builder, root))
    assert official["REAL/000"] == "train"
    assert official["REAL/001"] == "test"


def test_the_official_split_covers_every_compression(tmp_path):
    root = _make_synthetic_ffpp_root(tmp_path)
    _touch(root / "original_content" / "YouTube" / "c40" / "videos", "000")
    _write_official(root, {"train": [["000", "001"]], "val": [], "test": []})
    builder = FaceForensicsBuilder()
    records = collect_records(builder, root)
    assignment = assign_official(records, builder.official_splits(root, records))
    assert assignment[("REAL/000", "c23")] == assignment[("REAL/000", "c40")] == "train"


def test_the_official_split_needs_its_files(ffpp_root):
    _write_official(ffpp_root, {"train": [["000", "001"]], "test": []})
    builder = FaceForensicsBuilder()
    with pytest.raises(ConfigError, match=r"val\.json") as caught:
        builder.official_splits(ffpp_root, collect_records(builder, ffpp_root))
    assert ".official_files/splits" in caught.value.hint


def test_a_malformed_official_file_is_a_contract_error(ffpp_root):
    _write_official(ffpp_root, {"train": [], "val": [], "test": []})
    (ffpp_root / ".official_files" / "splits" / "val.json").write_text("{not json", "utf-8")
    builder = FaceForensicsBuilder()
    with pytest.raises(ContractError, match=r"val\.json"):
        builder.official_splits(ffpp_root, collect_records(builder, ffpp_root))


def test_an_official_file_that_is_not_a_list_is_a_contract_error(ffpp_root):
    _write_official(ffpp_root, {"train": [], "val": [], "test": []})
    (ffpp_root / ".official_files" / "splits" / "test.json").write_text('{"a": 1}', "utf-8")
    builder = FaceForensicsBuilder()
    with pytest.raises(ContractError, match=r"test\.json"):
        builder.official_splits(ffpp_root, collect_records(builder, ffpp_root))


# ------------------------------------------------------------------------------ pairs, labels


def test_pair_candidates_name_the_target_sequence(ffpp_root):
    builder = FaceForensicsBuilder()
    records = collect_records(builder, ffpp_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "FS_DF/000_870": "000",
        "FS_DF/001_872": "001",
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
        PairRecord("FS_DF/000_870", "REAL/000", "target-id"),
        PairRecord("FS_DF/001_872", "REAL/001", "target-id"),
    ]


def test_a_fake_whose_target_has_no_real_gets_no_pair(ffpp_root):
    _touch(ffpp_root / "manipulated_content" / "Deepfakes" / "c23" / "videos", "555_000")
    builder = FaceForensicsBuilder()
    records = collect_records(builder, ffpp_root)
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule="target-id",
    )
    assert "FS_DF/555_000" not in {p.fake_key for p in pairs}
    assert len(pairs) == 2


def test_label_vocab_covers_every_task():
    builder = FaceForensicsBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"FF-{task.abbr}" for task in builder.tasks}
    table = {
        key: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for key, v in vocab.items()
    }
    assert table == {
        "FF-REAL": (0, 0, 1, "real"),
        "FF-FS_DF": (1, 1, 2, "face-swap"),
        "FF-FR_F2F": (1, 1, 3, "face-reenactment"),
        "FF-FS_FSH": (1, 1, 4, "face-swap"),
        "FF-FS_FS": (1, 1, 5, "face-swap"),
        "FF-FR_NT": (1, 1, 6, "face-reenactment"),
    }
    assert vocab["FF-FS_DF"]["method"] == "Deepfakes"
    assert vocab["FF-REAL"]["method"] == "original"


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = FaceForensicsBuilder()
    assert builder.default_scheme == "official"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    # The benchmark is defined at c23, whatever other compressions are on disk.
    assert builder.benchmark == BenchmarkSpec(k_fake=500, compressions=("c23",))
    assert builder.pairing_rule == "target-id"
    assert builder.pairing_fanout is None


def test_the_benchmark_draws_from_the_official_test(tmp_path):
    root = tmp_path / "FaceForensics++"
    _touch(root / "original_content" / "YouTube" / "c23" / "videos", "000", "001", "002", "003")
    _touch(root / "manipulated_content" / "Deepfakes" / "c23" / "videos", "000_001", "002_003")
    _write_official(root, {"train": [["000", "001"]], "val": [], "test": [["002", "003"]]})
    builder = FaceForensicsBuilder()
    records = collect_records(builder, root)
    official = builder.official_splits(root, records)
    assert builder.benchmark is not None
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=[key for key, split in official.items() if split == "test"],
    )
    # One fake in the test pool, so one real is drawn to balance it.
    assert ("FS_DF/002_003", "c23") in chosen
    assert len(chosen) == 2
    assert {key for key, _ in chosen} <= {"FS_DF/002_003", "REAL/002", "REAL/003"}


def test_the_benchmark_ignores_the_other_compressions_on_disk(tmp_path):
    root = tmp_path / "FaceForensics++"
    for compression in ("raw", "c23", "c40"):
        _touch(root / "original_content" / "YouTube" / compression / "videos", "000", "001")
        _touch(root / "manipulated_content" / "Deepfakes" / compression / "videos", "000_001")
    _write_official(root, {"train": [], "val": [], "test": [["000", "001"]]})
    builder = FaceForensicsBuilder()
    records = collect_records(builder, root)
    official = builder.official_splits(root, records)
    assert builder.benchmark is not None
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=[key for key, split in official.items() if split == "test"],
    )
    assert {compression for _, compression in chosen} == {"c23"}
    assert len(chosen) == 2


def test_dataset_card():
    builder = FaceForensicsBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "ffpp"
    assert card.name == "FaceForensics++"
    assert card.compressions == ["raw", "c23", "c40"]
    assert card.default_scheme == "official"
    assert card.license.spdx is None
    assert card.paper is not None
    assert (card.paper.title, card.paper.venue, card.paper.year, card.paper.doi) == (
        "FaceForensics++: Learning to Detect Manipulated Facial Images",
        "ICCV",
        2019,
        "10.1109/ICCV.2019.00009",
    )
    assert list(builder.known_compressions) == card.compressions


def test_the_layout_names_the_official_files_and_the_shared_folder():
    text = FaceForensicsBuilder().describe_layout()
    assert "original_content/YouTube/{cX}/videos" in text
    assert ".official_files/splits" in text
    assert "raw, c23, c40" in text


def test_it_is_registered():
    assert isinstance(get_builder("ffpp"), FaceForensicsBuilder)
