"""The FFIW-10K inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, SchemeCard
from dfwb.preprocess.inventory.builders.ffiw10k import FFIW10KBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    local_key,
    task_of,
)

REAL_DIR = ("original_content", "source", "videos")
FAKE_DIR = ("manipulated_content", "target", "videos")


def _make_synthetic_ffiw10k_root(tmp_path: Path, *, with_split_pairs: bool = False) -> Path:
    """A tiny FFIW-10K tree: two reals and two fakes.

    With ``with_split_pairs``, also writes ``.official_files/splits/test.json``, which pairs
    source 1 with target 3 and source 2 with target 4.
    """
    root = tmp_path / "FFIW10K"
    real_dir = root.joinpath(*REAL_DIR)
    fake_dir = root.joinpath(*FAKE_DIR)
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    for stem in ("test_00000001", "test_00000002"):
        (real_dir / f"{stem}.mp4").write_bytes(b"\x00")
    for stem in ("test_00000003", "test_00000004"):
        (fake_dir / f"{stem}.mp4").write_bytes(b"\x00")
    if with_split_pairs:
        _write_pairs(root, "test", [[1, 3], [2, 4]])
    return root


def _write_pairs(root: Path, split: str, content: object) -> None:
    folder = root / ".official_files" / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{split}.json").write_text(json.dumps(content), encoding="utf-8")


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()


@pytest.fixture
def ffiw_root(tmp_path: Path) -> Path:
    """The standard tree, without the pair lists."""
    return _make_synthetic_ffiw10k_root(tmp_path)


@pytest.fixture
def ffiw_root_with_splits(tmp_path: Path) -> Path:
    """The standard tree with a pair list, to exercise pair_key."""
    return _make_synthetic_ffiw10k_root(tmp_path, with_split_pairs=True)


def _discover(root, **kwargs):
    return collect_records(FFIW10KBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(ffiw_root):
    # 2 reals + 2 fakes
    assert len(_discover(ffiw_root)) == 4


def test_entry_schema(ffiw_root):
    for rec in _discover(ffiw_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"REAL", "FS_FAKE"}
        assert rec.builder == BuilderRef("ffiw10k", FFIW10KBuilder.version)
        assert rec.attrs["task_name"] in {"source", "target"}
        assert not rec.relpath.startswith("/")
        # FFIW-10K has a single version: no compression
        assert rec.compression is None
        assert rec.folder is None


def test_no_media_field(ffiw_root):
    for rec in _discover(ffiw_root):
        assert rec.probe is None
        assert not hasattr(rec, "media")


def test_no_label_class(ffiw_root):
    for rec in _discover(ffiw_root):
        assert rec.label_key == f"FFIW-{task_of(rec.key)}"
        assert not hasattr(rec, "label")


def test_fields(ffiw_root):
    records = {rec.key: rec for rec in _discover(ffiw_root)}
    real = records["REAL/test_00000001"]
    # No actor ids ship with the dataset: the identity is the video's own stem.
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "test_00000001",
        None,
        None,
        None,
    )
    assert real.method == "original"
    assert real.relpath == "original_content/source/videos/test_00000001.mp4"
    assert real.attrs == {"task_name": "source"}
    fake = records["FS_FAKE/test_00000003"]
    assert (fake.identity, fake.target_id, fake.source_id) == ("test_00000003", None, None)
    assert fake.method == "unknown"
    assert fake.relpath == "manipulated_content/target/videos/test_00000003.mp4"
    assert fake.attrs == {"task_name": "target"}


def test_pair_key_populated_from_official_splits(ffiw_root_with_splits):
    fakes = [r for r in _discover(ffiw_root_with_splits) if task_of(r.key) != "REAL"]
    assert fakes
    assert [r for r in fakes if r.pair_key]
    by_key = {local_key(r.key): r for r in fakes}
    # The fake test_00000003 pairs with the source test_00000001.
    assert by_key["test_00000003"].pair_key == "test_00000001"
    assert by_key["test_00000004"].pair_key == "test_00000002"


def test_pair_key_absent_without_official_splits(ffiw_root):
    for rec in _discover(ffiw_root):
        assert rec.pair_key is None


def test_a_real_never_carries_a_pair_key(ffiw_root_with_splits):
    for rec in _discover(ffiw_root_with_splits):
        if task_of(rec.key) == "REAL":
            assert rec.pair_key is None


def test_the_pair_lists_map_both_ways_and_later_lists_win(ffiw_root):
    _touch(ffiw_root.joinpath(*FAKE_DIR), "test_00000001.mp4", "test_00000009.mp4")
    # train: 1 <-> 3; val: 3 <-> 9 (so 3 now points at 9); test: "7" <-> 1 (so 1 points at 7).
    _write_pairs(ffiw_root, "train", [[1, 3]])
    _write_pairs(ffiw_root, "val", [[3, 9]])
    _write_pairs(ffiw_root, "test", [["7", 1.0]])
    by_key = {rec.key: rec.pair_key for rec in _discover(ffiw_root)}
    assert by_key == {
        "REAL/test_00000001": None,
        "REAL/test_00000002": None,
        "FS_FAKE/test_00000001": "test_00000007",
        "FS_FAKE/test_00000003": "test_00000009",
        "FS_FAKE/test_00000004": None,
        "FS_FAKE/test_00000009": "test_00000003",
    }


def test_an_unreadable_pair_list_is_skipped_with_a_warning(ffiw_root, caplog):
    _write_pairs(ffiw_root, "train", [[2, 4]])
    (ffiw_root / ".official_files" / "splits" / "test.json").write_text("{oops", "utf-8")
    with caplog.at_level(logging.WARNING, logger="dfwb.preprocess.inventory.builders.ffiw10k"):
        by_key = {rec.key: rec.pair_key for rec in _discover(ffiw_root)}
    assert "test.json" in caplog.text
    assert by_key["FS_FAKE/test_00000004"] == "test_00000002"


@pytest.mark.parametrize("content", [{"1": 3}, [[1]], [["x", 3]], [7]])
def test_a_malformed_pair_list_is_a_contract_error(ffiw_root, content):
    _write_pairs(ffiw_root, "val", content)
    with pytest.raises(ContractError, match=r"val\.json"):
        _discover(ffiw_root)


def test_the_tasks_are_flat(ffiw_root):
    _touch(ffiw_root.joinpath(*FAKE_DIR, "nested"), "test_00000005.mp4")
    assert "FS_FAKE/test_00000005" not in {rec.key for rec in _discover(ffiw_root)}


def test_no_compression_can_be_requested(ffiw_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        _discover(ffiw_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ official split


def test_the_official_split_is_the_prefix_of_each_name(ffiw_root):
    _touch(ffiw_root.joinpath(*REAL_DIR), "train_00000000.mp4", "Val_00000000.mp4", "odd_1.mp4")
    _touch(ffiw_root.joinpath(*FAKE_DIR), "train_00000000.mp4", "Val_00000000.mp4")
    builder = FFIW10KBuilder()
    official = builder.official_splits(ffiw_root, collect_records(builder, ffiw_root))
    assert official == {
        "REAL/test_00000001": "test",
        "REAL/test_00000002": "test",
        "FS_FAKE/test_00000003": "test",
        "FS_FAKE/test_00000004": "test",
        "REAL/train_00000000": "train",
        "FS_FAKE/train_00000000": "train",
        # The prefix is matched ignoring case.
        "REAL/Val_00000000": "val",
        "FS_FAKE/Val_00000000": "val",
        # odd_1 names no split, so it is left out.
    }


def test_only_mp4_names_carry_the_split(ffiw_root):
    # A name counts when an .mp4 of that stem exists in either task.
    _touch(ffiw_root.joinpath(*FAKE_DIR), "train_00000007.avi", "train_00000008.MP4")
    _touch(ffiw_root.joinpath(*REAL_DIR), "train_00000009.mkv")
    _touch(ffiw_root.joinpath(*FAKE_DIR), "train_00000009.mp4")
    builder = FFIW10KBuilder()
    official = builder.official_splits(ffiw_root, collect_records(builder, ffiw_root))
    assert "FS_FAKE/train_00000007" not in official
    assert official["FS_FAKE/train_00000008"] == "train"
    assert official["REAL/train_00000009"] == "train"
    assert official["FS_FAKE/train_00000009"] == "train"


def test_the_official_scheme_assigns_every_named_video(ffiw_root):
    _touch(ffiw_root.joinpath(*REAL_DIR), "val_00000000.mp4")
    builder = FFIW10KBuilder()
    records = collect_records(builder, ffiw_root)
    assignment = assign_official(records, builder.official_splits(ffiw_root, records))
    assert assignment[("REAL/val_00000000", None)] == "val"
    assert len(assignment) == 5


# ------------------------------------------------------------------------------ pairs, labels


def test_nothing_pairs(ffiw_root_with_splits):
    builder = FFIW10KBuilder()
    assert builder.pairing_rule is None
    for rec in collect_records(builder, ffiw_root_with_splits):
        assert builder.pair_candidates(rec) is None


def test_label_vocab_covers_every_task():
    builder = FFIW10KBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"FFIW-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {"FFIW-REAL": (0, 0, 1, "real"), "FFIW-FS_FAKE": (1, 1, 2, "face-swap")}
    assert vocab["FFIW-FS_FAKE"]["method"] == "unknown"


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = FFIW10KBuilder()
    assert builder.default_scheme == "official"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=1000)
    assert builder.pairing_fanout is None
    assert builder.metadata_files == tuple(
        f".official_files/splits/{split}.json" for split in ("train", "val", "test")
    )


def test_the_benchmark_draws_from_the_official_test(ffiw_root):
    _touch(ffiw_root.joinpath(*FAKE_DIR), "train_00000000.mp4")
    builder = FFIW10KBuilder()
    records = collect_records(builder, ffiw_root)
    official = builder.official_splits(ffiw_root, records)
    assert builder.benchmark is not None
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=[key for key, split in official.items() if split == "test"],
    )
    # Fewer than 1,000 test fakes: all of them, and the reals; the train fake is not drawn.
    assert {key for key, _ in chosen} == {
        "REAL/test_00000001",
        "REAL/test_00000002",
        "FS_FAKE/test_00000003",
        "FS_FAKE/test_00000004",
    }


def test_dataset_card():
    builder = FFIW10KBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "ffiw10k"
    assert card.name == "FFIW-10K"
    assert {"FFIW10K", "FFIW"} <= set(card.aliases)
    assert card.compressions is None
    assert builder.known_compressions == ()
    assert card.default_scheme == "official"
    assert card.homepage == "https://github.com/tfzhou/FFIW"
    assert card.paper is not None
    assert (card.paper.title, card.paper.venue, card.paper.year, card.paper.doi) == (
        "Face Forensics in the Wild",
        "CVPR",
        2021,
        None,
    )
    assert card.license.spdx is None


def test_the_layout_names_the_split_prefix_and_the_pair_lists():
    text = FFIW10KBuilder().describe_layout()
    assert "'FFIW10K'" in text
    assert "original_content/source/videos/<video>" in text
    assert "manipulated_content/target/videos/<video>" in text
    assert "train_, val_ or test_" in text
    assert ".official_files/splits/" in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("ffiw10k"), FFIW10KBuilder)
    assert FFIW10KBuilder.expected_folder == "FFIW10K"
    assert FFIW10KBuilder.label_prefix == "FFIW"
