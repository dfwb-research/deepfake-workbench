"""The DFDC Preview inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.dfdcp import DFDCPreviewBuilder
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

REAL_DIR = ("original_content", "original", "videos")
METHOD_A_DIR = ("manipulated_content", "method_A", "videos")
METHOD_B_DIR = ("manipulated_content", "method_B", "videos")


def _make_synthetic_dfdcp_root(tmp_path: Path) -> Path:
    """A tiny DFDC Preview tree: two reals and two method_A fakes.

    As on the real release, videos are nested ``<target>/<target>_<suffix>/`` in every task.
    """
    root = tmp_path / "DFDC-P"
    real_1 = root.joinpath(*REAL_DIR, "1003254", "1003254_A")
    real_2 = root.joinpath(*REAL_DIR, "1004567", "1004567_B")
    fake_1 = root.joinpath(*METHOD_A_DIR, "1003254", "1003254_A")
    for folder in (real_1, real_2, fake_1):
        folder.mkdir(parents=True)
    # Reals: <identity>_<suffix>_<counter>.
    (real_1 / "1003254_A_001.mp4").write_bytes(b"\x00")
    (real_2 / "1004567_B_002.mp4").write_bytes(b"\x00")
    # Fakes: <swapped>_<target>_<suffix>_<counter>; the first pairs with 1003254_A_001.
    (fake_1 / "1255229_1003254_A_001.mp4").write_bytes(b"\x00")
    (fake_1 / "1255230_1003254_A_002.mp4").write_bytes(b"\x00")
    return root


def _touch(folder: Path, *stems: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (folder / f"{stem}.mp4").touch()


def _write_dataset_json(root: Path, content: object) -> None:
    """The preview's metadata: a JSON object keyed by video path, each with its ``set``."""
    folder = root / ".official_files"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dataset.json").write_text(json.dumps(content, indent=2), encoding="utf-8")


def _meta(split: str, label: str = "fake") -> dict:
    return {"augmentations": [], "label": label, "set": split}


@pytest.fixture
def dfdcp_root(tmp_path: Path) -> Path:
    return _make_synthetic_dfdcp_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(DFDCPreviewBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(dfdcp_root):
    # 2 reals + 2 fakes, found in their nested folders
    assert len(_discover(dfdcp_root)) == 4


def test_entry_schema(dfdcp_root):
    for rec in _discover(dfdcp_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"REAL", "FS_METHOD_A"}
        assert rec.builder == BuilderRef("dfdc-p", DFDCPreviewBuilder.version)
        assert rec.attrs["task_name"] in {"Original", "method_A"}
        assert not rec.relpath.startswith("/")
        # DFDC Preview has a single version: no compression
        assert rec.compression is None
        assert rec.folder is None


def test_no_media_no_label_class(dfdcp_root):
    for rec in _discover(dfdcp_root):
        assert rec.label_key == f"DFDCP-{task_of(rec.key)}"
        assert rec.probe is None
        assert not hasattr(rec, "label")
        assert not hasattr(rec, "media")


def test_pairing_attrs_present(dfdcp_root):
    records = _discover(dfdcp_root)
    fakes = [r for r in records if task_of(r.key) != "REAL"]
    assert fakes
    paired = [r for r in fakes if r.pair_key]
    assert paired
    sample = paired[0]
    assert sample.target_id
    assert sample.source_id
    # pair_key is "<target_id>_<suffix>_<counter>"
    assert sample.pair_key.startswith(sample.target_id + "_")
    # the fake's pair_key 1003254_A_001 is a real's local key in the same tree
    real_keys = {local_key(r.key) for r in records if task_of(r.key) == "REAL"}
    assert sample.pair_key in real_keys


def test_real_entry_attributes(dfdcp_root):
    reals = [r for r in _discover(dfdcp_root) if task_of(r.key) == "REAL"]
    assert reals
    for rec in reals:
        assert rec.identity is not None
        assert rec.target_id == rec.identity
        assert rec.source_id is None
        assert rec.pair_key is None
        assert rec.method == "original"


def test_fields_and_nested_relpaths(dfdcp_root):
    records = {rec.key: rec for rec in _discover(dfdcp_root)}
    fake = records["FS_METHOD_A/1255229_1003254_A_001"]
    assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
        "1003254",
        "1003254",
        "1255229",
        "1003254_A_001",
    )
    assert fake.method == "method_a"
    assert fake.attrs == {"task_name": "method_A"}
    assert fake.relpath == (
        "manipulated_content/method_A/videos/1003254/1003254_A/1255229_1003254_A_001.mp4"
    )
    real = records["REAL/1004567_B_002"]
    assert (real.identity, real.target_id) == ("1004567", "1004567")
    assert real.attrs == {"task_name": "Original"}
    assert real.relpath == "original_content/original/videos/1004567/1004567_B/1004567_B_002.mp4"


def test_method_b_is_discovered(dfdcp_root):
    _touch(dfdcp_root.joinpath(*METHOD_B_DIR, "1004567", "1004567_B"), "1300001_1004567_B_002")
    rec = {r.key: r for r in _discover(dfdcp_root)}["FS_METHOD_B/1300001_1004567_B_002"]
    assert rec.method == "method_b"
    assert rec.label_key == "DFDCP-FS_METHOD_B"
    assert rec.pair_key == "1004567_B_002"


def test_extra_parts_stay_in_the_counter(dfdcp_root):
    _touch(dfdcp_root.joinpath(*METHOD_A_DIR, "1003254"), "1255229_1003254_A_001_x")
    rec = {r.key: r for r in _discover(dfdcp_root)}["FS_METHOD_A/1255229_1003254_A_001_x"]
    assert (rec.target_id, rec.source_id, rec.pair_key) == (
        "1003254",
        "1255229",
        "1003254_A_001_x",
    )


def test_names_that_do_not_parse_are_kept_without_fields(dfdcp_root):
    _touch(dfdcp_root.joinpath(*METHOD_A_DIR), "abc_1003254_A_001", "1255229_x_A_001", "1_2_3")
    _touch(dfdcp_root.joinpath(*REAL_DIR), "1003254_A")
    records = {rec.key: rec for rec in _discover(dfdcp_root)}
    for key in (
        "FS_METHOD_A/abc_1003254_A_001",
        "FS_METHOD_A/1255229_x_A_001",
        "FS_METHOD_A/1_2_3",
        "REAL/1003254_A",
    ):
        rec = records[key]
        assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (None,) * 4, key


def test_a_real_needs_three_parts_but_not_a_numeric_identity(dfdcp_root):
    _touch(dfdcp_root.joinpath(*REAL_DIR), "actor_A_001")
    rec = {r.key: r for r in _discover(dfdcp_root)}["REAL/actor_A_001"]
    assert (rec.identity, rec.target_id) == ("actor", "actor")


def test_no_compression_can_be_requested(dfdcp_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        _discover(dfdcp_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ official split


def test_the_official_split_reads_each_videos_set(dfdcp_root):
    _write_dataset_json(
        dfdcp_root,
        {
            "original_videos/1003254/1003254_A_001.mp4": _meta("train", "real"),
            "original_videos/1004567/1004567_B_002.mp4": _meta(" Test ", "real"),
            "method_A/1003254/1003254_A/1255229_1003254_A_001.mp4": _meta("test"),
            "method_A/1003254/1003254_A/1255230_1003254_A_002.mp4": _meta("val"),  # ignored
            "method_A/9/9_A/1_9_A_001.mp4": _meta("train"),  # not on disk: ignored
        },
    )
    builder = DFDCPreviewBuilder()
    official = builder.official_splits(dfdcp_root, collect_records(builder, dfdcp_root))
    assert official == {
        "REAL/1003254_A_001": "train",
        "REAL/1004567_B_002": "test",
        "FS_METHOD_A/1255229_1003254_A_001": "test",
    }


def test_train_wins_when_a_stem_is_in_both_sets(dfdcp_root):
    _write_dataset_json(
        dfdcp_root,
        {
            "a/1255229_1003254_A_001.mp4": _meta("test"),
            "b/1255229_1003254_A_001.mp4": _meta("train"),
            "c/1255230_1003254_A_002.mp4": {"label": "fake"},  # no set: ignored
        },
    )
    builder = DFDCPreviewBuilder()
    official = builder.official_splits(dfdcp_root, collect_records(builder, dfdcp_root))
    assert official == {"FS_METHOD_A/1255229_1003254_A_001": "train"}


def test_the_default_scheme_carves_val_from_the_official_train(dfdcp_root):
    _write_dataset_json(
        dfdcp_root,
        {
            "o/1003254_A_001.mp4": _meta("train", "real"),
            "o/1004567_B_002.mp4": _meta("test", "real"),
            "m/1255229_1003254_A_001.mp4": _meta("train"),
        },
    )
    builder = DFDCPreviewBuilder()
    records = collect_records(builder, dfdcp_root)
    official = builder.official_splits(dfdcp_root, records)
    scheme = builder.schemes[builder.default_scheme]
    assert scheme.rule == "official+ident-80-20"
    assert scheme.params == {"policy": "official-train-test"}
    assignment = assign_official_plus_80_20(records, official, policy="official-train-test")
    expected = "val" if md5_mod_100("1003254") < 20 else "train"
    assert assignment == {
        ("REAL/1003254_A_001", None): expected,
        ("FS_METHOD_A/1255229_1003254_A_001", None): expected,  # the same identity
        ("REAL/1004567_B_002", None): "test",
        # 1255230_1003254_A_002 is in no official set, so it is left out
    }
    assert assign_official(records, official) == {
        ("REAL/1003254_A_001", None): "train",
        ("FS_METHOD_A/1255229_1003254_A_001", None): "train",
        ("REAL/1004567_B_002", None): "test",
    }


def test_the_official_split_needs_its_file(dfdcp_root):
    builder = DFDCPreviewBuilder()
    with pytest.raises(ConfigError, match=r"dataset\.json") as caught:
        builder.official_splits(dfdcp_root, collect_records(builder, dfdcp_root))
    assert ".official_files" in caught.value.hint


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (["a.mp4"], "list"),
        ({"a/1_2_A_001.mp4": "train"}, "1_2_A_001"),
        ({"a/1_2_A_001.mp4": {"set": 1}}, "1_2_A_001"),
    ],
)
def test_malformed_metadata_is_a_contract_error(dfdcp_root, content, match):
    builder = DFDCPreviewBuilder()
    records = collect_records(builder, dfdcp_root)
    _write_dataset_json(dfdcp_root, content)
    with pytest.raises(ContractError, match=match):
        builder.official_splits(dfdcp_root, records)


def test_unreadable_metadata_is_a_contract_error(dfdcp_root):
    builder = DFDCPreviewBuilder()
    records = collect_records(builder, dfdcp_root)
    _write_dataset_json(dfdcp_root, {})
    (dfdcp_root / ".official_files" / "dataset.json").write_text("{oops", "utf-8")
    with pytest.raises(ContractError, match=r"dataset\.json"):
        builder.official_splits(dfdcp_root, records)


# ------------------------------------------------------------------------------ pairs, labels


def test_pair_candidates_name_the_target_clip(dfdcp_root):
    builder = DFDCPreviewBuilder()
    _touch(dfdcp_root.joinpath(*METHOD_A_DIR, "1004567"), "1_2_3", "1300001_1004567_B_009")
    records = collect_records(builder, dfdcp_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "FS_METHOD_A/1255229_1003254_A_001": "1003254_A_001",
        "FS_METHOD_A/1255230_1003254_A_002": "1003254_A_002",
        "FS_METHOD_A/1300001_1004567_B_009": "1004567_B_009",
        "FS_METHOD_A/1_2_3": None,
    }
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    # Only 1003254_A_001 has its real on disk.
    assert pairs == [
        PairRecord("FS_METHOD_A/1255229_1003254_A_001", "REAL/1003254_A_001", "target-clip")
    ]


def test_label_vocab_covers_every_task():
    builder = DFDCPreviewBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"DFDCP-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {
        "DFDCP-REAL": (0, 0, 1, "real"),
        "DFDCP-FS_METHOD_A": (1, 1, 2, "face-swap"),
        "DFDCP-FS_METHOD_B": (1, 1, 3, "face-swap"),
    }
    assert vocab["DFDCP-FS_METHOD_B"]["method"] == "method_b"


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = DFDCPreviewBuilder()
    assert builder.default_scheme == "official+ident-80-20"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official+ident-80-20": "official+ident-80-20",
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=5, strata=("identity", "task"))
    assert builder.pairing_rule == "target-clip"
    assert builder.pairing_fanout is None
    assert builder.metadata_files == (".official_files/dataset.json",)
    assert all(task.recursive for task in builder.tasks)


def test_the_benchmark_draws_from_the_official_test(dfdcp_root):
    _touch(
        dfdcp_root.joinpath(*METHOD_A_DIR, "1004567"), *(f"13{i}_1004567_B_00{i}" for i in "123")
    )
    _write_dataset_json(
        dfdcp_root,
        {
            **{f"m/13{i}_1004567_B_00{i}.mp4": _meta("test") for i in "123"},
            "o/1004567_B_002.mp4": _meta("test", "real"),
            "m/1255229_1003254_A_001.mp4": _meta("train"),
        },
    )
    builder = DFDCPreviewBuilder()
    records = collect_records(builder, dfdcp_root)
    official = builder.official_splits(dfdcp_root, records)
    assert builder.benchmark is not None
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=[key for key, split in official.items() if split == "test"],
    )
    # One (identity, task) stratum of three fakes (all kept, fewer than 5) and the one real.
    assert {key for key, _ in chosen} == {
        "FS_METHOD_A/131_1004567_B_001",
        "FS_METHOD_A/132_1004567_B_002",
        "FS_METHOD_A/133_1004567_B_003",
        "REAL/1004567_B_002",
    }


def test_dataset_card():
    builder = DFDCPreviewBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "dfdc-p"
    assert card.name == "DFDC Preview"
    assert "DFDC-P" in card.aliases
    assert card.compressions is None
    assert builder.known_compressions == ()
    assert card.default_scheme == "official+ident-80-20"
    assert card.paper is None


def test_the_layout_names_the_nesting_and_the_metadata():
    text = DFDCPreviewBuilder().describe_layout()
    assert "'DFDC-P'" in text
    assert "original_content/original/videos/**/<video>" in text
    assert ".official_files/dataset.json" in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("dfdc-p"), DFDCPreviewBuilder)
    # The raw release's folder; DFDCP is only the label prefix.
    assert DFDCPreviewBuilder.expected_folder == "DFDC-P"
    assert DFDCPreviewBuilder.label_prefix == "DFDCP"
