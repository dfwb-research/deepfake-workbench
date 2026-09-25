"""The DFDC inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, SchemeCard
from dfwb.preprocess.inventory.builders.dfdc import DFDCBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    local_key,
    resolve_pairs,
    task_of,
)

REAL_DIR = ("original_content", "testing_real", "videos")
FAKE_DIR = ("manipulated_content", "testing_fake", "videos")


def _make_synthetic_dfdc_root(tmp_path: Path) -> Path:
    """A tiny DFDC tree: 2 reals and 3 fakes, flat, named by opaque 10-letter ids."""
    root = tmp_path / "DFDC"
    real_dir = root.joinpath(*REAL_DIR)
    fake_dir = root.joinpath(*FAKE_DIR)
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    (real_dir / "aamrozxzsq.mp4").write_bytes(b"\x00")
    (real_dir / "bbnsozyybt.mp4").write_bytes(b"\x00")
    # Fakes: no identity and no link to a real in the name.
    (fake_dir / "ccolpaaepe.mp4").write_bytes(b"\x00")
    (fake_dir / "ddooqzxxsa.mp4").write_bytes(b"\x00")
    (fake_dir / "eebbnnmzla.mp4").write_bytes(b"\x00")
    return root


def _write_metadata(root: Path, content: object) -> None:
    """The challenge's test metadata: a JSON object keyed by video file name."""
    folder = root / ".official_files"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "testing_metadata.json").write_text(json.dumps(content, indent=1), "utf-8")


def _entry(is_fake: int) -> dict:
    return {"augmentations": {"augmenter": {"framerate_change": {"fps": 20}}}, "is_fake": is_fake}


@pytest.fixture
def dfdc_root(tmp_path: Path) -> Path:
    return _make_synthetic_dfdc_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(DFDCBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(dfdc_root):
    records = _discover(dfdc_root)
    assert len(records) == 5
    assert len([r for r in records if task_of(r.key) == "REAL"]) == 2
    assert len([r for r in records if task_of(r.key) == "FS_FAKE"]) == 3


def test_entry_schema(dfdc_root):
    for rec in _discover(dfdc_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"REAL", "FS_FAKE"}
        assert rec.builder == BuilderRef("dfdc", DFDCBuilder.version)
        assert rec.attrs == {"task_name": "testing_real" if task == "REAL" else "testing_fake"}
        assert not rec.relpath.startswith("/")
        assert rec.folder is None


def test_no_media_or_label_field(dfdc_root):
    for rec in _discover(dfdc_root):
        assert rec.probe is None
        assert not hasattr(rec, "label")
        assert not hasattr(rec, "media")


def test_pairing_attrs_all_none(dfdc_root):
    # The names carry no identity and no link between a fake and a real.
    for rec in _discover(dfdc_root):
        assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (None,) * 4


def test_compression_always_null(dfdc_root):
    records = {rec.key: rec for rec in _discover(dfdc_root)}
    for rec in records.values():
        assert rec.compression is None
        assert "{cX}" not in rec.relpath
    assert records["REAL/aamrozxzsq"].relpath == (
        "original_content/testing_real/videos/aamrozxzsq.mp4"
    )
    assert records["FS_FAKE/ccolpaaepe"].relpath == (
        "manipulated_content/testing_fake/videos/ccolpaaepe.mp4"
    )


def test_label_key_and_method(dfdc_root):
    for rec in _discover(dfdc_root):
        assert rec.label_key == f"DFDC-{task_of(rec.key)}"
        if task_of(rec.key) == "REAL":
            assert rec.method == "original"
        else:
            assert rec.method == "deepfake"


def test_no_compression_can_be_requested(dfdc_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        _discover(dfdc_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ official split


def test_the_official_split_is_the_test_metadata(dfdc_root):
    _write_metadata(
        dfdc_root,
        {
            "aamrozxzsq.mp4": _entry(0),
            "ccolpaaepe.mp4": _entry(1),
            "ddooqzxxsa.mp4": _entry(0),  # the label in the file is not consulted
            "zzzzzzzzzz.mp4": _entry(1),  # listed, but not on disk: ignored
        },
    )
    builder = DFDCBuilder()
    records = collect_records(builder, dfdc_root)
    official = builder.official_splits(dfdc_root, records)
    # Only test exists; an unlisted video is left unassigned.
    assert official == {
        "REAL/aamrozxzsq": "test",
        "FS_FAKE/ccolpaaepe": "test",
        "FS_FAKE/ddooqzxxsa": "test",
    }
    assignment = assign_official(records, official)
    assert set(assignment.values()) == {"test"}
    assert ("REAL/bbnsozyybt", None) not in assignment


def test_the_metadata_is_matched_by_file_stem(dfdc_root):
    _write_metadata(dfdc_root, {"test/bbnsozyybt.mp4": _entry(0), "eebbnnmzla": _entry(1)})
    builder = DFDCBuilder()
    official = builder.official_splits(dfdc_root, collect_records(builder, dfdc_root))
    assert official == {"REAL/bbnsozyybt": "test", "FS_FAKE/eebbnnmzla": "test"}


def test_the_official_split_needs_the_metadata(dfdc_root):
    builder = DFDCBuilder()
    with pytest.raises(ConfigError, match=r"testing_metadata\.json") as caught:
        builder.official_splits(dfdc_root, collect_records(builder, dfdc_root))
    assert ".official_files" in caught.value.hint


def test_malformed_metadata_is_a_contract_error(dfdc_root):
    builder = DFDCBuilder()
    records = collect_records(builder, dfdc_root)
    _write_metadata(dfdc_root, ["aamrozxzsq.mp4"])
    with pytest.raises(ContractError, match="list"):
        builder.official_splits(dfdc_root, records)
    (dfdc_root / ".official_files" / "testing_metadata.json").write_text("{oops", "utf-8")
    with pytest.raises(ContractError, match=r"testing_metadata\.json"):
        builder.official_splits(dfdc_root, records)


# ------------------------------------------------------------------------------ pairs, labels


def test_nothing_pairs(dfdc_root):
    builder = DFDCBuilder()
    records = collect_records(builder, dfdc_root)
    assert builder.pairing_rule is None
    assert all(builder.pair_candidates(r) is None for r in records if not builder.is_real(r))
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule="none",
    )
    assert pairs == []


def test_label_vocab_covers_every_task():
    builder = DFDCBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"DFDC-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {"DFDC-REAL": (0, 0, 1, "real"), "DFDC-FS_FAKE": (1, 1, 2, "face-swap")}
    assert vocab["DFDC-FS_FAKE"]["method"] == "deepfake"


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = DFDCBuilder()
    assert builder.default_scheme == "official"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=1000)
    assert builder.metadata_files == (".official_files/testing_metadata.json",)


def test_the_benchmark_draws_from_the_official_test(dfdc_root):
    _write_metadata(
        dfdc_root,
        {"aamrozxzsq.mp4": _entry(0), "ccolpaaepe.mp4": _entry(1), "ddooqzxxsa.mp4": _entry(1)},
    )
    builder = DFDCBuilder()
    records = collect_records(builder, dfdc_root)
    official = builder.official_splits(dfdc_root, records)
    assert builder.benchmark is not None
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=[key for key, split in official.items() if split == "test"],
    )
    # Two fakes and one real in the pool: all fewer than k, so all are kept.
    assert {local_key(key) for key, _ in chosen} == {"aamrozxzsq", "ccolpaaepe", "ddooqzxxsa"}


def test_dataset_card():
    builder = DFDCBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "dfdc"
    assert card.name == "DFDC"
    assert card.compressions is None
    assert builder.known_compressions == ()
    assert card.default_scheme == "official"
    assert card.paper is None


def test_the_layout_names_the_metadata():
    text = DFDCBuilder().describe_layout()
    assert "original_content/testing_real/videos" in text
    assert ".official_files/testing_metadata.json" in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("dfdc"), DFDCBuilder)
    assert DFDCBuilder.expected_folder == "DFDC"
