"""The DFDM inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.dfdm import DFDMBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_72_14_14,
    assign_benchmark,
    local_key,
    resolve_pairs,
    task_of,
)

REAL_DIR = ("original_content", "Celeb-real", "videos")


def _fake_dir(root: Path, model: str = "DFaker", compression: str = "c0") -> Path:
    return root / "manipulated_content" / model / compression / "videos"


def _make_synthetic_dfdm_root(tmp_path: Path) -> Path:
    """A tiny DFDM tree: two reals (no compression level) and two DFaker fakes in ``c0``."""
    root = tmp_path / "DFDM"
    real_dir = root.joinpath(*REAL_DIR)
    fake_dir = _fake_dir(root)
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    # Reals: <id>_<rec>.
    (real_dir / "id28_0008.mp4").write_bytes(b"\x00")
    (real_dir / "id28_0009.mp4").write_bytes(b"\x00")
    # Fakes: <target>_<source>_<rec>_<tag>; the first pairs with id28_0008.
    (fake_dir / "id28_id20_0008_4dfaker.mp4").write_bytes(b"\x00")
    (fake_dir / "id28_id21_0009_4dfaker.mp4").write_bytes(b"\x00")
    return root


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()


@pytest.fixture
def dfdm_root(tmp_path: Path) -> Path:
    return _make_synthetic_dfdm_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(DFDMBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(dfdm_root):
    # 2 reals + 2 fakes in one compression.
    assert len(_discover(dfdm_root)) == 4


def test_entry_schema(dfdm_root):
    for rec in _discover(dfdm_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"REAL", "FS_DF"}
        assert rec.builder == BuilderRef("dfdm", DFDMBuilder.version)
        assert rec.attrs == {"task_name": {"REAL": "Celeb-real", "FS_DF": "DFaker"}[task]}
        assert not rec.relpath.startswith("/")
        assert rec.folder is None


def test_no_media_field_and_no_label_class(dfdm_root):
    for rec in _discover(dfdm_root):
        assert rec.label_key in {"DFDM-REAL", "DFDM-FS_DF"}
        assert rec.probe is None
        assert not hasattr(rec, "media")
        assert not hasattr(rec, "label")


def test_pairing_attrs_present(dfdm_root):
    fakes = [r for r in _discover(dfdm_root) if task_of(r.key) != "REAL"]
    assert fakes
    paired = [r for r in fakes if r.pair_key]
    assert paired
    sample = paired[0]
    assert sample.target_id
    assert sample.source_id
    # pair_key is "<target_id>_<rec>"
    assert sample.pair_key.startswith(sample.target_id + "_")


def test_fields_parsed_from_the_names(dfdm_root):
    records = {rec.key: rec for rec in _discover(dfdm_root)}
    fake = records["FS_DF/id28_id20_0008_4dfaker"]
    assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
        "id28",
        "id28",
        "id20",
        "id28_0008",
    )
    # A fake's method is its model's name as the folder spells it.
    assert fake.method == "DFaker"
    real = records["REAL/id28_0008"]
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "id28",
        "id28",
        None,
        None,
    )
    assert real.method == "original"


def test_the_relpath_is_concrete_and_reals_have_no_compression(dfdm_root):
    records = {rec.key: rec for rec in _discover(dfdm_root)}
    fake = records["FS_DF/id28_id20_0008_4dfaker"]
    assert fake.relpath == "manipulated_content/DFaker/c0/videos/id28_id20_0008_4dfaker.mp4"
    assert fake.compression == "c0"
    real = records["REAL/id28_0008"]
    assert real.relpath == "original_content/Celeb-real/videos/id28_0008.mp4"
    assert real.compression is None
    assert all("{cX}" not in rec.relpath for rec in records.values())


def test_a_fake_is_found_once_per_compression(dfdm_root):
    for compression in ("c10", "c23"):
        _touch(_fake_dir(dfdm_root, compression=compression), "id28_id20_0008_4dfaker.mp4")
    found = [
        rec.compression for rec in _discover(dfdm_root) if rec.key == "FS_DF/id28_id20_0008_4dfaker"
    ]
    assert found == ["c0", "c10", "c23"]


def test_requested_compressions_filter_the_fakes_only(dfdm_root):
    _touch(_fake_dir(dfdm_root, compression="c23"), "id28_id20_0008_4dfaker.mp4")
    records = _discover(dfdm_root, compressions=["c23"])
    assert {(task_of(r.key), r.compression) for r in records} == {
        ("REAL", None),
        ("FS_DF", "c23"),
    }
    with pytest.raises(ConfigError, match="c40"):
        _discover(dfdm_root, compressions=["c40"])


def test_every_model_is_discovered_with_its_method(dfdm_root):
    for model, tag in (
        ("DFL-H128", "5dfl-h128"),
        ("FaceSwap", "1original"),
        ("IAE", "3iae"),
        ("LightWeight", "2lightweight"),
    ):
        _touch(_fake_dir(dfdm_root, model, "c10"), f"id1_id2_0000_{tag}.mp4")
    methods = {(task_of(r.key), r.method) for r in _discover(dfdm_root)}
    assert methods == {
        ("REAL", "original"),
        ("FS_DF", "DFaker"),
        ("FS_DFL", "DFL-H128"),
        ("FS_FS", "FaceSwap"),
        ("FS_IAE", "IAE"),
        ("FS_LW", "LightWeight"),
    }


def test_names_that_do_not_parse_are_kept_without_fields(dfdm_root):
    fake_dir = _fake_dir(dfdm_root)
    _touch(fake_dir, "id28_id20_0008.mp4", "id28_x20_0008_tag.mp4", "x28_id20_0008_tag.mp4")
    _touch(dfdm_root.joinpath(*REAL_DIR), "id28.mp4", "x28_0001.mp4")
    records = {rec.key: rec for rec in _discover(dfdm_root)}
    for key in (
        "FS_DF/id28_id20_0008",
        "FS_DF/id28_x20_0008_tag",
        "FS_DF/x28_id20_0008_tag",
        "REAL/id28",
        "REAL/x28_0001",
    ):
        rec = records[key]
        assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (None,) * 4, key
    # A fake of more than four parts still parses on its first three.
    _touch(fake_dir, "id3_id4_0001_5dfl_h128.mp4")
    extra = {rec.key: rec for rec in _discover(dfdm_root)}["FS_DF/id3_id4_0001_5dfl_h128"]
    assert (extra.identity, extra.source_id, extra.pair_key) == ("id3", "id4", "id3_0001")


def test_the_model_folders_are_flat(dfdm_root):
    _touch(_fake_dir(dfdm_root) / "nested", "id1_id2_0000_4dfaker.mp4")
    assert "FS_DF/id1_id2_0000_4dfaker" not in {rec.key for rec in _discover(dfdm_root)}


# ------------------------------------------------------------------------------ splits, pairs


def test_there_is_no_official_split(dfdm_root):
    builder = DFDMBuilder()
    with pytest.raises(ContractError, match="no official split"):
        builder.official_splits(dfdm_root, collect_records(builder, dfdm_root))


def test_the_carve_keeps_an_identity_on_one_side(dfdm_root):
    _touch(_fake_dir(dfdm_root, compression="c23"), "id28_id20_0008_4dfaker.mp4")
    records = _discover(dfdm_root)
    assignment = assign_72_14_14(records)
    person = [r for r in records if r.identity == "id28"]
    assert {(task_of(r.key), r.compression) for r in person} == {
        ("REAL", None),
        ("FS_DF", "c0"),
        ("FS_DF", "c23"),
    }
    assert len({assignment[(r.key, r.compression)] for r in person}) == 1


def test_pair_candidates_name_the_target_recording(dfdm_root):
    builder = DFDMBuilder()
    _touch(_fake_dir(dfdm_root), "id28_id20_0008.mp4", "id5_id6_0001_4dfaker.mp4")
    records = collect_records(builder, dfdm_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "FS_DF/id28_id20_0008_4dfaker": "id28_0008",
        "FS_DF/id28_id21_0009_4dfaker": "id28_0009",
        "FS_DF/id28_id20_0008": None,
        "FS_DF/id5_id6_0001_4dfaker": "id5_0001",
    }
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    # id5_0001 has no real, so that fake has no pair.
    assert pairs == [
        PairRecord("FS_DF/id28_id20_0008_4dfaker", "REAL/id28_0008", "target-recording"),
        PairRecord("FS_DF/id28_id21_0009_4dfaker", "REAL/id28_0009", "target-recording"),
    ]


def test_label_vocab_covers_every_task():
    builder = DFDMBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"DFDM-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {
        "DFDM-REAL": (0, 0, 1, "real"),
        "DFDM-FS_DF": (1, 1, 2, "face-swap"),
        "DFDM-FS_DFL": (1, 1, 3, "face-swap"),
        "DFDM-FS_FS": (1, 1, 4, "face-swap"),
        "DFDM-FS_IAE": (1, 1, 5, "face-swap"),
        "DFDM-FS_LW": (1, 1, 6, "face-swap"),
    }


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = DFDMBuilder()
    assert builder.default_scheme == "ident-72-14-14"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "ident-72-14-14": "ident-72-14-14",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=50, strata=("task",))
    assert builder.benchmark.compressions is None  # drawn across every compression
    assert builder.pairing_rule == "target-recording"
    assert builder.pairing_fanout is None


def test_the_benchmark_draws_per_model_across_compressions(dfdm_root):
    fakes = [f"id1_id2_{n:04d}_4dfaker.mp4" for n in range(40)]
    for compression in ("c0", "c10"):
        _touch(_fake_dir(dfdm_root, compression=compression), *fakes)
    _touch(_fake_dir(dfdm_root, "IAE", "c23"), "id1_id2_0000_3iae.mp4")
    builder = DFDMBuilder()
    records = collect_records(builder, dfdm_root)
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=None,
    )
    by_task = [task_of(key) for key, _ in chosen]
    # DFaker has 82 fakes over two compressions and keeps 50; IAE keeps its one; the two reals
    # are all kept.
    assert (by_task.count("FS_DF"), by_task.count("FS_IAE"), by_task.count("REAL")) == (50, 1, 2)
    assert {local_key(key) for key, _ in chosen if task_of(key) == "REAL"} == {
        "id28_0008",
        "id28_0009",
    }


def test_the_known_compressions_are_listed_in_name_order():
    # Only the declared constant is checked: listed in name order, discovery order and the
    # benchmark's tie order (by compression name) are the same.
    known = DFDMBuilder.known_compressions
    assert known == ("c0", "c10", "c23")
    assert list(known) == sorted(known)


def test_dataset_card():
    builder = DFDMBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "dfdm"
    assert card.name == "DFDM"
    assert card.compressions == ["c0", "c10", "c23"]
    assert card.default_scheme == "ident-72-14-14"
    assert card.homepage == "https://github.com/shanface33/Deepfake_Model_Attribution"
    assert card.paper is not None
    assert (card.paper.title, card.paper.venue, card.paper.year, card.paper.doi) == (
        "Model Attribution of Face-Swap Deepfake Videos",
        "ICIP",
        2022,
        None,
    )
    assert card.license.spdx is None


def test_the_layout_names_the_release_folders_and_the_reals_source():
    text = DFDMBuilder().describe_layout()
    assert "'DFDM'" in text
    assert "manipulated_content/DFaker/{cX}/videos/<video>" in text
    assert "original_content/Celeb-real/videos/<video>" in text
    assert "{cX} is the compression: c0, c10, c23." in text
    assert "DFDM_crf0" in text
    assert "Celeb-DF v2" in text


def test_it_is_registered():
    assert isinstance(get_builder("dfdm"), DFDMBuilder)
    assert DFDMBuilder.expected_folder == "DFDM"
    assert DFDMBuilder.label_prefix == "DFDM"
