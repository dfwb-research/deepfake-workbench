"""The UADFV inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.uadfv import UADFVBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import local_key, resolve_pairs, task_of

REAL_DIR = ("original_content", "real", "videos")
FAKE_DIR = ("manipulated_content", "fake", "videos")


def _make_synthetic_uadfv_root(tmp_path: Path) -> Path:
    """A tiny UADFV tree: three reals and their three fakes, with no compression levels."""
    root = tmp_path / "UADFV"
    real_dir = root.joinpath(*REAL_DIR)
    fake_dir = root.joinpath(*FAKE_DIR)
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    for n in ("0001", "0002", "0003"):
        (real_dir / f"{n}.mp4").write_bytes(b"\x00")  # reals: <n>
        (fake_dir / f"{n}_fake.mp4").write_bytes(b"\x00")  # fakes: <n>_fake
    return root


@pytest.fixture
def uadfv_root(tmp_path: Path) -> Path:
    return _make_synthetic_uadfv_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(UADFVBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(uadfv_root):
    # 3 real + 3 fake stubs
    assert len(_discover(uadfv_root)) == 6


def test_entry_schema(uadfv_root):
    for rec in _discover(uadfv_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"REAL", "FS_FAKE"}
        assert rec.builder == BuilderRef("uadfv", UADFVBuilder.version)
        assert rec.attrs["task_name"] in {"real", "fake"}
        assert not rec.relpath.startswith("/")
        # UADFV has a single version: no compression
        assert rec.compression is None


def test_records_carry_a_label_key_but_no_label_or_media(uadfv_root):
    for rec in _discover(uadfv_root):
        assert rec.label_key in {"UADFV-REAL", "UADFV-FS_FAKE"}
        assert rec.probe is None
        assert not hasattr(rec, "label")
        assert not hasattr(rec, "media")


def test_pair_key_strips_fake_suffix(uadfv_root):
    records = _discover(uadfv_root)
    fakes = [r for r in records if task_of(r.key) != "REAL"]
    assert fakes
    real_keys = {local_key(r.key) for r in records if task_of(r.key) == "REAL"}
    for rec in fakes:
        pair_key = rec.pair_key
        assert pair_key
        assert local_key(rec.key) == f"{pair_key}_fake"
        assert pair_key in real_keys  # it names a real of this tree
        assert rec.target_id == pair_key
        assert rec.identity == pair_key
        assert rec.source_id is None  # UADFV records no swap source
        assert rec.method == "faceswap"


def test_real_entries_have_null_pair_key(uadfv_root):
    reals = [r for r in _discover(uadfv_root) if task_of(r.key) == "REAL"]
    assert reals
    for rec in reals:
        assert rec.pair_key is None
        assert rec.target_id is None
        assert rec.source_id is None
        assert rec.identity == local_key(rec.key)
        assert rec.method == "original"
        assert rec.label_key == "UADFV-REAL"


def test_relpath_is_relative_and_well_formed(uadfv_root):
    for rec in _discover(uadfv_root):
        assert rec.relpath.endswith(".mp4")
        if task_of(rec.key) == "REAL":
            assert rec.relpath == f"original_content/real/videos/{local_key(rec.key)}.mp4"
        else:
            assert rec.relpath == f"manipulated_content/fake/videos/{local_key(rec.key)}.mp4"
        assert "{cX}" not in rec.relpath


def test_a_fake_not_named_n_fake_is_kept_without_fields(uadfv_root):
    (uadfv_root.joinpath(*FAKE_DIR) / "0004.mp4").touch()
    rec = {r.key: r for r in _discover(uadfv_root)}["FS_FAKE/0004"]
    assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (None,) * 4


def test_no_compression_can_be_requested(uadfv_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        _discover(uadfv_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ splits, pairs


def test_there_is_no_official_split(uadfv_root):
    builder = UADFVBuilder()
    with pytest.raises(ContractError, match="no official split"):
        builder.official_splits(uadfv_root, collect_records(builder, uadfv_root))


def test_pair_candidates_strip_the_fake_suffix(uadfv_root):
    builder = UADFVBuilder()
    (uadfv_root.joinpath(*FAKE_DIR) / "0004.mp4").touch()
    records = collect_records(builder, uadfv_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "FS_FAKE/0001_fake": "0001",
        "FS_FAKE/0002_fake": "0002",
        "FS_FAKE/0003_fake": "0003",
        "FS_FAKE/0004": None,
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
        PairRecord(f"FS_FAKE/{n}_fake", f"REAL/{n}", "strip-fake-suffix")
        for n in ("0001", "0002", "0003")
    ]


def test_label_vocab_covers_every_task():
    builder = UADFVBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"UADFV-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {"UADFV-REAL": (0, 0, 1, "real"), "UADFV-FS_FAKE": (1, 1, 2, "face-swap")}


# ------------------------------------------------------------------------------ schemes, card


def test_every_video_is_test_and_there_is_no_benchmark():
    builder = UADFVBuilder()
    assert builder.default_scheme == "all-test"
    assert {name: s.rule for name, s in builder.schemes.items()} == {"all-test": "all-test"}
    assert builder.benchmark is None
    assert builder.pairing_rule == "strip-fake-suffix"
    assert builder.pairing_fanout is None


def test_dataset_card():
    builder = UADFVBuilder()
    card = builder.dataset_card({"all-test": SchemeCard(kind="subset", sha256="a" * 64)})
    assert card.id == "uadfv"
    assert card.name == "UADFV"
    assert card.compressions is None
    assert builder.known_compressions == ()
    assert card.default_scheme == "all-test"
    assert card.paper is not None
    assert (card.paper.title, card.paper.venue, card.paper.year, card.paper.doi) == (
        "Exposing Deep Fakes Using Inconsistent Head Poses",
        "ICASSP",
        2019,
        "10.1109/ICASSP.2019.8683164",
    )
    assert card.homepage is None
    # The owners' agreement form is the only way to the release.
    form = (
        "https://docs.google.com/forms/d/e/"
        "1FAIpQLScKPoOv15TIZ9Mn0nGScIVgKRM9tFWOmjh9eHKx57Yp-XcnxA/viewform"
    )
    assert f"({form});" in card.access
    assert card.license.summary == (
        "the Terms to use UADFV, agreed in the owners' agreement form; the terms need review"
    )


def test_the_layout_names_both_folders():
    text = UADFVBuilder().describe_layout()
    assert "original_content/real/videos" in text
    assert "manipulated_content/fake/videos" in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("uadfv"), UADFVBuilder)
