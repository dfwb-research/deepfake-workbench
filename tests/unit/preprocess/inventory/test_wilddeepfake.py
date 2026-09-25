"""The WildDeepfake inventory builder, run on synthetic trees of empty files.

WildDeepfake has no videos: each record is a directory of pre-cropped face frames. No test here
reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError
from dfwb.core.records import BuilderRef, SchemeCard
from dfwb.preprocess.inventory.builders.wilddeepfake import WildDeepfakeBuilder
from dfwb.preprocess.inventory.runner import (
    DatasetCopy,
    collect_records,
    get_builder,
    metadata_copy,
)
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    assign_official_plus_80_20,
    local_key,
    md5_mod_100,
    task_of,
)

_OG = "224w_224h_wild_precropped"
REAL_DIR = f"original_content/real/frames/{_OG}"
FAKE_DIR = f"manipulated_content/fake/frames/{_OG}"

# (sequence folder, key, n_frames)
_SEQS = [
    (REAL_DIR, "real_train_6_54", 3),
    (REAL_DIR, "real_test_104_10", 2),
    (FAKE_DIR, "fake_train_100_0", 4),
    (FAKE_DIR, "fake_test_100_0", 2),
]


def _make_synthetic_wdf_root(tmp_path: Path) -> Path:
    """A tiny WildDeepfake tree: two real and two fake sequences of PNG frames."""
    root = tmp_path / "WildDeepfake"
    for folder, key, n_frames in _SEQS:
        seq_dir = root / folder / key
        seq_dir.mkdir(parents=True)
        for i in range(n_frames):
            (seq_dir / f"{i:06d}.png").write_bytes(b"\x89PNG\x00")
    return root


@pytest.fixture
def wdf_root(tmp_path: Path) -> Path:
    return _make_synthetic_wdf_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(WildDeepfakeBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_one_entry_per_sequence(wdf_root):
    records = _discover(wdf_root)
    assert len(records) == 4
    assert {local_key(r.key) for r in records} == {key for _, key, _ in _SEQS}


def test_entry_schema(wdf_root):
    for rec in _discover(wdf_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert rec.label_key == f"WDF-{task}"
        assert rec.builder == BuilderRef("wilddeepfake", WildDeepfakeBuilder.version)
        assert rec.compression is None
        assert rec.folder is None
        assert not rec.relpath.startswith("/")


def test_no_media_or_label_field(wdf_root):
    for rec in _discover(wdf_root):
        assert rec.probe is None
        assert not hasattr(rec, "label")
        assert not hasattr(rec, "media")


def test_the_relpath_names_the_sequence_directory(wdf_root):
    records = {rec.key: rec for rec in _discover(wdf_root)}
    assert records["REAL/real_train_6_54"].relpath == f"{REAL_DIR}/real_train_6_54"
    assert records["FAKE/fake_test_100_0"].relpath == f"{FAKE_DIR}/fake_test_100_0"
    assert (wdf_root / records["REAL/real_train_6_54"].relpath).is_dir()


def test_split_and_counts_from_key(wdf_root):
    by_key = {local_key(r.key): r for r in _discover(wdf_root)}
    assert by_key["real_train_6_54"].attrs["split"] == "train"
    assert by_key["real_test_104_10"].attrs["split"] == "test"
    assert by_key["fake_train_100_0"].attrs["split"] == "train"
    # shard, sequence and frame count
    assert by_key["fake_train_100_0"].attrs == {
        "task_name": "fake",
        "split": "train",
        "shard": "100",
        "sequence": "0",
        "n_frames": 4,
    }


def test_no_source_or_pairing(wdf_root):
    """Wild fakes have no known source or donor, and nothing pairs: all four fields are None."""
    for rec in _discover(wdf_root):
        assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (None,) * 4


def test_real_vs_fake_method(wdf_root):
    by_key = {local_key(r.key): r for r in _discover(wdf_root)}
    assert by_key["real_train_6_54"].method == "original"
    assert by_key["fake_train_100_0"].method == "fake"
    assert task_of(by_key["real_train_6_54"].key) == "REAL"
    assert task_of(by_key["fake_train_100_0"].key) == "FAKE"


def test_names_that_do_not_parse_keep_empty_attributes(wdf_root):
    for key in ("odd", "real_val_1", "real_train"):
        (wdf_root / REAL_DIR / key).mkdir()
    by_key = {local_key(r.key): r for r in _discover(wdf_root)}
    assert by_key["odd"].attrs == {
        "task_name": "real",
        "split": None,
        "shard": None,
        "sequence": None,
        "n_frames": 0,
    }
    # Only "train" and "test" are splits; the shard and sequence are taken as they come.
    assert (by_key["real_val_1"].attrs["split"], by_key["real_val_1"].attrs["shard"]) == (None, "1")
    assert by_key["real_train"].attrs["split"] == "train"
    assert by_key["real_train"].attrs["shard"] is None


def test_only_directories_are_records_and_only_png_files_are_frames(wdf_root):
    (wdf_root / REAL_DIR / "stray.txt").touch()
    (wdf_root / REAL_DIR / ".hidden_seq").mkdir()
    seq = wdf_root / REAL_DIR / "real_train_6_54"
    (seq / "notes.txt").touch()
    (seq / "000009.PNG").touch()
    (seq / ".000010.png").touch()
    by_key = {local_key(r.key): r for r in _discover(wdf_root)}
    assert set(by_key) == {key for _, key, _ in _SEQS}
    # 3 frames, plus the upper-case .PNG; the text file and the hidden file are not frames.
    assert by_key["real_train_6_54"].attrs["n_frames"] == 4


def test_a_symlinked_sequence_directory_is_followed(wdf_root, tmp_path):
    elsewhere = tmp_path / "elsewhere" / "fake_test_7_1"
    elsewhere.mkdir(parents=True)
    (elsewhere / "000000.png").touch()
    (wdf_root / FAKE_DIR / "fake_test_7_1").symlink_to(elsewhere, target_is_directory=True)
    rec = {r.key: r for r in _discover(wdf_root)}["FAKE/fake_test_7_1"]
    assert rec.relpath == f"{FAKE_DIR}/fake_test_7_1"
    assert rec.attrs["n_frames"] == 1


def test_no_compression_can_be_requested(wdf_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        _discover(wdf_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ layout, copies


def test_the_layout_is_a_sequence_directory_holding_frames(tmp_path, wdf_root):
    builder = WildDeepfakeBuilder()
    assert builder.layout_dirs() == (REAL_DIR, FAKE_DIR)
    assert builder.layout_present(wdf_root)
    assert builder.videos_present(wdf_root) == (REAL_DIR, FAKE_DIR)

    # A processed copy with the folder names but no sequence directories, or only empty ones,
    # does not hold the raw layout.
    processed = tmp_path / "processed" / "WildDeepfake"
    (processed / REAL_DIR).mkdir(parents=True)
    (processed / FAKE_DIR / "fake_train_1_1").mkdir(parents=True)
    (processed / "original_content" / "real" / "frames" / "other_group" / "seq").mkdir(parents=True)
    (processed / "original_content" / "real" / "frames" / "other_group" / "seq" / "0.png").touch()
    assert not builder.layout_present(processed)
    assert builder.videos_present(processed) == ()

    # One task's sequences are enough.
    (processed / FAKE_DIR / "fake_train_1_1" / "000000.png").touch()
    assert builder.layout_present(processed)
    assert builder.videos_present(processed) == (FAKE_DIR,)


def test_copies_are_merged_per_task(tmp_path):
    """Reals in one copy and fakes in another: each task is scanned from the copy holding it."""
    first = tmp_path / "a" / "WildDeepfake"
    second = tmp_path / "b" / "WildDeepfake"
    (first / REAL_DIR / "real_test_1_1").mkdir(parents=True)
    (first / REAL_DIR / "real_test_1_1" / "000000.png").touch()
    (first / FAKE_DIR / "fake_test_9_9").mkdir(parents=True)  # empty: not a sequence of frames
    (second / FAKE_DIR / "fake_test_2_2").mkdir(parents=True)
    (second / FAKE_DIR / "fake_test_2_2" / "000000.png").touch()
    (second / REAL_DIR / "real_test_3_3").mkdir(parents=True)
    (second / REAL_DIR / "real_test_3_3" / "000000.png").touch()  # shadowed by the first copy

    builder = WildDeepfakeBuilder()
    copies = (first, second)
    chosen = builder.choose_copies(copies)
    assert chosen == {("REAL", None): first, ("FAKE", None): second}
    assert builder.copies_after(copies, first, builder.tasks[0], None) == [second]
    assert builder.copies_after(copies, second, builder.tasks[1], None) == []
    builder.bind_copies(copies, chosen)
    meta = metadata_copy(
        builder,
        [
            DatasetCopy(c, "test", builder.layout_present(c), builder.videos_present(c))
            for c in copies
        ],
    )
    assert meta.path == first
    assert [r.key for r in collect_records(builder, meta.path)] == [
        "FAKE/fake_test_2_2",
        "REAL/real_test_1_1",
    ]


def test_unbound_discovery_scans_the_given_root(wdf_root, tmp_path):
    builder = WildDeepfakeBuilder()
    assert len(list(builder.discover(wdf_root))) == 4
    assert list(builder.discover(tmp_path / "nothing")) == []


def test_the_layout_says_the_records_are_frame_directories():
    text = WildDeepfakeBuilder().describe_layout()
    assert "'WildDeepfake'" in text
    assert "frame directories" in text.lower()
    assert f"{REAL_DIR}/<label>_<split>_<shard>_<sequence>/<nnnnnn>.png" in text
    assert f"{FAKE_DIR}/<label>_<split>_<shard>_<sequence>/<nnnnnn>.png" in text
    assert "<video>" not in text
    # How the release's tar shards map onto that layout.
    assert "real_train, real_test, fake_train and fake_test" in text
    assert "<shard>.tar.gz" in text
    assert "<shard>/<label>/<sequence>/<frame>.png" in text
    assert "the shard file's name up to its first '.'" in text
    assert "zero-padded to six digits" in text
    assert "1919.png becomes 001919.png" in text
    assert "plain tar archives despite their .tar.gz names" in text
    assert "tar xf, not tar xzf" in text
    assert "keeps its stem, and its extension is written as lower-case .png" in text
    assert "The official split is the <split> in each folder name: train or test." in text
    assert "{cX}" not in text


# ------------------------------------------------------------------------------ official split


def test_the_official_split_is_the_split_named_in_each_key(wdf_root):
    (wdf_root / REAL_DIR / "real_val_1_1").mkdir()
    builder = WildDeepfakeBuilder()
    official = builder.official_splits(wdf_root, collect_records(builder, wdf_root))
    assert official == {
        "REAL/real_train_6_54": "train",
        "REAL/real_test_104_10": "test",
        "FAKE/fake_train_100_0": "train",
        "FAKE/fake_test_100_0": "test",
        # real_val_1_1 names no published split, so it is left out.
    }


def test_the_default_scheme_carves_val_from_the_official_train_by_key(wdf_root):
    builder = WildDeepfakeBuilder()
    records = collect_records(builder, wdf_root)
    official = builder.official_splits(wdf_root, records)
    scheme = builder.schemes[builder.default_scheme]
    assert scheme.rule == "official+ident-80-20"
    assert scheme.params == {"policy": "official-train-test"}

    def carved(key: str) -> str:
        # No identity: the carve hashes the sequence's own key.
        return "val" if md5_mod_100(key) < 20 else "train"

    assert assign_official_plus_80_20(records, official, policy="official-train-test") == {
        ("REAL/real_train_6_54", None): carved("real_train_6_54"),
        ("FAKE/fake_train_100_0", None): carved("fake_train_100_0"),
        ("REAL/real_test_104_10", None): "test",
        ("FAKE/fake_test_100_0", None): "test",
    }
    assert assign_official(records, official) == {
        ("REAL/real_train_6_54", None): "train",
        ("FAKE/fake_train_100_0", None): "train",
        ("REAL/real_test_104_10", None): "test",
        ("FAKE/fake_test_100_0", None): "test",
    }


# ------------------------------------------------------------------------------ pairs, labels


def test_nothing_pairs(wdf_root):
    builder = WildDeepfakeBuilder()
    assert builder.pairing_rule is None
    for rec in collect_records(builder, wdf_root):
        assert builder.pair_candidates(rec) is None


def test_label_vocab_covers_every_task():
    builder = WildDeepfakeBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"WDF-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    # The fakes come from the internet and their generator is not recorded.
    assert table == {"WDF-REAL": (0, 0, 1, "real"), "WDF-FAKE": (1, 1, 2, "unknown")}
    assert vocab["WDF-FAKE"]["method"] == "fake"


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = WildDeepfakeBuilder()
    assert builder.default_scheme == "official+ident-80-20"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official+ident-80-20": "official+ident-80-20",
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=100)
    assert builder.pairing_fanout is None
    assert builder.metadata_files == ()


def test_the_benchmark_draws_from_the_official_test(wdf_root):
    builder = WildDeepfakeBuilder()
    records = collect_records(builder, wdf_root)
    official = builder.official_splits(wdf_root, records)
    assert builder.benchmark is not None
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=[key for key, split in official.items() if split == "test"],
    )
    assert {key for key, _ in chosen} == {"REAL/real_test_104_10", "FAKE/fake_test_100_0"}


def test_dataset_card():
    builder = WildDeepfakeBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "wilddeepfake"
    assert card.name == "WildDeepfake"
    assert "WDF" in card.aliases
    assert card.compressions is None
    assert builder.known_compressions == ()
    assert card.default_scheme == "official+ident-80-20"
    assert card.paper is None
    assert card.license.spdx is None
    assert card.license.summary == (
        "access is gated by the authors; the release's README front matter declares "
        "apache-2.0; the access terms need review"
    )


def test_it_is_registered():
    assert isinstance(get_builder("wilddeepfake"), WildDeepfakeBuilder)
    assert WildDeepfakeBuilder.expected_folder == "WildDeepfake"
    assert WildDeepfakeBuilder.label_prefix == "WDF"
