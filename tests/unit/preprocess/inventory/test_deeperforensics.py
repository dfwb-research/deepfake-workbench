"""The DeeperForensics-1.0 inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.deeperforensics import DeeperForensicsBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    local_key,
    resolve_pairs,
    task_of,
)

REAL_DIR = ("original_content", "source_videos", "videos")
E2E_DIR = ("manipulated_content", "end_to_end", "videos")

# (task abbr, name, folder under manipulated_content/), in the release's order.
FAKE_TASKS = (
    ("FS_E2E", "End-to-End", "end_to_end"),
    ("FS_L1", "E2E-Level-1", "end_to_end_level_1"),
    ("FS_L2", "E2E-Level-2", "end_to_end_level_2"),
    ("FS_L3", "E2E-Level-3", "end_to_end_level_3"),
    ("FS_L4", "E2E-Level-4", "end_to_end_level_4"),
    ("FS_L5", "E2E-Level-5", "end_to_end_level_5"),
    ("FS_RND", "E2E-Random", "end_to_end_random_level"),
    ("FS_M2", "E2E-Mix-2", "end_to_end_mix_2_distortions"),
    ("FS_M3", "E2E-Mix-3", "end_to_end_mix_3_distortions"),
    ("FS_M4", "E2E-Mix-4", "end_to_end_mix_4_distortions"),
    ("FR_RP", "Reenact-Post", "reenact_postprocess"),
)


def _make_synthetic_defo_root(tmp_path: Path) -> Path:
    """A tiny DeeperForensics-1.0 tree: two nested reals and two flat fakes.

    A real's last folder (``camera_*``) is a directory, and its file name repeats the whole
    descriptive path, as in the release.
    """
    root = tmp_path / "DeeperForensics-1.0"
    real_base = root.joinpath(*REAL_DIR)
    fake_dir = root.joinpath(*E2E_DIR)

    # Nested real videos.
    m101 = real_base / "M101" / "light_down" / "contempt" / "camera_front"
    m101.mkdir(parents=True)
    (m101 / "M101_light_down_contempt_camera_front.mp4").write_bytes(b"\x00")
    w006 = real_base / "W006" / "light_uniform" / "neutral" / "camera_front"
    w006.mkdir(parents=True)
    (w006 / "W006_light_uniform_neutral_camera_front.mp4").write_bytes(b"\x00")

    # Flat fake videos.
    fake_dir.mkdir(parents=True)
    (fake_dir / "000_M101.mp4").write_bytes(b"\x00")
    (fake_dir / "001_W006.mp4").write_bytes(b"\x00")
    return root


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()


def _write_split_lists(root: Path, **lines: list[str]) -> None:
    """The release's split lists: one face-swap file name per line."""
    folder = root / ".official_files" / "lists" / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        text = "".join(f"{line}\n" for line in lines.get(split, []))
        (folder / f"{split}.txt").write_text(text, encoding="utf-8")


def _official(root: Path) -> dict[str, str]:
    builder = DeeperForensicsBuilder()
    return dict(builder.official_splits(root, collect_records(builder, root)))


@pytest.fixture
def defo_root(tmp_path: Path) -> Path:
    return _make_synthetic_defo_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(DeeperForensicsBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(defo_root):
    # 2 reals + 2 fakes
    assert len(_discover(defo_root)) == 4


def test_entry_schema(defo_root):
    for rec in _discover(defo_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"SR", "FS_E2E"}
        assert rec.builder == BuilderRef("deeperforensics", DeeperForensicsBuilder.version)
        assert rec.attrs["task_name"] in {"Source-Real", "End-to-End"}
        assert not rec.relpath.startswith("/")
        # DeeperForensics-1.0 has a single version: no compression
        assert rec.compression is None
        assert rec.folder is None


def test_no_media_no_label_class(defo_root):
    for rec in _discover(defo_root):
        assert rec.label_key == f"DeFo-{task_of(rec.key)}"
        assert rec.probe is None
        assert not hasattr(rec, "label")
        assert not hasattr(rec, "media")


def test_real_key_is_the_file_stem_not_the_joined_path(defo_root):
    reals = [r for r in _discover(defo_root) if task_of(r.key) == "SR"]
    assert {local_key(r.key) for r in reals} == {
        "M101_light_down_contempt_camera_front",
        "W006_light_uniform_neutral_camera_front",
    }
    by_key = {local_key(r.key): r for r in reals}
    # The identity is the leading actor token.
    assert by_key["M101_light_down_contempt_camera_front"].identity == "M101"
    assert by_key["W006_light_uniform_neutral_camera_front"].identity == "W006"


def test_real_entry_attributes(defo_root):
    reals = [r for r in _discover(defo_root) if task_of(r.key) == "SR"]
    assert reals
    for rec in reals:
        assert rec.target_id == rec.identity
        assert rec.source_id is None
        assert rec.pair_key is None
        assert rec.method == "original"
        assert rec.attrs == {"task_name": "Source-Real"}


def test_fake_pairing_attrs_present(defo_root):
    fakes = [r for r in _discover(defo_root) if task_of(r.key) != "SR"]
    by_key = {local_key(r.key): r for r in fakes}
    assert set(by_key) == {"000_M101", "001_W006"}
    # A fake "<n>_<actor>" names the actor whose face it shows.
    assert (by_key["000_M101"].identity, by_key["000_M101"].target_id) == ("M101", "M101")
    assert (by_key["001_W006"].identity, by_key["001_W006"].target_id) == ("W006", "W006")
    # No separate source identity, and pairing goes through the identity, not pair_key.
    for rec in fakes:
        assert rec.source_id is None
        assert rec.pair_key is None
        assert rec.method == "End-to-End"
        assert rec.attrs == {"task_name": "End-to-End"}


def test_real_and_fake_share_identity(defo_root):
    records = _discover(defo_root)
    real_ids = {r.identity for r in records if task_of(r.key) == "SR"}
    fake_ids = {r.identity for r in records if task_of(r.key) != "SR"}
    assert fake_ids <= real_ids


def test_relpaths_keep_the_real_nesting(defo_root):
    records = {rec.key: rec for rec in _discover(defo_root)}
    assert records["SR/M101_light_down_contempt_camera_front"].relpath == (
        "original_content/source_videos/videos/M101/light_down/contempt/camera_front/"
        "M101_light_down_contempt_camera_front.mp4"
    )
    assert (
        records["FS_E2E/000_M101"].relpath == "manipulated_content/end_to_end/videos/000_M101.mp4"
    )


def test_fakes_are_flat_and_reals_nested(defo_root):
    # A fake in a sub-folder is not scanned; a real at any depth is.
    _touch(defo_root.joinpath(*E2E_DIR, "extra"), "002_M101.mp4")
    _touch(defo_root.joinpath(*REAL_DIR, "M101", "BlendShape", "camera_down"), "M101_x.mp4")
    keys = {rec.key for rec in _discover(defo_root)}
    assert "FS_E2E/002_M101" not in keys
    assert "SR/M101_x" in keys


def test_every_task_is_discovered_with_its_method(defo_root):
    for _, _, folder in FAKE_TASKS[1:]:
        _touch(defo_root / "manipulated_content" / folder / "videos", "000_M101.mp4")
    records = {rec.key: rec for rec in _discover(defo_root)}
    for abbr, name, _ in FAKE_TASKS:
        rec = records[f"{abbr}/000_M101"]
        assert rec.method == name
        assert rec.label_key == f"DeFo-{abbr}"
        assert rec.attrs == {"task_name": name}
        assert rec.identity == "M101"
    assert [t.abbr for t in DeeperForensicsBuilder.tasks] == ["SR"] + [a for a, _, _ in FAKE_TASKS]


def test_unusual_names_keep_the_old_parsing(defo_root):
    _touch(defo_root.joinpath(*E2E_DIR), "nounderscore.mp4", "7_M101_extra.mp4")
    _touch(defo_root.joinpath(*REAL_DIR, "W006"), "W006.mp4")
    records = {rec.key: rec for rec in _discover(defo_root)}
    # A fake without "_" has no fields; everything after the first "_" is the actor.
    lone = records["FS_E2E/nounderscore"]
    assert (lone.identity, lone.target_id, lone.source_id, lone.pair_key) == (None,) * 4
    assert records["FS_E2E/7_M101_extra"].identity == "M101_extra"
    # A real's identity is everything before its first "_" (the whole stem if there is none).
    assert records["SR/W006"].identity == "W006"


def test_no_compression_can_be_requested(defo_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        _discover(defo_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ official split


def test_a_fake_takes_the_split_of_its_listed_file_name(defo_root):
    _touch(defo_root / "manipulated_content" / "end_to_end_level_1" / "videos", "000_M101.mp4")
    _write_split_lists(
        defo_root,
        train=["000_M101.mp4", "", "   "],
        val=["  001_W006.mp4  "],
        test=["999_M101.mp4"],  # not on disk: names no record, but still names an actor
    )
    assert _official(defo_root) == {
        # Every version of a listed face swap shares its file name, so all of them are listed.
        "FS_E2E/000_M101": "train",
        "FS_L1/000_M101": "train",
        "FS_E2E/001_W006": "val",
        # A real takes the split of its actor's fakes: M101 has a test fake (999), so test.
        "SR/M101_light_down_contempt_camera_front": "test",
        "SR/W006_light_uniform_neutral_camera_front": "val",
    }


def test_a_real_takes_its_actors_split_test_before_val_before_train(defo_root):
    _write_split_lists(defo_root, train=["000_M101.mp4", "001_W006.mp4"], val=["002_W006.mp4"])
    official = _official(defo_root)
    assert official["SR/M101_light_down_contempt_camera_front"] == "train"
    assert official["SR/W006_light_uniform_neutral_camera_front"] == "val"


def test_a_fake_on_several_lists_takes_test_before_val_before_train(defo_root):
    _write_split_lists(
        defo_root, train=["000_M101.mp4", "001_W006.mp4"], val=["000_M101.mp4", "001_W006.mp4"]
    )
    official = _official(defo_root)
    assert official["FS_E2E/000_M101"] == "val"
    assert official["FS_E2E/001_W006"] == "val"
    _write_split_lists(defo_root, train=["000_M101.mp4"], test=["000_M101.mp4"])
    assert _official(defo_root)["FS_E2E/000_M101"] == "test"


def test_a_real_whose_actor_has_no_listed_fake_is_left_out(defo_root):
    _write_split_lists(defo_root, test=["001_W006.mp4"])
    assert _official(defo_root) == {
        "FS_E2E/001_W006": "test",
        "SR/W006_light_uniform_neutral_camera_front": "test",
        # M101 has fakes on disk, but none of them is on a list: its real has no split.
    }


def test_the_actor_of_a_listed_name_is_its_second_underscore_part(defo_root):
    _touch(defo_root.joinpath(*E2E_DIR), "7_M101_extra.mp4")
    _touch(defo_root.joinpath(*REAL_DIR, "W006"), "W006.mp4")
    _write_split_lists(defo_root, val=["7_M101_extra.mp4", "lonely.mp4"], test=["x_W006.mp4"])
    assert _official(defo_root) == {
        "FS_E2E/7_M101_extra": "val",
        # "7_M101_extra.mp4" names actor M101 (the part between the first two "_").
        "SR/M101_light_down_contempt_camera_front": "val",
        # A real's actor is the part of its file name before the first "_", extension
        # included when there is no "_": "W006.mp4" is not actor W006.
        "SR/W006_light_uniform_neutral_camera_front": "test",
    }


def test_only_mp4_files_are_listed_but_a_listed_stem_covers_any_file(defo_root):
    fl1 = defo_root / "manipulated_content" / "end_to_end_level_1" / "videos"
    _touch(fl1, "000_M101.avi")  # same stem as a listed .mp4: matched by its stem
    _touch(defo_root.joinpath(*E2E_DIR), "003_M101.avi")  # listed, but not an .mp4
    _touch(defo_root.joinpath(*REAL_DIR, "M101", "extra"), "M101_extra.MP4")  # not .mp4
    _write_split_lists(defo_root, train=["000_M101.mp4", "003_M101.avi"])
    official = _official(defo_root)
    assert official["FS_L1/000_M101"] == "train"
    assert "FS_E2E/003_M101" not in official
    assert "SR/M101_extra" not in official
    assert official["SR/M101_light_down_contempt_camera_front"] == "train"


def test_a_stem_listed_in_two_splits_goes_to_the_first_in_train_val_test_order(defo_root):
    # A real of actor W006 whose stem equals a listed fake's: the real is test (W006 has a test
    # fake), the fake is train; the shared stem then goes to train for both.
    _touch(defo_root.joinpath(*REAL_DIR, "W006"), "W006_M101.mp4")
    _touch(defo_root.joinpath(*E2E_DIR), "W006_M101.mp4")
    _write_split_lists(defo_root, train=["W006_M101.mp4"], test=["001_W006.mp4"])
    official = _official(defo_root)
    assert official["SR/W006_M101"] == "train"
    assert official["FS_E2E/W006_M101"] == "train"
    assert official["SR/W006_light_uniform_neutral_camera_front"] == "test"


def test_the_official_scheme_assigns_the_listed_videos(defo_root):
    _write_split_lists(defo_root, train=["000_M101.mp4"], test=["001_W006.mp4"])
    builder = DeeperForensicsBuilder()
    records = collect_records(builder, defo_root)
    assert assign_official(records, builder.official_splits(defo_root, records)) == {
        ("FS_E2E/000_M101", None): "train",
        ("SR/M101_light_down_contempt_camera_front", None): "train",
        ("SR/W006_light_uniform_neutral_camera_front", None): "test",
        ("FS_E2E/001_W006", None): "test",
    }


def test_the_official_split_needs_every_release_list(defo_root):
    _write_split_lists(defo_root, train=["000_M101.mp4"])
    (defo_root / ".official_files" / "lists" / "splits" / "val.txt").unlink()
    # Per-video lists made elsewhere are not read.
    csv_dir = defo_root / ".official_files" / "official_splits"
    csv_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (csv_dir / f"{split}.csv").write_text("manipulated_videos/end_to_end/000_M101.mp4")
    builder = DeeperForensicsBuilder()
    records = collect_records(builder, defo_root)
    with pytest.raises(ConfigError, match=r"val\.txt") as caught:
        builder.official_splits(defo_root, records)
    assert ".official_files/lists/splits" in caught.value.hint


def test_an_unreadable_list_is_a_contract_error(defo_root):
    _write_split_lists(defo_root)
    (defo_root / ".official_files" / "lists" / "splits" / "test.txt").write_bytes(b"\xff\xfe\x00")
    builder = DeeperForensicsBuilder()
    with pytest.raises(ContractError, match=r"test\.txt"):
        builder.official_splits(defo_root, collect_records(builder, defo_root))


# ------------------------------------------------------------------------------ pairs, labels


def test_pair_candidates_name_the_actor(defo_root):
    _touch(defo_root.joinpath(*E2E_DIR), "7_M101_extra.mp4", "8_X101.mp4", "nounderscore.mp4")
    builder = DeeperForensicsBuilder()
    records = collect_records(builder, defo_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "FS_E2E/000_M101": "M101",
        "FS_E2E/001_W006": "W006",
        # Only "<digits>_<M or W><digits>" names an actor to pair with.
        "FS_E2E/7_M101_extra": None,
        "FS_E2E/8_X101": None,
        "FS_E2E/nounderscore": None,
    }


def test_each_fake_pairs_with_the_first_real_of_its_actor(defo_root):
    _touch(
        defo_root.joinpath(*REAL_DIR, "M101", "BlendShape", "camera_down"), "M101_BlendShape.mp4"
    )
    _touch(defo_root / "manipulated_content" / "reenact_postprocess" / "videos", "000_M101.mp4")
    builder = DeeperForensicsBuilder()
    records = collect_records(builder, defo_root)
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    # M101 has two reals; the cap of one keeps the first by key.
    assert pairs == [
        PairRecord("FR_RP/000_M101", "SR/M101_BlendShape", "identity-fanout"),
        PairRecord("FS_E2E/000_M101", "SR/M101_BlendShape", "identity-fanout"),
        PairRecord(
            "FS_E2E/001_W006", "SR/W006_light_uniform_neutral_camera_front", "identity-fanout"
        ),
    ]


def test_label_vocab_covers_every_task():
    builder = DeeperForensicsBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"DeFo-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {
        "DeFo-SR": (0, 0, 1, "real"),
        "DeFo-FS_E2E": (1, 1, 2, "face-swap"),
        "DeFo-FS_L1": (1, 1, 3, "face-swap"),
        "DeFo-FS_L2": (1, 1, 4, "face-swap"),
        "DeFo-FS_L3": (1, 1, 5, "face-swap"),
        "DeFo-FS_L4": (1, 1, 6, "face-swap"),
        "DeFo-FS_L5": (1, 1, 7, "face-swap"),
        "DeFo-FS_RND": (1, 1, 8, "face-swap"),
        "DeFo-FS_M2": (1, 1, 9, "face-swap"),
        "DeFo-FS_M3": (1, 1, 10, "face-swap"),
        "DeFo-FS_M4": (1, 1, 11, "face-swap"),
        "DeFo-FR_RP": (1, 1, 12, "face-reenactment"),
    }
    assert vocab["DeFo-SR"]["method"] == "original"
    assert vocab["DeFo-FR_RP"]["method"] == "Reenact-Post"


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = DeeperForensicsBuilder()
    assert builder.default_scheme == "official"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=100, strata=("task",))
    assert builder.pairing_rule == "identity-fanout"
    assert builder.pairing_fanout == 1
    assert builder.metadata_files == tuple(
        f".official_files/lists/splits/{split}.txt" for split in ("train", "val", "test")
    )
    assert [t.recursive for t in builder.tasks] == [True] + [False] * len(FAKE_TASKS)


def test_the_benchmark_draws_every_task_from_the_official_test(defo_root):
    for _, _, folder in FAKE_TASKS[:2]:
        _touch(defo_root / "manipulated_content" / folder / "videos", "002_M101.mp4")
    # M101 has a test fake, so its real is test; W006 has only a train fake.
    _write_split_lists(defo_root, train=["000_M101.mp4", "001_W006.mp4"], test=["002_M101.mp4"])
    builder = DeeperForensicsBuilder()
    records = collect_records(builder, defo_root)
    official = builder.official_splits(defo_root, records)
    assert builder.benchmark is not None
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=[key for key, split in official.items() if split == "test"],
    )
    # Every test fake (fewer than 100 per task) and the one test real.
    assert {key for key, _ in chosen} == {
        "FS_E2E/002_M101",
        "FS_L1/002_M101",
        "SR/M101_light_down_contempt_camera_front",
    }


def test_dataset_card():
    builder = DeeperForensicsBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "deeperforensics"
    assert card.name == "DeeperForensics-1.0"
    assert "DeFo" in card.aliases
    assert card.compressions is None
    assert builder.known_compressions == ()
    assert card.default_scheme == "official"
    assert card.homepage == "https://github.com/EndlessSora/DeeperForensics-1.0"
    assert card.paper is not None
    assert (card.paper.title, card.paper.venue, card.paper.year, card.paper.doi) == (
        "DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection",
        "CVPR",
        2020,
        None,
    )
    assert card.license.spdx is None


def test_the_layout_names_the_nesting_and_the_split_lists():
    text = DeeperForensicsBuilder().describe_layout()
    assert "'DeeperForensics-1.0'" in text
    assert "original_content/source_videos/videos/**/<video>" in text
    assert "manipulated_content/reenact_postprocess/videos/<video>" in text
    assert ".official_files/lists/splits/{train,val,test}.txt" in text
    assert "official_splits" not in text
    assert "test before val before train" in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("deeperforensics"), DeeperForensicsBuilder)
    assert DeeperForensicsBuilder.expected_folder == "DeeperForensics-1.0"
    assert DeeperForensicsBuilder.label_prefix == "DeFo"
