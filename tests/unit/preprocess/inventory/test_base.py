"""Tests for ``dfwb.preprocess.inventory.base``: the builder base class and its helpers."""

from __future__ import annotations

import pytest
from tests.unit.preprocess.inventory._demo import DemoBuilder, make_demo_tree

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, SchemeCard, VideoRecord
from dfwb.preprocess.inventory.base import (
    VIDEO_SUFFIXES,
    BaseBuilder,
    InventoryBuilder,
    LabelSpec,
    SchemeSpec,
    TaskSpec,
    expand_compressions,
    has_videos,
    scan_videos,
)
from dfwb.preprocess.inventory.runner import collect_records

REAL, FAKE = DemoBuilder.tasks


# ------------------------------------------------------------------------------ scan_videos


def test_scan_skips_clutter_and_is_deterministic(tmp_path):
    for name in (".DS_Store", ".hidden.mp4", "Thumbs.db", "notes.txt", "a.MP4", "b.mp4"):
        (tmp_path / name).touch()
    (tmp_path / "c.mp4").symlink_to(tmp_path / "b.mp4")
    expected = [tmp_path / "a.MP4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
    assert scan_videos(tmp_path) == expected
    assert scan_videos(tmp_path) == expected


def test_scan_of_a_missing_directory_is_empty(tmp_path):
    assert scan_videos(tmp_path / "absent") == []


def test_scan_ignores_directories_and_dangling_links(tmp_path):
    (tmp_path / "folder.mp4").mkdir()
    (tmp_path / "gone.mp4").symlink_to(tmp_path / "never-existed.mp4")
    (tmp_path / "x.webm").touch()
    assert scan_videos(tmp_path) == [tmp_path / "x.webm"]


def test_scan_recursive_follows_linked_folders_without_looping(tmp_path):
    (tmp_path / "a" / "deep").mkdir(parents=True)
    (tmp_path / "a" / "deep" / "1.mp4").touch()
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "2.mp4").touch()
    elsewhere = tmp_path.parent / f"{tmp_path.name}-elsewhere"
    elsewhere.mkdir()
    (elsewhere / "3.mkv").touch()
    (tmp_path / "linked").symlink_to(elsewhere)
    (tmp_path / "a" / "loop").symlink_to(tmp_path)  # a cycle back to the top
    found = scan_videos(tmp_path, recursive=True)
    assert found == [tmp_path / "a" / "deep" / "1.mp4", tmp_path / "linked" / "3.mkv"]
    assert scan_videos(tmp_path) == []  # flat by default


def test_video_suffixes_are_lower_case():
    assert all(s == s.lower() and s.startswith(".") for s in VIDEO_SUFFIXES)
    assert {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"} == VIDEO_SUFFIXES


# ------------------------------------------------------------------------------ compressions


def test_expand_compressions():
    known = ("c0", "c23", "c40")
    assert expand_compressions("videos", known, None) == [None]
    assert expand_compressions("videos", known, ["c23"]) == [None]
    assert expand_compressions("a/{cX}/videos", known, None) == ["c0", "c23", "c40"]
    assert expand_compressions("a/{cX}/videos", known, ["c40", "c0"]) == ["c0", "c40"]
    assert expand_compressions("a/{cX}/videos", known, ["c23", "c23"]) == ["c23"]
    assert expand_compressions("a/{cX}/videos", known, []) == []
    with pytest.raises(
        ConfigError, match=r"unknown compression 'c32' \(did you mean 'c23'"
    ) as info:
        expand_compressions("a/{cX}/videos", known, ["c32"])
    assert "c0, c23, c40" in info.value.hint
    with pytest.raises(ConfigError, match="unknown compression"):
        expand_compressions("videos", (), ["c23"])


# ------------------------------------------------------------------------------ specs


def test_specs_reject_bad_values():
    with pytest.raises(ContractError, match="kind"):
        TaskSpec("REAL", "Real", "genuine", "videos", "original")  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="abbr"):
        TaskSpec("A/B", "Real", "real", "videos", "original")
    with pytest.raises(ContractError, match="abbr"):
        TaskSpec("", "Real", "real", "videos", "original")
    with pytest.raises(ContractError, match="binary"):
        LabelSpec(binary=2, binary_av=0, multiclass=0, family="real")
    with pytest.raises(ContractError, match="rule"):
        SchemeSpec("ident-70-30", "derived")  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="kind"):
        SchemeSpec("official", "published")  # type: ignore[arg-type]


# ------------------------------------------------------------------------------ records


def test_record_builds_keys_and_label_keys():
    builder = DemoBuilder()
    record = builder.record(
        FAKE,
        "000_001",
        "swapped/c23/000_001.mp4",
        "c23",
        identity="000",
        source_id="001",
        target_id="000",
        pair_key="000",
        attrs={"extra": 1},
    )
    assert record.key == "FS_SWAP/000_001"
    assert record.label_key == "DEMO-FS_SWAP"
    assert record.method == "Swapper"
    assert record.compression == "c23"
    assert record.relpath == "swapped/c23/000_001.mp4"
    assert record.builder == BuilderRef("demo", "1")
    assert (record.identity, record.source_id, record.target_id, record.pair_key) == (
        "000",
        "001",
        "000",
        "000",
    )
    assert record.attrs == {"task_name": "Swapper", "extra": 1}
    assert record.folder is None

    real = builder.record(REAL, "000", "originals/c23/000.mp4", "c23")
    assert (real.key, real.label_key, real.method) == ("REAL/000", "DEMO-REAL", "original")
    assert real.attrs == {"task_name": "Originals"}


def test_record_rejects_an_empty_key_and_a_bad_relpath():
    builder = DemoBuilder()
    with pytest.raises(ContractError, match="empty"):
        builder.record(REAL, "", "originals/c23/x.mp4", "c23")
    with pytest.raises(ContractError, match="relpath"):
        builder.record(REAL, "x", "../elsewhere/x.mp4", None)


def test_default_discover_walks_every_task_and_compression(tmp_path):
    make_demo_tree(tmp_path)
    (tmp_path / "swapped" / "c23" / "stray.mp4").touch()  # the demo builder skips it
    (tmp_path / "originals" / "c23" / "README.txt").touch()
    records = list(DemoBuilder().discover(tmp_path))
    assert [(r.key, r.compression) for r in records] == [
        ("REAL/000", "c23"),
        ("REAL/001", "c23"),
        ("REAL/000", "c40"),
        ("REAL/001", "c40"),
        ("FS_SWAP/000_001", "c23"),
        ("FS_SWAP/001_000", "c23"),
        ("FS_SWAP/000_001", "c40"),
        ("FS_SWAP/001_000", "c40"),
    ]
    assert records[0].relpath == "originals/c23/000.mp4"
    assert records[4].source_id == "001"


def test_default_discover_filters_compressions_and_tolerates_missing_tasks(tmp_path):
    make_demo_tree(tmp_path, fakes=(), compressions=("c23",))
    records = list(DemoBuilder().discover(tmp_path, compressions=["c23"]))
    assert [(r.key, r.compression) for r in records] == [("REAL/000", "c23"), ("REAL/001", "c23")]
    with pytest.raises(ConfigError, match="c32"):
        list(DemoBuilder().discover(tmp_path, compressions=["c32"]))


def test_the_base_hook_keys_a_video_by_its_file_stem(tmp_path):
    class Plain(BaseBuilder):
        dataset_id = "plain"
        expected_folder = "Plain"
        label_prefix = "PL"
        tasks = (TaskSpec("REAL", "Real", "real", "real", "original"),)
        labels = {"REAL": LabelSpec(0, 0, 0, "real")}
        schemes = {"all-test": SchemeSpec("all-test", "subset")}
        default_scheme = "all-test"
        card_info = {}

    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "clip.01.mov").touch()
    (record,) = Plain().discover(tmp_path)
    assert (record.key, record.relpath, record.compression) == (
        "REAL/clip.01",
        "real/clip.01.mov",
        None,
    )
    assert isinstance(Plain(), InventoryBuilder)


# ------------------------------------------------------------------------------ hooks


def test_default_official_splits_raises(tmp_path):
    with pytest.raises(ContractError, match="demo has no official split"):
        DemoBuilder().official_splits(tmp_path, [])


def test_default_pair_candidates_is_none():
    builder = DemoBuilder()
    fake = builder.record(FAKE, "000_001", "swapped/c23/000_001.mp4", "c23")
    assert builder.pair_candidates(fake) is None


def test_is_real_uses_the_visual_binary_label():
    builder = DemoBuilder()
    assert builder.is_real(builder.record(REAL, "000", "originals/c23/000.mp4", "c23"))
    assert not builder.is_real(VideoRecord("FS_SWAP/000_001", "c23", "DEMO-FS_SWAP", "Swapper"))
    with pytest.raises(ContractError, match="NOPE"):
        builder.is_real(VideoRecord("NOPE/1", None, "DEMO-NOPE", "x"))


def test_task_rank_follows_the_task_table():
    assert DemoBuilder().task_rank() == {"REAL": 0, "FS_SWAP": 1}


def test_label_vocab_has_the_four_mappings():
    vocab = DemoBuilder().label_vocab()
    assert vocab.vocab == {
        "DEMO-REAL": {
            "binary": 0,
            "binary_av": 0,
            "multiclass": 0,
            "family": "real",
            "method": "original",
            "task": "REAL",
        },
        "DEMO-FS_SWAP": {
            "binary": 1,
            "binary_av": 1,
            "multiclass": 1,
            "family": "face-swap",
            "method": "Swapper",
            "task": "FS_SWAP",
        },
    }
    assert {name: spec.from_ for name, spec in vocab.mappings.items()} == {
        "binary": "binary",
        "audiovisual-binary": "binary_av",
        "multiclass": "multiclass",
        "family": "family",
    }
    assert all(not spec.override for spec in vocab.mappings.values())


def test_label_vocab_needs_a_label_for_every_task():
    class Unlabelled(DemoBuilder):
        labels = {"REAL": LabelSpec(0, 0, 0, "real")}

    with pytest.raises(ContractError, match="FS_SWAP"):
        Unlabelled().label_vocab()


def test_dataset_card_has_undecided_distribution():
    scheme = SchemeCard(kind="subset", rule="all-test", sha256="0" * 64)
    card = DemoBuilder().dataset_card({"all-test": scheme})
    assert card.id == "demo"
    assert card.name == "Demo"
    assert card.distribution == "undecided"
    assert card.default_scheme == "all-test"
    assert card.schemes == {"all-test": scheme}
    assert card.license.summary == "synthetic test data"
    assert card.terms.reviewed is None


def test_dataset_card_rejects_unknown_or_missing_schemes_and_bad_info():
    scheme = SchemeCard(kind="subset", sha256="0" * 64)
    builder = DemoBuilder()
    with pytest.raises(ContractError, match="'ident-72-14-14'"):
        builder.dataset_card({"all-test": scheme, "ident-72-14-14": scheme})
    with pytest.raises(ContractError, match="default scheme 'all-test'"):
        builder.dataset_card({"benchmark": scheme})

    class NoName(DemoBuilder):
        card_info = {k: v for k, v in DemoBuilder.card_info.items() if k != "name"}

    with pytest.raises(ContractError, match="name"):
        NoName().dataset_card({"all-test": scheme})

    class Overreach(DemoBuilder):
        card_info = {**DemoBuilder.card_info, "distribution": "list"}

    with pytest.raises(ContractError, match="distribution"):
        Overreach().dataset_card({"all-test": scheme})


def test_describe_layout_lists_every_task():
    text = DemoBuilder().describe_layout()
    assert "Demo" in text
    assert "'Demo'" in text  # the expected folder
    assert "originals/{cX}/" in text
    assert "swapped/{cX}/" in text
    assert "c23, c40" in text
    assert "<task>/<file stem>" in text
    assert "Fakes are named <target>_<source>." in text
    assert text == DemoBuilder().describe_layout()


def test_layout_dirs_expands_compressions_over_every_task():
    assert DemoBuilder().layout_dirs() == (
        "originals/c23",
        "originals/c40",
        "swapped/c23",
        "swapped/c40",
    )


def test_layout_present_needs_at_least_one_video_file(tmp_path):
    builder = DemoBuilder()
    assert not builder.layout_present(tmp_path)  # nothing at all
    (tmp_path / "originals" / "c23").mkdir(parents=True)
    assert not builder.layout_present(tmp_path)  # an empty placeholder directory is not enough
    (tmp_path / "originals" / "c23" / "000.mp4").touch()
    assert builder.layout_present(tmp_path)
    assert builder.videos_present(tmp_path) == ("originals/c23",)


def test_layout_present_ignores_a_task_dir_that_is_actually_a_file(tmp_path):
    class _FlatDemo(DemoBuilder):
        tasks = (TaskSpec("REAL", "Originals", "real", "originals", "original"),)
        known_compressions = ()

    builder = _FlatDemo()
    (tmp_path / "originals").touch()  # a file, not a directory
    assert not builder.layout_present(tmp_path)
    assert builder.layout_dirs() == ("originals",)


def test_metadata_files_defaults_to_empty():
    assert DemoBuilder().metadata_files == ()


def test_video_copy_for_picks_the_first_copy_with_a_video(tmp_path):
    builder = DemoBuilder()
    empty, full = tmp_path / "empty", tmp_path / "full"
    (empty / "originals" / "c23").mkdir(parents=True)  # a placeholder, no video
    make_demo_tree(full, compressions=("c23",))
    task = REAL
    assert builder.video_copy_for((empty, full), task, "c23") == full
    assert builder.video_copy_for((full, empty), task, "c23") == full
    assert builder.video_copy_for((empty,), task, "c23") is None


def test_unbound_discover_scans_only_the_given_root(tmp_path):
    # No bind_copies call: discover(root) scans exactly root, nothing else.
    make_demo_tree(tmp_path, compressions=("c23",))
    keys = [r.key for r in DemoBuilder().discover(tmp_path, compressions=["c23"])]
    assert keys == ["REAL/000", "REAL/001", "FS_SWAP/000_001", "FS_SWAP/001_000"]


def test_bind_copies_merges_records_across_copies_per_task_and_compression(tmp_path):
    # One copy holds c23 originals only; another holds c40 originals and every fake.
    copy_a, copy_b = tmp_path / "a", tmp_path / "b"
    make_demo_tree(copy_a, reals=("000", "001"), fakes=(), compressions=("c23",))
    make_demo_tree(copy_b, reals=("000", "001"), compressions=("c40",))

    builder = DemoBuilder()
    builder.bind_copies((copy_a, copy_b))
    records = sorted(builder.discover(copy_a), key=lambda r: (r.key, r.compression))

    assert [(r.key, r.compression) for r in records] == [
        ("FS_SWAP/000_001", "c40"),
        ("FS_SWAP/001_000", "c40"),
        ("REAL/000", "c23"),
        ("REAL/000", "c40"),
        ("REAL/001", "c23"),
        ("REAL/001", "c40"),
    ]
    # c23 originals come from copy_a's tree, not copy_b's (which has no c23 at all).
    c23_real = next(r for r in records if r.key == "REAL/000" and r.compression == "c23")
    assert (copy_a / c23_real.relpath).is_file()


def test_bind_copies_falls_back_to_a_later_copy_when_the_first_lacks_a_video(tmp_path):
    copy_a, copy_b = tmp_path / "a", tmp_path / "b"
    (copy_a / "originals" / "c23").mkdir(parents=True)  # a placeholder, no video
    make_demo_tree(copy_b, reals=("000",), fakes=(), compressions=("c23",))

    builder = DemoBuilder()
    builder.bind_copies((copy_a, copy_b))
    records = list(builder.discover(copy_a, compressions=["c23"]))
    assert [r.key for r in records] == ["REAL/000"]


def test_two_video_suffixes_sharing_a_stem_raise_a_contract_error(tmp_path):
    # discover() yields both 000.mp4 and 000.avi; collect_records's duplicate-key check catches
    # the resulting repeated (key, compression), exactly as it would for one copy on its own.
    make_demo_tree(tmp_path, fakes=(), compressions=("c23",))
    (tmp_path / "originals" / "c23" / "000.avi").touch()
    with pytest.raises(ContractError, match=r"duplicate record 'REAL/000' \(compression 'c23'\)"):
        collect_records(DemoBuilder(), tmp_path)


def test_a_repeated_known_compression_is_rejected_loudly(tmp_path):
    class _RepeatedCompression(DemoBuilder):
        known_compressions = ("c23", "c23")

    with pytest.raises(ContractError, match="known_compressions has a repeated entry"):
        list(_RepeatedCompression().discover(tmp_path))


def test_has_videos_matches_scan_videos_content_check(tmp_path):
    assert not has_videos(tmp_path)  # nothing at all
    (tmp_path / "notes.txt").touch()
    assert not has_videos(tmp_path)  # clutter only
    (tmp_path / "a.mp4").touch()
    assert has_videos(tmp_path)
    assert has_videos(tmp_path) == bool(scan_videos(tmp_path))


def test_has_videos_recursive_follows_sub_folders(tmp_path):
    (tmp_path / "a" / "deep").mkdir(parents=True)
    assert not has_videos(tmp_path, recursive=True)
    (tmp_path / "a" / "deep" / "1.mp4").touch()
    assert has_videos(tmp_path, recursive=True)
    assert not has_videos(tmp_path)  # flat: doesn't look inside "a"


def test_copies_property_is_bound_or_falls_back_to_the_prepare_root(tmp_path):
    builder = DemoBuilder()
    assert builder.copies == ()
    make_demo_tree(tmp_path, compressions=("c23",))
    collect_records(builder, tmp_path)  # calls prepare(tmp_path); bind_copies was never called
    assert builder.copies == (tmp_path,)

    bound = DemoBuilder()
    elsewhere = tmp_path / "elsewhere"
    bound.bind_copies((tmp_path, elsewhere))
    assert bound.copies == (tmp_path, elsewhere)


def test_choose_copies_picks_per_task_and_compression(tmp_path):
    copy_a, copy_b = tmp_path / "a", tmp_path / "b"
    make_demo_tree(copy_a, reals=("000",), fakes=(), compressions=("c23",))
    make_demo_tree(copy_b, reals=("000",), compressions=("c40",))

    chosen = DemoBuilder().choose_copies((copy_a, copy_b))

    assert chosen[("REAL", "c23")] == copy_a
    assert chosen[("REAL", "c40")] == copy_b
    assert chosen[("FS_SWAP", "c40")] == copy_b
    assert ("FS_SWAP", "c23") not in chosen  # neither copy has a c23 fake


def test_bind_copies_with_a_precomputed_choice_is_reused_by_discover(tmp_path, monkeypatch):
    copy_a, copy_b = tmp_path / "a", tmp_path / "b"
    make_demo_tree(copy_a, reals=("000",), fakes=(), compressions=("c23",))
    make_demo_tree(copy_b, reals=("000",), compressions=("c40",))
    builder = DemoBuilder()
    chosen = builder.choose_copies((copy_a, copy_b))

    calls: list[tuple[str, str | None]] = []
    original = DemoBuilder.video_copy_for

    def counting(self, copies, task, compression):
        calls.append((task.abbr, compression))
        return original(self, copies, task, compression)

    monkeypatch.setattr(DemoBuilder, "video_copy_for", counting)
    builder.bind_copies((copy_a, copy_b), chosen)
    records = list(builder.discover(copy_a))

    assert {r.key for r in records} == {"REAL/000", "FS_SWAP/000_001", "FS_SWAP/001_000"}
    # Every combination the pre-computed choice covered was reused, not re-probed.
    assert ("REAL", "c23") not in calls
    assert ("REAL", "c40") not in calls
    assert ("FS_SWAP", "c40") not in calls


def test_the_demo_builder_satisfies_the_protocol():
    builder = DemoBuilder()
    assert isinstance(builder, InventoryBuilder)
    assert not isinstance(object(), InventoryBuilder)


def test_scan_follows_a_symlinked_dataset_folder(tmp_path):
    real = make_demo_tree(tmp_path / "real-location", compressions=("c23",))
    link = tmp_path / "Demo"
    link.symlink_to(real)
    keys = [r.key for r in DemoBuilder().discover(link, compressions=["c23"])]
    assert keys == ["REAL/000", "REAL/001", "FS_SWAP/000_001", "FS_SWAP/001_000"]


# ------------------------------------------------------------------------------ nested layouts


class _Nested(BaseBuilder):
    """A flat task next to a nested one (videos in per-actor sub-folders)."""

    dataset_id = "nested"
    expected_folder = "Nested"
    label_prefix = "NE"
    tasks = (
        TaskSpec("REAL", "Real", "real", "real", "original"),
        TaskSpec("FAKE", "Fake", "fake", "fake/{cX}", "swap", recursive=True),
    )
    known_compressions = ("c23",)
    labels = {"REAL": LabelSpec(0, 0, 0, "real"), "FAKE": LabelSpec(1, 1, 1, "face-swap")}
    schemes = {"all-test": SchemeSpec("all-test", "subset")}
    default_scheme = "all-test"
    card_info = {}


def test_tasks_are_flat_by_default():
    assert REAL.recursive is False
    assert TaskSpec("X", "X", "fake", "x", "x", recursive=True).recursive is True


def test_a_nested_task_scans_sub_folders_and_keeps_their_path(tmp_path):
    for rel in ("fake/c23/actor1/a.mp4", "fake/c23/actor2/deep/b.mp4", "fake/c23/c.mp4"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    records = [r for r in _Nested().discover(tmp_path) if r.key.startswith("FAKE/")]
    assert [(r.key, r.relpath, r.compression) for r in records] == [
        ("FAKE/a", "fake/c23/actor1/a.mp4", "c23"),
        ("FAKE/b", "fake/c23/actor2/deep/b.mp4", "c23"),
        ("FAKE/c", "fake/c23/c.mp4", "c23"),
    ]


def test_a_builder_can_mix_flat_and_nested_tasks(tmp_path):
    for rel in ("real/r1.mp4", "real/sub/ignored.mp4", "fake/c23/actor1/f1.mp4"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    records = list(_Nested().discover(tmp_path))
    assert [(r.key, r.relpath) for r in records] == [
        ("REAL/r1", "real/r1.mp4"),  # the flat task does not descend into real/sub
        ("FAKE/f1", "fake/c23/actor1/f1.mp4"),
    ]
    assert "recursive" in _Nested().describe_layout()


# ------------------------------------------------------------------------------ record options


def test_record_can_override_the_task_method():
    builder = DemoBuilder()
    record = builder.record(FAKE, "000_001", "swapped/c23/000_001.mp4", "c23", method="Swapper-v2")
    assert record.method == "Swapper-v2"
    assert record.label_key == "DEMO-FS_SWAP"  # the label still comes from the task
    assert builder.record(FAKE, "000_001", "swapped/c23/000_001.mp4", "c23").method == "Swapper"


def test_record_folder_must_be_one_path_segment():
    builder = DemoBuilder()
    ok = builder.record(REAL, "000", "original/000.mp4", None, folder="FaceForensics++")
    assert ok.folder == "FaceForensics++"
    for bad in ("", "a/b", "a\\b", "..", ".", "/abs"):
        with pytest.raises(ContractError, match="folder"):
            builder.record(REAL, "000", "original/000.mp4", None, folder=bad)
