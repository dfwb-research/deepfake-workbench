"""``VideoIndex``: the join of a protocol split with a processed face store.

Every test here builds its own tiny, hand-written processed store next to the ``toyone`` fixture
pack (``tests/unit/protocols/conftest.py``): a few ``ProcessedRecord`` rows in ``index.jsonl``,
plus a frame directory where a video is meant to have frames on disk. No test touches the face
pipeline's own ``Store`` -- ``dfwb.data`` never imports ``dfwb.preprocess``, so neither do these
fixtures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from tests.unit.data.conftest import (
    processed_record,
    write_inventory_meta,
    write_provenance,
    write_store_frames,
    write_store_index,
)

from dfwb.core.errors import ConfigError, ContractError
from dfwb.data.index import SourceSpec, VideoIndex, VideoItem, _resolve_store_dir

PROFILE_ID = "toy-crop-face-a1b2c3d4"
PROFILE_SLUG = "toy-crop-face"


def _store_dir(work_root: Path, dataset: str = "toyone", profile_id: str = PROFILE_ID) -> Path:
    return work_root / dataset / "processed" / profile_id


def _seed_store(work_root: Path, records: list, *, profile_id: str = PROFILE_ID) -> Path:
    """Write ``index.jsonl`` plus a frame directory for every ``status="ok"`` record."""
    store_dir = _store_dir(work_root, profile_id=profile_id)
    write_store_index(store_dir, records)
    for record in records:
        if record.status == "ok" and record.n_frames > 0:
            write_store_frames(store_dir, record)
    return store_dir


# ------------------------------------------------------------------------------------------- join


def test_build_joins_split_and_where_filters(tmp_path, toyone_pack):
    work_root = tmp_path / "work"
    _seed_store(work_root, [processed_record("FAKE_A/a3", n_frames=3)])

    index = VideoIndex.build(
        [SourceSpec("toyone-pack:toyone/official", "test", where={"task": "FAKE_A"})],
        profile=PROFILE_ID,
        labels="binary",
        work_root=work_root,
    )

    assert index.excluded == []
    assert len(index.items) == 1
    item = index.items[0]
    assert item == VideoItem(
        source=0,
        dataset="toyone",
        key="FAKE_A/a3",
        compression=None,
        label=1,
        label_key="TOYONE-FAKE_A",
        method="FakeA",
        video_dir=_store_dir(work_root) / "FAKE_A/a3/_",
        frame_indices=[0, 1, 2],
    )


def test_build_applies_the_label_mapping_and_excludes_fake_b(tmp_path, toyone_pack):
    work_root = tmp_path / "work"
    records = [
        processed_record("REAL/r1", compression="c23"),
        processed_record("REAL/r1", compression="c40"),
        processed_record("FAKE_A/a1"),
        processed_record("FAKE_A/a4"),
        processed_record("FAKE_B/b1"),
        processed_record("FAKE_B/b4"),
    ]
    _seed_store(work_root, records)

    index = VideoIndex.build(
        [SourceSpec("toyone-pack:toyone/official", "train")],
        profile=PROFILE_ID,
        labels="binary",
        work_root=work_root,
    )

    included_keys = {item.key for item in index.items}
    assert included_keys == {"REAL/r1", "FAKE_A/a1", "FAKE_A/a4"}
    assert {item.label for item in index.items if item.key == "REAL/r1"} == {0}
    assert {item.label for item in index.items if item.key.startswith("FAKE_A")} == {1}

    assert sorted(index.excluded) == sorted(
        [
            (0, "FAKE_B/b1", None, "label-excluded"),
            (0, "FAKE_B/b4", None, "label-excluded"),
        ]
    )

    summary = index.summary()["sources"][0]
    assert summary["counts"] == {
        "in_split": 6,
        "included": 4,
        "excluded": {"label-excluded": 2},
    }
    assert summary["labels"] == {"0": 2, "1": 2}


def test_videoindex_counts_missing_and_label_excluded(tmp_path, toyone_pack):
    """The pinned scenario: a store still catching up counts its gaps but does not stop training."""
    work_root = tmp_path / "work"
    records = [
        processed_record("REAL/r1", compression="c23"),
        # REAL/r1 c40 is never written to the store at all: not-processed.
        processed_record("FAKE_A/a1"),
        # FAKE_A/a4 is never written to the store either: not-processed.
        processed_record("FAKE_B/b1"),
        processed_record("FAKE_B/b4"),
    ]
    _seed_store(work_root, records)

    index = VideoIndex.build(
        [SourceSpec("toyone-pack:toyone/official", "train")],
        profile=PROFILE_ID,
        labels="binary",
        work_root=work_root,
    )

    # training still runs: the videos that are processed and not excluded are still usable.
    assert {(item.key, item.compression, item.label) for item in index.items} == {
        ("REAL/r1", "c23", 0),
        ("FAKE_A/a1", None, 1),
    }

    assert sorted(index.excluded) == sorted(
        [
            (0, "REAL/r1", "c40", "not-processed"),
            (0, "FAKE_A/a4", None, "not-processed"),
            (0, "FAKE_B/b1", None, "label-excluded"),
            (0, "FAKE_B/b4", None, "label-excluded"),
        ]
    )

    summary = index.summary()["sources"][0]
    assert summary["counts"] == {
        "in_split": 6,
        "included": 2,
        "excluded": {"label-excluded": 2, "not-processed": 2},
    }


def test_build_treats_a_store_with_no_index_file_yet_as_nothing_processed(tmp_path, toyone_pack):
    # the store's directory may exist (e.g. its profile.json was written) before anything has
    # actually been processed into it -- that must behave exactly like an empty index.
    (_store_dir(tmp_path / "work")).mkdir(parents=True)

    index = VideoIndex.build(
        [SourceSpec("toyone-pack:toyone/official", "test", where={"task": "FAKE_A"})],
        profile=PROFILE_ID,
        labels="binary",
        work_root=tmp_path / "work",
    )

    assert index.items == []
    assert index.excluded == [(0, "FAKE_A/a3", None, "not-processed")]


def test_build_counts_processing_failed_and_no_frames(tmp_path, toyone_pack):
    work_root = tmp_path / "work"
    store_dir = _store_dir(work_root)
    failed = processed_record("REAL/r2", compression="c23", status="decode_error", n_frames=0)
    empty = processed_record("REAL/r2", compression="c40", status="ok", n_frames=0)
    missing_dir = processed_record("FAKE_A/a2", status="ok", n_frames=4)
    write_store_index(store_dir, [failed, empty, missing_dir])
    # `missing_dir` claims 4 frames but its directory is never written -- still `no-frames`.

    index = VideoIndex.build(
        [SourceSpec("toyone-pack:toyone/official", "val", where={"task": ["REAL", "FAKE_A"]})],
        profile=PROFILE_ID,
        labels="binary",
        work_root=work_root,
    )

    assert index.items == []
    assert sorted(index.excluded) == sorted(
        [
            (0, "REAL/r2", "c23", "processing-failed:decode_error"),
            (0, "REAL/r2", "c40", "no-frames"),
            (0, "FAKE_A/a2", None, "no-frames"),
        ]
    )
    summary = index.summary()["sources"][0]
    assert summary["counts"] == {
        "in_split": 3,
        "included": 0,
        "excluded": {"no-frames": 2, "processing-failed:decode_error": 1},
    }


def test_build_raises_when_a_label_mapping_value_is_not_int_or_exclude(tmp_path, toyone_pack):
    work_root = tmp_path / "work"
    _seed_store(work_root, [processed_record("REAL/r1", compression="c23")])

    with pytest.raises(ContractError, match="family"):
        VideoIndex.build(
            [SourceSpec("toyone-pack:toyone/official", "train", where={"task": "REAL"})],
            profile=PROFILE_ID,
            labels="family",
            work_root=work_root,
        )


# --------------------------------------------------------------------------------- multiple sources


def test_build_joins_multiple_sources_and_tags_each_items_source_index(tmp_path, toyone_pack):
    work_root = tmp_path / "work"
    _seed_store(
        work_root,
        [
            processed_record("FAKE_A/a3"),
            processed_record("REAL/r3", compression="c23"),
            processed_record("REAL/r3", compression="c40"),
        ],
    )

    index = VideoIndex.build(
        [
            SourceSpec("toyone-pack:toyone/official", "test", where={"task": "FAKE_A"}),
            SourceSpec("toyone-pack:toyone/official", "test", where={"task": "REAL"}),
        ],
        profile=PROFILE_ID,
        labels="binary",
        work_root=work_root,
    )

    by_source = {0: [], 1: []}
    for item in index.items:
        by_source[item.source].append(item.key)
    assert by_source == {0: ["FAKE_A/a3"], 1: ["REAL/r3", "REAL/r3"]}

    summaries = index.summary()["sources"]
    assert len(summaries) == 2
    assert summaries[0]["counts"]["in_split"] == 1
    assert summaries[1]["counts"]["in_split"] == 2


# ------------------------------------------------------------------------------------- determinism


def test_items_are_sorted_by_source_key_and_compression(tmp_path, toyone_pack):
    work_root = tmp_path / "work"
    # audiovisual-binary has no override, so FAKE_B is included too -- every task appears, and the
    # dataset's own file order (REAL, then FAKE_A, then FAKE_B) is not alphabetical, so a passing
    # assertion here only holds if VideoIndex actually sorts, rather than echoing file order.
    records = [
        processed_record("REAL/r1", compression="c23"),
        processed_record("REAL/r1", compression="c40"),
        processed_record("FAKE_A/a1"),
        processed_record("FAKE_A/a4"),
        processed_record("FAKE_B/b1"),
        processed_record("FAKE_B/b4"),
    ]
    _seed_store(work_root, records)

    index = VideoIndex.build(
        [SourceSpec("toyone-pack:toyone/official", "train")],
        profile=PROFILE_ID,
        labels="audiovisual-binary",
        work_root=work_root,
    )

    assert [(item.key, item.compression) for item in index.items] == [
        ("FAKE_A/a1", None),
        ("FAKE_A/a4", None),
        ("FAKE_B/b1", None),
        ("FAKE_B/b4", None),
        ("REAL/r1", "c23"),
        ("REAL/r1", "c40"),
    ]


# ------------------------------------------------------------------------------------- version pin


def test_build_warns_and_records_a_version_mismatch(tmp_path, toyone_pack, caplog):
    work_root = tmp_path / "work"
    _seed_store(work_root, [processed_record("FAKE_A/a3")])
    write_provenance(toyone_pack, builder_version="2")
    write_inventory_meta(work_root, "toyone", builder_version="1")

    with caplog.at_level(logging.WARNING):
        index = VideoIndex.build(
            [SourceSpec("toyone-pack:toyone/official", "test", where={"task": "FAKE_A"})],
            profile=PROFILE_ID,
            labels="binary",
            work_root=work_root,
        )

    assert any("toyone" in message and "version" in message for message in caplog.messages)
    version_check = index.summary()["sources"][0]["version_check"]
    assert version_check == {"status": "mismatch", "pack": "2", "store": "1"}


def test_build_records_unknown_version_check_without_warning_when_inventory_meta_is_missing(
    tmp_path, toyone_pack, caplog
):
    work_root = tmp_path / "work"
    _seed_store(work_root, [processed_record("FAKE_A/a3")])
    write_provenance(toyone_pack, builder_version="2")
    # no inventory.meta.json is written at all.

    with caplog.at_level(logging.WARNING):
        index = VideoIndex.build(
            [SourceSpec("toyone-pack:toyone/official", "test", where={"task": "FAKE_A"})],
            profile=PROFILE_ID,
            labels="binary",
            work_root=work_root,
        )

    assert not any("version" in message for message in caplog.messages)
    version_check = index.summary()["sources"][0]["version_check"]
    assert version_check == {"status": "unknown", "pack": "2", "store": None}


def test_build_records_a_matching_version_check_without_warning(tmp_path, toyone_pack, caplog):
    work_root = tmp_path / "work"
    _seed_store(work_root, [processed_record("FAKE_A/a3")])
    write_provenance(toyone_pack, builder_version="7")
    write_inventory_meta(work_root, "toyone", builder_version="7")

    with caplog.at_level(logging.WARNING):
        index = VideoIndex.build(
            [SourceSpec("toyone-pack:toyone/official", "test", where={"task": "FAKE_A"})],
            profile=PROFILE_ID,
            labels="binary",
            work_root=work_root,
        )

    assert not any("version" in message for message in caplog.messages)
    version_check = index.summary()["sources"][0]["version_check"]
    assert version_check == {"status": "match", "pack": "7", "store": "7"}


# ----------------------------------------------------------------------------- profile resolution


def test_resolve_store_dir_matches_a_full_profile_id(tmp_path):
    store_dir = tmp_path / "work" / "toyone" / "processed" / PROFILE_ID
    store_dir.mkdir(parents=True)
    assert _resolve_store_dir(tmp_path / "work", "toyone", PROFILE_ID) == store_dir


def test_resolve_store_dir_matches_a_single_slug_prefixed_directory(tmp_path):
    processed_dir = tmp_path / "work" / "toyone" / "processed"
    store_dir = processed_dir / PROFILE_ID
    store_dir.mkdir(parents=True)
    assert _resolve_store_dir(tmp_path / "work", "toyone", PROFILE_SLUG) == store_dir


def test_resolve_store_dir_raises_when_nothing_is_processed_yet(tmp_path):
    with pytest.raises(ConfigError, match="nothing has been processed"):
        _resolve_store_dir(tmp_path / "work", "toyone", PROFILE_SLUG)


def test_resolve_store_dir_raises_when_the_slug_matches_several_directories(tmp_path):
    processed_dir = tmp_path / "work" / "toyone" / "processed"
    (processed_dir / f"{PROFILE_SLUG}-aaaaaaaa").mkdir(parents=True)
    (processed_dir / f"{PROFILE_SLUG}-bbbbbbbb").mkdir(parents=True)

    with pytest.raises(ConfigError, match=f"{PROFILE_SLUG}-aaaaaaaa"):
        _resolve_store_dir(tmp_path / "work", "toyone", PROFILE_SLUG)


def test_build_raises_a_config_error_naming_what_exists_for_an_unresolved_profile(
    tmp_path, toyone_pack
):
    work_root = tmp_path / "work"
    (work_root / "toyone" / "processed" / "other-profile-deadbeef").mkdir(parents=True)

    with pytest.raises(ConfigError, match="other-profile-deadbeef"):
        VideoIndex.build(
            [SourceSpec("toyone-pack:toyone/official", "test", where={"task": "FAKE_A"})],
            profile=PROFILE_SLUG,
            labels="binary",
            work_root=work_root,
        )


# -------------------------------------------------------------------------------------- SourceSpec


def test_source_spec_defaults_where_to_none_and_weight_to_one():
    spec = SourceSpec("toyone-pack:toyone/official", "train")
    assert spec.where is None
    assert spec.weight == 1.0


def test_the_latest_index_row_for_a_video_wins(tmp_path, toyone_pack):
    work_root = tmp_path / "work"
    store_dir = _store_dir(work_root)
    ok = processed_record("REAL/r2", compression="c23", status="ok", n_frames=2)
    failed = processed_record("REAL/r2", compression="c23", status="decode_error", n_frames=0)
    source = SourceSpec("toyone-pack:toyone/official", "val", where={"task": ["REAL"]})

    write_store_index(store_dir, [ok, failed])
    write_store_frames(store_dir, ok)
    first = VideoIndex.build([source], profile=PROFILE_ID, labels="binary", work_root=work_root)
    assert (0, "REAL/r2", "c23", "processing-failed:decode_error") in first.excluded

    write_store_index(store_dir, [failed, ok])
    second = VideoIndex.build([source], profile=PROFILE_ID, labels="binary", work_root=work_root)
    assert [(item.key, item.compression) for item in second.items] == [("REAL/r2", "c23")]


def test_summary_returns_a_copy_its_caller_can_change(tmp_path, toyone_pack):
    work_root = tmp_path / "work"
    _seed_store(work_root, [processed_record("REAL/r2", compression="c23", n_frames=2)])
    index = VideoIndex.build(
        [SourceSpec("toyone-pack:toyone/official", "val", where={"task": ["REAL"]})],
        profile=PROFILE_ID,
        labels="binary",
        work_root=work_root,
    )
    index.summary()["sources"][0]["counts"]["included"] = -1
    assert index.summary()["sources"][0]["counts"]["included"] != -1


# ----------------------------------------------------------------------------- store index hash


def test_store_index_sha256_is_none_for_a_store_with_no_index_yet(tmp_path):
    from dfwb.data.index import store_index_sha256

    _store_dir(tmp_path / "work").mkdir(parents=True)

    assert store_index_sha256(tmp_path / "work", "toyone", PROFILE_ID) is None


def test_store_index_sha256_hashes_the_index_and_changes_when_it_grows(tmp_path):
    from dfwb.core.hashing import sha256_file
    from dfwb.data.index import store_index_sha256

    work_root = tmp_path / "work"
    store_dir = _seed_store(work_root, [processed_record("REAL/a")])
    before = store_index_sha256(work_root, "toyone", PROFILE_SLUG)
    assert before == sha256_file(store_dir / "index.jsonl")

    _seed_store(work_root, [processed_record("REAL/a"), processed_record("FAKE/b")])

    assert store_index_sha256(work_root, "toyone", PROFILE_ID) != before
