"""Combining shard files, and reporting a processed store's status.

``merge``'s low-level behaviour (combining, the tie-break, sorting, atomicity, the refusal to mix
different ``n``) is tested here directly against hand-built index files; the round trip against a
real, sharded run (a merged store equalling a single-machine run's) is
``test_shard_merge_equals_single_run`` in ``test_runner.py``, which also reuses the Demo dataset,
inventory and protocol pack fixtures ``status`` is tested against below.
"""

from __future__ import annotations

import pytest
from tests.unit.preprocess.face.test_runner import PROFILE, _run, _store_path

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError
from dfwb.core.records import ProcessedRecord, read_jsonl, write_jsonl
from dfwb.preprocess.face.shard import MergeSummary, StatusRow, merge, status, store_root

# ------------------------------------------------------------------------------------- merge


def _rec(key: str, *, compression: str | None = None, status: str = "ok") -> ProcessedRecord:
    return ProcessedRecord(
        key=key,
        compression=compression,
        status=status,  # type: ignore[arg-type]
        n_frames=1,
        frame_indices=[0],
        relpath=f"{key}/{compression or '_'}",
        track=None,
        reason=None,
    )


def test_merge_with_nothing_to_merge_is_a_no_op(tmp_path):
    store = tmp_path / "never-created"
    summary = merge(store)
    assert summary == MergeSummary(store=store, n_records=0, n_shards=0)
    assert not store.exists()


def test_merge_rewrites_an_existing_index_when_there_are_no_shards(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "index.jsonl", [_rec("a/1"), _rec("a/2")])

    summary = merge(store)

    assert summary == MergeSummary(store=store, n_records=2, n_shards=0)
    assert [r.key for r in read_jsonl(store / "index.jsonl", ProcessedRecord)] == ["a/1", "a/2"]


def test_merge_combines_shard_files_and_an_existing_index_then_removes_the_shards(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "index.jsonl", [_rec("a/1"), _rec("a/2")])
    write_jsonl(store / "index.shard-0-of-2.jsonl", [_rec("a/3")])
    write_jsonl(store / "index.shard-1-of-2.jsonl", [_rec("a/4")])

    summary = merge(store)

    assert summary == MergeSummary(store=store, n_records=4, n_shards=2)
    rows = read_jsonl(store / "index.jsonl", ProcessedRecord)
    assert [r.key for r in rows] == ["a/1", "a/2", "a/3", "a/4"]
    assert not (store / "index.shard-0-of-2.jsonl").exists()
    assert not (store / "index.shard-1-of-2.jsonl").exists()


def test_merge_prefers_an_ok_row_already_in_the_index_over_a_failing_shard_row(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "index.jsonl", [_rec("a/1", status="ok")])
    write_jsonl(store / "index.shard-0-of-1.jsonl", [_rec("a/1", status="decode_error")])

    merge(store)

    (row,) = read_jsonl(store / "index.jsonl", ProcessedRecord)
    assert row.status == "ok"


def test_merge_prefers_an_ok_shard_row_over_a_failing_row_already_in_the_index(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "index.jsonl", [_rec("a/1", status="decode_error")])
    write_jsonl(store / "index.shard-0-of-1.jsonl", [_rec("a/1", status="ok")])

    merge(store)

    (row,) = read_jsonl(store / "index.jsonl", ProcessedRecord)
    assert row.status == "ok"


def test_merge_prefers_the_shard_row_when_neither_is_ok(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "index.jsonl", [_rec("a/1", status="no_face")])
    write_jsonl(store / "index.shard-0-of-1.jsonl", [_rec("a/1", status="decode_error")])

    merge(store)

    (row,) = read_jsonl(store / "index.jsonl", ProcessedRecord)
    assert row.status == "decode_error"  # the shard's row, the later attempt, wins


def test_merge_sorts_the_merged_index_by_key_and_compression(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(
        store / "index.shard-0-of-1.jsonl",
        [_rec("b/1"), _rec("a/1", compression="c40"), _rec("a/1", compression="c23")],
    )

    merge(store)

    rows = read_jsonl(store / "index.jsonl", ProcessedRecord)
    assert [(r.key, r.compression) for r in rows] == [("a/1", "c23"), ("a/1", "c40"), ("b/1", None)]


def test_merge_refuses_shards_with_different_n_and_changes_nothing(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "index.shard-0-of-2.jsonl", [_rec("a/1")])
    write_jsonl(store / "index.shard-0-of-3.jsonl", [_rec("a/2")])

    with pytest.raises(ContractError, match="disagree"):
        merge(store)

    assert not (store / "index.jsonl").exists()
    assert (store / "index.shard-0-of-2.jsonl").is_file()
    assert (store / "index.shard-0-of-3.jsonl").is_file()


def test_merge_ignores_a_file_that_only_looks_like_a_shard_file(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "index.shard-x-of-y.jsonl").write_text("not really a shard file\n")
    write_jsonl(store / "index.shard-0-of-1.jsonl", [_rec("a/1")])

    summary = merge(store)

    assert summary.n_shards == 1
    assert (store / "index.shard-x-of-y.jsonl").is_file()  # untouched


def test_merge_leaves_no_temporary_file_behind(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "index.shard-0-of-1.jsonl", [_rec("a/1")])

    merge(store)

    assert [p.name for p in store.iterdir() if p.name.startswith(".")] == []


# ------------------------------------------------------------------------------------ status


def test_store_root_is_where_run_writes_its_store(env):
    assert store_root("demo", PROFILE) == _store_path(env.work)


def test_status_without_a_protocol_counts_by_task_and_status_including_not_processed(env):
    _run(where={"task": "REAL"})  # 5 REAL videos processed; the 2 FS_SWAP ones are untouched

    table = status("demo", PROFILE)

    assert set(table.rows) == {
        StatusRow(task="REAL", split=None, status="ok", count=5),
        StatusRow(task="FS_SWAP", split=None, status="not-processed", count=2),
    }


def test_status_with_a_protocol_breaks_counts_down_by_split(env):
    _run(protocol="demo/official", split="test")  # REAL/001 c23, REAL/001 c40, FS_SWAP/000_001 c23

    table = status("demo", PROFILE, protocol="demo/official")

    assert table.rows == (
        StatusRow(task="FS_SWAP", split="test", status="ok", count=1),
        StatusRow(task="FS_SWAP", split="val", status="not-processed", count=1),
        StatusRow(task="REAL", split="test", status="ok", count=2),
        StatusRow(task="REAL", split="train", status="not-processed", count=2),
    )
    # REAL/002 c23 (split "exclude") is out of scope entirely, like a run without --split.
    assert sum(row.count for row in table.rows) == 6


def test_status_with_a_protocol_and_split_only_counts_that_split(env):
    table = status("demo", PROFILE, protocol="demo/official", split="test")

    assert table.rows == (
        StatusRow(task="FS_SWAP", split="test", status="not-processed", count=1),
        StatusRow(task="REAL", split="test", status="not-processed", count=2),
    )


def test_status_is_read_only(env):
    status("demo", PROFILE)
    assert not (env.work / "demo" / "processed").exists()


def test_status_does_not_fold_in_an_unmerged_shard_file(env):
    _run(shard=(0, 1))  # every video, into index.shard-0-of-1.jsonl, never merged
    table = status("demo", PROFILE)
    assert {row.status for row in table.rows} == {"not-processed"}


def test_status_propagates_scoping_errors_like_run(env):
    with pytest.raises(ConfigError, match="protocol"):
        status("demo", PROFILE, split="test")


def test_status_of_an_unknown_profile_is_refused(env):
    with pytest.raises(UnknownKeyError):
        status("demo", "no-such-profile")
