"""Combining shard files, and reporting a processed store's status.

``merge``'s low-level behaviour (combining, the tie-break, sorting, atomicity, the refusal to mix
different ``n``) is tested here directly against hand-built index files; the round trip against a
real, sharded run (a merged store equalling a single-machine run's) is
``test_shard_merge_equals_single_run`` in ``test_runner.py``, which also reuses the Demo dataset,
inventory and protocol pack fixtures ``status`` is tested against below.
"""

from __future__ import annotations

import json

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


def test_merge_heals_a_stale_shard_file_left_beside_an_already_merged_index(tmp_path):
    # A crash between the index rename and the shard-file deletes: the shard's row is already
    # folded into index.jsonl, but the shard file itself is still sitting there. A second merge
    # must give the same index and finish removing it.
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "index.jsonl", [_rec("a/1"), _rec("a/2")])
    write_jsonl(store / "index.shard-0-of-1.jsonl", [_rec("a/2")])

    summary = merge(store)

    assert summary == MergeSummary(store=store, n_records=2, n_shards=1)
    assert [r.key for r in read_jsonl(store / "index.jsonl", ProcessedRecord)] == ["a/1", "a/2"]
    assert not (store / "index.shard-0-of-1.jsonl").exists()


def test_merge_is_idempotent(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_jsonl(store / "index.shard-0-of-2.jsonl", [_rec("a/1")])
    write_jsonl(store / "index.shard-1-of-2.jsonl", [_rec("a/2")])

    first = merge(store)
    assert first == MergeSummary(store=store, n_records=2, n_shards=2)
    first_index = (store / "index.jsonl").read_bytes()

    second = merge(store)

    assert second == MergeSummary(store=store, n_records=2, n_shards=0)
    assert (store / "index.jsonl").read_bytes() == first_index


# ------------------------------------------------------------------------------- live shard runs


def _write_marker(store, index: int, count: int, *, host: str = "host-1", pid: int = 4242):
    store.mkdir(parents=True, exist_ok=True)
    marker = store / f"index.shard-{index}-of-{count}.jsonl.running"
    marker.write_text(json.dumps({"host": host, "pid": pid, "started_at": "2026-01-01T00:00:00"}))
    return marker


def test_merge_refuses_while_a_shard_run_marker_exists(tmp_path):
    store = tmp_path / "store"
    marker = _write_marker(store, 0, 1, host="worker-3", pid=9999)
    write_jsonl(store / "index.shard-0-of-1.jsonl", [_rec("a/1")])

    with pytest.raises(ContractError, match="worker-3") as caught:
        merge(store)

    assert "9999" in str(caught.value)
    assert "delete the marker" in caught.value.hint
    assert not (store / "index.jsonl").exists()
    assert (store / "index.shard-0-of-1.jsonl").is_file()  # nothing touched
    assert marker.is_file()


def test_merge_refuses_even_with_nothing_else_to_merge(tmp_path):
    store = tmp_path / "store"
    _write_marker(store, 0, 1)

    with pytest.raises(ContractError, match="live"):
        merge(store)


def test_merge_names_an_unreadable_markers_host_and_pid_as_unknown(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    marker = store / "index.shard-0-of-1.jsonl.running"
    marker.write_text("not json")

    with pytest.raises(ContractError, match="unknown"):
        merge(store)


def test_merge_proceeds_once_the_marker_is_gone(tmp_path):
    store = tmp_path / "store"
    marker = _write_marker(store, 0, 1)
    write_jsonl(store / "index.shard-0-of-1.jsonl", [_rec("a/1")])
    marker.unlink()

    summary = merge(store)

    assert summary == MergeSummary(store=store, n_records=1, n_shards=1)


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


def test_status_folds_index_jsonl_and_every_shard_file(env):
    # index.jsonl holds the REAL videos; the FS_SWAP ones are only in an unmerged shard file.
    _run(where={"task": "REAL"})
    _run(where={"task": "FS_SWAP"}, shard=(0, 1))
    store = _store_path(env.work)
    assert (store / "index.jsonl").is_file()
    assert (store / "index.shard-0-of-1.jsonl").is_file()

    table = status("demo", PROFILE)

    assert set(table.rows) == {
        StatusRow(task="REAL", split=None, status="ok", count=5),
        StatusRow(task="FS_SWAP", split=None, status="ok", count=2),
    }


def test_status_uses_merges_own_precedence_ok_over_a_failing_shard_row(env):
    store = _store_path(env.work)
    store.mkdir(parents=True)
    write_jsonl(store / "index.jsonl", [_rec("REAL/000", compression="c23", status="ok")])
    write_jsonl(
        store / "index.shard-0-of-1.jsonl",
        [_rec("REAL/000", compression="c23", status="decode_error")],
    )

    table = status("demo", PROFILE)

    real_counts = {row.status: row.count for row in table.rows if row.task == "REAL"}
    assert real_counts == {"ok": 1, "not-processed": 4}


def test_status_works_while_a_shard_run_marker_exists(env):
    _run(shard=(0, 1))
    store = _store_path(env.work)
    marker = _write_marker(store, 0, 1)

    table = status("demo", PROFILE)

    assert sum(row.count for row in table.rows if row.status == "ok") == 7
    assert marker.is_file()  # status never touches it


def test_status_propagates_scoping_errors_like_run(env):
    with pytest.raises(ConfigError, match="protocol"):
        status("demo", PROFILE, split="test")


def test_status_of_an_unknown_profile_is_refused(env):
    with pytest.raises(UnknownKeyError):
        status("demo", "no-such-profile")
