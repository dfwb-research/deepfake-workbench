"""Split rules: carve keys, the 72/14/14 and 80/20 carves, official and official+80/20.

The assertions about the rules themselves are ported from the reference split builder's tests; the
ones about its output files and folders are not (pack files are written by
``dfwb.protocols.writer`` and tested in ``test_writer.py``).
"""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dfwb.core.errors import ContractError
from dfwb.core.records import BuilderRef, InventoryRecord, VideoRecord
from dfwb.protocols.rules import (
    assign_72_14_14,
    assign_80_20,
    assign_all_test,
    assign_official,
    assign_official_plus_80_20,
    carve_key,
    local_key,
    md5_mod_100,
    task_of,
)


def rec(key, identity=None, compression=None):
    return VideoRecord(
        key=key,
        compression=compression,
        label_key="X-" + key.split("/")[0],
        method="m",
        identity=identity,
    )


def _md5(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 100


def test_md5_mod_100_matches_the_legacy_formula():
    for text in ["id0", "000", "kodf_actor_17", "Ünïcode"]:
        assert md5_mod_100(text) == int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 100


def test_md5_mod_100_pinned_values():
    # Frozen values: a change here changes every carved scheme hash.
    assert md5_mod_100("id0") == 23
    assert md5_mod_100("000") == 64
    assert md5_mod_100("kodf_actor_17") == 31
    assert md5_mod_100("Ünïcode") == 3  # UTF-8, not Latin-1
    assert md5_mod_100("real_train_6_54") == 26


def test_local_key_and_task_of():
    assert local_key("FS_DF/000_003") == "000_003"
    assert local_key("a/b/c") == "b/c"  # only the first "/" separates the task
    assert local_key("noslash") == "noslash"
    assert task_of("FS_DF/000_003") == "FS_DF"
    assert task_of("a/b/c") == "a"
    assert task_of("noslash") == ""


def test_carve_key_prefers_identity_then_local_key():
    assert carve_key(rec("FS_DF/000_003", identity="000")) == "000"
    assert carve_key(rec("REAL/real_train_6_54")) == "real_train_6_54"
    assert local_key("noslash") == "noslash"


def test_carve_key_treats_an_empty_identity_as_missing():
    assert carve_key(rec("REAL/v1", identity="")) == "v1"


def test_72_14_14_thresholds():
    rows = assign_72_14_14([rec(f"REAL/v{i}", identity=f"id{i}") for i in range(500)])
    for (key, _), split in rows.items():
        h = md5_mod_100(f"id{key.split('v')[1]}")
        assert split == ("val" if h < 14 else "test" if h < 28 else "train")


def test_72_14_14_bucket_edges():
    # id11 -> 13 (last val), id0 -> 23 (test), id5 -> 28 (first train), id8 -> 0 (val).
    records = [rec(f"REAL/v{i}", identity=f"id{i}") for i in (0, 5, 8, 11)]
    assert assign_72_14_14(records) == {
        ("REAL/v0", None): "test",
        ("REAL/v5", None): "train",
        ("REAL/v8", None): "val",
        ("REAL/v11", None): "val",
    }


def test_80_20_thresholds():
    rows = assign_80_20([rec(f"REAL/v{i}", identity=f"id{i}") for i in range(500)])
    assert set(rows.values()) == {"train", "val"}
    for (key, _), split in rows.items():
        h = md5_mod_100(f"id{key.split('v')[1]}")
        assert split == ("val" if h < 20 else "train")


def test_80_20_bucket_edges():
    # id11 -> 13 (val), id0 -> 23 (train), id8 -> 0 (val), id3 -> 25 (train).
    records = [rec(f"REAL/v{i}", identity=f"id{i}") for i in (0, 3, 8, 11)]
    assert assign_80_20(records) == {
        ("REAL/v0", None): "train",
        ("REAL/v3", None): "train",
        ("REAL/v8", None): "val",
        ("REAL/v11", None): "val",
    }


def test_carves_apply_the_same_split_to_every_compression():
    records = [rec("REAL/v1", identity="id8", compression=c) for c in ("c23", "c40")]
    assert assign_80_20(records) == {("REAL/v1", "c23"): "val", ("REAL/v1", "c40"): "val"}
    assert assign_72_14_14(records) == {("REAL/v1", "c23"): "val", ("REAL/v1", "c40"): "val"}


def test_carve_without_identity_hashes_the_local_key_not_the_task():
    # The same local key in two tasks lands in the same bucket: md5("id0") == 23.
    records = [rec("FS_A/id0"), rec("FS_B/id0")]
    assert assign_72_14_14(records) == {("FS_A/id0", None): "test", ("FS_B/id0", None): "test"}
    assert assign_80_20(records) == {("FS_A/id0", None): "train", ("FS_B/id0", None): "train"}


def test_carves_accept_inventory_records():
    inventory = [
        InventoryRecord(
            key=f"REAL/v{i}",
            compression=None,
            label_key="X-REAL",
            method="m",
            relpath=f"real/v{i}.mp4",
            builder=BuilderRef("fixture", "1"),
            identity=f"id{i}",
        )
        for i in range(20)
    ]
    videos = [rec(r.key, identity=r.identity) for r in inventory]
    assert assign_72_14_14(inventory) == assign_72_14_14(videos)
    assert assign_80_20(inventory) == assign_80_20(videos)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["REAL", "FS_A", "FS_B"]), st.integers(0, 40), st.integers(0, 999)
        ),
        min_size=1,
        max_size=200,
        unique_by=lambda t: (t[0], t[2]),
    )
)
def test_identity_disjointness_property(items):
    records = [rec(f"{task}/v{n}", identity=f"id{ident}") for task, ident, n in items]
    for assign in (assign_72_14_14, assign_80_20):
        by_identity: dict[str, set[str]] = {}
        for r, split in ((r, assign([r])[(r.key, None)]) for r in records):
            by_identity.setdefault(r.identity, set()).add(split)
        assert all(len(s) == 1 for s in by_identity.values())


def test_all_test_assigns_every_record_to_test():
    # Ported: an all-to-test rule tags every entry "test" and nothing else.
    records = [rec(f"REAL/actor_{i}_001", identity=f"actor_{i}") for i in range(5)]
    records.append(rec("REAL/actor_0_001", identity="actor_0", compression="c23"))
    assignment = assign_all_test(records)
    assert len(assignment) == 6
    assert set(assignment.values()) == {"test"}
    assert ("REAL/actor_0_001", "c23") in assignment


def test_official_applies_its_split_to_every_compression_and_skips_the_rest():
    records = [
        rec("REAL/r1", compression="c23"),
        rec("REAL/r1", compression="c40"),
        rec("REAL/r2", compression="c23"),
        rec("FS_A/f1"),
    ]
    official = {"REAL/r1": "train", "FS_A/f1": "test", "REAL/not-on-disk": "val"}
    assert assign_official(records, official) == {
        ("REAL/r1", "c23"): "train",
        ("REAL/r1", "c40"): "train",
        ("FS_A/f1", None): "test",
    }


def test_official_rejects_an_unknown_split():
    with pytest.raises(ContractError, match="validation"):
        assign_official([rec("REAL/r1")], {"REAL/r1": "validation"})


def test_official_plus_80_20_policies():
    records = [rec(f"REAL/v{i}", identity=f"id{i % 7}") for i in range(40)]
    official = {r.key: "test" for r in records[:10]}
    a = assign_official_plus_80_20(records, official, policy="test-only-official")
    assert {k for (k, _), s in a.items() if s == "test"} == set(official)
    assert all(s in {"train", "val"} for (k, _), s in a.items() if k not in official)
    official2 = {**official, **{r.key: "train" for r in records[10:30]}}
    b = assign_official_plus_80_20(records, official2, policy="official-train-test")
    assert {k for (k, _), _s in b.items()} == set(official2)  # records outside official unassigned


def test_test_only_official_carves_everything_else_by_identity():
    # Ported: 10 videos, official test = the first 3 keys; the other 7 are carved 80/20 by
    # md5(identity), val below 20, train at or above it.
    records = [rec(f"CR/id{i}_0001", identity=f"id{i}") for i in range(10)]
    official = {r.key: "test" for r in records[:3]}
    assignment = assign_official_plus_80_20(records, official, policy="test-only-official")

    assert {k for (k, _), s in assignment.items() if s == "test"} == set(official)
    carved = {r.key: r.identity for r in records[3:]}
    assert {k for (k, _), s in assignment.items() if s in {"train", "val"}} == set(carved)
    for key, identity in carved.items():
        split = assignment[(key, None)]
        assert split == ("val" if _md5(identity) < 20 else "train"), (key, split)


def test_test_only_official_counts():
    # Ported: 8 videos with an official test of 2 -> 2 test rows and 6 carved rows.
    records = [rec(f"CR/id{i}_0001", identity=f"id{i}") for i in range(8)]
    official = {records[0].key: "test", records[1].key: "test"}
    assignment = assign_official_plus_80_20(records, official, policy="test-only-official")
    splits = list(assignment.values())
    assert splits.count("test") == 2
    assert splits.count("train") + splits.count("val") == 6


def test_test_only_official_without_an_official_test_carves_everything_deterministically():
    # Ported: with no official list the whole dataset is carved, and two runs agree.
    records = [rec(f"CR/id{i}_0001", identity=f"id{i}") for i in range(20)]
    first = assign_official_plus_80_20(records, {}, policy="test-only-official")
    second = assign_official_plus_80_20(list(reversed(records)), {}, policy="test-only-official")
    assert first == second
    assert first == assign_80_20(records)
    assert set(first.values()) == {"train", "val"}  # 20 identities reach both buckets


def test_test_only_official_carves_official_train_rows_too():
    # Only the official *test* is kept; any other official label is re-carved.
    records = [rec(f"CR/id{i}_0001", identity=f"id{i}") for i in range(10)]
    official = {records[0].key: "test", records[8].key: "train"}
    assignment = assign_official_plus_80_20(records, official, policy="test-only-official")
    assert assignment[(records[8].key, None)] == "val"  # md5("id8") == 0
    assert len(assignment) == 10


def test_official_train_test_carves_only_the_official_train():
    records = [
        rec(f"CR/id{i}_0001", identity=f"id{i}", compression=c)
        for i in range(12)
        for c in ("c23", "c40")
    ]
    keys = sorted({r.key for r in records})
    official = {keys[0]: "test", keys[1]: "test"} | {k: "train" for k in keys[2:9]}
    assignment = assign_official_plus_80_20(records, official, policy="official-train-test")

    assert {k for (k, _), s in assignment.items() if s == "test"} == {keys[0], keys[1]}
    for r in records:
        if r.key in keys[9:]:
            assert (r.key, r.compression) not in assignment
        elif official[r.key] == "train":
            expected = "val" if _md5(r.identity) < 20 else "train"
            assert assignment[(r.key, r.compression)] == expected


@pytest.mark.parametrize("policy", ["test-only-official", "official-train-test"])
def test_official_plus_80_20_rejects_an_official_val(policy):
    records = [rec("CR/a", identity="a"), rec("CR/b", identity="b")]
    with pytest.raises(ContractError, match="val"):
        assign_official_plus_80_20(records, {"CR/a": "test", "CR/b": "val"}, policy=policy)


@pytest.mark.parametrize("policy", ["test-only-official", "official-train-test"])
def test_official_plus_80_20_keeps_explicit_exclusions(policy):
    records = [rec("CR/a", identity="id0"), rec("CR/b", identity="id8"), rec("CR/c", identity="x")]
    official = {"CR/a": "test", "CR/b": "exclude", "CR/c": "train"}
    assignment = assign_official_plus_80_20(records, official, policy=policy)
    assert assignment[("CR/a", None)] == "test"
    assert assignment[("CR/b", None)] == "exclude"
    assert assignment[("CR/c", None)] in {"train", "val"}


def test_official_plus_80_20_rejects_an_unknown_policy():
    with pytest.raises(ContractError, match="policy"):
        assign_official_plus_80_20([rec("CR/a")], {}, policy="official-full")  # type: ignore[arg-type]
