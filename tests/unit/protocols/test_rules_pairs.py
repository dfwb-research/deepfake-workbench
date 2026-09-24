"""The pairing rule (J7): fakes resolved to reals by a real's local key or by identity fan-out.

The rule assertions are ported from the reference pairs builder's tests; the ones about its output
files, folders and per-dataset rule table are not (each dataset's rule lives on its builder).
"""

from __future__ import annotations

import pytest

from dfwb.core.errors import ContractError
from dfwb.core.records import BuilderRef, InventoryRecord, PairRecord, VideoRecord
from dfwb.protocols.rules import resolve_pairs, task_of

REAL_TASKS = frozenset({"REAL", "REAL_A", "REAL_B"})
RANK = {"REAL_A": 0, "REAL_B": 1, "FS_A": 2}


def real(key, identity=None, *, compression=None):
    return VideoRecord(
        key=key,
        compression=compression,
        label_key="X-" + task_of(key),
        method="original",
        identity=identity,
        target_id=identity,
    )


def ffpp_fake(target, source, *, task="FS_DF", compression="c23"):
    stem = f"{target}_{source}"
    return VideoRecord(
        key=f"{task}/{stem}",
        compression=compression,
        label_key=f"FF-{task}",
        method="Deepfakes",
        identity=target,
        target_id=target,
        source_id=source,
        pair_key=stem,
    )


def fake(key, *, identity=None, target_id=None, compression=None):
    return VideoRecord(
        key=key,
        compression=compression,
        label_key="X-" + task_of(key),
        method="m",
        identity=identity,
        target_id=target_id,
    )


def is_real(record) -> bool:
    return task_of(record.key) in REAL_TASKS


def by_target(record):
    return record.target_id


def pairs_of(records, candidates=by_target, *, fanout_cap=None, rule="r", **kwargs):
    return resolve_pairs(
        records, is_real=is_real, candidates=candidates, fanout_cap=fanout_cap, rule=rule, **kwargs
    )


def test_ffpp_target_id_rule_two_reals_four_fakes():
    # Ported: 2 reals + 4 fakes (2 per real) -> 4 pairs with the right keys and rule.
    reals = [real("REAL/001", "001"), real("REAL/002", "002")]
    fakes = [
        ffpp_fake("001", "003"),
        ffpp_fake("001", "004"),
        ffpp_fake("002", "005"),
        ffpp_fake("002", "006"),
    ]
    pairs = pairs_of(reals + fakes, rule="ffpp_target_id")

    assert len(pairs) == 4
    assert all(isinstance(p, PairRecord) for p in pairs)
    assert {p.rule for p in pairs} == {"ffpp_target_id"}
    assert {task_of(p.real_key) for p in pairs} == {"REAL"}
    assert {task_of(p.fake_key) for p in pairs} == {"FS_DF"}
    assert {p.fake_key: p.real_key for p in pairs} == {
        "FS_DF/001_003": "REAL/001",
        "FS_DF/001_004": "REAL/001",
        "FS_DF/002_005": "REAL/002",
        "FS_DF/002_006": "REAL/002",
    }


def test_no_candidates_means_no_pairs():
    # Ported: a dataset without a pairing rule yields no pairs (the writer then writes no file).
    records = [real("DFDC-REAL/real_001", "x"), fake("DFDC-FAKE/fake_001", identity="x")]
    assert pairs_of(records, lambda _fake: None) == []


def test_output_is_deduplicated_and_sorted_by_real_then_fake():
    reals = [real("REAL/002", "002"), real("REAL/001", "001")]
    fakes = [
        ffpp_fake("002", "009", compression="c23"),
        ffpp_fake("002", "009", compression="c40"),  # same key: one pair
        ffpp_fake("001", "008"),
        ffpp_fake("001", "003"),
    ]
    pairs = pairs_of(reals + fakes)
    assert [(p.real_key, p.fake_key) for p in pairs] == [
        ("REAL/001", "FS_DF/001_003"),
        ("REAL/001", "FS_DF/001_008"),
        ("REAL/002", "FS_DF/002_009"),
    ]


def test_the_same_local_fake_key_in_two_tasks_gets_a_pair_each():
    reals = [real("REAL/000", "000")]
    fakes = [ffpp_fake("000", "003", task="FS_DF"), ffpp_fake("000", "003", task="FS_FS")]
    assert [(p.real_key, p.fake_key) for p in pairs_of(reals + fakes)] == [
        ("REAL/000", "FS_DF/000_003"),
        ("REAL/000", "FS_FS/000_003"),
    ]


def test_a_real_with_several_compressions_is_one_pair():
    reals = [real("REAL/001", "001", compression=c) for c in ("c23", "c40")]
    assert len(pairs_of([*reals, ffpp_fake("001", "002")])) == 1


def test_an_unknown_value_gets_no_pair():
    records = [real("REAL/001", "001"), ffpp_fake("404", "002")]
    assert pairs_of(records) == []


def test_identity_fan_out_pairs_every_real_of_that_identity():
    reals = [real(f"REAL/actor1_rec{i}", "actor1") for i in (3, 1, 2)] + [
        real("REAL/actor2_rec1", "actor2"),
        real("REAL/anonymous"),  # a real without an identity is never an identity candidate
    ]
    fakes = [fake("FS_A/f1", identity="actor1")]
    pairs = pairs_of(reals + fakes, lambda f: f.identity)
    assert [p.real_key for p in pairs] == [
        "REAL/actor1_rec1",
        "REAL/actor1_rec2",
        "REAL/actor1_rec3",
    ]


@pytest.mark.parametrize(("cap", "expected"), [(1, ["rec1"]), (2, ["rec1", "rec2"])])
def test_fan_out_cap_keeps_the_first_reals_by_local_key(cap, expected):
    reals = [
        real(f"REAL/rec{i}", "actor1", compression=c) for i in (3, 1, 2) for c in ("c23", "c40")
    ]
    fakes = [fake(f"FS_A/f{i}", identity="actor1") for i in range(3)]
    pairs = pairs_of(reals + fakes, lambda f: f.identity, fanout_cap=cap)
    assert len(pairs) == 3 * cap
    for fake_key in ("FS_A/f0", "FS_A/f1", "FS_A/f2"):
        assert [p.real_key for p in pairs if p.fake_key == fake_key] == [
            f"REAL/{name}" for name in expected
        ]


def test_fan_out_cap_does_not_limit_a_local_key_match():
    # The cap limits identity fan-out only; a direct match to a real's local key is always kept.
    reals = [real("REAL_A/x", "a"), real("REAL_B/x", "b")]
    pairs = pairs_of([*reals, fake("FS_A/f", target_id="x")], fanout_cap=1, task_rank=RANK)
    assert [p.real_key for p in pairs] == ["REAL_A/x", "REAL_B/x"]


def test_fan_out_ties_follow_task_rank():
    reals = [real("REAL_A/x", "actor"), real("REAL_B/x", "actor")]
    fakes = [fake("FS_A/f", identity="actor")]
    first_b = {"REAL_B": 0, "REAL_A": 1, "FS_A": 2}

    def one(rank):
        pairs = pairs_of(reals + fakes, lambda f: f.identity, fanout_cap=1, task_rank=rank)
        return [p.real_key for p in pairs]

    assert one(RANK) == ["REAL_A/x"]
    assert one(first_b) == ["REAL_B/x"]
    assert one(None) == ["REAL_A/x"]  # without a rank: by task name


def test_fan_out_needs_a_rank_for_every_real_task():
    reals = [real("REAL_A/x", "actor"), real("REAL_B/x", "actor")]
    with pytest.raises(ContractError, match="REAL_B"):
        pairs_of(
            [*reals, fake("FS_A/f", identity="actor")],
            lambda f: f.identity,
            task_rank={"REAL_A": 0},
        )


def test_a_local_key_match_wins_over_identity_fan_out():
    reals = [real("REAL/actor1", "someone"), real("REAL/actor1_rec1", "actor1")]
    pairs = pairs_of([*reals, fake("FS_A/f", identity="actor1")], lambda f: f.identity)
    assert [p.real_key for p in pairs] == ["REAL/actor1"]


def test_several_candidate_values_are_each_resolved():
    reals = [real("REAL/tgt", "t"), real("REAL/a1", "actor"), real("REAL/a2", "actor")]
    pairs = pairs_of(
        [*reals, fake("FS_A/f")], lambda _f: ["tgt", "actor", "", "missing"], fanout_cap=1
    )
    assert [p.real_key for p in pairs] == ["REAL/a1", "REAL/tgt"]


def test_is_real_decides_which_side_a_record_is_on():
    # A fake is never paired as a real, and a real is never looked up as a fake.
    records = [real("REAL/001", "001"), ffpp_fake("001", "002"), fake("FS_A/001", identity="001")]
    pairs = pairs_of(records)
    assert [(p.real_key, p.fake_key) for p in pairs] == [("REAL/001", "FS_DF/001_002")]


def test_resolve_pairs_accepts_inventory_records():
    videos = [real("REAL/001", "001"), ffpp_fake("001", "002")]
    inventory = [
        InventoryRecord(
            key=v.key,
            compression=v.compression,
            label_key=v.label_key,
            method=v.method,
            relpath=f"{v.key}.mp4",
            builder=BuilderRef("fixture", "1"),
            identity=v.identity,
            target_id=v.target_id,
        )
        for v in videos
    ]
    assert pairs_of(inventory) == pairs_of(videos)


def test_a_negative_fan_out_cap_is_rejected():
    with pytest.raises(ContractError, match="fan-out"):
        pairs_of([real("REAL/001", "001")], fanout_cap=-1)
