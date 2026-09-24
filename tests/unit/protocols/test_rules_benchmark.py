"""The benchmark rule: a seeded, stratified test subset with balanced reals.

The rule assertions are ported from the reference benchmark selector's tests and those of its
sampling helpers; the ones about its source files, output files and folders are not.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from dfwb.core.errors import ContractError
from dfwb.core.records import BuilderRef, InventoryRecord, VideoRecord
from dfwb.protocols.rules import BenchmarkSpec, assign_benchmark, local_key, task_of

TASK_RANK = {"REAL": 0, "RVFA": 1, "FS_A": 2, "FS_B": 3, "FR_DAGAN": 4, "FS_SBI": 5}
REAL_TASKS = frozenset({"REAL", "RVFA"})


def rec(key, *, compression=None, identity=None, target_id=None, source_id=None):
    return VideoRecord(
        key=key,
        compression=compression,
        label_key="X-" + task_of(key),
        method="m",
        identity=identity,
        target_id=target_id,
        source_id=source_id,
    )


def is_real(record) -> bool:
    return task_of(record.key) in REAL_TASKS


def run(records, *, pool_keys=None, task_rank=TASK_RANK, **spec):
    return assign_benchmark(
        records,
        spec=BenchmarkSpec(**spec),
        is_real=is_real,
        task_rank=task_rank,
        pool_keys=pool_keys,
    )


def picked(assignment) -> list[tuple[str, str | None]]:
    return sorted(assignment)


def fakes_of(assignment) -> list[str]:
    return sorted(k for k, _ in assignment if task_of(k) not in REAL_TASKS)


def reals_of(assignment) -> list[str]:
    return sorted(k for k, _ in assignment if task_of(k) in REAL_TASKS)


def _sort_key(r):
    return (local_key(r.key), TASK_RANK[task_of(r.key)], r.compression or "")


# ---------------------------------------------------------------------------------------------
# Exact draws
# ---------------------------------------------------------------------------------------------


def test_benchmark_is_seed_deterministic_and_matches_a_hand_computed_draw():
    fakes = [rec(f"FS_A/f{i:02d}") for i in range(30)]
    reals = [rec(f"REAL/r{i:02d}") for i in range(50)]
    records = fakes + reals
    random.Random(7).shuffle(records)  # input order must not matter

    rng = random.Random(0)
    expected_fakes = rng.sample(sorted(fakes, key=_sort_key), 5)
    expected_reals = rng.sample(sorted(reals, key=_sort_key), 5)
    expected = {(r.key, r.compression): "test" for r in [*expected_fakes, *expected_reals]}

    assert run(records, k_fake=5) == expected
    assert run(list(reversed(records)), k_fake=5) == expected
    assert run(records, k_fake=5, seed=1) != expected


def test_benchmark_draws_strata_in_sorted_group_order_then_reals_on_the_same_rng():
    # Groups arrive out of order, one has a missing identity (-> ""), and two are larger than k.
    fakes = (
        [rec(f"FS_B/b{i}", identity="zed") for i in range(4)]
        + [rec(f"FS_A/a{i}", identity="amy") for i in range(5)]
        + [rec(f"FS_A/n{i}") for i in range(3)]
        + [rec(f"FS_B/c{i}", identity="amy") for i in range(2)]
    )
    reals = [rec(f"REAL/r{i:02d}") for i in range(20)]
    records = reals + fakes

    groups: dict[tuple[str, ...], list[VideoRecord]] = {}
    for r in sorted(fakes, key=_sort_key):
        groups.setdefault((r.identity or "", task_of(r.key)), []).append(r)
    rng = random.Random(0)
    expected_fakes: list[VideoRecord] = []
    for group in sorted(groups):  # ("", FS_A) < (amy, FS_A) < (amy, FS_B) < (zed, FS_B)
        pool = groups[group]
        expected_fakes.extend(pool if len(pool) <= 2 else rng.sample(pool, 2))
    expected_reals = rng.sample(sorted(reals, key=_sort_key), len(expected_fakes))

    assignment = run(records, k_fake=2, strata=("identity", "task"))
    assert fakes_of(assignment) == sorted(r.key for r in expected_fakes)
    assert reals_of(assignment) == sorted(r.key for r in expected_reals)
    assert len(expected_fakes) == 8  # 2 + 2 + 2 + 2


def test_benchmark_ties_follow_task_rank():
    # The same local key in two tasks: the pool order, and so the draw, follows the task rank.
    records = [rec("FS_A/x"), rec("FS_B/x")]
    rank_b_first = {**TASK_RANK, "FS_B": 1, "FS_A": 9}
    rank_a_first = {**TASK_RANK, "FS_A": 1, "FS_B": 9}

    by_b = random.Random(0).sample(["FS_B/x", "FS_A/x"], 1)
    by_a = random.Random(0).sample(["FS_A/x", "FS_B/x"], 1)
    assert by_a != by_b  # the fixture would prove nothing otherwise

    assert fakes_of(run(records, k_fake=1, task_rank=rank_b_first)) == by_b
    assert fakes_of(run(records, k_fake=1, task_rank=rank_a_first)) == by_a


def test_benchmark_ties_within_a_task_follow_compression():
    records = [rec("FS_A/x", compression="c40"), rec("FS_A/x", compression="c23")]
    expected = random.Random(0).sample(["c23", "c40"], 1)
    assert [c for _, c in run(records, k_fake=1)] == expected


def test_benchmark_needs_a_rank_for_every_task():
    with pytest.raises(ContractError, match="FS_Z"):
        run([rec("FS_Z/x"), rec("REAL/r")], k_fake=1)


# ---------------------------------------------------------------------------------------------
# Ported: the benchmark selector
# ---------------------------------------------------------------------------------------------


def test_stratified_sample_one_per_cell():
    fakes = (
        [rec(f"FR_DAGAN/f_id1_t1_{i}", identity="id1") for i in range(3)]
        + [rec(f"FS_SBI/f_id1_t2_{i}", identity="id1") for i in range(3)]
        + [rec(f"FR_DAGAN/f_id2_t1_{i}", identity="id2") for i in range(3)]
    )
    assignment = run(fakes, k_fake=1, strata=("identity", "task"))
    assert len(assignment) == 3  # 3 cells x 1 each, no reals
    by_key = {r.key: r for r in fakes}
    cells = {(by_key[k].identity, task_of(k)) for k, _ in assignment}
    assert cells == {("id1", "FR_DAGAN"), ("id1", "FS_SBI"), ("id2", "FR_DAGAN")}


def test_all_reals_unless_greater_rule():
    reals = [rec(f"REAL/r{i:03d}", identity=f"id{i}") for i in range(10)]

    # 10 reals, 2 fakes -> reals trimmed to 2.
    fakes_small = [rec(f"FR_DAGAN/fA{i}", identity=f"id{i}") for i in range(2)]
    a = run(reals + fakes_small, k_fake=10)
    assert len(fakes_of(a)) == 2
    assert len(reals_of(a)) == 2
    assert len(a) == 4

    # 10 reals, 50 fakes -> all 10 reals kept.
    fakes_big = [rec(f"FR_DAGAN/fB{i:03d}", identity=f"id_b{i}") for i in range(50)]
    b = run(reals + fakes_big, k_fake=50)
    assert len(fakes_of(b)) == 50
    assert len(reals_of(b)) == 10


def test_k_real_cap_clamps():
    reals = [rec(f"REAL/r{i:03d}", identity=f"id{i}") for i in range(20)]
    fakes = [rec(f"FR_DAGAN/f{i:03d}", identity=f"idf{i}") for i in range(30)]
    assignment = run(reals + fakes, k_fake=30, k_real_cap=5)
    assert len(fakes_of(assignment)) == 30  # k_fake == pool: every fake kept
    assert len(reals_of(assignment)) == 5  # all 20 reals, then clamped by the cap
    assert len(assignment) == 35


def test_excluded_tasks_are_dropped_before_the_draw():
    records = [
        rec("REAL/real_1", identity="id00"),
        rec("REAL/real_2", identity="id01"),
        rec("FS_AM_TM/fs_1", identity="id00"),
        rec("LS_AM_TM/ls_1", identity="id01"),
        rec("RVC_TM/rvc_drop", identity="id00"),
        rec("TTS_TG/tts_g_drop", identity="id00"),
        rec("TTS_TM/tts_m_drop", identity="id00"),
    ]
    rank = {"REAL": 0, "FS_AM_TM": 1, "LS_AM_TM": 2, "RVC_TM": 3, "TTS_TG": 4, "TTS_TM": 5}
    assignment = run(
        records,
        task_rank=rank,
        k_fake=5,
        strata=("identity", "task"),
        exclude_tasks=("RVC_TM", "TTS_TG", "TTS_TM"),
    )
    assert {task_of(k) for k, _ in assignment}.isdisjoint({"RVC_TM", "TTS_TG", "TTS_TM"})
    assert fakes_of(assignment) == ["FS_AM_TM/fs_1", "LS_AM_TM/ls_1"]  # two distinct cells


def test_every_selected_record_is_test():
    reals = [rec(f"REAL/r{i}", identity=f"id{i}") for i in range(5)]
    fakes = [rec(f"FR_DAGAN/f{i}", identity=f"idx{i}") for i in range(3)]
    assignment = run(reals + fakes, k_fake=2)
    assert set(assignment.values()) == {"test"}
    assert len(assignment) == 4


# ---------------------------------------------------------------------------------------------
# Ported: the sampling helpers
# ---------------------------------------------------------------------------------------------


def test_reals_all_pass_when_pool_is_not_larger_than_n_fakes():
    reals = [rec("REAL/r3"), rec("REAL/r1"), rec("REAL/r2")]
    fakes = [rec(f"FS_A/f{i}") for i in range(5)]
    assert reals_of(run(reals + fakes, k_fake=5)) == ["REAL/r1", "REAL/r2", "REAL/r3"]


def test_reals_are_subsampled_to_n_fakes():
    reals = [rec(f"REAL/r{i:02d}") for i in range(10)]
    fakes = [rec(f"FS_A/f{i}") for i in range(3)]  # 3 <= k_fake: kept without a draw
    expected = random.Random(0).sample([f"REAL/r{i:02d}" for i in range(10)], 3)
    assert reals_of(run(reals + fakes, k_fake=3)) == sorted(expected)


def test_real_cap_is_a_second_draw_on_the_balanced_sample():
    reals = [rec(f"REAL/r{i:02d}") for i in range(10)]
    fakes = [rec(f"FS_A/f{i}") for i in range(8)]
    rng = random.Random(0)
    balanced = rng.sample([f"REAL/r{i:02d}" for i in range(10)], 8)
    expected = rng.sample(balanced, 4)  # samples the draw order, not a re-sorted list

    first = run(reals + fakes, k_fake=8, k_real_cap=4)
    assert reals_of(first) == sorted(expected)
    assert run(reals + fakes, k_fake=8, k_real_cap=4) == first


def test_real_cap_is_a_no_op_below_the_threshold():
    reals = [rec(f"REAL/r{i:02d}") for i in range(3)]
    fakes = [rec(f"FS_A/f{i}") for i in range(10)]
    assert reals_of(run(reals + fakes, k_fake=10, k_real_cap=100)) == [
        "REAL/r00",
        "REAL/r01",
        "REAL/r02",
    ]


def test_stratified_draw_is_independent_of_input_order():
    fakes = [rec(f"FS_A/{cell}_{j}", identity=cell) for cell in "ABC" for j in range(5)]
    out1 = run(fakes, k_fake=2, strata=("identity",))
    out2 = run(list(reversed(fakes)), k_fake=2, strata=("identity",))
    assert out1 == out2
    assert len(out1) == 6
    assert Counter(local_key(k)[0] for k, _ in out1) == {"A": 2, "B": 2, "C": 2}


def test_stratified_draw_keeps_a_cell_smaller_than_k():
    fakes = [rec("FS_A/X_0", identity="X")] + [rec(f"FS_A/Y_{i}", identity="Y") for i in range(3)]
    assert fakes_of(run(fakes, k_fake=3, strata=("identity",))) == [
        "FS_A/X_0",
        "FS_A/Y_0",
        "FS_A/Y_1",
        "FS_A/Y_2",
    ]


@pytest.mark.parametrize("strata", [(), ("identity",)])
def test_k_fake_zero_selects_nothing(strata):
    records = [rec("FS_A/a", identity="A"), rec("REAL/r", identity="A")]
    assert run(records, k_fake=0, strata=strata) == {}


def test_stratified_draw_supports_several_strata():
    fakes = [
        rec("FS_A/a", identity="X", target_id="Y1"),
        rec("FS_A/b", identity="X", target_id="Y2"),
        rec("FS_A/c", identity="Z", target_id="Y1"),
    ]
    assert len(run(fakes, k_fake=1, strata=("identity", "target_id"))) == 3


def test_source_id_is_a_stratum_too():
    fakes = [rec(f"FS_A/f{i}", source_id=f"s{i % 2}") for i in range(6)]
    assert len(run(fakes, k_fake=1, strata=("source_id",))) == 2


def test_is_real_decides_the_pools():
    # A visually untouched task counts as real when is_real says so (e.g. real video, fake audio).
    records = [rec(f"RVFA/v{i}") for i in range(4)] + [rec(f"FS_A/f{i}") for i in range(2)]
    assignment = run(records, k_fake=2)
    assert len(fakes_of(assignment)) == 2
    assert len(reals_of(assignment)) == 2
    assert all(task_of(k) == "RVFA" for k in reals_of(assignment))


# ---------------------------------------------------------------------------------------------
# The source pool
# ---------------------------------------------------------------------------------------------


def test_pool_keys_restrict_the_pool_to_every_compression_of_those_keys():
    records = [
        rec("REAL/r1", compression="c23"),
        rec("REAL/r1", compression="c40"),
        rec("REAL/r2", compression="c23"),
        rec("FS_A/f1", compression="c23"),
        rec("FS_A/f1", compression="c40"),
        rec("FS_A/f2", compression="c23"),
    ]
    assignment = run(records, k_fake=5, pool_keys={"REAL/r1", "FS_A/f1"})
    assert picked(assignment) == [
        ("FS_A/f1", "c23"),
        ("FS_A/f1", "c40"),
        ("REAL/r1", "c23"),
        ("REAL/r1", "c40"),
    ]


@pytest.mark.parametrize("pool_keys", [None, set(), []])
def test_no_official_test_means_every_record(pool_keys):
    # None, or an empty official test (a dataset whose official scheme publishes no test), means
    # the whole dataset is the pool.
    records = [rec("REAL/r1"), rec("FS_A/f1")]
    assert picked(run(records, k_fake=5, pool_keys=pool_keys)) == [
        ("FS_A/f1", None),
        ("REAL/r1", None),
    ]


def test_benchmark_accepts_inventory_records():
    videos = [rec(f"FS_A/f{i:02d}", identity=f"id{i % 3}") for i in range(12)] + [
        rec(f"REAL/r{i:02d}", identity=f"id{i}") for i in range(12)
    ]
    inventory = [
        InventoryRecord(
            key=v.key,
            compression=v.compression,
            label_key=v.label_key,
            method=v.method,
            relpath=f"{v.key}.mp4",
            builder=BuilderRef("fixture", "1"),
            identity=v.identity,
        )
        for v in videos
    ]
    spec = {"k_fake": 2, "strata": ("identity",)}
    assert run(inventory, **spec) == run(videos, **spec)


# ---------------------------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------------------------


def test_benchmark_spec_defaults():
    spec = BenchmarkSpec(k_fake=100)
    assert spec.strata == ()
    assert spec.k_real_cap is None
    assert spec.exclude_tasks == ()
    assert spec.seed == 0


def test_benchmark_spec_rejects_an_unknown_stratum():
    with pytest.raises(ContractError, match=r"attributes\.identity"):
        BenchmarkSpec(k_fake=2, strata=("attributes.identity",))
