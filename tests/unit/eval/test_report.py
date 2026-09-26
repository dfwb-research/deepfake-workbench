"""``evaluate()``: the files/breakdown/seeds/suite tables, and the coverage exit code."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dfwb.core.records import ScoreRow, write_scores
from dfwb.eval.metrics import auc
from dfwb.eval.report import evaluate
from dfwb.eval.suites import Suite

from .conftest import make_meta, make_rows


def _write(tmp_path, name, rows, meta):
    path, _ = write_scores(tmp_path / name, rows, meta)
    return path


def test_evaluate_basic_file_table(tmp_path):
    rows = make_rows(20, 20)
    meta = make_meta(coverage={"expected": 40, "ok": 40, "missing": 0, "error": 0})
    path = _write(tmp_path, "a.scores.csv", rows, meta)
    result = evaluate([path], metrics=["auc", "acc@thr=0.5"], bootstrap=100)
    assert result.exit_code == 0
    file_row = result.tables["files"][0]
    assert file_row["n"] == 40
    assert file_row["coverage"] == 1.0
    assert 0.9 < file_row["metrics"]["auc"]["value"] <= 1.0
    assert file_row["metrics"]["auc"]["ci_lo"] <= file_row["metrics"]["auc"]["value"]
    assert "breakdown" not in result.tables
    assert "seeds" not in result.tables
    assert "suite" not in result.tables


def test_evaluate_to_json_round_trips_through_json(tmp_path):
    rows = make_rows(10, 10)
    meta = make_meta(coverage={"expected": 20, "ok": 20, "missing": 0, "error": 0})
    path = _write(tmp_path, "a.scores.csv", rows, meta)
    result = evaluate([path], metrics=["auc"], bootstrap=50)
    payload = json.loads(json.dumps(result.to_json()))
    assert payload["exit_code"] == 0
    assert payload["tables"]["files"][0]["metrics"]["auc"]["value"] == pytest.approx(
        result.tables["files"][0]["metrics"]["auc"]["value"]
    )


def test_evaluate_file_table_n_reflects_the_missing_policy_not_all_rows(tmp_path):
    rows = [*make_rows(10, 10), ScoreRow("d", "m0", None, 0, None, "missing")]
    meta = make_meta(coverage={"expected": 21, "ok": 20, "missing": 1, "error": 0})
    path = _write(tmp_path, "a.scores.csv", rows, meta)

    excluded = evaluate([path], metrics=["auc"], bootstrap=50, missing="exclude")
    excluded_row = excluded.tables["files"][0]
    assert excluded_row["n"] == 20  # only the "ok" rows were fed to the metric
    assert excluded_row["expected"] == 21  # coverage's total is unaffected by the policy

    chance = evaluate([path], metrics=["auc"], bootstrap=50, missing="as-chance")
    chance_row = chance.tables["files"][0]
    assert chance_row["n"] == 21  # the missing row was filled in and fed to the metric too
    assert chance_row["expected"] == 21


def test_evaluate_breakdown_by_method(tmp_path, family_pack):
    rows = [
        *[_row("REAL", i, "c23", 0, 0.15, "TOYONE-REAL", "original") for i in range(10)],
        *[_row("FAKE_A", i, None, 1, 0.85, "TOYONE-FAKE_A", "FakeA") for i in range(10)],
        *[_row("FAKE_B", i, None, 1, 0.75, "TOYONE-FAKE_B", "FakeB") for i in range(10)],
    ]
    meta = make_meta(coverage={"expected": 30, "ok": 30, "missing": 0, "error": 0})
    path = _write(tmp_path, "a.scores.csv", rows, meta)
    result = evaluate([path], metrics=["acc@thr=0.5"], by="method", bootstrap=100)
    by_group = {row["group"]: row for row in result.tables["breakdown"]}
    assert set(by_group) == {"original", "FakeA", "FakeB"}
    # "original" (the real group) is a plain partition: nothing to combine it with.
    assert by_group["original"]["n"] == 10
    # each fake method's group is evaluated against every real row too (n = 10 reals + its own
    # 10 fakes), not just its own rows.
    assert by_group["FakeA"]["n"] == 20
    assert by_group["FakeB"]["n"] == 20
    for row in by_group.values():
        assert row["metrics"]["acc@thr=0.5"]["value"] == pytest.approx(1.0)


def test_evaluate_breakdown_combines_each_fake_group_with_all_reals(tmp_path, family_pack):
    reals = [
        _row("REAL", i, "c23", 0, 0.1 + 0.01 * i, "TOYONE-REAL", "original") for i in range(20)
    ]
    fake_a = [
        _row("FAKE_A", i, None, 1, 0.55 + 0.01 * i, "TOYONE-FAKE_A", "FakeA") for i in range(10)
    ]
    fake_b = [
        _row("FAKE_B", i, None, 1, 0.80 + 0.01 * i, "TOYONE-FAKE_B", "FakeB") for i in range(10)
    ]
    rows = reals + fake_a + fake_b
    meta = make_meta(coverage={"expected": 40, "ok": 40, "missing": 0, "error": 0})
    path = _write(tmp_path, "a.scores.csv", rows, meta)
    result = evaluate([path], metrics=["auc"], by="method", bootstrap=200)
    by_group = {row["group"]: row for row in result.tables["breakdown"]}

    # a per-method AUC is, by hand, this method's fakes evaluated against *every* real row --
    # never just the method's own rows (which would have no reals to compare against at all).
    y_a = np.array([0] * len(reals) + [1] * len(fake_a))
    p_a = np.array([r.score for r in reals] + [r.score for r in fake_a])
    assert by_group["FakeA"]["n"] == 30
    assert by_group["FakeA"]["metrics"]["auc"]["value"] == pytest.approx(auc(y_a, p_a))

    y_b = np.array([0] * len(reals) + [1] * len(fake_b))
    p_b = np.array([r.score for r in reals] + [r.score for r in fake_b])
    assert by_group["FakeB"]["n"] == 30
    assert by_group["FakeB"]["metrics"]["auc"]["value"] == pytest.approx(auc(y_b, p_b))

    # the real group itself is a plain partition (nothing is added to it), so a two-class metric
    # like auc has no data to be defined on and is left out of its row, as before.
    assert by_group["original"]["n"] == 20
    assert "auc" not in by_group["original"]["metrics"]


def test_evaluate_breakdown_by_family_combines_fake_families_with_all_reals(tmp_path, family_pack):
    rows = [
        *[_row("REAL", i, "c23", 0, 0.15, "TOYONE-REAL", "original") for i in range(10)],
        *[_row("FAKE_A", i, None, 1, 0.85, "TOYONE-FAKE_A", "FakeA") for i in range(10)],
    ]
    meta = make_meta(coverage={"expected": 20, "ok": 20, "missing": 0, "error": 0})
    path = _write(tmp_path, "a.scores.csv", rows, meta)
    result = evaluate([path], metrics=["auc", "acc@thr=0.5"], by="family", bootstrap=50)
    by_group = {row["group"]: row for row in result.tables["breakdown"]}
    # "face-swap" (FakeA's family) now has both classes, via the added reals -- auc is defined.
    assert by_group["face-swap"]["n"] == 20
    assert by_group["face-swap"]["metrics"]["auc"]["value"] == pytest.approx(1.0)
    assert "acc@thr=0.5" in by_group["face-swap"]["metrics"]
    # "real" is the real group itself: still a plain partition, still no two-class metric.
    assert by_group["real"]["n"] == 10
    assert "auc" not in by_group["real"]["metrics"]
    assert "acc@thr=0.5" in by_group["real"]["metrics"]


def test_evaluate_breakdown_by_compression_stays_a_plain_partition(tmp_path, family_pack):
    rows = [
        *[_row("REAL", i, "c23", 0, 0.2, "TOYONE-REAL", "original") for i in range(5)],
        *[_row("REAL", i, "c40", 0, 0.3, "TOYONE-REAL", "original") for i in range(5, 10)],
        *[_row("FAKE_A", i, "c23", 1, 0.8, "TOYONE-FAKE_A", "FakeA") for i in range(5)],
    ]
    meta = make_meta(coverage={"expected": 15, "ok": 15, "missing": 0, "error": 0})
    path = _write(tmp_path, "a.scores.csv", rows, meta)
    result = evaluate([path], metrics=["auc"], by="compression", bootstrap=50)
    by_group = {row["group"]: row for row in result.tables["breakdown"]}
    # c23 already has both classes on its own (5 real + 5 fake) -- no reals are added to it.
    assert by_group["c23"]["n"] == 10
    # c40 is real-only and stays that way: no fakes exist at c40 to combine with, so auc is
    # simply undefined for it, not silently borrowed from another compression level.
    assert by_group["c40"]["n"] == 5
    assert "auc" not in by_group["c40"]["metrics"]


def test_evaluate_multi_seed_summary(tmp_path):
    rows_a = make_rows(20, 20, fake_score=0.8)
    rows_b = make_rows(20, 20, fake_score=0.9)
    cov = {"expected": 40, "ok": 40, "missing": 0, "error": 0}
    path_a = _write(tmp_path, "seed0.scores.csv", rows_a, make_meta(seed=0, coverage=cov))
    path_b = _write(tmp_path, "seed1.scores.csv", rows_b, make_meta(seed=1, coverage=cov))
    result = evaluate([path_a, path_b], metrics=["auc"], bootstrap=50)
    assert len(result.tables["seeds"]) == 1
    seed_row = result.tables["seeds"][0]
    assert seed_row["metric"] == "auc"
    assert seed_row["seeds"] == [0, 1]
    assert seed_row["mean"] == pytest.approx((seed_row["values"][0] + seed_row["values"][1]) / 2)
    file_points = [r["metrics"]["auc"]["value"] for r in result.tables["files"]]
    assert seed_row["values"] == pytest.approx(file_points)


def test_evaluate_different_protocols_are_not_treated_as_multi_seed(tmp_path):
    rows = make_rows(10, 10)
    cov = {"expected": 20, "ok": 20, "missing": 0, "error": 0}
    path_a = _write(
        tmp_path,
        "a.scores.csv",
        rows,
        make_meta(seed=0, protocol={**_PROTO, "id": "x/official"}, coverage=cov),
    )
    path_b = _write(
        tmp_path,
        "b.scores.csv",
        rows,
        make_meta(seed=1, protocol={**_PROTO, "id": "y/official"}, coverage=cov),
    )
    result = evaluate([path_a, path_b], metrics=["auc"], bootstrap=20)
    assert "seeds" not in result.tables


_PROTO = {
    "id": "toyone/official",
    "split": "test",
    "where": {},
    "pack": "toyone-pack",
    "pack_version": "1.0.0",
    "scheme_sha256": "e" * 64,
}


def test_evaluate_suite_aggregate(tmp_path):
    rows = make_rows(15, 15)
    cov = {"expected": 30, "ok": 30, "missing": 0, "error": 0}
    path_a = _write(
        tmp_path,
        "a.scores.csv",
        rows,
        make_meta(protocol={**_PROTO, "id": "alpha/official"}, coverage=cov),
    )
    path_b = _write(
        tmp_path,
        "b.scores.csv",
        make_rows(15, 15, fake_score=0.6),
        make_meta(protocol={**_PROTO, "id": "beta/official"}, coverage=cov),
    )
    suite = Suite.model_validate(
        {
            "name": "demo",
            "entries": [
                {"protocol": "alpha/official", "split": "test", "group": "in-domain"},
                {"protocol": "beta/official", "split": "test", "group": "cross-dataset"},
            ],
            "aggregates": [
                {"group": "in-domain", "metric": "auc", "how": "mean"},
                {"group": "cross-dataset", "metric": "auc", "how": "mean"},
            ],
        }
    )
    result = evaluate([path_a, path_b], metrics=["auc"], suite=suite, bootstrap=50)
    suite_rows = {row["group"]: row for row in result.tables["suite"]}
    file_points = {r["file"]: r["metrics"]["auc"]["value"] for r in result.tables["files"]}
    assert suite_rows["in-domain"]["value"] == pytest.approx(file_points["a.scores.csv"])
    assert suite_rows["cross-dataset"]["value"] == pytest.approx(file_points["b.scores.csv"])


def _row(task, i, compression, label, score, label_key, method):
    return ScoreRow(
        "toyone",
        f"{task}/v{i:03d}",
        compression,
        label,
        score,
        "ok",
        label_key=label_key,
        method=method,
    )


# ------------------------------------------------------------------ list-valued ``where`` filters


def test_evaluate_seed_grouping_handles_a_list_valued_where(tmp_path):
    """An any-of ``where`` (``--where method=a --where method=b``) is stored as a list; grouping
    files into "the same run, another seed" must not need it to be hashable."""
    cov = {"expected": 40, "ok": 40, "missing": 0, "error": 0}
    where = {"method": ["Deepfakes", "original"], "compression": "c23"}
    protocol = {**_PROTO, "where": where}
    path_a = _write(
        tmp_path,
        "s0.scores.csv",
        make_rows(20, 20),
        make_meta(seed=0, protocol=protocol, coverage=cov),
    )
    path_b = _write(
        tmp_path,
        "s1.scores.csv",
        make_rows(20, 20, fake_score=0.9),
        make_meta(seed=1, protocol=protocol, coverage=cov),
    )

    result = evaluate([path_a, path_b], metrics=["auc"], bootstrap=0)

    assert [row["seeds"] for row in result.tables["seeds"]] == [[0, 1]]


def test_evaluate_seed_grouping_treats_list_order_as_irrelevant(tmp_path):
    cov = {"expected": 40, "ok": 40, "missing": 0, "error": 0}
    first = {**_PROTO, "where": {"method": ["Deepfakes", "original"]}}
    second = {**_PROTO, "where": {"method": ["original", "Deepfakes"]}}
    path_a = _write(
        tmp_path,
        "s0.scores.csv",
        make_rows(20, 20),
        make_meta(seed=0, protocol=first, coverage=cov),
    )
    path_b = _write(
        tmp_path,
        "s1.scores.csv",
        make_rows(20, 20),
        make_meta(seed=1, protocol=second, coverage=cov),
    )

    result = evaluate([path_a, path_b], metrics=["auc"], bootstrap=0)

    assert [row["seeds"] for row in result.tables["seeds"]] == [[0, 1]]


def test_evaluate_suite_matches_a_list_where_whatever_its_order(tmp_path):
    """A suite entry written the way dfwb-protocols' ``in-domain-ffpp`` writes one (``method:
    [original, Deepfakes]``) matches the file ``dfwb score`` wrote for it, whose meta holds the
    same list in canonical (sorted) order."""
    cov = {"expected": 40, "ok": 40, "missing": 0, "error": 0}
    stored = {"method": ["Deepfakes", "original"], "compression": "c23"}
    path = _write(
        tmp_path,
        "df.scores.csv",
        make_rows(20, 20),
        make_meta(
            seed=None, protocol={**_PROTO, "id": "ffpp/official", "where": stored}, coverage=cov
        ),
    )
    suite = Suite.model_validate(
        {
            "name": "in-domain-like",
            "entries": [
                {
                    "protocol": "ffpp/official",
                    "split": "test",
                    "where": {"method": ["original", "Deepfakes"], "compression": "c23"},
                    "group": "in-domain",
                }
            ],
            "aggregates": [{"group": "in-domain", "metric": "auc", "how": "mean"}],
        }
    )

    result = evaluate([path], metrics=["auc"], suite=suite, bootstrap=0)

    (row,) = result.tables["suite"]
    assert row["n_entries"] == 1
    assert row["value"] == pytest.approx(result.tables["files"][0]["metrics"]["auc"]["value"])


# ------------------------------------------------------------------------- undefined metric cells


def _all_fake_rows(n):
    return [ScoreRow("d", f"fake/{i:04d}", None, 1, 0.8, "ok") for i in range(n)]


def test_a_single_class_file_gets_undefined_cells_not_an_abort(tmp_path):
    good = _write(
        tmp_path,
        "good.scores.csv",
        make_rows(10, 10),
        make_meta(seed=None, coverage={"expected": 20, "ok": 20, "missing": 0, "error": 0}),
    )
    one_class = _write(
        tmp_path,
        "fakes.scores.csv",
        _all_fake_rows(10),
        make_meta(seed=None, coverage={"expected": 10, "ok": 10, "missing": 0, "error": 0}),
    )

    result = evaluate([good, one_class], metrics=["auc", "brier"], bootstrap=0)

    assert result.exit_code == 0
    rows = {row["file"]: row for row in result.tables["files"]}
    assert rows["good.scores.csv"]["metrics"]["auc"]["value"] == pytest.approx(1.0)
    undefined = rows["fakes.scores.csv"]["metrics"]["auc"]
    assert undefined["value"] is None
    assert undefined["ci_lo"] is None
    assert undefined["ci_hi"] is None
    assert "every label is 'fake'" in undefined["undefined"]
    # a metric that is defined on one class is still reported for the same file
    assert rows["fakes.scores.csv"]["metrics"]["brier"]["value"] is not None
    json.loads(json.dumps(result.to_json(), allow_nan=False))


def test_an_undefined_metric_is_left_out_of_the_seed_and_suite_numbers(tmp_path):
    cov = {"expected": 20, "ok": 20, "missing": 0, "error": 0}
    path_a = _write(tmp_path, "s0.scores.csv", make_rows(10, 10), make_meta(seed=0, coverage=cov))
    path_b = _write(
        tmp_path,
        "s1.scores.csv",
        _all_fake_rows(20),
        make_meta(seed=1, coverage=cov),
    )

    result = evaluate([path_a, path_b], metrics=["auc"], bootstrap=0)

    assert "seeds" not in result.tables  # only one seed has a defined auc


def test_evaluate_raises_only_when_no_metric_is_defined_for_any_file(tmp_path):
    from dfwb.eval.metrics import MetricUndefined

    cov = {"expected": 10, "ok": 10, "missing": 0, "error": 0}
    path_a = _write(
        tmp_path, "a.scores.csv", _all_fake_rows(10), make_meta(seed=None, coverage=cov)
    )
    path_b = _write(
        tmp_path, "b.scores.csv", _all_fake_rows(10), make_meta(seed=None, coverage=cov)
    )

    with pytest.raises(MetricUndefined, match="no requested metric") as info:
        evaluate([path_a, path_b], metrics=["auc", "eer"], bootstrap=0)

    assert info.value.exit_code == 4
    assert "every label is 'fake'" in info.value.message


# ------------------------------------------------------------------ the seeds table's identity


def test_two_files_with_one_identity_and_seed_warn_and_keep_the_first(tmp_path, caplog):
    """``#best`` and ``#last`` of one run share the detector source and the seed; folding both
    into the seeds table as if they were two seeds would be wrong, and so is silently dropping
    one. The first file given is kept, and a warning says which was ignored."""
    import logging

    cov = {"expected": 40, "ok": 40, "missing": 0, "error": 0}
    best = _write(tmp_path, "best.scores.csv", make_rows(20, 20), make_meta(seed=0, coverage=cov))
    last = _write(
        tmp_path,
        "last.scores.csv",
        make_rows(20, 20, fake_score=0.1),  # an AUC of 0.0: easy to tell apart from best's 1.0
        make_meta(seed=0, coverage=cov),
    )
    other_seed = _write(
        tmp_path, "s1.scores.csv", make_rows(20, 20), make_meta(seed=1, coverage=cov)
    )

    with caplog.at_level(logging.WARNING, logger="dfwb"):
        result = evaluate([best, last, other_seed], metrics=["auc"], bootstrap=0)

    (seed_row,) = result.tables["seeds"]
    assert seed_row["seeds"] == [0, 1]
    assert seed_row["values"] == pytest.approx([1.0, 1.0])  # best's value, not last's
    warnings = [m for m in caplog.messages if "last.scores.csv" in m]
    assert len(warnings) == 1
    assert "best.scores.csv" in warnings[0]
    assert "keeping" in warnings[0]


@pytest.mark.parametrize(
    "change",
    [
        {"labels": "family"},
        {"aggregation": {"clip_to_video": "max", "clips_per_video": 1}},
        {"processing_profile": {"id": "other-00000000", "sha256": "f" * 64}},
    ],
    ids=["labels", "aggregation", "processing_profile"],
)
def test_files_scored_differently_are_not_seeds_of_one_run(tmp_path, change):
    cov = {"expected": 40, "ok": 40, "missing": 0, "error": 0}
    path_a = _write(tmp_path, "s0.scores.csv", make_rows(20, 20), make_meta(seed=0, coverage=cov))
    path_b = _write(
        tmp_path, "s1.scores.csv", make_rows(20, 20), make_meta(seed=1, coverage=cov, **change)
    )

    result = evaluate([path_a, path_b], metrics=["auc"], bootstrap=0)

    assert "seeds" not in result.tables
