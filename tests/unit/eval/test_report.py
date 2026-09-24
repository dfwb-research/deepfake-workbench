"""``evaluate()``: the files/breakdown/seeds/suite tables, and the coverage exit code."""

from __future__ import annotations

import json

import pytest

from dfwb.core.records import write_scores
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


def test_evaluate_breakdown_by_method(tmp_path, family_pack):
    rows = [
        *[_row("REAL", i, "c23", 0, 0.15, "TOYONE-REAL", "original") for i in range(10)],
        *[_row("FAKE_A", i, None, 1, 0.85, "TOYONE-FAKE_A", "FakeA") for i in range(10)],
        *[_row("FAKE_B", i, None, 1, 0.75, "TOYONE-FAKE_B", "FakeB") for i in range(10)],
    ]
    meta = make_meta(coverage={"expected": 30, "ok": 30, "missing": 0, "error": 0})
    path = _write(tmp_path, "a.scores.csv", rows, meta)
    result = evaluate([path], metrics=["acc@thr=0.5"], by="method", bootstrap=100)
    groups = {row["group"] for row in result.tables["breakdown"]}
    assert groups == {"original", "FakeA", "FakeB"}
    for row in result.tables["breakdown"]:
        assert row["n"] == 10
        assert row["metrics"]["acc@thr=0.5"]["value"] == pytest.approx(1.0)


def test_evaluate_breakdown_by_family_skips_undefined_metric(tmp_path, family_pack):
    # "auc" needs both classes; a per-family group (all real, or all one fake method) never has
    # both, so it must be left out of that group's row rather than raising.
    rows = [
        *[_row("REAL", i, "c23", 0, 0.15, "TOYONE-REAL", "original") for i in range(10)],
        *[_row("FAKE_A", i, None, 1, 0.85, "TOYONE-FAKE_A", "FakeA") for i in range(10)],
    ]
    meta = make_meta(coverage={"expected": 20, "ok": 20, "missing": 0, "error": 0})
    path = _write(tmp_path, "a.scores.csv", rows, meta)
    result = evaluate([path], metrics=["auc", "acc@thr=0.5"], by="family", bootstrap=50)
    for row in result.tables["breakdown"]:
        assert "auc" not in row["metrics"]
        assert "acc@thr=0.5" in row["metrics"]


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
    from dfwb.core.records import ScoreRow

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
