"""Coverage policy: the four ``--missing`` options, and the C5 coverage exit code."""

from __future__ import annotations

import pytest

from dfwb.core.errors import ConfigError
from dfwb.core.records import ScoreRow
from dfwb.eval.coverage import MISSING_POLICIES, coverage_of, labels_and_scores
from dfwb.eval.report import evaluate

ROWS = [
    ScoreRow("d", "r0", None, 0, 0.1, "ok"),
    ScoreRow("d", "r1", None, 0, 0.2, "ok"),
    ScoreRow("d", "r2", None, 1, 0.8, "ok"),
    ScoreRow("d", "r3", None, 1, 0.9, "ok"),
    ScoreRow("d", "m0", None, 0, None, "missing"),
    ScoreRow("d", "e0", None, 1, None, "error"),
]


def test_coverage_of_counts_every_status():
    coverage = coverage_of(ROWS)
    assert coverage.as_dict() == {
        "expected": 6,
        "ok": 4,
        "missing": 1,
        "error": 1,
        "coverage": pytest.approx(4 / 6),
    }


def test_coverage_empty_file_is_full_coverage():
    assert coverage_of([]).fraction == 1.0


@pytest.mark.parametrize("policy", MISSING_POLICIES)
def test_labels_and_scores_every_missing_policy(policy):
    y, p, kept = labels_and_scores(ROWS, missing=policy)
    if policy == "exclude":
        assert len(kept) == 4
        assert list(y) == [0, 0, 1, 1]
    else:
        assert len(kept) == 6
        fill = {"as-real": 0.0, "as-fake": 1.0, "as-chance": 0.5}[policy]
        # the missing row (label 0) and the error row (label 1) both get the fill score
        assert p[-2] == fill
        assert p[-1] == fill
        assert list(y) == [0, 0, 1, 1, 0, 1]
    assert len(y) == len(p) == len(kept)


def test_labels_and_scores_rejects_unknown_policy():
    with pytest.raises(ConfigError, match="unknown --missing policy"):
        labels_and_scores(ROWS, missing="drop")


@pytest.mark.parametrize("policy", MISSING_POLICIES)
def test_coverage_policy_and_exit_code(tmp_path, policy):
    """Every ``--missing`` option: metrics are still computed, and coverage/exit_code reflect the
    fraction of rows that were actually scored, independent of how non-ok rows are filled in."""
    path = tmp_path / "t.scores.csv"
    path.write_text(
        "dataset,key,compression,label,score,status\n"
        + "".join(f"d,r{i},,0,0.1,ok\n" for i in range(49))
        + "".join(f"d,f{i},,1,0.9,ok\n" for i in range(49))
        + "d,m0,,0,,missing\n"
        + "d,m1,,1,,missing\n"
    )
    result = evaluate([path], metrics=["auc"], bootstrap=50, missing=policy)
    file_row = result.tables["files"][0]
    assert file_row["expected"] == 100
    assert file_row["ok"] == 98
    assert file_row["coverage"] == pytest.approx(0.98)
    assert result.exit_code == 3  # 0.98 < the default min_coverage of 0.99

    lenient = evaluate([path], metrics=["auc"], bootstrap=50, missing=policy, min_coverage=0.5)
    assert lenient.exit_code == 0
