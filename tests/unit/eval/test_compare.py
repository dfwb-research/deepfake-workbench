"""Paired comparison: intersection semantics, DeLong vs. a naive reference, and Holm correction."""

from __future__ import annotations

import numpy as np
import pytest

from dfwb.core.errors import ConfigError, ContractError, InstallationError
from dfwb.eval.compare import compare, delong_test, holm_correction
from dfwb.eval.metrics import MetricUndefined, auc

pytest.importorskip("scipy")


def _write(tmp_path, name, rows):
    path = tmp_path / name
    lines = ["dataset,key,compression,label,score,status"]
    lines.extend(rows)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_compare_needs_at_least_two_files(tmp_path):
    a = _write(tmp_path, "a.scores.csv", ["d,k0,,0,0.1,ok"])
    with pytest.raises(ConfigError, match="at least two"):
        compare([a], metrics=["auc"])


def test_compare_uses_intersection(tmp_path):
    # a has k0..k4 (all ok); b has k2..k6, and k4 is `missing` in b so it drops out of `ok` too.
    a = _write(
        tmp_path,
        "a.scores.csv",
        [f"d,k{i},,{i % 2},{0.1 + 0.05 * i},ok" for i in range(5)],
    )
    b_rows = [f"d,k{i},,{i % 2},{0.2 + 0.05 * i},ok" for i in range(2, 7) if i != 4]
    b_rows.append("d,k4,,0,,missing")
    b = _write(tmp_path, "b.scores.csv", b_rows)
    result = compare([a, b], metrics=["acc@thr=0.5"])
    assert len(result.comparisons) == 1
    comparison = result.comparisons[0]
    # common ok keys: k2, k3 (k4 is missing in b, k5/k6 absent from a, k0/k1 absent from b)
    assert comparison.n == 2
    assert comparison.a == "a.scores.csv"
    assert comparison.b == "b.scores.csv"
    assert "acc@thr=0.5" in comparison.metrics


def test_compare_handles_a_key_with_both_a_null_and_a_named_compression(tmp_path):
    # the same (dataset, key) with one None-compression row and one "c23" row must not raise
    # when sorting the intersection (comparing None to a string).
    a = _write(
        tmp_path,
        "a.scores.csv",
        ["d,k0,,0,0.1,ok", "d,k0,c23,0,0.15,ok", "d,k1,,1,0.9,ok"],
    )
    b = _write(
        tmp_path,
        "b.scores.csv",
        ["d,k0,,0,0.2,ok", "d,k0,c23,0,0.25,ok", "d,k1,,1,0.8,ok"],
    )
    result = compare([a, b], metrics=["acc@thr=0.5"])
    assert result.comparisons[0].n == 3


def test_compare_with_no_bootstrap_reports_the_delta_without_an_interval(tmp_path):
    # --bootstrap 0 means "no confidence intervals", as in dfwb eval, never a crash
    a = _write(tmp_path, "a.scores.csv", [f"d,k{i},,{i % 2},{0.1 + 0.05 * i},ok" for i in range(6)])
    b = _write(tmp_path, "b.scores.csv", [f"d,k{i},,{i % 2},{0.2 + 0.1 * i},ok" for i in range(6)])
    result = compare([a, b], metrics=["auc"], bootstrap=0)
    row = result.comparisons[0].metrics["auc"]
    assert row["delta"] == pytest.approx(row["b"] - row["a"])
    assert row["delta_lo"] is None
    assert row["delta_hi"] is None


def test_compare_empty_intersection_raises_a_clear_error(tmp_path):
    a = _write(tmp_path, "a.scores.csv", ["d,k0,,0,0.1,ok", "d,k1,,1,0.9,ok"])
    b = _write(tmp_path, "b.scores.csv", ["d,k2,,0,0.2,ok", "d,k3,,1,0.8,ok"])
    with pytest.raises(ContractError, match="share no 'ok' rows") as info:
        compare([a, b], metrics=["auc"])
    assert "a.scores.csv" in info.value.message
    assert "b.scores.csv" in info.value.message


def test_compare_one_class_intersection_raises_a_clear_metric_undefined(tmp_path):
    # both files' shared "ok" rows are all real -- "auc" cannot be defined on them, and the error
    # must say so clearly (naming the files and the shared-row count), not just the generic
    # "every label is 'fake'" a bare `compute()` call would give with no context.
    a = _write(tmp_path, "a.scores.csv", ["d,k0,,0,0.1,ok", "d,k1,,0,0.2,ok"])
    b = _write(tmp_path, "b.scores.csv", ["d,k0,,0,0.3,ok", "d,k1,,0,0.4,ok"])
    with pytest.raises(MetricUndefined) as info:
        compare([a, b], metrics=["auc"])
    assert "a.scores.csv" in info.value.message
    assert "b.scores.csv" in info.value.message
    assert "2 shared" in info.value.message


def test_compare_one_class_intersection_does_not_affect_single_class_metrics(tmp_path):
    # a metric that does not need both classes must still work fine over a one-class intersection.
    a = _write(tmp_path, "a.scores.csv", ["d,k0,,0,0.1,ok", "d,k1,,0,0.2,ok"])
    b = _write(tmp_path, "b.scores.csv", ["d,k0,,0,0.3,ok", "d,k1,,0,0.4,ok"])
    result = compare([a, b], metrics=["brier"])
    assert result.comparisons[0].n == 2
    assert "brier" in result.comparisons[0].metrics


def test_compare_reports_delong_for_auc(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    p_a = np.clip(np.r_[rng.normal(0.3, 0.2, n // 2), rng.normal(0.6, 0.2, n // 2)], 0.001, 0.999)
    p_b = np.clip(np.r_[rng.normal(0.3, 0.2, n // 2), rng.normal(0.85, 0.1, n // 2)], 0.001, 0.999)
    rows_a = [f"d,k{i},,{int(y[i])},{p_a[i]},ok" for i in range(n)]
    rows_b = [f"d,k{i},,{int(y[i])},{p_b[i]},ok" for i in range(n)]
    a = _write(tmp_path, "a.scores.csv", rows_a)
    b = _write(tmp_path, "b.scores.csv", rows_b)
    result = compare([a, b], metrics=["auc"], bootstrap=200)
    row = result.comparisons[0].metrics["auc"]
    assert row["a"] == pytest.approx(auc(y, p_a))
    assert row["b"] == pytest.approx(auc(y, p_b))
    assert 0.0 <= row["delong_p"] <= 1.0
    assert result.holm_applied is False  # only one pair


# --------------------------------------------------------------------------- DeLong reference


def _naive_components(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """O(n*m) structural components (DeLong et al. 1988): brute-force pairwise comparisons."""
    m, n = len(pos), len(neg)
    v10 = np.array([(np.sum(x > neg) + 0.5 * np.sum(x == neg)) / n for x in pos])
    v01 = np.array([(np.sum(pos > x) + 0.5 * np.sum(pos == x)) / m for x in neg])
    return v10, v01


def _naive_delong(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> tuple[float, float]:
    from scipy.stats import norm

    order = np.argsort(-y, kind="mergesort")
    y_sorted = y[order]
    m = int(np.sum(y_sorted == 1))
    aucs = []
    v10s = []
    v01s = []
    for p in (p_a[order], p_b[order]):
        pos, neg = p[:m], p[m:]
        v10, v01 = _naive_components(pos, neg)
        aucs.append(v10.mean())
        v10s.append(v10)
        v01s.append(v01)
    cov = np.cov(np.vstack(v01s)) / m + np.cov(np.vstack(v10s)) / (len(y) - m)
    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    p_value = 2 * norm.sf(abs(z))
    return float(z), float(p_value)


def test_delong_matches_a_naive_reference_implementation():
    rng = np.random.default_rng(7)
    n = 50
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    p_a = rng.uniform(0, 1, n)
    p_b = np.clip(p_a + rng.normal(0, 0.1, n), 0, 1)
    z, p_value = delong_test(y, p_a, p_b)
    z_naive, p_naive = _naive_delong(y, p_a, p_b)
    assert z == pytest.approx(z_naive, abs=1e-9)
    assert p_value == pytest.approx(p_naive, abs=1e-9)


def test_delong_agrees_with_the_auc_metric():
    rng = np.random.default_rng(8)
    n = 40
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    p_a = rng.uniform(0, 1, n)
    p_b = rng.uniform(0, 1, n)
    # the fast-DeLong AUC (from midranks) must equal dfwb.eval.metrics.auc exactly
    order = np.argsort(-y, kind="mergesort")
    from dfwb.eval.compare import _fast_delong

    m = int(np.sum(y[order] == 1))
    aucs, _cov = _fast_delong(np.vstack([p_a[order], p_b[order]]), m)
    assert aucs[0] == pytest.approx(auc(y, p_a))
    assert aucs[1] == pytest.approx(auc(y, p_b))
    delong_test(y, p_a, p_b)  # exercised for its own sake too


def test_delong_needs_scipy(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "scipy.stats" or name.startswith("scipy"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    with pytest.raises(InstallationError, match="needs scipy") as info:
        delong_test(y, p, p)
    assert "deepfake-workbench[eval]" in info.value.hint


# --------------------------------------------------------------------------- Holm correction


def test_holm_correction_matches_the_textbook_example():
    # the standard worked example (R's p.adjust(method="holm") on 0.01..0.05)
    adjusted = holm_correction([0.01, 0.02, 0.03, 0.04, 0.05])
    assert adjusted == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])


def test_holm_correction_single_value_is_unadjusted():
    assert holm_correction([0.03]) == pytest.approx([0.03])


def test_compare_holm_correction_for_three_files(tmp_path):
    rng = np.random.default_rng(11)
    n = 80
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    scores = {
        "a": np.clip(np.r_[rng.normal(0.3, 0.2, n // 2), rng.normal(0.55, 0.2, n // 2)], 0, 1),
        "b": np.clip(np.r_[rng.normal(0.3, 0.2, n // 2), rng.normal(0.70, 0.2, n // 2)], 0, 1),
        "c": np.clip(np.r_[rng.normal(0.3, 0.2, n // 2), rng.normal(0.90, 0.1, n // 2)], 0, 1),
    }
    paths = [
        _write(
            tmp_path,
            f"{name}.scores.csv",
            [f"d,k{i},,{int(y[i])},{p[i]},ok" for i in range(n)],
        )
        for name, p in scores.items()
    ]
    result = compare(paths, metrics=["auc"], bootstrap=100)
    assert len(result.comparisons) == 3  # C(3, 2)
    assert result.holm_applied is True
    raw = [c.metrics["auc"]["delong_p"] for c in result.comparisons]
    expected = holm_correction(raw)
    got = [c.metrics["auc"]["delong_p_holm"] for c in result.comparisons]
    assert got == pytest.approx(expected)
    for holm_p, raw_p in zip(got, raw, strict=True):
        assert holm_p >= raw_p - 1e-12
