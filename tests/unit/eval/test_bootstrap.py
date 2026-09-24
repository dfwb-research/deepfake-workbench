"""Stratified bootstrap CIs: reproducible by seed, and narrower with more data."""

from __future__ import annotations

import time

import numpy as np
import pytest

from dfwb.eval.bootstrap import (
    _fast_auc_from_indices,
    _prepare_fast_auc,
    bootstrap_ci,
    stratified_resample,
    summarize_seeds,
)
from dfwb.eval.metrics import auc, compute


def _synthetic(n_per_class: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([0] * n_per_class + [1] * n_per_class)
    p = np.clip(
        np.r_[rng.normal(0.3, 0.15, n_per_class), rng.normal(0.7, 0.15, n_per_class)], 0.0, 1.0
    )
    return y, p


def test_stratified_resample_preserves_class_counts():
    y = np.array([0, 0, 0, 1, 1])
    rng = np.random.default_rng(0)
    for _ in range(20):
        idx = stratified_resample(y, rng)
        resampled = y[idx]
        assert (resampled == 0).sum() == 3
        assert (resampled == 1).sum() == 2


def test_bootstrap_ci_is_reproducible_by_seed():
    rng = np.random.default_rng(1)
    y, p = _synthetic(40, rng)
    first = bootstrap_ci("auc", y, p, n_boot=200, seed=42)
    second = bootstrap_ci("auc", y, p, n_boot=200, seed=42)
    assert first == second


def test_bootstrap_ci_differs_across_seeds():
    rng = np.random.default_rng(2)
    y, p = _synthetic(40, rng)
    a = bootstrap_ci("auc", y, p, n_boot=200, seed=0)
    b = bootstrap_ci("auc", y, p, n_boot=200, seed=1)
    assert (a.lo, a.hi) != (b.lo, b.hi)


def test_bootstrap_ci_width_shrinks_as_n_grows():
    rng = np.random.default_rng(3)
    y_small, p_small = _synthetic(15, rng)
    y_large, p_large = _synthetic(500, rng)
    small = bootstrap_ci("auc", y_small, p_small, n_boot=500, seed=0)
    large = bootstrap_ci("auc", y_large, p_large, n_boot=500, seed=0)
    assert (large.hi - large.lo) < (small.hi - small.lo)


def test_bootstrap_ci_covers_the_point_estimate():
    rng = np.random.default_rng(4)
    y, p = _synthetic(100, rng)
    result = bootstrap_ci("auc", y, p, n_boot=500, seed=0)
    assert result.lo <= result.point <= result.hi


def test_summarize_seeds_mean_and_sd():
    summary = summarize_seeds("auc", {0: 0.8, 1: 0.9, 2: 1.0})
    assert summary.seeds == (0, 1, 2)
    assert summary.values == (0.8, 0.9, 1.0)
    assert summary.mean == pytest.approx(0.9)
    assert summary.sd == pytest.approx(np.std([0.8, 0.9, 1.0], ddof=1))


def test_summarize_seeds_single_seed_has_zero_sd():
    summary = summarize_seeds("auc", {7: 0.5})
    assert summary.mean == 0.5
    assert summary.sd == 0.0


# --------------------------------------------------------------------------- fast AUC bootstrap


def test_fast_auc_matches_the_generic_auc_metric_on_many_resamples():
    """The O(n) per-resample AUC formula must agree with ``auc()`` on every single resample it
    would ever be asked to score, not just on average -- this is what "identical to the current
    per-resample definition" means, checked directly rather than through a bootstrap CI."""
    rng = np.random.default_rng(9)
    y, p = _synthetic(25, rng)
    bin_of_all, n_neg, n_pos, n_bins = _prepare_fast_auc(y, p)
    resample_rng = np.random.default_rng(3)
    for _ in range(500):
        idx = stratified_resample(y, resample_rng)
        fast_value = _fast_auc_from_indices(idx, bin_of_all, n_neg, n_pos, n_bins)
        naive_value = auc(y[idx], p[idx])
        assert fast_value == pytest.approx(naive_value, abs=1e-12)


def test_fast_auc_matches_with_heavy_ties_and_repeats():
    # small n and few distinct scores, so resampling with replacement produces lots of ties --
    # exactly the case where getting the bin/rank bookkeeping wrong would show up.
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
    bin_of_all, n_neg, n_pos, n_bins = _prepare_fast_auc(y, p)
    rng = np.random.default_rng(11)
    for _ in range(300):
        idx = stratified_resample(y, rng)
        assert _fast_auc_from_indices(idx, bin_of_all, n_neg, n_pos, n_bins) == pytest.approx(
            auc(y[idx], p[idx]), abs=1e-12
        )


def test_bootstrap_ci_auc_fast_path_matches_the_naive_per_resample_loop():
    """``bootstrap_ci``'s fast path for "auc" must give the exact same CI a naive per-resample
    ``compute("auc", ...)`` loop would, for the same seed (same resampling indices)."""
    rng = np.random.default_rng(6)
    y, p = _synthetic(60, rng)
    fast = bootstrap_ci("auc", y, p, n_boot=300, seed=13)

    naive_rng = np.random.default_rng(13)
    naive_values = np.empty(300)
    for i in range(300):
        idx = stratified_resample(y, naive_rng)
        naive_values[i] = compute("auc", y[idx], p[idx])
    lo, hi = np.quantile(naive_values, [0.025, 0.975])
    assert fast.point == pytest.approx(compute("auc", y, p))
    assert fast.lo == pytest.approx(lo)
    assert fast.hi == pytest.approx(hi)


@pytest.mark.slow
def test_bootstrap_ci_auc_is_fast_at_scale():
    rng = np.random.default_rng(0)
    n_per_class = 25_000
    y, p = _synthetic(n_per_class, rng)
    start = time.perf_counter()
    bootstrap_ci("auc", y, p, n_boot=2000, seed=0)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, (
        f"bootstrap_ci('auc', n={2 * n_per_class}, n_boot=2000) took {elapsed:.2f}s"
    )
