"""Stratified bootstrap CIs: reproducible by seed, and narrower with more data."""

from __future__ import annotations

import numpy as np
import pytest

from dfwb.eval.bootstrap import bootstrap_ci, stratified_resample, summarize_seeds


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
