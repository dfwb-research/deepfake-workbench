"""Stratified bootstrap over videos, and mean/sd summaries across several seeds.

A confidence interval on a single point estimate says nothing about how much the number would
move if the same detector were re-run on a resampled split. Every metric here gets one by
resampling videos with replacement, holding each class's count fixed (a video's label never
changes), recomputing the metric on each resample, and reading off the percentiles of that
distribution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dfwb.eval.metrics import compute

__all__ = [
    "BootstrapResult",
    "SeedSummary",
    "bootstrap_ci",
    "stratified_resample",
    "summarize_seeds",
]

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate plus its percentile bootstrap confidence interval."""

    point: float
    lo: float
    hi: float
    n_boot: int
    seed: int
    alpha: float


def stratified_resample(y: IntArray, rng: np.random.Generator) -> NDArray[np.intp]:
    """Indices of one resample of ``y``, drawn with replacement separately within each class.

    Every class keeps exactly its original count, so a resample of data with both classes present
    always has both classes present too (a metric that needs both never sees an empty class from
    resampling alone).
    """
    indices = np.empty(y.shape[0], dtype=np.intp)
    filled = 0
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        draw = rng.integers(0, cls_idx.shape[0], size=cls_idx.shape[0])
        indices[filled : filled + cls_idx.shape[0]] = cls_idx[draw]
        filled += cls_idx.shape[0]
    return indices


def bootstrap_ci(
    metric: str,
    y: IntArray,
    p: FloatArray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """The point estimate of ``metric`` on ``(y, p)`` and its percentile bootstrap CI.

    Resampling is stratified by label (see :func:`stratified_resample`) and reproducible: the same
    ``(metric, y, p, n_boot, seed)`` always retraces the same resamples, because the generator is
    seeded once from ``seed`` and drawn in a fixed order.

    Args:
        metric: A metric spec understood by :func:`~dfwb.eval.metrics.compute`, e.g. ``"auc"`` or
            ``"tpr@fpr=0.01"``.
        n_boot: Number of resamples.
        seed: Seed for the resampling generator.
        alpha: The CI is the ``[alpha/2, 1 - alpha/2]`` percentile interval (default: 95%).

    Raises:
        MetricUndefined, ContractError, ConfigError: as :func:`~dfwb.eval.metrics.compute` raises
            them, from the point estimate on the unresampled data.
    """
    point = compute(metric, y, p)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = stratified_resample(y, rng)
        values[i] = compute(metric, y[idx], p[idx])
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        point=point, lo=float(lo), hi=float(hi), n_boot=n_boot, seed=seed, alpha=alpha
    )


@dataclass(frozen=True)
class SeedSummary:
    """Mean and sample standard deviation of one metric across several seeded runs."""

    metric: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    mean: float
    sd: float


def summarize_seeds(metric: str, per_seed: Mapping[int, float]) -> SeedSummary:
    """Mean +/- sd of ``metric`` across seeds, plus the per-seed values (sorted by seed).

    The sample standard deviation (``ddof=1``) is the usual way multi-seed results are reported;
    it is ``0.0`` for a single seed rather than undefined.
    """
    seeds = tuple(sorted(per_seed))
    values = tuple(per_seed[s] for s in seeds)
    array = np.asarray(values, dtype=np.float64)
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return SeedSummary(metric=metric, seeds=seeds, values=values, mean=float(np.mean(array)), sd=sd)
