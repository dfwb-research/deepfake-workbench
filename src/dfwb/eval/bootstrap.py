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

from dfwb.core.plugins import get_registry
from dfwb.eval.metrics import compute, parse_metric_spec

__all__ = [
    "BootstrapResult",
    "SeedSummary",
    "bootstrap_ci",
    "stratified_resample",
    "summarize_seeds",
]

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate, plus its percentile bootstrap confidence interval -- or, when
    ``n_boot == 0`` was asked for (no resampling, no interval), ``lo``/``hi`` of ``None``."""

    point: float
    lo: float | None
    hi: float | None
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


def _prepare_fast_auc(y: IntArray, p: FloatArray) -> tuple[IndexArray, int, int, int]:
    """Precompute, once, the sorted-distinct-score bin id of every sample.

    A resample only changes which original samples are drawn, never their values, so two
    resampled scores tie exactly when the original samples they came from tie -- the bin id (a
    sample's rank among the *distinct* scores) can therefore be computed once, outside the
    resample loop, instead of re-sorting the resampled scores from scratch on every draw.
    """
    _, bin_of_all = np.unique(p, return_inverse=True)
    n_bins = int(bin_of_all.max()) + 1 if bin_of_all.size else 0
    n_neg = int(np.sum(y == 0))
    n_pos = int(np.sum(y == 1))
    return bin_of_all.astype(np.intp), n_neg, n_pos, n_bins


def _fast_auc_from_indices(
    idx: IndexArray, bin_of_all: IndexArray, n_neg: int, n_pos: int, n_bins: int
) -> float:
    """The AUC of the resample ``idx`` describes, in ``O(n)`` instead of ``O(n log n)``.

    ``idx`` is :func:`stratified_resample`'s output for a binary ``y``: negatives (label 0) fill
    the first ``n_neg`` slots, positives (label 1) the rest (``np.unique`` sorts ``{0, 1}``
    ascending). Counting, per distinct-score bin, how many resampled negatives and positives
    landed there (via :func:`numpy.bincount`) and taking a cumulative sum gives the same
    pairwise-comparison count the rank-based AUC computes -- see
    ``test_fast_auc_matches_the_generic_auc_metric`` for a direct, per-resample check against
    :func:`~dfwb.eval.metrics.auc`.
    """
    neg_bins = bin_of_all[idx[:n_neg]]
    pos_bins = bin_of_all[idx[n_neg:]]
    c_neg = np.bincount(neg_bins, minlength=n_bins).astype(np.float64)
    c_pos = np.bincount(pos_bins, minlength=n_bins).astype(np.float64)
    cum_neg_below = np.cumsum(c_neg) - c_neg  # negatives strictly below each bin
    return float(np.sum(c_pos * (cum_neg_below + 0.5 * c_neg)) / (n_pos * n_neg))


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

    The metric's spec is parsed and its parameters validated once, before the resample loop, not
    on every resample. ``"auc"`` (no parameters) additionally uses an ``O(n)``-per-resample formula
    (:func:`_fast_auc_from_indices`) instead of a fresh sort and rank computation each time, since
    a resample only ever repeats scores that were already there -- see the module's tests for a
    direct check that this gives the same value as :func:`~dfwb.eval.metrics.auc` on every
    resample. Other metrics still recompute from scratch each time, but skip the registry lookup
    and parameter validation :func:`~dfwb.eval.metrics.compute` would otherwise repeat.

    Args:
        metric: A metric spec understood by :func:`~dfwb.eval.metrics.compute`, e.g. ``"auc"`` or
            ``"tpr@fpr=0.01"``.
        n_boot: Number of resamples; ``0`` means "no confidence interval": the point estimate is
            still computed and returned, ``lo``/``hi`` are ``None``, and no resampling happens at
            all (no generator draws, so a later, positive ``n_boot`` at the same ``seed`` is not
            "the first ``n_boot`` resamples of a longer run" -- there is no notion of a shared
            prefix between two different ``n_boot`` values to begin with).
        seed: Seed for the resampling generator.
        alpha: The CI is the ``[alpha/2, 1 - alpha/2]`` percentile interval (default: 95%).

    Raises:
        MetricUndefined, ContractError, ConfigError: as :func:`~dfwb.eval.metrics.compute` raises
            them, from the point estimate on the unresampled data.
    """
    point = compute(metric, y, p)
    if n_boot == 0:
        return BootstrapResult(point=point, lo=None, hi=None, n_boot=0, seed=seed, alpha=alpha)
    name, params = parse_metric_spec(metric)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=np.float64)
    if name == "auc" and not params:
        bin_of_all, n_neg, n_pos, n_bins = _prepare_fast_auc(y, p)
        for i in range(n_boot):
            idx = stratified_resample(y, rng)
            values[i] = _fast_auc_from_indices(idx, bin_of_all, n_neg, n_pos, n_bins)
    else:
        registry = get_registry("metrics")
        kwargs = registry.validate(name, **params)
        metric_fn = registry.load(name)
        for i in range(n_boot):
            idx = stratified_resample(y, rng)
            values[i] = float(metric_fn(y[idx], p[idx], **kwargs))
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
