"""Paired comparison between two or more C5 score files.

``dfwb eval compare`` never re-joins two files against the pack: it works from what they already
agree on, the intersection of their ``ok`` rows keyed by ``(dataset, key, compression)``, and
always reports how big that intersection is, so a key present in only one file is counted rather
than silently misaligned. For AUC specifically, the DeLong test needs the ``[eval]`` extra
(``scipy``), imported lazily so the rest of comparison works without it; every other metric's
paired bootstrap needs nothing beyond numpy.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dfwb.core.errors import ConfigError, ContractError, InstallationError
from dfwb.core.plugins import get_registry
from dfwb.core.records import ScoreRow, read_scores
from dfwb.core.registry import install_hint
from dfwb.eval.bootstrap import _fast_auc_from_indices, _prepare_fast_auc, stratified_resample
from dfwb.eval.metrics import MetricUndefined, compute, parse_metric_spec

__all__ = [
    "CompareResult",
    "PairComparison",
    "compare",
    "delong_test",
    "holm_correction",
]

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

_RowKey = tuple[str, str, str | None]


def _row_key(row: ScoreRow) -> _RowKey:
    return (row.dataset, row.key, row.compression)


def _ok_rows(rows: Sequence[ScoreRow]) -> dict[_RowKey, ScoreRow]:
    return {_row_key(row): row for row in rows if row.status == "ok"}


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the same order as ``p_values``.

    The smallest raw p-value is multiplied by ``m`` (the count of tests), the next by ``m - 1``,
    and so on; each adjustment is then floored at the previous (more conservative) one so the
    adjusted values are monotone non-decreasing in the sorted order, and capped at ``1.0``. With a
    single p-value this is the identity (no correction to make).
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def _compute_midranks(x: FloatArray) -> FloatArray:
    """Midranks of ``x`` (1-based), tied values sharing the mean rank of their run."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = sorted_x.shape[0]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _fast_delong(predictions: FloatArray, n_pos: int) -> tuple[FloatArray, FloatArray]:
    """The fast DeLong algorithm: two raters' AUCs and their 2x2 covariance.

    ``predictions`` is ``(2, n)``, one row per rater, columns ordered positives (fake) first, then
    negatives, the same order for both rows. From the structural components of DeLong et al.
    (1988), computed via midranks as in Sun and Xu (2014) rather than the original O(n^2) placement
    counts.
    """
    n_neg = predictions.shape[1] - n_pos
    positives = predictions[:, :n_pos]
    negatives = predictions[:, n_pos:]
    tx = np.vstack([_compute_midranks(row) for row in positives])
    ty = np.vstack([_compute_midranks(row) for row in negatives])
    tz = np.vstack([_compute_midranks(row) for row in predictions])
    aucs = tz[:, :n_pos].sum(axis=1) / (n_pos * n_neg) - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    return aucs, cov


def delong_test(y: IntArray, p_a: FloatArray, p_b: FloatArray) -> tuple[float, float]:
    """The DeLong test for ``AUC(p_a) == AUC(p_b)`` on the same paired samples ``y``.

    Returns ``(z, p_value)`` for a two-sided test, using the fast (midrank) DeLong algorithm for
    the AUCs' covariance and the standard normal for the p-value.

    Raises:
        InstallationError: scipy (the ``[eval]`` extra) is not installed.
    """
    try:
        from scipy.stats import norm
    except ModuleNotFoundError as exc:
        raise InstallationError(
            "the DeLong test needs scipy, which is not installed", hint=install_hint("scipy")
        ) from exc
    order = np.argsort(-y, kind="mergesort")  # positives (label 1) first, negatives after
    y_sorted = y[order]
    n_pos = int(np.sum(y_sorted == 1))
    predictions = np.vstack([np.asarray(p_a)[order], np.asarray(p_b)[order]])
    aucs, cov = _fast_delong(predictions, n_pos)
    variance = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if variance <= 0.0:
        return 0.0, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(variance))
    p_value = float(2.0 * norm.sf(abs(z)))
    return z, p_value


def _paired_bootstrap_delta(
    metric: str, y: IntArray, p_a: FloatArray, p_b: FloatArray, *, n_boot: int, seed: int
) -> tuple[float, float]:
    """The paired-bootstrap CI of ``metric(y, p_b) - metric(y, p_a)``.

    The metric's spec is resolved once, before the resample loop (see
    :func:`~dfwb.eval.bootstrap.bootstrap_ci`); ``"auc"`` additionally uses the same ``O(n)``
    per-resample formula that module uses, prepared once for each of ``p_a`` and ``p_b``.
    """
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    name, params = parse_metric_spec(metric)
    if name == "auc" and not params:
        auc_a = _prepare_fast_auc(y, p_a)
        auc_b = _prepare_fast_auc(y, p_b)
        for i in range(n_boot):
            idx = stratified_resample(y, rng)
            deltas[i] = _fast_auc_from_indices(idx, *auc_b) - _fast_auc_from_indices(idx, *auc_a)
    else:
        registry = get_registry("metrics")
        kwargs = registry.validate(name, **params)
        metric_fn = registry.load(name)
        for i in range(n_boot):
            idx = stratified_resample(y, rng)
            deltas[i] = float(metric_fn(y[idx], p_b[idx], **kwargs)) - float(
                metric_fn(y[idx], p_a[idx], **kwargs)
            )
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(lo), float(hi)


@dataclass(frozen=True)
class PairComparison:
    """One pair's result: per-metric values, the paired delta CI, and DeLong for AUC."""

    a: str
    b: str
    n: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class CompareResult:
    """The result of :func:`compare`: every pairwise comparison, Holm-corrected if there are >2."""

    files: tuple[str, ...]
    comparisons: tuple[PairComparison, ...]
    holm_applied: bool

    def to_json(self) -> dict[str, Any]:
        """A JSON-friendly, full-precision rendering of every comparison."""
        return {
            "files": list(self.files),
            "holm_applied": self.holm_applied,
            "comparisons": [
                {"a": c.a, "b": c.b, "n": c.n, "metrics": c.metrics} for c in self.comparisons
            ],
        }


def compare(
    files: Sequence[str | os.PathLike[str]],
    *,
    metrics: Sequence[str],
    bootstrap: int = 2000,
    seed: int = 0,
) -> CompareResult:
    """Compare two or more C5 score files, pairwise, on the intersection of their ``ok`` rows.

    For every metric and every pair, this reports both files' point values, the paired-bootstrap
    CI of their difference (``b - a``, stratified by label over the shared rows), and, for
    ``"auc"``, the DeLong test. With more than two files (more than one pair), every pair's DeLong
    p-value is Holm-corrected across all of them (:attr:`CompareResult.holm_applied` says so).

    Raises:
        ConfigError: fewer than two files are given.
        ContractError: a file cannot be read (see :func:`~dfwb.core.records.read_scores`), or a
            pair's ``ok`` rows share no key at all.
        MetricUndefined: a pair's shared rows are all one class and a metric needs both (named
            with the two files and how many rows they share, not just "every label is 'fake'").
        InstallationError: ``"auc"`` is among ``metrics`` and scipy is not installed.
    """
    if len(files) < 2:
        raise ConfigError("compare needs at least two score files", hint="pass 2 or more files")
    names = [Path(f).name for f in files]
    ok_rows = [_ok_rows(read_scores(f).rows) for f in files]
    pairs = list(itertools.combinations(range(len(files)), 2))
    comparisons: list[PairComparison] = []
    delong_by_pair: dict[int, float] = {}
    for pair_index, (i, j) in enumerate(pairs):
        # sort key coerces a possibly-``None`` compression to "" -- comparing the raw 3-tuples
        # would raise if two keys share (dataset, key) but differ in compression being None.
        common = sorted(set(ok_rows[i]) & set(ok_rows[j]), key=lambda k: (k[0], k[1], k[2] or ""))
        if not common:
            raise ContractError(
                f"{names[i]!r} and {names[j]!r} share no 'ok' rows to compare "
                "(their (dataset, key, compression) intersection is empty)",
                hint="check both files were scored on the same split with matching keys",
            )
        y = np.asarray([ok_rows[i][k].label for k in common], dtype=np.int64)
        p_a = np.asarray([ok_rows[i][k].score for k in common], dtype=np.float64)
        p_b = np.asarray([ok_rows[j][k].score for k in common], dtype=np.float64)
        metric_rows: dict[str, dict[str, float]] = {}
        for metric in metrics:
            try:
                value_a = compute(metric, y, p_a)
                value_b = compute(metric, y, p_b)
            except MetricUndefined as exc:
                raise MetricUndefined(
                    f"{exc.message} ({names[i]!r} vs {names[j]!r}, over their {len(common)} "
                    "shared 'ok' rows)",
                    hint=exc.hint,
                ) from exc
            lo, hi = _paired_bootstrap_delta(metric, y, p_a, p_b, n_boot=bootstrap, seed=seed)
            row = {
                "a": value_a,
                "b": value_b,
                "delta": value_b - value_a,
                "delta_lo": lo,
                "delta_hi": hi,
            }
            if parse_metric_spec(metric)[0] == "auc":
                z, p_value = delong_test(y, p_a, p_b)
                row["delong_z"] = z
                row["delong_p"] = p_value
                delong_by_pair[pair_index] = p_value
            metric_rows[metric] = row
        comparisons.append(PairComparison(names[i], names[j], len(common), metric_rows))
    holm_applied = len(delong_by_pair) > 1
    if holm_applied:
        pair_indices = list(delong_by_pair)
        adjusted = holm_correction([delong_by_pair[idx] for idx in pair_indices])
        for pair_index, p_holm in zip(pair_indices, adjusted, strict=True):
            comparisons[pair_index].metrics["auc"]["delong_p_holm"] = p_holm
    return CompareResult(tuple(names), tuple(comparisons), holm_applied)
