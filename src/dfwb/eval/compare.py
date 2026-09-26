"""Paired comparison between two or more C5 score files.

``dfwb eval compare`` never re-joins two files against the pack: it works from what they already
agree on, the intersection of their ``ok`` rows keyed by ``(dataset, key, compression)``, and
always reports how big that intersection is and how many ``ok`` rows each file has that the other
does not, so a key present in only one file is counted rather than silently misaligned. Two files
that give a shared row different labels are refused rather than compared on either one's labels.
For AUC specifically, the DeLong test needs the ``[eval]`` extra (``scipy``), imported lazily so
the rest of comparison works without it; every other metric's paired bootstrap needs nothing
beyond numpy.

JSON has no infinity and no NaN, so :meth:`CompareResult.to_json` writes an infinite value (a
DeLong ``z`` when the two AUCs differ but their paired variance is zero) as the string ``"inf"``
or ``"-inf"``, and a DeLong test that cannot be computed as ``null`` with the reason in
``delong_undefined``.
"""

from __future__ import annotations

import itertools
import math
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
    the AUCs' covariance and the standard normal for the p-value. When the paired variance of the
    AUC difference is zero (both raters' structural components are constant: a constant score, or
    a perfect separator), the answer is exact rather than estimated: equal AUCs give ``(0.0,
    1.0)``, and unequal ones give ``z = inf`` (``-inf`` when ``AUC(p_a)`` is the smaller) with
    ``p = 0.0`` -- the AUCs certainly differ.

    Raises:
        InstallationError: scipy (the ``[eval]`` extra) is not installed.
        MetricUndefined: a class has fewer than two rows, so the covariance the test needs cannot
            be estimated (or it is not finite for any other reason).
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
    n_neg = int(y_sorted.shape[0]) - n_pos
    if n_pos < 2 or n_neg < 2:
        raise MetricUndefined(
            f"delong: undefined with fewer than two rows in a class ({n_pos} fake, {n_neg} real)",
            hint="the DeLong covariance needs at least two fake and two real rows",
        )
    predictions = np.vstack([np.asarray(p_a)[order], np.asarray(p_b)[order]])
    aucs, cov = _fast_delong(predictions, n_pos)
    variance = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if not math.isfinite(variance):
        raise MetricUndefined(
            "delong: undefined, the paired variance of the AUC difference is not finite",
            hint="check the scores are finite and both classes have at least two rows",
        )
    difference = float(aucs[0] - aucs[1])
    if variance <= 0.0:
        if math.isclose(difference, 0.0, abs_tol=1e-12):
            return 0.0, 1.0
        return math.copysign(math.inf, difference), 0.0
    z = difference / math.sqrt(variance)
    p_value = float(2.0 * norm.sf(abs(z)))
    return z, p_value


def _paired_bootstrap_delta(
    metric: str, y: IntArray, p_a: FloatArray, p_b: FloatArray, *, n_boot: int, seed: int
) -> tuple[float | None, float | None]:
    """The paired-bootstrap CI of ``metric(y, p_b) - metric(y, p_a)``; ``(None, None)`` when
    ``n_boot`` is 0 (no resampling, so no interval).

    The metric's spec is resolved once, before the resample loop (see
    :func:`~dfwb.eval.bootstrap.bootstrap_ci`); ``"auc"`` additionally uses the same ``O(n)``
    per-resample formula that module uses, prepared once for each of ``p_a`` and ``p_b``.
    """
    if n_boot == 0:
        return None, None
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
    """One pair's result: per-metric values, the paired delta CI, and DeLong for AUC.

    ``n`` is how many ``ok`` rows the two files share (what every metric here is computed over);
    ``only_a``/``only_b`` count the ``ok`` rows of ``a``/``b`` the other file has no ``ok`` row
    for, which the comparison leaves out.
    """

    a: str
    b: str
    n: int
    metrics: dict[str, dict[str, float | str | None]] = field(default_factory=dict)
    only_a: int = 0
    only_b: int = 0


def _json_value(value: float | str | None) -> float | str | None:
    """``value`` as valid JSON: an infinity becomes ``"inf"``/``"-inf"`` and a NaN ``None``,
    since JSON has no literal for either."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "inf" if value > 0 else "-inf"
    return value


@dataclass(frozen=True)
class CompareResult:
    """The result of :func:`compare`: every pairwise comparison, Holm-corrected if there are >2."""

    files: tuple[str, ...]
    comparisons: tuple[PairComparison, ...]
    holm_applied: bool

    def to_json(self) -> dict[str, Any]:
        """A JSON-friendly, full-precision rendering of every comparison; always valid JSON (see
        the module docstring for how an infinite or undefined value is written)."""
        return {
            "files": list(self.files),
            "holm_applied": self.holm_applied,
            "comparisons": [
                {
                    "a": c.a,
                    "b": c.b,
                    "n": c.n,
                    "only_a": c.only_a,
                    "only_b": c.only_b,
                    "metrics": {
                        metric: {name: _json_value(value) for name, value in row.items()}
                        for metric, row in c.metrics.items()
                    },
                }
                for c in self.comparisons
            ],
        }


def _check_labels_agree(
    common: Sequence[_RowKey],
    rows_a: dict[_RowKey, ScoreRow],
    rows_b: dict[_RowKey, ScoreRow],
    names: tuple[str, str],
) -> None:
    disagree = [key for key in common if rows_a[key].label != rows_b[key].label]
    if not disagree:
        return
    sample = ", ".join(
        f"{dataset}/{key}"
        + (f" ({compression})" if compression else "")
        + f": {rows_a[(dataset, key, compression)].label} vs "
        + f"{rows_b[(dataset, key, compression)].label}"
        for dataset, key, compression in disagree[:5]
    )
    raise ContractError(
        f"{names[0]!r} and {names[1]!r} give {len(disagree)} shared row(s) different labels "
        f"(e.g. {sample})",
        hint="compare files scored under the same label mapping and pack version",
    )


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
    ``"auc"``, the DeLong test (see :func:`delong_test`; a test that cannot be computed is
    reported as ``None`` with its reason in ``delong_undefined``, not raised). With more than one
    DeLong p-value (more than one pair), every one is Holm-corrected across all of them
    (:attr:`CompareResult.holm_applied` says so). Each pair also reports how many ``ok`` rows
    each file has that the other does not (``only_a``/``only_b``).

    Raises:
        ConfigError: fewer than two files are given.
        ContractError: a file cannot be read (see :func:`~dfwb.core.records.read_scores`), a
            pair's ``ok`` rows share no key at all, or a pair gives a shared row two different
            labels.
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
        _check_labels_agree(common, ok_rows[i], ok_rows[j], (names[i], names[j]))
        y = np.asarray([ok_rows[i][k].label for k in common], dtype=np.int64)
        p_a = np.asarray([ok_rows[i][k].score for k in common], dtype=np.float64)
        p_b = np.asarray([ok_rows[j][k].score for k in common], dtype=np.float64)
        metric_rows: dict[str, dict[str, float | str | None]] = {}
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
            row: dict[str, float | str | None] = {
                "a": value_a,
                "b": value_b,
                "delta": value_b - value_a,
                "delta_lo": lo,
                "delta_hi": hi,
            }
            if parse_metric_spec(metric)[0] == "auc":
                try:
                    z, p_value = delong_test(y, p_a, p_b)
                except MetricUndefined as exc:
                    row["delong_z"] = None
                    row["delong_p"] = None
                    row["delong_undefined"] = exc.message
                else:
                    row["delong_z"] = z
                    row["delong_p"] = p_value
                    delong_by_pair[pair_index] = p_value
            metric_rows[metric] = row
        comparisons.append(
            PairComparison(
                names[i],
                names[j],
                len(common),
                metric_rows,
                only_a=len(ok_rows[i]) - len(common),
                only_b=len(ok_rows[j]) - len(common),
            )
        )
    holm_applied = len(delong_by_pair) > 1
    if holm_applied:
        pair_indices = list(delong_by_pair)
        adjusted = holm_correction([delong_by_pair[idx] for idx in pair_indices])
        for pair_index, p_holm in zip(pair_indices, adjusted, strict=True):
            comparisons[pair_index].metrics["auc"]["delong_p_holm"] = p_holm
    return CompareResult(tuple(names), tuple(comparisons), holm_applied)
