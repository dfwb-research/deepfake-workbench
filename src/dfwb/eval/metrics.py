"""Metrics over label/score pairs (registry ``metrics``): exact, documented, torch-free.

Every metric takes ``y`` (ground-truth labels, 0 = real, 1 = fake) and ``p`` (the model's
``P(fake)`` for each corresponding sample, in ``[0, 1]``) and returns one float. A metric that
needs a threshold, a bin count or an operating point takes it as a keyword parameter with a
default, so ``dfwb.eval.aggregate`` and the CLI can spell it as ``name@k=v,k=v`` through
:func:`parse_metric_spec` and :func:`compute`.

Metrics that compare across classes (``auc``, ``ap``, ``eer``, ``tpr``, ``fpr``) need both a
positive and a negative example to mean anything. Older code in this space quietly returned a
placeholder (an AUC of 0.5, say) when that was not true. This module never does that: it raises
:class:`MetricUndefined` instead, so a caller sees the data problem rather than a number that
looks real.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dfwb.core.errors import ConfigError, ContractError, DFWBError
from dfwb.core.plugins import get_registry

__all__ = [
    "Metric",
    "MetricUndefined",
    "acc",
    "ap",
    "auc",
    "aurc",
    "brier",
    "compute",
    "ece",
    "eer",
    "eer_point",
    "fpr",
    "nll",
    "parse_metric_spec",
    "tpr",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_EPS = 1e-7


class MetricUndefined(DFWBError):
    """A metric's definition does not apply to the given data (exit code 4).

    Raised instead of a placeholder value (an AUC of 0.5, for instance) when a metric compares
    the fake and real classes and the data has only one of them.
    """

    exit_code = 4


class Metric(Protocol):
    """The shape every registered metric has: labels, scores, and its own keyword parameters."""

    def __call__(self, y: IntArray, p: FloatArray, /, **params: object) -> float: ...


# --------------------------------------------------------------------------- parameter syntax


def parse_metric_spec(spec: str) -> tuple[str, dict[str, bool | int | float | str]]:
    """Parse ``name@k=v,k=v`` into ``(name, params)``.

    ``name`` alone (no ``@``) is a metric or aggregation mode with no parameters. Each value is
    read as a bool (``true``/``false``, case-insensitive), else an int, else a float, else left
    as a string; the registry that owns ``name`` then validates and coerces it against that
    target's own parameter types.

    Raises:
        ConfigError: ``spec`` is empty, has no name before ``@``, has ``@`` with nothing after
            it, or a parameter piece is not ``key=value``.
    """
    if not spec.strip():
        raise ConfigError("empty metric spec", hint="use e.g. 'auc' or 'tpr@fpr=0.01'")
    name, sep, rest = spec.partition("@")
    name = name.strip()
    if not name:
        raise ConfigError(f"{spec!r}: missing a name before '@'", hint="use e.g. 'tpr@fpr=0.01'")
    params: dict[str, bool | int | float | str] = {}
    if sep:
        if not rest.strip():
            raise ConfigError(f"{spec!r}: '@' with no parameters", hint="use e.g. 'fpr=0.01'")
        for piece in rest.split(","):
            piece = piece.strip()
            if not piece:
                raise ConfigError(f"{spec!r}: empty parameter", hint="use e.g. 'k=v,k=v'")
            key, eq, value = piece.partition("=")
            key = key.strip()
            if not eq or not key:
                raise ConfigError(
                    f"{spec!r}: parameter {piece!r} is not key=value", hint="use e.g. 'fpr=0.01'"
                )
            if key in params:
                raise ConfigError(
                    f"{spec!r}: parameter {key!r} given twice", hint="pass each parameter once"
                )
            params[key] = _coerce(value.strip())
    return name, params


def _coerce(value: str) -> bool | int | float | str:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def compute(spec: str, y: IntArray, p: FloatArray) -> float:
    """Look up ``spec``'s metric in the ``metrics`` registry and call it on ``(y, p)``.

    ``spec`` is parsed by :func:`parse_metric_spec`; its parameters are validated against the
    metric's own signature (unknown parameters raise :class:`~dfwb.core.errors.ConfigError` with
    a did-you-mean, exactly as any other registry does).
    """
    name, params = parse_metric_spec(spec)
    registry = get_registry("metrics")
    kwargs = registry.validate(name, **params)
    metric: Metric = registry.load(name)
    return float(metric(np.asarray(y), np.asarray(p, dtype=np.float64), **kwargs))


# --------------------------------------------------------------------------- shared helpers


def _require_both_classes(y: IntArray, name: str) -> tuple[int, int]:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        only = "fake" if n_neg == 0 else "real"
        raise MetricUndefined(
            f"{name}: undefined because every label is {only!r}",
            hint="include at least one real and one fake example, or report coverage instead",
        )
    return n_pos, n_neg


def _require_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(
            f"{name} must be between 0 and 1, got {value!r}", hint=f"pass 0 <= {name} <= 1"
        )


def _describe_non_finite(p: FloatArray) -> str:
    n_nan = int(np.sum(np.isnan(p)))
    n_inf = int(np.sum(np.isinf(p)))
    parts = [f"{count} {label}" for count, label in ((n_nan, "NaN"), (n_inf, "inf")) if count]
    return ", ".join(parts)


def validate_scores(p: FloatArray, *, name: str, bounded: bool) -> FloatArray:
    """Check ``p`` is a 1-D array of finite scores, and, if ``bounded``, that they lie in [0, 1].

    Shared by every metric and by :func:`dfwb.eval.aggregate.aggregate`, so a NaN, an infinity or
    an out-of-range score is caught at the same point and described the same way everywhere,
    rather than silently producing a nonsense result (an AUC above 1, a NaN that propagates).

    Raises:
        ContractError: ``p`` is not 1-D, has a non-finite value, or (``bounded``) a value outside
            ``[0, 1]``.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim != 1:
        raise ContractError(
            f"{name}: scores must be 1-D, got shape {p_arr.shape}", hint="pass a 1-D array"
        )
    if not np.all(np.isfinite(p_arr)):
        raise ContractError(
            f"{name}: scores must be finite; found {_describe_non_finite(p_arr)}",
            hint="remove or fix NaN/inf scores before computing a metric",
        )
    if bounded and p_arr.size:
        out_of_range = p_arr[(p_arr < 0.0) | (p_arr > 1.0)]
        if out_of_range.size:
            raise ContractError(
                f"{name}: scores must be probabilities in [0, 1]; found {out_of_range[0].item()!r}",
                hint="scores are P(fake) in [0, 1]",
            )
    return p_arr


def _validate_labels(y: IntArray, *, name: str) -> IntArray:
    """Check ``y`` is a 1-D array of labels in ``{0, 1}`` (bool or int dtypes are both fine).

    Raises:
        ContractError: ``y`` is not 1-D, or has a value other than 0 or 1.
    """
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ContractError(
            f"{name}: labels must be 1-D, got shape {y_arr.shape}", hint="pass a 1-D array"
        )
    invalid = y_arr[~np.isin(y_arr, (0, 1))]
    if invalid.size:
        raise ContractError(
            f"{name}: labels must be 0 or 1; found {invalid[0].item()!r}",
            hint="use 0 for real, 1 for fake",
        )
    return y_arr


def _validate(
    y: IntArray, p: FloatArray, *, name: str, bounded: bool
) -> tuple[IntArray, FloatArray]:
    """Check ``y`` and ``p`` are same-length, valid 1-D arrays (see :func:`validate_scores` and
    :func:`_validate_labels`); the one entry point every metric calls before computing anything.

    Raises:
        ContractError: either array fails its own check, or they differ in length.
    """
    y_arr = _validate_labels(y, name=name)
    p_arr = validate_scores(p, name=name, bounded=bounded)
    if y_arr.shape[0] != p_arr.shape[0]:
        raise ContractError(
            f"{name}: y and p must be the same length, got {y_arr.shape[0]} and {p_arr.shape[0]}",
            hint="pass one label and one score per sample",
        )
    return y_arr, p_arr


def _average_ranks(x: FloatArray) -> FloatArray:
    """Ranks of ``x`` from 1 to ``len(x)``, tied values sharing their group's mean rank."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = sorted_x.shape[0]
    ranks_sorted = np.arange(1, n + 1, dtype=np.float64)
    _, start_idx, counts = np.unique(sorted_x, return_index=True, return_counts=True)
    for start, count in zip(start_idx, counts, strict=True):
        if count > 1:
            ranks_sorted[start : start + count] = ranks_sorted[start : start + count].mean()
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _roc_points(y: IntArray, p: FloatArray, name: str) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The ROC curve as ``(fpr, tpr, threshold)``, from ``(0, 0)`` to ``(1, 1)``, one point per
    threshold: rank-only, so any finite real score is accepted (not just ``[0, 1]``).

    Thresholds are every distinct score in ``p``, descending, plus a leading ``+inf`` (the
    threshold at which nothing is predicted fake, giving the ``(0, 0)`` point); tied scores share
    one point, so a run of ties never looks like several thresholds.
    """
    y, p = _validate(y, p, name=name, bounded=False)
    n_pos, n_neg = _require_both_classes(y, name)
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order].astype(np.float64)
    p_sorted = p[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    distinct = np.flatnonzero(np.diff(p_sorted))
    idx = np.r_[distinct, p_sorted.shape[0] - 1]
    thresholds = np.r_[np.inf, p_sorted[idx]]
    tps = np.r_[0.0, tps[idx]]
    fps = np.r_[0.0, fps[idx]]
    return fps / n_neg, tps / n_pos, thresholds


# --------------------------------------------------------------------------- metrics


def auc(y: IntArray, p: FloatArray, /) -> float:
    """ROC AUC: ``P(score of a random fake > score of a random real)``, ties counted as a half.

    Computed from the rank-sum (Mann-Whitney U) statistic: average the ranks of every sample
    (tied scores share their group's mean rank), then ``AUC = (sum of the fakes' ranks -
    n_pos*(n_pos+1)/2) / (n_pos*n_neg)``. Equivalent to the area under the ROC curve, without
    building the curve. Rank-only: any finite real score is accepted, not just ``[0, 1]``.

    Raises:
        MetricUndefined: ``y`` has no real example, or no fake one.
        ContractError: ``y`` or ``p`` is invalid (see :func:`validate_scores`).
    """
    y, p = _validate(y, p, name="auc", bounded=False)
    n_pos, n_neg = _require_both_classes(y, "auc")
    ranks = _average_ranks(p)
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ap(y: IntArray, p: FloatArray, /) -> float:
    """Average precision: the step-wise area under the precision-recall curve.

    At every distinct score, taken from the highest down (ties resolved together, as one
    threshold), precision and recall are the precision/recall of predicting fake for exactly the
    samples scored at or above it. ``AP = sum over those points of (recall - previous recall) *
    precision``, with recall starting at 0. This is the same step function scikit-learn's
    ``average_precision_score`` integrates. Rank-only: any finite real score is accepted, not
    just ``[0, 1]``.

    Raises:
        MetricUndefined: ``y`` has no real example, or no fake one.
        ContractError: ``y`` or ``p`` is invalid (see :func:`validate_scores`).
    """
    y, p = _validate(y, p, name="ap", bounded=False)
    n_pos, _ = _require_both_classes(y, "ap")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order].astype(np.float64)
    p_sorted = p[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    distinct = np.flatnonzero(np.diff(p_sorted))
    idx = np.r_[distinct, p_sorted.shape[0] - 1]
    tps = tps[idx]
    fps = fps[idx]
    precision = tps / (tps + fps)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def eer_point(y: IntArray, p: FloatArray, /) -> tuple[float, float]:
    """Equal error rate and its threshold: ``(eer, threshold)``, both linearly interpolated.

    The ROC curve's false-positive and false-negative rates move from ``(FPR, FNR) = (0, 1)`` to
    ``(1, 0)``; ``FPR - FNR`` is non-decreasing along it, so it crosses zero at most once. This
    walks to the pair of ROC points bracketing that crossing and interpolates linearly between
    them, rather than snapping to the closest one (which only agrees up to the gap between
    thresholds): with fraction ``t`` along the bracketing segment, ``eer`` is the mean of the
    interpolated FPR and FNR there (equal by construction, since that is exactly the crossing),
    and ``threshold`` is the score interpolated by that same ``t`` between the segment's two
    thresholds.

    One segment has no second threshold to interpolate towards: the one starting at ``+inf`` (no
    sample scored high enough to be the first predicted fake), since a threshold cannot be
    interpolated towards infinity. When the crossing falls there, ``threshold`` is simply the
    first real threshold reached (the highest score in ``p``), rather than a value pulled towards
    infinity.

    Rank-only: any finite real score is accepted, not just ``[0, 1]``.

    Raises:
        MetricUndefined: ``y`` has no real example, or no fake one.
        ContractError: ``y`` or ``p`` is invalid (see :func:`validate_scores`).
    """
    fpr_arr, tpr_arr, thresholds = _roc_points(y, p, "eer")
    fnr_arr = 1.0 - tpr_arr
    diff = fpr_arr - fnr_arr
    # _roc_points always starts at (fpr, tpr) = (0, 0), so diff[0] = 0 - 1 = -1 and this search
    # never lands on index 0: there is always a point to the left of the crossing to interpolate
    # from.
    idx = int(np.searchsorted(diff, 0.0))
    x0, x1 = diff[idx - 1], diff[idx]
    t = 0.0 if x1 == x0 else float(-x0 / (x1 - x0))
    fpr_at = fpr_arr[idx - 1] + t * (fpr_arr[idx] - fpr_arr[idx - 1])
    fnr_at = fnr_arr[idx - 1] + t * (fnr_arr[idx] - fnr_arr[idx - 1])
    eer_value = float((fpr_at + fnr_at) / 2)
    if np.isinf(thresholds[idx - 1]):
        threshold_at = float(thresholds[idx])
    else:
        threshold_at = float(thresholds[idx - 1] + t * (thresholds[idx] - thresholds[idx - 1]))
    return eer_value, threshold_at


def eer(y: IntArray, p: FloatArray, /) -> float:
    """Equal error rate: the point on the ROC curve where ``FPR == FNR``, linearly interpolated.

    See :func:`eer_point` for the threshold at that point and the exact interpolation; this is
    just its first element.

    Raises:
        MetricUndefined: ``y`` has no real example, or no fake one.
        ContractError: ``y`` or ``p`` is invalid (see :func:`validate_scores`).
    """
    return eer_point(y, p)[0]


def acc(y: IntArray, p: FloatArray, /, *, thr: float = 0.5) -> float:
    """Accuracy of predicting fake iff ``p >= thr`` (default ``thr=0.5``).

    ``p`` must be a probability in ``[0, 1]`` (see :func:`validate_scores`).

    Raises:
        ContractError: ``y`` or ``p`` is invalid.
    """
    y, p = _validate(y, p, name="acc", bounded=True)
    predicted = (p >= thr).astype(np.int64)
    return float(np.mean(predicted == y))


def tpr(y: IntArray, p: FloatArray, /, *, fpr: float, interp: bool = False) -> float:
    """The largest TPR achievable while keeping FPR at or below ``fpr`` (registry key ``tpr``).

    Conservative by default: it reports an operating point the ROC curve actually reaches, never
    one interpolated between two thresholds (which could overstate what is achievable). Pass
    ``interp=True`` to linearly interpolate the ROC curve at ``fpr`` instead. Rank-only: any
    finite real score is accepted, not just ``[0, 1]``.

    Raises:
        MetricUndefined: ``y`` has no real example, or no fake one.
        ContractError: ``y`` or ``p`` is invalid (see :func:`validate_scores`).
    """
    _require_unit_interval(fpr, "fpr")
    fpr_arr, tpr_arr, _ = _roc_points(y, p, "tpr")
    if interp:
        return float(np.interp(fpr, fpr_arr, tpr_arr))
    reachable = tpr_arr[fpr_arr <= fpr]
    return float(np.max(reachable))


def fpr(y: IntArray, p: FloatArray, /, *, tpr: float, interp: bool = False) -> float:
    """The smallest FPR needed to reach at least ``tpr`` (registry key ``fpr``); symmetric to `tpr`.

    Conservative by default: it reports an operating point the ROC curve actually reaches, never
    one interpolated between two thresholds (which could understate the FPR needed). Pass
    ``interp=True`` to linearly interpolate the ROC curve at ``tpr`` instead. Rank-only: any
    finite real score is accepted, not just ``[0, 1]``.

    Raises:
        MetricUndefined: ``y`` has no real example, or no fake one.
        ContractError: ``y`` or ``p`` is invalid (see :func:`validate_scores`).
    """
    _require_unit_interval(tpr, "tpr")
    fpr_arr, tpr_arr, _ = _roc_points(y, p, "fpr")
    if interp:
        return float(np.interp(tpr, tpr_arr, fpr_arr))
    reachable = fpr_arr[tpr_arr >= tpr]
    return float(np.min(reachable))


def ece(y: IntArray, p: FloatArray, /, *, bins: int = 15, adaptive: bool = False) -> float:
    """Expected calibration error: the weighted mean gap between confidence and accuracy.

    Samples are split into ``bins`` groups by their score ``p`` -- equal-width ``[0, 1]``
    intervals by default (the last one closed at both ends), or, with ``adaptive=True``,
    equal-count groups from the quantiles of ``p``. ``ECE = sum over non-empty bins of
    (n_bin/n) * |mean(p in the bin) - mean(y in the bin)|``. ``p`` must be a probability in
    ``[0, 1]`` (see :func:`validate_scores`).

    Raises:
        ConfigError: ``bins`` is not a positive integer.
        ContractError: ``y`` or ``p`` is invalid.
    """
    if bins < 1:
        raise ConfigError(f"bins must be a positive integer, got {bins!r}", hint="pass bins>=1")
    y, p = _validate(y, p, name="ece", bounded=True)
    y = y.astype(np.float64)
    n = p.shape[0]
    edges = (
        np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
        if adaptive
        else np.linspace(0.0, 1.0, bins + 1)
    )
    total = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p <= hi if b == bins - 1 else p < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        total += (count / n) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return total


def brier(y: IntArray, p: FloatArray, /) -> float:
    """Brier score: the mean squared error of ``p`` against ``y``, ``mean((p - y) ** 2)``.

    ``p`` must be a probability in ``[0, 1]`` (see :func:`validate_scores`).

    Raises:
        ContractError: ``y`` or ``p`` is invalid.
    """
    y, p = _validate(y, p, name="brier", bounded=True)
    return float(np.mean((p - y.astype(np.float64)) ** 2))


def nll(y: IntArray, p: FloatArray, /) -> float:
    """Binary cross-entropy (negative log-likelihood) of ``p`` against ``y``, clipped at ``1e-7``.

    ``p`` is clipped to ``[1e-7, 1 - 1e-7]`` before taking logarithms, so a score of exactly 0 or
    1 never makes this infinite. ``p`` must itself already be a probability in ``[0, 1]`` (see
    :func:`validate_scores`) -- the clip only guards the logarithm, not out-of-range input.

    Raises:
        ContractError: ``y`` or ``p`` is invalid.
    """
    y, p = _validate(y, p, name="nll", bounded=True)
    yf = y.astype(np.float64)
    clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(yf * np.log(clipped) + (1.0 - yf) * np.log(1.0 - clipped)))


def aurc(y: IntArray, p: FloatArray, /) -> float:
    """Area under the risk-coverage curve, for selective prediction.

    Confidence is ``conf = max(p, 1-p)`` and a prediction (fake iff ``p >= 0.5``) is correct iff
    it matches ``y``. Sorted from most to least confident, at coverage ``k/n`` the risk is the
    error rate of the ``k`` most confident predictions; ``AURC`` is the mean of that risk over
    every coverage level ``k = 1..n``. ``p`` must be a probability in ``[0, 1]`` (see
    :func:`validate_scores`): ``max(p, 1-p)`` and ``p >= 0.5`` both assume it.

    Raises:
        ContractError: ``y`` or ``p`` is invalid.
    """
    y, p = _validate(y, p, name="aurc", bounded=True)
    y = y.astype(np.float64)
    n = p.shape[0]
    conf = np.maximum(p, 1.0 - p)
    predicted = (p >= 0.5).astype(np.float64)
    correct = (predicted == y).astype(np.float64)
    order = np.argsort(-conf, kind="mergesort")
    correct_sorted = correct[order]
    k = np.arange(1, n + 1, dtype=np.float64)
    risks = np.cumsum(1.0 - correct_sorted) / k
    return float(np.mean(risks))
