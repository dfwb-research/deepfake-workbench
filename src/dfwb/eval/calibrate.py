"""Post-hoc calibration of P(fake) scores: temperature, Platt and isotonic (numpy only).

None of these methods is a research contribution; they are the standard, generic tools for making
an already-trained detector's scores better probabilities, fit on one score file (typically a
validation split) and applied to another. All three work on scores already in ``[0, 1]``, and none
needs anything beyond numpy at runtime (a golden-section search stands in for a bounded scalar
optimiser, and a small Newton loop for a two-parameter logistic fit). :func:`calibrate_file` turns
a fit and an apply score file into a new C5 file: the same rows, with ``ok`` scores recalibrated
and the calibration's method, parameters and the fit file's hash recorded in its meta.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from dfwb.core.errors import ConfigError, ContractError, did_you_mean
from dfwb.core.hashing import sha256_file
from dfwb.core.records import ScoreFile, ScoreMeta, ScoreRow, read_scores
from dfwb.core.records.scores import CalibrationInfo
from dfwb.core.records.scores import GitState as _MetaGitState
from dfwb.core.runmeta import RunInfo, collect_run_info
from dfwb.eval.coverage import labels_and_scores
from dfwb.eval.metrics import validate_scores

__all__ = [
    "CALIBRATION_METHODS",
    "CalibratedResult",
    "Calibration",
    "apply_calibration",
    "calibrate_file",
    "fit_calibration",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: The three post-hoc calibration methods, in the order the CLI documents them.
CALIBRATION_METHODS: Final = ("temperature", "platt", "isotonic")

_EPS: Final = 1e-6
_T_LO: Final = 0.05
_T_HI: Final = 20.0


@dataclass(frozen=True)
class Calibration:
    """A fitted calibrator: which method, and its parameters (JSON-friendly, C5 meta-ready)."""

    method: str
    params: dict[str, Any]


@dataclass(frozen=True)
class CalibratedResult:
    """The result of :func:`calibrate_file`: the apply file's rows, recalibrated, plus its meta."""

    rows: list[ScoreRow]
    meta: ScoreMeta
    calibration: Calibration


def _logit(p: FloatArray, eps: float = _EPS) -> FloatArray:
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(z: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-z))


def _parse_created(text: str) -> datetime.datetime:
    """``created``'s ``dfwb.core.runmeta.utc_now()`` string as an aware ``datetime``.

    ``ScoreMeta.created`` is validated as ``AwareDatetime``; ``model_copy(update=...)`` (used to
    refresh a calibrated file's provenance) never re-validates, so the plain string
    :func:`~dfwb.core.runmeta.collect_run_info` returns must be parsed by hand here.
    """
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC)


def _validate_labels(y: IntArray) -> IntArray:
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ContractError(
            f"calibrate: labels must be 1-D, got shape {y_arr.shape}", hint="pass a 1-D array"
        )
    invalid = y_arr[~np.isin(y_arr, (0, 1))]
    if invalid.size:
        raise ContractError(
            f"calibrate: labels must be 0 or 1; found {invalid[0].item()!r}",
            hint="use 0 for real, 1 for fake",
        )
    return y_arr


def _golden_section_minimize(
    f: Callable[[float], float], lo: float, hi: float, *, tol: float = 1e-6, max_iter: int = 200
) -> float:
    """The minimiser of a unimodal ``f`` on ``[lo, hi]``, bounded scalar search, no scipy needed."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _bce_nll(y: FloatArray, z: FloatArray) -> float:
    """Mean binary cross-entropy of pre-sigmoid logits ``z`` against labels ``y`` in ``{0, 1}``.

    Computed as ``mean(softplus((1 - 2y) * z))`` via :func:`numpy.logaddexp`, which never
    overflows even for large ``|z|`` (unlike forming ``sigmoid(z)`` and taking its log directly).
    """
    sign = 1.0 - 2.0 * y
    return float(np.mean(np.logaddexp(0.0, sign * z)))


def _fit_temperature(y: IntArray, p: FloatArray) -> Calibration:
    """Temperature scaling: the ``T`` minimising NLL(sigmoid(logit(p) / T)) over ``[0.05, 20]``."""
    z = _logit(p)
    yf = y.astype(np.float64)

    def nll(t: float) -> float:
        return _bce_nll(yf, z / t)

    t = _golden_section_minimize(nll, _T_LO, _T_HI)
    return Calibration("temperature", {"T": t})


def _fit_platt(
    y: IntArray, p: FloatArray, *, max_iter: int = 100, tol: float = 1e-12
) -> Calibration:
    """Platt scaling: ``sigmoid(a * logit(p) + b)``, ``(a, b)`` fit by Newton's method.

    Plain maximum-likelihood logistic regression of ``y`` on the single feature ``logit(p)``
    (no regularisation), which is what "Platt scaling" means outside the original paper's own
    out-of-sample label-smoothing heuristic. The Hessian gets a tiny ridge so a near-degenerate
    fit (very few points, or near-perfect separation) never divides by zero.
    """
    x = _logit(p)
    yf = y.astype(np.float64)
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        z = a * x + b
        pi = _sigmoid(z)
        w = pi * (1.0 - pi)
        g_a = float(np.mean((pi - yf) * x))
        g_b = float(np.mean(pi - yf))
        h_aa = float(np.mean(w * x * x)) + 1e-10
        h_ab = float(np.mean(w * x))
        h_bb = float(np.mean(w)) + 1e-10
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-18:
            break
        delta_a = (h_bb * g_a - h_ab * g_b) / det
        delta_b = (h_aa * g_b - h_ab * g_a) / det
        a -= delta_a
        b -= delta_b
        if abs(delta_a) < tol and abs(delta_b) < tol:
            break
    return Calibration("platt", {"a": a, "b": b})


def _pool_adjacent_violators(values: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted pool-adjacent-violators: the non-decreasing least-squares fit of ``values``."""
    blocks: list[list[float]] = []  # each: [value, weight, start, end]
    for index in range(values.shape[0]):
        block = [float(values[index]), float(weights[index]), index, index]
        blocks.append(block)
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = left[1] + right[1]
            merged_value = (left[0] * left[1] + right[0] * right[1]) / merged_weight
            blocks.append([merged_value, merged_weight, left[2], right[3]])
    result = np.empty(values.shape[0], dtype=np.float64)
    for value, _weight, start, end in blocks:
        result[int(start) : int(end) + 1] = value
    return result


def _fit_isotonic(y: IntArray, p: FloatArray) -> Calibration:
    """Isotonic regression of ``y`` on ``p`` (PAV), clipped to ``[0, 1]``.

    Tied ``p`` values are first collapsed to their mean label (a monotone function can only take
    one value at one point), then pooled; :func:`apply_calibration` linearly interpolates between
    the resulting knots, matching :class:`sklearn.isotonic.IsotonicRegression`'s own behaviour.
    """
    x = np.asarray(p, dtype=np.float64)
    yf = y.astype(np.float64)
    order = np.argsort(x, kind="mergesort")
    unique_x, inverse, counts = np.unique(x[order], return_inverse=True, return_counts=True)
    sums = np.zeros(unique_x.shape[0], dtype=np.float64)
    np.add.at(sums, inverse, yf[order])
    means = sums / counts
    fitted = np.clip(_pool_adjacent_violators(means, counts.astype(np.float64)), 0.0, 1.0)
    return Calibration("isotonic", {"x": unique_x.tolist(), "y": fitted.tolist()})


_FITTERS: dict[str, Callable[[IntArray, FloatArray], Calibration]] = {
    "temperature": _fit_temperature,
    "platt": _fit_platt,
    "isotonic": _fit_isotonic,
}


def fit_calibration(method: str, y: IntArray, p: FloatArray) -> Calibration:
    """Fit ``method`` (one of :data:`CALIBRATION_METHODS`) on ``(y, p)``.

    Raises:
        ConfigError: ``method`` is not one of :data:`CALIBRATION_METHODS`.
        ContractError: ``y`` or ``p`` is invalid, or they differ in length.
    """
    if method not in _FITTERS:
        raise ConfigError(
            f"unknown calibration method {method!r}{did_you_mean(method, CALIBRATION_METHODS)}",
            hint="methods: " + ", ".join(CALIBRATION_METHODS),
        )
    y_arr = _validate_labels(y)
    p_arr = validate_scores(
        np.asarray(p, dtype=np.float64), name=f"calibrate/{method}", bounded=True
    )
    if y_arr.shape[0] != p_arr.shape[0]:
        raise ContractError(
            f"calibrate/{method}: y and p must be the same length, got {y_arr.shape[0]} and "
            f"{p_arr.shape[0]}",
            hint="pass one label and one score per sample",
        )
    if y_arr.shape[0] == 0:
        raise ContractError(
            f"calibrate/{method}: no rows to fit on", hint="pass a non-empty fit file"
        )
    return _FITTERS[method](y_arr, p_arr)


def apply_calibration(calibration: Calibration, p: FloatArray) -> FloatArray:
    """Map raw scores ``p`` through a fitted :class:`Calibration`; result stays in ``[0, 1]``.

    Raises:
        ConfigError: ``calibration.method`` is not one of :data:`CALIBRATION_METHODS`.
        ContractError: ``p`` is not a valid array of scores.
    """
    p_arr = validate_scores(
        np.asarray(p, dtype=np.float64), name=f"calibrate/{calibration.method}", bounded=True
    )
    if calibration.method == "temperature":
        t = float(calibration.params["T"])
        return _sigmoid(_logit(p_arr) / t)
    if calibration.method == "platt":
        a = float(calibration.params["a"])
        b = float(calibration.params["b"])
        return _sigmoid(a * _logit(p_arr) + b)
    if calibration.method == "isotonic":
        knots_x = np.asarray(calibration.params["x"], dtype=np.float64)
        knots_y = np.asarray(calibration.params["y"], dtype=np.float64)
        return np.clip(np.interp(p_arr, knots_x, knots_y), 0.0, 1.0)
    raise ConfigError(
        f"unknown calibration method {calibration.method!r}"
        f"{did_you_mean(calibration.method, CALIBRATION_METHODS)}",
        hint="methods: " + ", ".join(CALIBRATION_METHODS),
    )


def _recalibrated_row(row: ScoreRow, calibration: Calibration) -> ScoreRow:
    if row.status != "ok":
        return row
    assert row.score is not None  # invariant of ScoreRow: ok always carries a score
    new_score = float(apply_calibration(calibration, np.array([row.score]))[0])
    return dataclasses.replace(row, score=new_score, logit=None)


def calibrate_file(
    fit: str | os.PathLike[str],
    apply: str | os.PathLike[str],
    method: str,
    *,
    run_info: RunInfo | None = None,
) -> CalibratedResult:
    """Fit ``method`` on ``fit``'s ``ok`` rows and apply it to ``apply``'s rows.

    Every row of ``apply`` is kept: ``ok`` rows get a recalibrated ``score`` (and a cleared
    ``logit``, since the original one no longer matches it); ``missing``/``error`` rows pass
    through unchanged. The result's meta is ``apply``'s meta with ``calibration`` set to the
    method, its parameters, and ``fit``'s sha256 (so the calibrated file's provenance is
    self-contained), and ``env``/``command``/``created``/``git`` refreshed for this run.

    Raises:
        ContractError: either file cannot be read (see
            :func:`~dfwb.core.records.read_scores`), ``apply`` has no ``.meta.json``, or ``fit``
            has no ``ok`` rows.
        ConfigError: ``method`` is not one of :data:`CALIBRATION_METHODS`.
    """
    fit_file: ScoreFile = read_scores(fit)
    apply_file: ScoreFile = read_scores(apply)
    if apply_file.meta is None:
        raise ContractError(
            f"{Path(apply).name}: needs a .meta.json to calibrate (no provenance to build on)",
            hint="keep .meta.json alongside .scores.csv",
        )
    y, p, _kept = labels_and_scores(fit_file.rows, missing="exclude")
    if y.size == 0:
        raise ContractError(
            f"{Path(fit).name}: no 'ok' rows to fit calibration on",
            hint="pass a fit file with scored rows",
        )
    calibration = fit_calibration(method, y, p)
    rows = [_recalibrated_row(row, calibration) for row in apply_file.rows]

    info = run_info or collect_run_info()
    git = _MetaGitState(commit=info.git.commit, dirty=info.git.dirty) if info.git else None
    meta = apply_file.meta.model_copy(
        update={
            "calibration": CalibrationInfo(
                method=calibration.method,  # type: ignore[arg-type]  # one of CALIBRATION_METHODS
                fit_file_sha256=sha256_file(fit),
                params=calibration.params,
            ),
            "env": info.env,
            "git": git,
            "command": info.command,
            "created": _parse_created(info.created),
        }
    )
    return CalibratedResult(rows=rows, meta=meta, calibration=calibration)
