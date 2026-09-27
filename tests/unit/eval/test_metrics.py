"""Metric definitions: parameter syntax, hand-built fixtures, and adversarial inputs."""

from __future__ import annotations

import numpy as np
import pytest

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError
from dfwb.core.plugins import get_registry
from dfwb.eval.metrics import (
    MetricUndefined,
    acc,
    ap,
    auc,
    aurc,
    brier,
    compute,
    ece,
    eer,
    eer_point,
    fpr,
    nll,
    parse_metric_spec,
    tpr,
    validate_scores,
)

# ------------------------------------------------------------------- parameter syntax


@pytest.mark.parametrize(
    ("spec", "name", "params"),
    [
        ("auc", "auc", {}),
        ("tpr@fpr=0.01", "tpr", {"fpr": 0.01}),
        ("tpr@fpr=0.01,interp=true", "tpr", {"fpr": 0.01, "interp": True}),
        ("vote@thr=0.5", "vote", {"thr": 0.5}),
        ("ece@bins=10,adaptive=false", "ece", {"bins": 10, "adaptive": False}),
        (" auc ", "auc", {}),
        ("thr@k=abc", "thr", {"k": "abc"}),  # not bool/int/float: left as a plain string
    ],
)
def test_parse_metric_spec(spec, name, params):
    assert parse_metric_spec(spec) == (name, params)


@pytest.mark.parametrize(
    "spec",
    ["", "   ", "@fpr=0.01", "tpr@", "tpr@fpr", "tpr@fpr=0.01,", "tpr@fpr=0.01,fpr=0.02"],
)
def test_parse_metric_spec_rejects_malformed_specs(spec):
    with pytest.raises(ConfigError):
        parse_metric_spec(spec)


def test_compute_unknown_metric_name_suggests_close_match():
    y = np.array([0, 1])
    p = np.array([0.1, 0.9])
    with pytest.raises(UnknownKeyError) as info:
        compute("acu", y, p)
    assert "auc" in info.value.message


def test_compute_unknown_parameter_raises_config_error():
    y = np.array([0, 1])
    p = np.array([0.1, 0.9])
    with pytest.raises(ConfigError) as info:
        compute("auc@x=1", y, p)
    assert "unknown parameter" in info.value.message


def test_compute_typo_parameter_suggests_close_match():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ConfigError) as info:
        compute("tpr@fp=0.01", y, p)
    assert "fpr" in info.value.message


def test_compute_dispatches_to_the_registered_metric():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    assert compute("acc", y, p) == pytest.approx(acc(np.asarray(y), np.asarray(p, dtype=float)))
    assert compute("acc@thr=0.85", y, p) == pytest.approx(0.75)


def test_every_metric_is_registered_once():
    keys = get_registry("metrics").keys()
    assert set(keys) == {"auc", "ap", "eer", "acc", "tpr", "fpr", "ece", "brier", "nll", "aurc"}


# ------------------------------------------------------------------- single-class -> undefined


def test_auc_single_class_raises_metric_undefined():
    y = np.zeros(5, dtype=np.int64)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(MetricUndefined) as info:
        auc(y, p)
    assert info.value.exit_code == 4
    assert info.value.hint


@pytest.mark.parametrize("metric_fn", [auc, ap, eer])
@pytest.mark.parametrize("label", [0, 1])
def test_rank_metrics_undefined_for_single_class(metric_fn, label):
    y = np.full(6, label, dtype=np.int64)
    p = np.linspace(0.1, 0.9, 6)
    with pytest.raises(MetricUndefined):
        metric_fn(y, p)


def test_tpr_and_fpr_undefined_for_single_class():
    y = np.zeros(4, dtype=np.int64)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(MetricUndefined):
        tpr(y, p, fpr=0.1)
    with pytest.raises(MetricUndefined):
        fpr(y, p, tpr=0.1)


# ------------------------------------------------------------------- tpr@fpr / fpr@tpr, hand-built

# A tie at score 0.5 between one fake and one real example creates a genuine diagonal segment of
# the ROC curve (from (fpr, tpr) = (0.5, 0.5) to (0.75, 0.75)): the only case where a threshold
# moves both axes at once, so it is the only place "conservative" and "interp=true" can disagree.
_Y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
_P = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3])


def test_tpr_at_fpr_conservative_versus_interpolated():
    assert tpr(_Y, _P, fpr=0.6) == pytest.approx(0.5)
    assert tpr(_Y, _P, fpr=0.6, interp=True) == pytest.approx(0.6)


def test_fpr_at_tpr_conservative_versus_interpolated():
    assert fpr(_Y, _P, tpr=0.6) == pytest.approx(0.75)
    assert fpr(_Y, _P, tpr=0.6, interp=True) == pytest.approx(0.6)


def test_tpr_at_fpr_zero_is_the_first_reachable_point():
    # No threshold can guarantee FPR < the smallest step above 0, so the conservative reading of
    # fpr=0 is whatever TPR is reached before the first false positive: one fake (0.9) is scored
    # above the first real (0.8), for a TPR of 1/4.
    assert tpr(_Y, _P, fpr=0.0) == pytest.approx(0.25)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_tpr_and_fpr_reject_out_of_range_targets(value):
    with pytest.raises(ConfigError):
        tpr(_Y, _P, fpr=value)
    with pytest.raises(ConfigError):
        fpr(_Y, _P, tpr=value)


# ------------------------------------------------------------------- eer: linear interpolation vs.
# the old nearest-grid-point approximation


def _old_style_eer_approximation(y: np.ndarray, p: np.ndarray) -> float:
    """``i = argmin|fpr - fnr|; eer = (fpr[i] + fnr[i]) / 2`` over the raw ROC points."""
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order].astype(float)
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr_arr = np.r_[0.0, tps / tps[-1]]
    fpr_arr = np.r_[0.0, fps / fps[-1]]
    fnr_arr = 1 - tpr_arr
    i = int(np.argmin(np.abs(fpr_arr - fnr_arr)))
    return float((fpr_arr[i] + fnr_arr[i]) / 2)


def test_eer_interpolated_agrees_with_the_old_approximation_within_the_step_size():
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    p = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25])
    new_eer = eer(y, p)
    old_eer = _old_style_eer_approximation(y, p)
    assert new_eer == pytest.approx(0.4)
    assert old_eer == pytest.approx(11 / 30)
    step = max(1 / int(np.sum(y == 1)), 1 / int(np.sum(y == 0)))
    assert abs(new_eer - old_eer) < step
    assert new_eer != pytest.approx(old_eer, abs=1e-9)  # the two methods really do differ here


def test_eer_zero_for_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert eer(y, p) == pytest.approx(0.0)


def test_eer_point_reports_the_threshold_at_an_exact_grid_crossing():
    # The tie fixture used above for tpr@fpr/fpr@tpr: the FPR=FNR crossing lands exactly on the
    # ROC point (fpr, tpr) = (0.5, 0.5) (t=1.0 in the interpolation, i.e. no interpolation is
    # actually needed), one step before the tied score's diagonal segment begins. That point's
    # threshold is the real score 0.6, not the tied score 0.5.
    assert eer_point(_Y, _P) == pytest.approx((0.5, 0.6))


def test_eer_point_interpolates_the_threshold_fractionally():
    # Same fixture as test_eer_interpolated_agrees_with_the_old_approximation_within_the_step_size:
    # the crossing falls 4/5 of the way from threshold 0.75 to 0.65 (no ties, so a real fractional
    # interpolation, not just a boundary case): 0.75 + 0.8*(0.65-0.75) = 0.67.
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    p = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25])
    eer_value, threshold = eer_point(y, p)
    assert eer_value == pytest.approx(0.4)
    assert threshold == pytest.approx(0.67)


def test_eer_matches_eer_points_first_element():
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    p = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25])
    assert eer(y, p) == eer_point(y, p)[0]


# ------------------------------------------------------------------- ece, hand-computed


def test_ece_15_bins_by_hand():
    # bin 0 [0, 1/15): 0.02, 0.03, 0.04 (label 0, 0, 1)
    # bin 7 [7/15, 8/15): 0.48, 0.50   (label 1, 1)
    # bin 14 [14/15, 1]:  0.97          (label 1)
    y = np.array([0, 0, 1, 1, 1, 1])
    p = np.array([0.02, 0.03, 0.04, 0.48, 0.50, 0.97])
    bin0 = (3 / 6) * abs(0.03 - 1 / 3)
    bin7 = (2 / 6) * abs(0.49 - 1.0)
    bin14 = (1 / 6) * abs(0.97 - 1.0)
    assert ece(y, p) == pytest.approx(bin0 + bin7 + bin14, abs=1e-9)


def test_ece_adaptive_uses_quantile_bins():
    # Median-split into two equal-count bins: the low half has mean p = mean y = 0.25, the high
    # half has mean p = mean y = 0.75, so a coarse 2-bin adaptive split reads as well calibrated
    # even though the equal-*width* default (bins=15) would not group these the same way.
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    assert ece(y, p, bins=2, adaptive=True) == pytest.approx(0.0, abs=1e-9)


def test_ece_rejects_non_positive_bins():
    with pytest.raises(ConfigError):
        ece(np.array([0, 1]), np.array([0.1, 0.9]), bins=0)


# ------------------------------------------------------------------- aurc, against the old formula


def _old_style_aurc(y: np.ndarray, p: np.ndarray) -> float:
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == y
    order = np.argsort(-conf, kind="mergesort")
    correct_sorted = correct[order]
    n = len(p)
    risks = np.cumsum(1 - correct_sorted) / np.arange(1, n + 1)
    return float(np.mean(risks))


@pytest.mark.parametrize("seed", range(5))
def test_aurc_matches_the_old_formula(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 50)
    y = rng.integers(0, 2, size=n)
    p = rng.random(n)
    assert aurc(y, p) == pytest.approx(_old_style_aurc(y, p))


def test_aurc_single_sample():
    assert aurc(np.array([1]), np.array([0.9])) == pytest.approx(0.0)
    assert aurc(np.array([0]), np.array([0.9])) == pytest.approx(1.0)


# ------------------------------------------------------------------- adversarial inputs shared
# across every metric: all ties, n=1, and perfect separation


def test_all_ties_auc_and_eer():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.full(6, 0.5)
    assert auc(y, p) == pytest.approx(0.5)
    assert eer(y, p) == pytest.approx(0.5)


def test_all_ties_ap_matches_the_positive_rate():
    y = np.array([0, 1, 0, 1, 1])
    p = np.full(5, 0.5)
    assert ap(y, p) == pytest.approx(3 / 5)


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auc(y, p) == pytest.approx(1.0)
    assert ap(y, p) == pytest.approx(1.0)
    assert eer(y, p) == pytest.approx(0.0)
    assert tpr(y, p, fpr=0.0) == pytest.approx(1.0)
    assert fpr(y, p, tpr=1.0) == pytest.approx(0.0)


@pytest.mark.parametrize("well_defined", [acc, ece, brier, nll, aurc])
def test_metrics_well_defined_for_n_equals_one(well_defined):
    y = np.array([1])
    p = np.array([0.7])
    assert isinstance(well_defined(y, p), float)


@pytest.mark.parametrize("undefined_for_single_sample", [auc, ap, eer])
def test_rank_metrics_undefined_for_n_equals_one(undefined_for_single_sample):
    y = np.array([1])
    p = np.array([0.7])
    with pytest.raises(MetricUndefined):
        undefined_for_single_sample(y, p)


# ------------------------------------------------------------------- brier, nll, acc: exact math


def test_brier_is_mean_squared_error():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.4, 0.6])
    assert brier(y, p) == pytest.approx(np.mean((p - y) ** 2))


def test_nll_clips_extreme_scores():
    y = np.array([1, 0])
    p = np.array([0.0, 1.0])  # would be -inf/inf without clipping
    value = nll(y, p)
    assert np.isfinite(value)
    eps = 1e-7
    expected = -np.mean(
        y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))
    )
    assert value == pytest.approx(expected)


def test_acc_at_default_and_custom_threshold():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.6, 0.4])
    assert acc(y, p) == pytest.approx(0.5)  # thr=0.5: predictions [0,1,1,0] vs [0,1,0,1]
    assert acc(y, p, thr=0.65) == pytest.approx(0.75)  # predictions [0,1,0,0] vs [0,1,0,1]


# ------------------------------------------------------------------- input validation


_ALL_METRICS = [auc, ap, eer, tpr, fpr, acc, ece, brier, nll, aurc]
# auc/ap/eer/tpr/fpr are rank-only (any finite real score); acc/ece/brier/nll/aurc need [0, 1].
_BOUNDED_METRICS = [acc, ece, brier, nll, aurc]
_RANK_ONLY_METRICS = [auc, ap, eer, tpr, fpr]


def _call(metric_fn, y, p):
    if metric_fn in (tpr,):
        return metric_fn(y, p, fpr=0.5)
    if metric_fn in (fpr,):
        return metric_fn(y, p, tpr=0.5)
    return metric_fn(y, p)


@pytest.mark.parametrize("metric_fn", _ALL_METRICS)
def test_labels_outside_zero_one_are_rejected(metric_fn):
    y = np.array([0, 1, 2, 1])
    p = np.array([0.1, 0.9, 0.5, 0.7])
    with pytest.raises(ContractError) as info:
        _call(metric_fn, y, p)
    assert "labels must be 0 or 1; found 2" in info.value.message
    assert info.value.exit_code == 4
    assert info.value.hint


@pytest.mark.parametrize("metric_fn", _ALL_METRICS)
def test_nan_scores_are_rejected(metric_fn):
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, np.nan, 0.5, 0.7])
    with pytest.raises(ContractError) as info:
        _call(metric_fn, y, p)
    assert "scores must be finite; found 1 NaN" in info.value.message
    assert info.value.exit_code == 4


@pytest.mark.parametrize("metric_fn", _ALL_METRICS)
def test_inf_scores_are_rejected(metric_fn):
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, np.inf, 0.5, -np.inf])
    with pytest.raises(ContractError) as info:
        _call(metric_fn, y, p)
    assert "scores must be finite; found 2 inf" in info.value.message


@pytest.mark.parametrize("metric_fn", _ALL_METRICS)
def test_mismatched_lengths_are_rejected(metric_fn):
    y = np.array([0, 1, 0])
    p = np.array([0.1, 0.9])
    with pytest.raises(ContractError) as info:
        _call(metric_fn, y, p)
    assert "same length" in info.value.message


@pytest.mark.parametrize("metric_fn", _BOUNDED_METRICS)
def test_bounded_metrics_reject_scores_outside_unit_interval(metric_fn):
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 1.3, 0.5, 0.7])
    with pytest.raises(ContractError) as info:
        _call(metric_fn, y, p)
    assert "scores must be probabilities in [0, 1]; found 1.3" in info.value.message


@pytest.mark.parametrize("metric_fn", _RANK_ONLY_METRICS)
def test_rank_only_metrics_accept_scores_outside_unit_interval(metric_fn):
    # A score of 2.0 would break acc/ece/brier/nll/aurc's arithmetic, but rank-only metrics only
    # ever compare scores to each other, so any finite real value is fine.
    y = np.array([0, 1, 0, 1])
    p = np.array([-3.0, 2.0, -1.0, 5.0])
    result = _call(metric_fn, y, p)
    assert isinstance(result, float)


def test_bool_and_int_label_dtypes_are_both_accepted():
    p = np.array([0.1, 0.9, 0.2, 0.8])
    assert auc(np.array([False, True, False, True]), p) == auc(
        np.array([0, 1, 0, 1], dtype=np.int64), p
    )


def test_validate_scores_rejects_non_1d_arrays():
    with pytest.raises(ContractError, match="1-D"):
        validate_scores(np.zeros((2, 2)), name="x", bounded=False)


def test_non_1d_labels_are_rejected():
    with pytest.raises(ContractError, match="1-D"):
        auc(np.zeros((2, 2), dtype=np.int64), np.array([0.1, 0.2, 0.3, 0.4]))
