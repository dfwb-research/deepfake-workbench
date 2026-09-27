"""Every metric checked against scikit-learn, on 200 random cases plus adversarial ones.

This file is the only place scikit-learn is imported: it is a dev-only dependency, used solely
to cross-check the pure-numpy definitions in ``dfwb.eval.metrics``, never at runtime.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from dfwb.eval.metrics import MetricUndefined, ap, auc, brier, eer, eer_point, fpr, nll, tpr

# Coarse (2-decimal) scores make ties common, so a single strategy exercises both the generic
# random case and the tied-score case hypothesis is asked for separately.
_LABELLED_SCORES = st.integers(min_value=2, max_value=40).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n),
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, exclude_min=False).map(
                lambda x: round(x, 2)
            ),
            min_size=n,
            max_size=n,
        ),
    )
)


def _both_classes(case: tuple[list[int], list[float]]) -> bool:
    y, _ = case
    return 0 < sum(y) < len(y)


@given(_LABELLED_SCORES.filter(_both_classes))
@settings(max_examples=200, deadline=None)
def test_auc_matches_sklearn(case):
    y_list, p_list = case
    y = np.array(y_list, dtype=np.int64)
    p = np.array(p_list, dtype=np.float64)
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p))


@given(_LABELLED_SCORES.filter(_both_classes))
@settings(max_examples=200, deadline=None)
def test_ap_matches_sklearn(case):
    y_list, p_list = case
    y = np.array(y_list, dtype=np.int64)
    p = np.array(p_list, dtype=np.float64)
    assert ap(y, p) == pytest.approx(average_precision_score(y, p))


@given(_LABELLED_SCORES.filter(_both_classes))
@settings(max_examples=200, deadline=None)
def test_eer_brackets_a_sign_change_of_fpr_minus_fnr(case):
    # There is no direct scikit-learn EER; instead this checks the interpolated point against
    # scikit-learn's own ROC curve (undropped, so it has exactly the same points ours does): the
    # result must lie between the two ROC points that bracket the FPR=FNR crossing.
    y_list, p_list = case
    y = np.array(y_list, dtype=np.int64)
    p = np.array(p_list, dtype=np.float64)
    fpr_arr, tpr_arr, _ = roc_curve(y, p, drop_intermediate=False)
    fnr_arr = 1 - tpr_arr
    diff = fpr_arr - fnr_arr
    idx = int(np.searchsorted(diff, 0.0))
    lo = max(idx - 1, 0)
    hi = min(idx, len(diff) - 1)
    result = eer(y, p)
    assert min(fpr_arr[lo], fpr_arr[hi]) - 1e-9 <= result <= max(fpr_arr[lo], fpr_arr[hi]) + 1e-9


@given(_LABELLED_SCORES.filter(_both_classes))
@settings(max_examples=200, deadline=None)
def test_eer_point_first_element_matches_eer(case):
    y_list, p_list = case
    y = np.array(y_list, dtype=np.int64)
    p = np.array(p_list, dtype=np.float64)
    assert eer_point(y, p)[0] == eer(y, p)


@given(_LABELLED_SCORES.filter(_both_classes), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=200, deadline=None)
def test_tpr_at_fpr_is_conservative_against_sklearns_roc_curve(case, target):
    y_list, p_list = case
    y = np.array(y_list, dtype=np.int64)
    p = np.array(p_list, dtype=np.float64)
    fpr_arr, tpr_arr, _ = roc_curve(y, p, drop_intermediate=False)
    expected = float(np.max(tpr_arr[fpr_arr <= target]))
    assert tpr(y, p, fpr=target) == pytest.approx(expected)


@given(_LABELLED_SCORES.filter(_both_classes), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=200, deadline=None)
def test_fpr_at_tpr_is_conservative_against_sklearns_roc_curve(case, target):
    y_list, p_list = case
    y = np.array(y_list, dtype=np.int64)
    p = np.array(p_list, dtype=np.float64)
    fpr_arr, tpr_arr, _ = roc_curve(y, p, drop_intermediate=False)
    expected = float(np.min(fpr_arr[tpr_arr >= target]))
    assert fpr(y, p, tpr=target) == pytest.approx(expected)


@given(_LABELLED_SCORES)
@settings(max_examples=200, deadline=None)
def test_brier_matches_sklearn(case):
    y_list, p_list = case
    y = np.array(y_list, dtype=np.int64)
    p = np.array(p_list, dtype=np.float64)
    assert brier(y, p) == pytest.approx(brier_score_loss(y, p))


@given(
    st.integers(min_value=1, max_value=40).flatmap(
        lambda n: st.tuples(
            st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n),
            # kept away from the clipping boundary so sklearn's own (different) clip never fires
            st.lists(
                st.floats(min_value=0.01, max_value=0.99, allow_nan=False).map(
                    lambda x: round(x, 2)
                ),
                min_size=n,
                max_size=n,
            ),
        )
    )
)
@settings(max_examples=200, deadline=None)
def test_nll_matches_sklearn_away_from_the_clip_boundary(case):
    y_list, p_list = case
    y = np.array(y_list, dtype=np.int64)
    p = np.array(p_list, dtype=np.float64)
    assert nll(y, p) == pytest.approx(log_loss(y, p, labels=[0, 1]))


# ------------------------------------------------------------------- adversarial cases, against
# the same library


@pytest.mark.parametrize("n", [1, 2, 5])
def test_all_ties_match_sklearn(n):
    y = np.array([0, 1] * n, dtype=np.int64)
    p = np.full(2 * n, 0.5)
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert ap(y, p) == pytest.approx(average_precision_score(y, p))


def test_perfect_separation_matches_sklearn():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    p = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert ap(y, p) == pytest.approx(average_precision_score(y, p))


def test_n_equals_one_raises_metric_undefined_where_sklearn_is_silent():
    y = np.array([1], dtype=np.int64)
    p = np.array([0.9])
    with pytest.raises(MetricUndefined):
        auc(y, p)
    # scikit-learn does not raise for a single class; it warns and returns nan instead. This is
    # exactly the silent-placeholder behaviour dfwb.eval.metrics is built to avoid.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(roc_auc_score(y, p))
