"""Post-hoc calibration: temperature, Platt and isotonic, reference-tested and round-tripped
through a C5 file's provenance.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfwb.core.errors import ConfigError
from dfwb.core.records import ScoreRow, read_scores, write_scores
from dfwb.eval.calibrate import (
    CALIBRATION_METHODS,
    Calibration,
    apply_calibration,
    calibrate_file,
    fit_calibration,
)

from .conftest import make_meta

sklearn = pytest.importorskip("sklearn")


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # a systematically overconfident detector: push scores away from 0.5, then add noise.
    z = (y * 2 - 1) * 1.3 + rng.normal(scale=1.0, size=n)
    p = 1.0 / (1.0 + np.exp(-z * 2.0))
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return y.astype(np.int64), p.astype(np.float64)


def _logit(p, eps=1e-6):
    c = np.clip(p, eps, 1 - eps)
    return np.log(c / (1 - c))


def test_temperature_reference_against_scipy_minimize_scalar():
    scipy_opt = pytest.importorskip("scipy.optimize")
    y, p = _synthetic()
    calibration = fit_calibration("temperature", y, p)
    assert calibration.method == "temperature"
    t_ours = calibration.params["T"]

    z = _logit(p)

    def nll(t):
        sign = 1.0 - 2.0 * y.astype(np.float64)
        return float(np.mean(np.logaddexp(0.0, sign * (z / t))))

    ref = scipy_opt.minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    assert t_ours == pytest.approx(ref.x, abs=1e-2)
    assert nll(t_ours) == pytest.approx(nll(ref.x), abs=1e-6)


def test_temperature_bounds_are_respected():
    y, p = _synthetic()
    calibration = fit_calibration("temperature", y, p)
    assert 0.05 <= calibration.params["T"] <= 20.0


def test_platt_reference_against_sklearn_logistic_regression_on_logits():
    from sklearn.linear_model import LogisticRegression

    y, p = _synthetic(n=800, seed=1)
    calibration = fit_calibration("platt", y, p)
    assert calibration.method == "platt"

    x = _logit(p).reshape(-1, 1)
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-12)
    clf.fit(x, y)
    ref_a = clf.coef_[0, 0]
    ref_b = clf.intercept_[0]

    assert calibration.params["a"] == pytest.approx(ref_a, abs=1e-2)
    assert calibration.params["b"] == pytest.approx(ref_b, abs=1e-2)

    applied = apply_calibration(calibration, p)
    ref_applied = clf.predict_proba(x)[:, 1]
    assert applied == pytest.approx(ref_applied, abs=1e-4)


def test_isotonic_reference_against_sklearn_isotonic_regression():
    from sklearn.isotonic import IsotonicRegression

    y, p = _synthetic(n=300, seed=2)
    calibration = fit_calibration("isotonic", y, p)
    assert calibration.method == "isotonic"

    ref = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ref.fit(p, y)

    probe = np.linspace(0.0, 1.0, 101)
    ours = apply_calibration(calibration, probe)
    theirs = ref.predict(probe)
    assert ours == pytest.approx(theirs, abs=1e-6)


def test_isotonic_output_is_monotonic_and_in_unit_interval():
    y, p = _synthetic(n=200, seed=3)
    calibration = fit_calibration("isotonic", y, p)
    probe = np.sort(np.linspace(0.0, 1.0, 50))
    applied = apply_calibration(calibration, probe)
    assert np.all(applied >= 0.0)
    assert np.all(applied <= 1.0)
    assert np.all(np.diff(applied) >= -1e-12)


def test_apply_calibration_identity_ish_for_temperature_one():
    calibration = Calibration("temperature", {"T": 1.0})
    p = np.array([0.1, 0.5, 0.9])
    assert apply_calibration(calibration, p) == pytest.approx(p, abs=1e-9)


def test_unknown_method_rejected():
    with pytest.raises(ConfigError, match="unknown calibration method"):
        fit_calibration("nope", np.array([0, 1]), np.array([0.1, 0.9]))


def test_calibration_methods_tuple():
    assert CALIBRATION_METHODS == ("temperature", "platt", "isotonic")


# --------------------------------------------------------------------------- calibrate_file


def _write(tmp_path, name, rows, meta):
    path, _ = write_scores(tmp_path / name, rows, meta)
    return path


def _rows(n, seed):
    y, p = _synthetic(n=n, seed=seed)
    return [ScoreRow("d", f"v{i:04d}", None, int(y[i]), float(p[i]), "ok") for i in range(len(y))]


def _cov(rows):
    return {"expected": len(rows), "ok": len(rows), "missing": 0, "error": 0}


def test_calibrate_file_writes_provenance_into_meta(tmp_path):
    fit_rows = _rows(200, 10)
    fit_path = _write(tmp_path, "val.scores.csv", fit_rows, make_meta(coverage=_cov(fit_rows)))
    apply_rows = _rows(100, 11)
    apply_path = _write(
        tmp_path,
        "test.scores.csv",
        apply_rows,
        make_meta(coverage={"expected": 100, "ok": 100, "missing": 0, "error": 0}),
    )

    result = calibrate_file(fit_path, apply_path, "temperature")

    assert result.meta.calibration is not None
    assert result.meta.calibration.method == "temperature"
    assert len(result.meta.calibration.fit_file_sha256) == 64
    assert "T" in result.meta.calibration.params


def test_calibrate_file_meta_fit_hash_matches_the_fit_file(tmp_path):
    from dfwb.core.hashing import sha256_file

    fit_rows = _rows(150, 20)
    fit_path = _write(tmp_path, "val.scores.csv", fit_rows, make_meta(coverage=_cov(fit_rows)))
    apply_path = _write(
        tmp_path,
        "test.scores.csv",
        _rows(80, 21),
        make_meta(coverage={"expected": 80, "ok": 80, "missing": 0, "error": 0}),
    )

    result = calibrate_file(fit_path, apply_path, "platt")
    assert result.meta.calibration.fit_file_sha256 == sha256_file(fit_path)
    assert result.meta.calibration.method == "platt"
    assert set(result.meta.calibration.params) == {"a", "b"}


def test_calibrate_file_preserves_non_ok_rows_and_recalibrates_ok_rows(tmp_path):
    fit_rows = _rows(150, 30)
    fit_path = _write(tmp_path, "val.scores.csv", fit_rows, make_meta(coverage=_cov(fit_rows)))
    apply_rows = [
        *_rows(20, 31),
        ScoreRow("d", "missing0", None, 0, None, "missing"),
        ScoreRow("d", "error0", None, 1, None, "error"),
    ]
    apply_path = _write(
        tmp_path,
        "test.scores.csv",
        apply_rows,
        make_meta(coverage={"expected": 22, "ok": 20, "missing": 1, "error": 1}),
    )

    result = calibrate_file(fit_path, apply_path, "isotonic")
    by_key = {r.key: r for r in result.rows}
    assert by_key["missing0"].status == "missing"
    assert by_key["missing0"].score is None
    assert by_key["error0"].status == "error"
    assert by_key["error0"].score is None

    # Every ok row's new score must be exactly what applying the fitted calibration to its
    # *original* score gives -- not merely "some number in [0, 1]" -- and, since the underlying
    # detector is deliberately miscalibrated (see `_synthetic`), at least one must actually move.
    original_by_key = {r.key: r for r in apply_rows}
    any_score_changed = False
    for row in result.rows:
        if row.status != "ok":
            continue
        original_score = original_by_key[row.key].score
        expected = float(apply_calibration(result.calibration, np.array([original_score]))[0])
        assert row.score == pytest.approx(expected)
        assert 0.0 <= row.score <= 1.0
        if abs(row.score - original_score) > 1e-9:
            any_score_changed = True
    assert any_score_changed


def test_calibrate_file_round_trips_through_write_scores(tmp_path):
    fit_rows = _rows(150, 40)
    fit_path = _write(tmp_path, "val.scores.csv", fit_rows, make_meta(coverage=_cov(fit_rows)))
    apply_path = _write(
        tmp_path,
        "test.scores.csv",
        _rows(60, 41),
        make_meta(coverage={"expected": 60, "ok": 60, "missing": 0, "error": 0}),
    )
    result = calibrate_file(fit_path, apply_path, "temperature")
    out_path, _ = write_scores(tmp_path / "test.temperature.scores.csv", result.rows, result.meta)
    reloaded = read_scores(out_path)
    assert reloaded.meta.calibration.method == "temperature"


def test_calibrate_file_requires_meta_on_the_apply_file(tmp_path):
    from dfwb.core.errors import ContractError

    fit_rows = _rows(50, 50)
    fit_path = _write(tmp_path, "val.scores.csv", fit_rows, make_meta(coverage=_cov(fit_rows)))
    apply_path = tmp_path / "noraw.scores.csv"
    apply_path.write_text("dataset,key,compression,label,score,status\nd,v0,,0,0.2,ok\n")

    with pytest.raises(ContractError, match="meta"):
        calibrate_file(fit_path, apply_path, "temperature")
