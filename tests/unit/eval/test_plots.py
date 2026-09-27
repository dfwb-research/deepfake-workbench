"""ROC, DET, reliability and risk-coverage plots, written as PNG and PDF."""

from __future__ import annotations

import numpy as np
import pytest

from dfwb.core.errors import InstallationError
from dfwb.eval.plots import PLOT_KINDS, write_plots

matplotlib = pytest.importorskip("matplotlib")


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    p = np.clip(0.5 + (y * 2 - 1) * 0.3 + rng.normal(scale=0.15, size=n), 1e-3, 1 - 1e-3)
    return y, p.astype(np.float64)


def test_write_plots_creates_png_and_pdf_for_every_kind(tmp_path):
    y, p = _synthetic()
    written = write_plots(y, p, tmp_path, "run")
    for kind in PLOT_KINDS:
        for ext in ("png", "pdf"):
            path = tmp_path / f"run-{kind}.{ext}"
            assert path in written
            assert path.is_file()
            assert path.stat().st_size > 0


def test_write_plots_skips_roc_and_det_for_a_single_class(tmp_path):
    y = np.zeros(20, dtype=np.int64)
    p = np.full(20, 0.3)
    written = write_plots(y, p, tmp_path, "onec")
    names = {path.name for path in written}
    assert not any(name.startswith("onec-roc") for name in names)
    assert not any(name.startswith("onec-det") for name in names)
    assert any(name.startswith("onec-reliability") for name in names)
    assert any(name.startswith("onec-risk-coverage") for name in names)


def test_write_plots_creates_output_directory(tmp_path):
    y, p = _synthetic()
    out = tmp_path / "nested" / "plots"
    write_plots(y, p, out, "run")
    assert (out / "run-roc.png").is_file()


def test_write_plots_raises_installation_error_without_matplotlib(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    y, p = _synthetic()
    with pytest.raises(InstallationError, match="matplotlib"):
        write_plots(y, p, tmp_path, "run")
