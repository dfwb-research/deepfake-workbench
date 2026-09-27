"""ROC, DET, reliability and risk-coverage plots for score files (the ``[eval]`` extra).

Every plot takes plain ``(y, p)`` arrays, so it works for a whole file or for any subset a caller
has already applied a ``--missing`` policy to. matplotlib is imported lazily, behind
:class:`~dfwb.core.errors.InstallationError`, so the rest of ``dfwb.eval`` (including every other
name in this module) stays usable without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import NDArray

from dfwb.core.errors import InstallationError
from dfwb.core.registry import install_hint
from dfwb.eval.metrics import _roc_points  # the same ROC construction eer/tpr/fpr already use

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["PLOT_KINDS", "write_plots"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: The four curves :func:`write_plots` draws, in the order it draws them.
PLOT_KINDS: Final = ("roc", "det", "reliability", "risk-coverage")

_RELIABILITY_BINS = 15


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as exc:
        raise InstallationError(
            "plots need matplotlib, which is not installed", hint=install_hint("matplotlib")
        ) from exc
    return plt


def _draw_roc(ax: Axes, y: IntArray, p: FloatArray) -> None:
    fpr, tpr, _thresholds = _roc_points(y, p, "roc plot")
    ax.plot(fpr, tpr)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=0.75, color="gray")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")


def _draw_det(ax: Axes, y: IntArray, p: FloatArray) -> None:
    fpr, tpr, _thresholds = _roc_points(y, p, "det plot")
    fnr = 1.0 - tpr
    eps = 1e-4
    ax.plot(np.clip(fpr, eps, 1.0), np.clip(fnr, eps, 1.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FPR (log)")
    ax.set_ylabel("FNR (log)")
    ax.set_title("DET")


def _draw_reliability(
    ax: Axes, y: IntArray, p: FloatArray, *, bins: int = _RELIABILITY_BINS
) -> None:
    yf = y.astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    xs: list[float] = []
    ys: list[float] = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p <= hi if b == bins - 1 else p < hi)
        if not np.any(mask):
            continue
        xs.append(float(np.mean(p[mask])))
        ys.append(float(np.mean(yf[mask])))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=0.75, color="gray")
    ax.plot(xs, ys, marker="o")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean predicted P(fake)")
    ax.set_ylabel("observed fraction fake")
    ax.set_title("Reliability")


def _draw_risk_coverage(ax: Axes, y: IntArray, p: FloatArray) -> None:
    yf = y.astype(np.float64)
    n = p.shape[0]
    confidence = np.maximum(p, 1.0 - p)
    predicted = (p >= 0.5).astype(np.float64)
    correct = (predicted == yf).astype(np.float64)
    order = np.argsort(-confidence, kind="mergesort")
    correct_sorted = correct[order]
    k = np.arange(1, n + 1, dtype=np.float64)
    risk = np.cumsum(1.0 - correct_sorted) / k
    coverage = k / n
    ax.plot(coverage, risk)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk")
    ax.set_title("Risk-coverage")


_DRAW: Final = {
    "roc": _draw_roc,
    "det": _draw_det,
    "reliability": _draw_reliability,
    "risk-coverage": _draw_risk_coverage,
}
#: The curves that are undefined (need both classes) and so are skipped for single-class data.
_NEEDS_BOTH_CLASSES: Final = frozenset({"roc", "det"})


def write_plots(y: IntArray, p: FloatArray, out_dir: str | Path, prefix: str) -> list[Path]:
    """Write ROC, DET, reliability and risk-coverage curves for ``(y, p)``, as PNG and PDF.

    ROC and DET are skipped when ``y`` has only one class (they are undefined then, like the
    ``auc``/``eer`` metrics); reliability and risk-coverage need only ``p`` and are always drawn.
    Files are named ``<out_dir>/<prefix>-<kind>.{png,pdf}``; ``out_dir`` is created if needed.

    Raises:
        InstallationError: matplotlib (the ``[eval]`` extra) is not installed.
    """
    plt = _pyplot()
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    has_both_classes = bool(np.any(y == 0)) and bool(np.any(y == 1))
    written: list[Path] = []
    for kind in PLOT_KINDS:
        if kind in _NEEDS_BOTH_CLASSES and not has_both_classes:
            continue
        fig, ax = plt.subplots()
        try:
            _DRAW[kind](ax, y, p)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                out_path = target / f"{prefix}-{kind}.{ext}"
                fig.savefig(out_path)
                written.append(out_path)
        finally:
            plt.close(fig)
    return written
