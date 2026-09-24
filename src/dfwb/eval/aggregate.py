"""Frame-to-video aggregation: reduce one score per frame to one score per video.

Score files carry frame-level rows keyed by whatever the caller uses to identify a video (a
plain string, or a ``(dataset, key)`` pair -- :func:`aggregate` does not care). This groups rows
by that key and reduces each group's scores to a single float, by one of a fixed set of modes
parsed with the same ``name@k=v`` syntax as a metric spec.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable

import numpy as np
from numpy.typing import NDArray

from dfwb.core.errors import ConfigError, did_you_mean
from dfwb.eval.metrics import parse_metric_spec

__all__ = ["aggregate"]

FloatArray = NDArray[np.float64]

_EPS = 1e-7


def _mean_prob(scores: FloatArray) -> float:
    return float(np.mean(scores))


def _mean_logit(scores: FloatArray) -> float:
    clipped = np.clip(scores, _EPS, 1.0 - _EPS)
    logits = np.log(clipped / (1.0 - clipped))
    return float(1.0 / (1.0 + np.exp(-np.mean(logits))))


def _max_score(scores: FloatArray) -> float:
    return float(np.max(scores))


def _median_score(scores: FloatArray) -> float:
    return float(np.median(scores))


def _vote(scores: FloatArray, *, thr: float = 0.5) -> float:
    return float(np.mean(scores >= thr))


_MODES: dict[str, tuple[Callable[..., float], frozenset[str]]] = {
    "mean-prob": (_mean_prob, frozenset()),
    "mean-logit": (_mean_logit, frozenset()),
    "max": (_max_score, frozenset()),
    "median": (_median_score, frozenset()),
    "vote": (_vote, frozenset({"thr"})),
}


def aggregate[K: Hashable](frame_rows: Iterable[tuple[K, float]], mode: str) -> dict[K, float]:
    """Reduce ``(video_key, frame_score)`` rows to one score per key, by ``mode``.

    Modes (parsed by :func:`~dfwb.eval.metrics.parse_metric_spec`, so ``vote@thr=0.4`` works):

    - ``mean-prob``: the mean of the frame scores.
    - ``mean-logit``: frame scores mapped to logits (``log(p / (1-p))``, clipped away from 0
      and 1), averaged, then mapped back through the sigmoid.
    - ``max``: the largest frame score.
    - ``median``: the median frame score.
    - ``vote@thr=T`` (default ``T=0.5``): the fraction of frames with a score at or above ``T``.

    Rows for the same key need not be adjacent; every row is read once, in whatever order
    ``frame_rows`` yields them.

    Raises:
        ConfigError: ``mode``'s name is not one of the above, or gives it a parameter it does
            not accept.
    """
    name, params = parse_metric_spec(mode)
    if name not in _MODES:
        raise ConfigError(
            f"aggregate: unknown mode {name!r}{did_you_mean(name, _MODES)}",
            hint="modes: " + ", ".join(sorted(_MODES)),
        )
    fn, allowed = _MODES[name]
    for param_name in params:
        if param_name not in allowed:
            raise ConfigError(
                f"aggregate/{name}: {param_name}: unknown parameter"
                f"{did_you_mean(param_name, allowed)}",
                hint="accepted parameters: " + (", ".join(sorted(allowed)) or "none"),
            )
    groups: dict[K, list[float]] = {}
    for key, score in frame_rows:
        groups.setdefault(key, []).append(float(score))
    return {
        key: fn(np.asarray(values, dtype=np.float64), **params) for key, values in groups.items()
    }
