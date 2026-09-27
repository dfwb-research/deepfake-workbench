"""Frame-to-video aggregation: reduce one score per frame to one score per video.

Score files carry frame-level rows keyed by whatever the caller uses to identify a video (a
plain string, or a ``(dataset, key)`` pair -- :func:`aggregate` does not care). This groups rows
by that key and reduces each group's scores to a single float, by one of a fixed set of modes
parsed with the same ``name@k=v`` syntax as a metric spec.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dfwb.core.errors import ConfigError, DFWBError, did_you_mean
from dfwb.eval.metrics import check_metric_spec, parse_metric_spec, validate_scores

__all__ = ["aggregate", "check_eval_config", "check_mode"]

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


_MODES: dict[str, tuple[Callable[..., float], frozenset[str], bool]] = {
    # (function, accepted parameters, whether scores must be probabilities in [0, 1])
    "mean-prob": (_mean_prob, frozenset(), True),
    "mean-logit": (_mean_logit, frozenset(), True),
    "max": (_max_score, frozenset(), False),
    "median": (_median_score, frozenset(), False),
    "vote": (_vote, frozenset({"thr"}), True),
}


def _parse_mode(mode: str) -> tuple[str, dict[str, Any]]:
    name, params = parse_metric_spec(mode)
    if name not in _MODES:
        raise ConfigError(
            f"aggregate: unknown mode {name!r}{did_you_mean(name, _MODES)}",
            hint="modes: " + ", ".join(sorted(_MODES)),
        )
    allowed = _MODES[name][1]
    for param_name in params:
        if param_name not in allowed:
            raise ConfigError(
                f"aggregate/{name}: {param_name}: unknown parameter"
                f"{did_you_mean(param_name, allowed)}",
                hint="accepted parameters: " + (", ".join(sorted(allowed)) or "none"),
            )
    return name, params


def check_mode(mode: str) -> None:
    """Check an aggregation mode without aggregating anything (see :func:`aggregate`).

    Raises:
        ConfigError: The mode is unknown, or given a parameter it does not accept.
    """
    _parse_mode(mode)


def check_eval_config(metrics: Sequence[str], aggregate_mode: str) -> None:
    """Check a training config's ``eval:`` section -- every metric spec and the aggregation mode
    -- before anything is trained or scored, so a typo costs nothing.

    Raises:
        ConfigError: One line per problem, each at its config path (``eval.metrics[1]: ...``,
            ``eval.aggregate: ...``).
    """
    problems: list[tuple[str, DFWBError]] = []
    for index, spec in enumerate(metrics):
        try:
            check_metric_spec(spec)
        except DFWBError as exc:
            problems.append((f"eval.metrics[{index}]", exc))
    try:
        check_mode(aggregate_mode)
    except DFWBError as exc:
        problems.append(("eval.aggregate", exc))
    if not problems:
        return
    lines = [f"  {where}: {spec_problem.message}" for where, spec_problem in problems]
    hints = {spec_problem.hint for _, spec_problem in problems}
    raise ConfigError(
        f"{len(lines)} invalid eval value(s)\n" + "\n".join(lines),
        hint=hints.pop()
        if len(hints) == 1
        else "`dfwb plugins list` shows every metric; aggregation modes: "
        + ", ".join(sorted(_MODES)),
    )


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
    ``frame_rows`` yields them. Every score must be finite; ``mean-prob``, ``mean-logit`` and
    ``vote`` additionally require scores to be probabilities in ``[0, 1]`` (``max`` and
    ``median``, being order statistics, do not) -- see
    :func:`~dfwb.eval.metrics.validate_scores`.

    Raises:
        ConfigError: ``mode``'s name is not one of the above, or gives it a parameter it does
            not accept.
        ContractError: a frame score is not finite, or (for a mode that requires it) not in
            ``[0, 1]``.
    """
    name, params = _parse_mode(mode)
    fn, _, bounded = _MODES[name]
    groups: dict[K, list[float]] = {}
    for key, score in frame_rows:
        groups.setdefault(key, []).append(float(score))
    return {
        key: fn(
            validate_scores(np.asarray(values), name=f"aggregate/{name}", bounded=bounded),
            **params,
        )
        for key, values in groups.items()
    }
