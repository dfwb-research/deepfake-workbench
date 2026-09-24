"""Frame-to-video aggregation modes, against hand-computed fixtures."""

from __future__ import annotations

import math

import pytest

from dfwb.core.errors import ConfigError
from dfwb.eval.aggregate import aggregate

_FRAMES = [
    ("vid-a", 0.2),
    ("vid-a", 0.4),
    ("vid-a", 0.9),
    ("vid-b", 0.6),
    ("vid-b", 0.6),
]


def test_mean_prob():
    result = aggregate(_FRAMES, "mean-prob")
    assert result == pytest.approx({"vid-a": (0.2 + 0.4 + 0.9) / 3, "vid-b": 0.6})


def test_max():
    result = aggregate(_FRAMES, "max")
    assert result == pytest.approx({"vid-a": 0.9, "vid-b": 0.6})


def test_median():
    result = aggregate(_FRAMES, "median")
    assert result == pytest.approx({"vid-a": 0.4, "vid-b": 0.6})


def test_vote_default_threshold():
    # vid-a: 0.2,0.4 < 0.5 <= 0.9 -> 1/3 voted fake. vid-b: both >= 0.5 -> 2/2.
    result = aggregate(_FRAMES, "vote")
    assert result == pytest.approx({"vid-a": 1 / 3, "vid-b": 1.0})


def test_vote_custom_threshold():
    result = aggregate(_FRAMES, "vote@thr=0.95")
    assert result == pytest.approx({"vid-a": 0.0, "vid-b": 0.0})


def test_mean_logit_by_hand():
    frames = [("vid-a", 0.2), ("vid-a", 0.8)]

    def logit(x: float) -> float:
        return math.log(x / (1 - x))

    expected = 1.0 / (1.0 + math.exp(-(logit(0.2) + logit(0.8)) / 2))
    result = aggregate(frames, "mean-logit")
    assert result["vid-a"] == pytest.approx(expected)
    # 0.2 and 0.8 are symmetric around 0.5, so the mean logit is exactly 0 and the aggregated
    # score is exactly 0.5.
    assert result["vid-a"] == pytest.approx(0.5)


def test_mean_logit_clips_extreme_scores():
    frames = [("vid-a", 0.0), ("vid-a", 1.0)]
    result = aggregate(frames, "mean-logit")
    assert math.isfinite(result["vid-a"])


def test_empty_input_returns_empty_dict():
    assert aggregate([], "mean-prob") == {}


def test_grouping_does_not_require_adjacent_rows():
    shuffled = [_FRAMES[3], _FRAMES[0], _FRAMES[4], _FRAMES[1], _FRAMES[2]]
    assert aggregate(shuffled, "mean-prob") == pytest.approx(aggregate(_FRAMES, "mean-prob"))


def test_tuple_keys_are_supported():
    frames = [(("ds", "vid-a"), 0.1), (("ds", "vid-a"), 0.3)]
    result = aggregate(frames, "mean-prob")
    assert result == pytest.approx({("ds", "vid-a"): 0.2})


def test_unknown_mode_suggests_close_match():
    with pytest.raises(ConfigError) as info:
        aggregate(_FRAMES, "meen-prob")
    assert "mean-prob" in info.value.message


def test_unknown_parameter_for_a_mode_raises_config_error():
    with pytest.raises(ConfigError) as info:
        aggregate(_FRAMES, "vote@thrr=0.5")
    assert "unknown parameter" in info.value.message
    assert "thr" in info.value.message


def test_max_and_median_reject_parameters():
    with pytest.raises(ConfigError):
        aggregate(_FRAMES, "max@thr=0.5")
