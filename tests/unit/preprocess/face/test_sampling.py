import numpy as np
import pytest

from dfwb.preprocess.face.sampling import sample_indices


def test_uniform_matches_legacy_example():
    total, frames = 396, 64
    expected = [int(i) for i in np.unique(np.linspace(0, total - 1, frames, dtype=int))]

    result = sample_indices("uniform", total, frames=frames)

    assert result == expected
    assert result[:5] == [0, 6, 12, 18, 25]
    assert result[-2:] == [388, 395]


def test_uniform_short_video_collapses():
    result = sample_indices("uniform", 10, frames=32)

    assert result == list(range(10))


def test_uniform_requires_frames():
    with pytest.raises(ValueError, match="frames"):
        sample_indices("uniform", 100)


def test_first_consecutive_takes_the_first_n():
    result = sample_indices("first-consecutive", 100, frames=5)

    assert result == [0, 1, 2, 3, 4]


def test_first_consecutive_short_video_collapses():
    result = sample_indices("first-consecutive", 3, frames=64)

    assert result == [0, 1, 2]


def test_first_consecutive_requires_frames():
    with pytest.raises(ValueError, match="frames"):
        sample_indices("first-consecutive", 100)


def test_stride_takes_every_nth_frame():
    result = sample_indices("stride", 20, stride=5)

    assert result == [0, 5, 10, 15]


def test_stride_requires_stride():
    with pytest.raises(ValueError, match="stride"):
        sample_indices("stride", 100)


def test_all_takes_every_frame():
    result = sample_indices("all", 7)

    assert result == [0, 1, 2, 3, 4, 5, 6]


def test_all_of_an_empty_source_is_empty():
    assert sample_indices("all", 0) == []


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("all", {}),
        ("uniform", {"frames": 5}),
        ("first-consecutive", {"frames": 5}),
        ("stride", {"stride": 3}),
    ],
)
def test_an_empty_source_samples_nothing_for_every_mode(mode, kwargs):
    # The old pipeline never received an empty source, so it never had a formula for this case;
    # "uniform" in particular would otherwise divide up a source that has nothing in it and end up
    # with a made-up negative index. This is the one place sample_indices departs from the old
    # formulas: every mode simply has nothing to pick from.
    assert sample_indices(mode, 0, **kwargs) == []
