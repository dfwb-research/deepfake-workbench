"""``ClipSpec`` and window arithmetic: which stored-frame positions a clip decodes.

Every expected window below is either a literal, hand-worked-out list, or drawn from a second
``random.Random``/``numpy.linspace`` call seeded and invoked the same way the rules describe --
never by calling :func:`clip_windows` itself and trusting its own answer.
"""

from __future__ import annotations

import random

import pytest

from dfwb.core.config.schema import ClipSection
from dfwb.core.config.schema import ClipsPerVideo as ConfigClipsPerVideo
from dfwb.core.errors import ContractError
from dfwb.data.clips import ClipSpec, ClipsPerVideo, clip_windows, clip_windows_padded

# ------------------------------------------------------------------------------------- ClipSpec


def test_from_config_copies_frames_sampling_and_clips_per_video():
    section = ClipSection(
        frames=8, sampling="consecutive", clips_per_video=ConfigClipsPerVideo(train=4, eval=10)
    )
    spec = ClipSpec.from_config(section)
    assert spec == ClipSpec(
        frames=8,
        sampling="consecutive",
        clips_per_video=ClipsPerVideo(train=4, eval=10),
        stride=1,
    )


def test_from_config_takes_an_explicit_stride():
    section = ClipSection(
        frames=2, sampling="uniform", clips_per_video=ConfigClipsPerVideo(train=1, eval=1)
    )
    spec = ClipSpec.from_config(section, stride=3)
    assert spec.stride == 3


def test_clips_per_mode_picks_train_or_eval():
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=2, eval=32))
    assert spec.clips_per_mode(train=True) == 2
    assert spec.clips_per_mode(train=False) == 32


# ---------------------------------------------------------------------------------------- uniform


def test_uniform_t1_reproduces_32_evenly_spaced_frames():
    # the thesis's 32FA regime: T=1, 32 evenly spaced frames per video.
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=32, eval=32))
    n = 200
    windows = clip_windows(n, spec, train=False, rng=None)
    assert len(windows) == 32
    assert all(len(w) == 1 for w in windows)
    expected = [int(k * (n - 1) / 31) for k in range(32)]
    assert [w[0] for w in windows] == expected


def test_uniform_splits_ct_positions_into_c_clips_of_t_in_order():
    spec = ClipSpec(frames=8, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=2))
    n = 100
    windows = clip_windows(n, spec, train=False, rng=None)
    assert len(windows) == 2
    # linspace(0, 99, 16, dtype=int) hand-evaluated: floor(k * 99 / 15) for k in range(16).
    all_positions = [int(k * 99 / 15) for k in range(16)]
    assert windows[0] == all_positions[:8]
    assert windows[1] == all_positions[8:]


def test_uniform_is_identical_whichever_mode_uses_the_same_count():
    spec = ClipSpec(frames=4, sampling="uniform", clips_per_video=ClipsPerVideo(train=5, eval=5))
    n = 37
    assert clip_windows(n, spec, train=True, rng=None) == clip_windows(
        n, spec, train=False, rng=None
    )


def test_uniform_never_needs_padding_even_for_a_short_video():
    spec = ClipSpec(frames=8, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))
    windows = clip_windows_padded(3, spec, train=False, rng=None)
    assert len(windows) == 1
    positions, padded = windows[0]
    assert padded is False
    assert max(positions) <= 2


def test_uniform_train_never_needs_an_rng():
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=4, eval=4))
    # no ValueError, even though train=True and rng=None: uniform is rng-free.
    windows = clip_windows(10, spec, train=True, rng=None)
    assert len(windows) == 4


# ------------------------------------------------------------------------------------ consecutive


def test_consecutive_eval_windows_match_a_hand_worked_example():
    spec = ClipSpec(
        frames=3, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=4), stride=1
    )
    # n=10, T=3, stride=1: max_start = 7; linspace(0, 7, 4, dtype=int) = [0, 2, 4, 7].
    windows = clip_windows(10, spec, train=False, rng=None)
    assert windows == [[0, 1, 2], [2, 3, 4], [4, 5, 6], [7, 8, 9]]


def test_consecutive_eval_windows_respect_stride():
    spec = ClipSpec(
        frames=3, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=1), stride=2
    )
    # n=10, T=3, stride=2: max_start = max(10 - 6, 0) = 4; linspace(0, 4, 1) = [0].
    windows = clip_windows(10, spec, train=False, rng=None)
    assert windows == [[0, 2, 4]]


def test_consecutive_train_windows_are_drawn_by_the_given_rng_in_order():
    spec = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=3, eval=1), stride=1
    )
    n = 10
    max_start = n - spec.frames * spec.stride  # 8, so no clamping is needed here

    windows = clip_windows(n, spec, train=True, rng=random.Random("clip-seed"))

    # an independently seeded Random, called the same number of times in the same order.
    independent = random.Random("clip-seed")
    expected_starts = [independent.randint(0, max_start) for _ in range(3)]
    assert windows == [[s, s + 1] for s in expected_starts]


def test_consecutive_train_missing_rng_raises_value_error():
    spec = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=1)
    )
    with pytest.raises(ValueError, match="rng"):
        clip_windows(10, spec, train=True, rng=None)


# --------------------------------------------------------------------------------- random-window


def test_random_window_train_matches_consecutive_train_given_the_same_rng():
    spec_consecutive = ClipSpec(
        frames=3, sampling="consecutive", clips_per_video=ClipsPerVideo(train=4, eval=1), stride=1
    )
    spec_random = ClipSpec(
        frames=3,
        sampling="random-window",
        clips_per_video=ClipsPerVideo(train=4, eval=1),
        stride=1,
    )
    n = 15
    a = clip_windows(n, spec_consecutive, train=True, rng=random.Random("shared"))
    b = clip_windows(n, spec_random, train=True, rng=random.Random("shared"))
    assert a == b


def test_random_window_eval_matches_consecutive_eval():
    spec_consecutive = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=5), stride=1
    )
    spec_random = ClipSpec(
        frames=2, sampling="random-window", clips_per_video=ClipsPerVideo(train=1, eval=5), stride=1
    )
    n = 15
    assert clip_windows(n, spec_consecutive, train=False, rng=None) == clip_windows(
        n, spec_random, train=False, rng=None
    )


# -------------------------------------------------------------------------------- short videos


def test_short_video_repeats_its_last_frame_and_is_marked_padded():
    spec = ClipSpec(
        frames=4, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=1), stride=1
    )
    # n=2, T=4: max_start = max(2 - 4, 0) = 0; raw window [0, 1, 2, 3], clamped past n-1=1.
    windows = clip_windows_padded(2, spec, train=False, rng=None)
    positions, padded = windows[0]
    assert positions == [0, 1, 1, 1]
    assert padded is True


def test_long_enough_video_is_not_padded():
    spec = ClipSpec(
        frames=3, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=2), stride=1
    )
    windows = clip_windows_padded(10, spec, train=False, rng=None)
    assert all(padded is False for _, padded in windows)


def test_zero_stored_frames_raises_contract_error():
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))
    with pytest.raises(ContractError, match="zero stored frames"):
        clip_windows(0, spec, train=False, rng=None)


# ------------------------------------------------------------------------------- clip_windows


def test_clip_windows_is_just_the_positions_of_clip_windows_padded():
    spec = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=3), stride=1
    )
    padded_result = clip_windows_padded(6, spec, train=False, rng=None)
    assert clip_windows(6, spec, train=False, rng=None) == [w for w, _ in padded_result]
