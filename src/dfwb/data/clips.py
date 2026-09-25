"""``ClipSpec`` and the window arithmetic that turns a video's stored frame count into clip
positions.

A "position" is an index into a video's *stored* frame list (0 .. n-1), not the frame's own number
in the source video -- :class:`~dfwb.data.dataset.ClipDataset` looks the real frame number up
afterwards, from ``VideoItem.frame_indices[position]``, before it ever opens a file. Keeping this
module free of anything torch- or image-decoding related means the window arithmetic, and its
tests, stay independent of whatever decodes the actual pixels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

from dfwb.core.config.schema import ClipSection
from dfwb.core.errors import ContractError

__all__ = ["ClipSpec", "ClipsPerVideo", "Sampling", "clip_windows", "clip_windows_padded"]

Sampling = Literal["uniform", "consecutive", "random-window"]


@dataclass(frozen=True)
class ClipsPerVideo:
    """How many clips one video contributes, in each mode."""

    train: int
    eval: int

    def for_mode(self, *, train: bool) -> int:
        return self.train if train else self.eval


@dataclass(frozen=True)
class ClipSpec:
    """How a video's stored frames turn into clips: how many frames each clip has, how their
    positions are chosen, and how many clips a video contributes in each mode."""

    frames: int
    sampling: Sampling
    clips_per_video: ClipsPerVideo
    stride: int = 1

    @classmethod
    def from_config(cls, section: ClipSection, *, stride: int | None = None) -> ClipSpec:
        """Build from a config's ``clip:`` section. ``stride`` comes from ``section.stride`` by
        default; pass it explicitly only to override the section's own value."""
        return cls(
            frames=section.frames,
            sampling=section.sampling,
            clips_per_video=ClipsPerVideo(
                train=section.clips_per_video.train,
                eval=section.clips_per_video.eval,
            ),
            stride=section.stride if stride is None else stride,
        )

    def clips_per_mode(self, *, train: bool) -> int:
        return self.clips_per_video.for_mode(train=train)


def clip_windows_padded(
    n: int, spec: ClipSpec, *, train: bool, rng: random.Random | None
) -> list[tuple[list[int], bool]]:
    """One ``(positions, padded)`` pair per clip: ``positions`` index a video's stored frame list
    (0 .. n-1); ``padded`` says whether any of them had to be clamped there because the video is
    shorter than the window it needed.

    ``uniform`` spreads ``clips_per_mode(train=train) * spec.frames`` positions evenly across the
    whole video with ``numpy.linspace``, then slices them into clips in order. That spread is
    always bounded by ``n - 1``, so it never needs to pad: a short video's frames simply repeat as
    the spacing rounds to whole indices. ``consecutive`` and ``random-window`` share one formula:
    each clip is a run of ``spec.frames`` positions, ``spec.stride`` apart, starting at an evenly
    spaced point (eval) or a point ``rng.randint`` draws (train, seeded by the caller) between 0
    and the last position a full window can start at without running past the video. When even
    that latest start cannot fit a whole window -- the video has fewer stored frames than the
    window needs -- every start collapses to 0 and the run's tail is clamped to the last stored
    frame.

    Args:
        n: How many frames the video has stored.
        train: ``True`` draws ``clips_per_video.train`` windows (and, for ``consecutive`` /
            ``random-window``, needs ``rng``); ``False`` draws ``clips_per_video.eval`` windows at
            fixed, evenly spaced positions.
        rng: The source of randomness for ``train`` windows of ``consecutive`` /
            ``random-window``; unused (and may be ``None``) for ``uniform`` or for eval windows.

    Raises:
        ContractError: ``n`` is 0 -- a video with no stored frames should already have been
            excluded before it reaches here.
        ValueError: ``train`` is ``True``, the sampling needs randomness to place its windows, and
            ``rng`` is ``None``.
    """
    if n <= 0:
        raise ContractError(
            "clip_windows: a video has zero stored frames",
            hint="VideoIndex excludes frameless videos with reason 'no-frames' before this point",
        )
    frames = spec.frames
    stride = spec.stride

    if spec.sampling == "uniform":
        count = spec.clips_per_mode(train=train)
        positions = [int(p) for p in np.linspace(0, n - 1, count * frames, dtype=int)]
        return [(positions[k * frames : (k + 1) * frames], False) for k in range(count)]

    max_start = max(n - frames * stride, 0)
    if train:
        if rng is None:
            raise ValueError("clip_windows: train windows of this sampling need an rng")
        count = spec.clips_per_mode(train=True)
        starts = [rng.randint(0, max_start) for _ in range(count)]
    else:
        count = spec.clips_per_mode(train=False)
        starts = [int(s) for s in np.linspace(0, max_start, count, dtype=int)]

    windows: list[tuple[list[int], bool]] = []
    for start in starts:
        raw = [start + step * stride for step in range(frames)]
        padded = any(position >= n for position in raw)
        windows.append(([min(position, n - 1) for position in raw], padded))
    return windows


def clip_windows(
    n: int, spec: ClipSpec, *, train: bool, rng: random.Random | None
) -> list[list[int]]:
    """Just the positions from :func:`clip_windows_padded`, for callers that do not need to know
    which clips were padded."""
    return [positions for positions, _ in clip_windows_padded(n, spec, train=train, rng=rng)]
