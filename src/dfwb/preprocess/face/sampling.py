"""Which frame indices to decode from a video, for each sampling mode.

These are the same formulas the private face pipeline used per video, kept unchanged so that a
profile built to reproduce an old run samples exactly the same frames. None of this touches a
decoder: it only turns ``(mode, total_frames)`` into the list of indices a decoder should look for.

Numpy is only needed for the ``uniform`` formula's ``linspace``, so it is imported inside that
branch rather than at module scope, matching the rest of the framework: nothing here pulls in a
heavy dependency just by being imported.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["sample_indices"]

SamplingMode = Literal["uniform", "stride", "first-consecutive", "all"]


def sample_indices(
    mode: SamplingMode,
    total_frames: int,
    *,
    frames: int | None = None,
    stride: int | None = None,
) -> list[int]:
    """The ascending list of frame indices ``mode`` selects out of ``total_frames``.

    A source shorter than what was asked for is never padded: ``uniform`` and ``first-consecutive``
    both just return every index the source actually has.

    Args:
        mode: ``"all"`` takes every frame; ``"uniform"`` spreads ``frames`` picks evenly across the
            source (numpy ``linspace`` rounded to whole indices, then de-duplicated); ``"stride"``
            takes every ``stride``-th frame starting at 0; ``"first-consecutive"`` takes the first
            ``frames`` indices.
        total_frames: How many frames the source reports.
        frames: How many frames to pick; required for ``"uniform"`` and ``"first-consecutive"``.
        stride: The gap between picks; required for ``"stride"``.

    Raises:
        ValueError: ``frames`` or ``stride`` is missing for a mode that needs it.
    """
    if mode == "all":
        return list(range(total_frames))
    if mode == "uniform":
        if frames is None:
            raise ValueError("frames is required when mode is 'uniform'")
        import numpy as np

        picked = np.linspace(0, total_frames - 1, frames, endpoint=True, dtype=int)
        return [int(index) for index in np.unique(picked)]
    if mode == "first-consecutive":
        if frames is None:
            raise ValueError("frames is required when mode is 'first-consecutive'")
        return list(range(min(frames, total_frames)))
    if stride is None:
        raise ValueError("stride is required when mode is 'stride'")
    return list(range(0, total_frames, stride))
