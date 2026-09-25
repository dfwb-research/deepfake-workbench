"""Choosing one face per frame and following it through a clip.

These are the rules the earlier face pipeline used, carried over unchanged so that a store built
with this framework picks the same face on every frame as a store built before it:

1. A face whose detection score is below ``min_score`` counts as not there at all.
2. With a subject embedding (the ``identity-cluster`` strategy), the face most similar to the
   subject is chosen, and the frame fails when even that face is not similar enough. If no face in
   the frame carries an embedding, rules 3 and 4 decide instead.
3. Otherwise, once a face has been chosen, the face overlapping the last chosen face's box most
   (intersection over union) is chosen, unless even that overlap is below the profile's threshold.
4. Otherwise (the first chosen frame, or nothing overlapping enough) the largest face is chosen.

A frame with no usable face is dropped, never filled in from its neighbours, and tracking carries
on from the last face that was chosen. Overlap is always measured on the detector's own box; the
optional smoothing only changes the box that is reported for cropping.

When rule 4 has to step in after a face has already been chosen, the track may have jumped to a
different person, so the clip is flagged as a possible identity switch.

Numpy is only needed to compare embeddings with the subject, so it is imported there rather than
at module scope, matching the rest of the framework.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from dfwb.core.errors import ConfigError, did_you_mean

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dfwb.core.records.local import TrackSpec
    from dfwb.preprocess.face.types import Face

__all__ = ["TrackResult", "select_track"]

Box = tuple[float, float, float, float]

_STRATEGIES = ("largest-then-iou", "identity-cluster")

# The strategy that writes one clip per face in the frame is part of the profile format, but is
# not implemented in this release.
_RESERVED_STRATEGIES = ("all-faces",)

# The lowest similarity (the dot product of the unit-length face and subject embeddings) at which
# a face still counts as the subject.
_SUBJECT_MIN_SIMILARITY = 0.3

# Smoothing starts afresh when more than this many frame indices separate two chosen faces: the
# earlier box says too little about where the face is now.
_SMOOTHING_MAX_GAP = 2

_STRATEGY_HINT = "use track.strategy 'largest-then-iou' or 'identity-cluster'"


@dataclass(frozen=True)
class TrackResult:
    """The face chosen on each frame of a clip, and whether the track may have changed person.

    Attributes:
        frames: One ``(frame index, face, failure reason)`` per input frame, in input order (see
            :func:`select_track`).
        identity_switch: True when, on some frame after the first chosen one, no face overlapped
            the last chosen face enough and the largest face was taken instead. That fallback
            counts even when it lands on the only face in the frame, since a face that jumped
            that far cannot be told apart from a different one. Frames chosen by similarity to
            the subject never set it.
    """

    frames: list[tuple[int, Face | None, str | None]]
    identity_switch: bool


def select_track(
    per_frame: Sequence[tuple[int, list[Face]]],
    spec: TrackSpec,
    *,
    min_score: float,
    subject: npt.NDArray[Any] | None = None,
) -> TrackResult:
    """Choose at most one face on each frame of a clip, following the same face across frames.

    Args:
        per_frame: ``(frame index, detected faces)`` for each decoded frame, in increasing index
            order. An empty face list means nothing was detected on that frame.
        spec: The profile's tracking settings. ``spec.iou`` is the overlap below which tracking
            gives up on the previous face and takes the largest one; ``spec.ema``, when set, is
            the weight of the new box when smoothing the reported box (see below).
        min_score: Faces scoring below this are treated as not detected.
        subject: The clip's subject embedding, as found by
            :func:`~dfwb.preprocess.face.identity.cluster_subject`; only the ``identity-cluster``
            strategy takes one. Without it, ``identity-cluster`` tracks exactly like
            ``largest-then-iou``, as the earlier pipeline did when it could not find a subject.

    Returns:
        A :class:`TrackResult`. Its ``frames`` hold one ``(frame index, face, failure reason)``
        per input frame, in input order. A frame that failed has face ``None`` and one of these
        reasons: ``"no-face"`` (nothing detected), ``"low-score"`` (every face scored below
        ``min_score``) or ``"no-subject-match"`` (no face is similar enough to ``subject``). A
        frame that succeeded has reason ``None``.

        With ``spec.ema`` unset, the face returned is the detector's own. With it set, the
        returned face is a copy whose box is ``ema * raw + (1 - ema) * previous smoothed box``,
        except where the smoothing restarts: on the first chosen frame, and on any chosen frame
        whose index is more than 2 past the previous chosen frame's. There the detector's own
        face is returned, and its box becomes the new starting point.

    Raises:
        ConfigError: ``spec.strategy`` is not one this release implements.
        ValueError: ``subject`` was given to a strategy other than ``identity-cluster``, or the
            frame indices do not increase.
    """
    _check_strategy(spec.strategy)
    if subject is not None and spec.strategy != "identity-cluster":
        raise ValueError(
            f"a subject embedding is only used by the 'identity-cluster' strategy, "
            f"not {spec.strategy!r}"
        )
    subject_vector = None if subject is None else _as_float32(subject)
    ema = spec.ema
    # The previous box's weight, worked out in decimal: 1 - 0.7 in binary floating point is
    # 0.30000000000000004, and the earlier pipeline multiplied by a literal 0.3.
    keep = None if ema is None else float(Decimal(1) - Decimal(repr(ema)))

    results: list[tuple[int, Face | None, str | None]] = []
    previous_raw: Box | None = None
    previous_smoothed: Box | None = None
    previous_good_index: int | None = None
    previous_index: int | None = None
    identity_switch = False

    for index, faces in per_frame:
        if previous_index is not None and index <= previous_index:
            raise ValueError(
                f"frame indices must be strictly increasing, got {index} after {previous_index}"
            )
        previous_index = index

        if not faces:
            results.append((index, None, "no-face"))
            continue
        usable = [face for face in faces if face.score >= min_score]
        if not usable:
            results.append((index, None, "low-score"))
            continue
        chosen, fell_back = _choose(usable, previous_raw, subject_vector, spec.iou)
        # the first chosen face is always the largest; that starts the track, it does not switch it
        if fell_back and previous_raw is not None:
            identity_switch = True
        if chosen is None:
            results.append((index, None, "no-subject-match"))
            continue

        raw = chosen.bbox
        reported = chosen
        if ema is not None and keep is not None:
            if (
                previous_smoothed is None
                or previous_good_index is None
                or index - previous_good_index > _SMOOTHING_MAX_GAP
            ):
                smoothed = raw
            else:
                smoothed = (
                    ema * raw[0] + keep * previous_smoothed[0],
                    ema * raw[1] + keep * previous_smoothed[1],
                    ema * raw[2] + keep * previous_smoothed[2],
                    ema * raw[3] + keep * previous_smoothed[3],
                )
                reported = dataclasses.replace(chosen, bbox=smoothed)
            previous_smoothed = smoothed
        previous_good_index = index
        previous_raw = raw
        results.append((index, reported, None))

    return TrackResult(frames=results, identity_switch=identity_switch)


def _check_strategy(strategy: str) -> None:
    if strategy in _STRATEGIES:
        return
    if strategy in _RESERVED_STRATEGIES:
        raise ConfigError(
            f"track strategy {strategy!r} is not available in this release "
            "(it is reserved for a later one)",
            hint=_STRATEGY_HINT,
        )
    known = (*_STRATEGIES, *_RESERVED_STRATEGIES)
    raise ConfigError(
        f"track strategy {strategy!r} is not available: it is not a known strategy"
        f"{did_you_mean(strategy, known)}",
        hint=_STRATEGY_HINT,
    )


def _as_float32(vector: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    import numpy as np

    return np.asarray(vector, dtype=np.float32)


def _choose(
    faces: list[Face],
    previous_bbox: Box | None,
    subject: npt.NDArray[np.float32] | None,
    iou_threshold: float,
) -> tuple[Face | None, bool]:
    """The face to take on one frame, and whether it was taken as the largest face.

    The face is ``None`` only when the subject is not among ``faces``.
    """
    if subject is not None:
        import numpy as np

        best: Face | None = None
        best_similarity = -float("inf")
        any_embedding = False
        for face in faces:
            if face.embedding is None:
                continue
            any_embedding = True
            similarity = float(np.dot(np.asarray(face.embedding, dtype=np.float32), subject))
            if similarity > best_similarity:
                best_similarity = similarity
                best = face
        if any_embedding:
            return (best if best_similarity >= _SUBJECT_MIN_SIMILARITY else None), False
        # no face on this frame carries an embedding: fall through to overlap and size

    if previous_bbox is not None:
        best_overlap_face = faces[0]
        best_overlap = -1.0
        for face in faces:
            overlap = _iou(face.bbox, previous_bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_face = face
        if best_overlap >= iou_threshold:
            return best_overlap_face, False

    # max() keeps the first of several equally large faces
    return max(faces, key=_area), True


def _area(face: Face) -> float:
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def _iou(box_a: Box, box_b: Box) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    intersection = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = float(area_a + area_b - intersection)
    if union == 0:
        # two zero-area boxes: there is nothing to overlap
        return 0.0
    return intersection / union
