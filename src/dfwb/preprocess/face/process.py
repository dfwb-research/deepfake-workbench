"""Processing one video into a lossless output directory of frames plus ``clip.json``.

:func:`process_video` is the per-video unit the runner (not part of this module) schedules across
many videos and workers: it decodes the frames a profile asks for, tracks one face across them,
crops and writes each one as a lossless PNG, and describes the whole clip in ``clip.json``.

Frames are decoded and processed in fixed-size windows (:data:`_WINDOW`), not all at once: a long,
high-resolution video sampled with ``mode: all`` can have far more frames than fit comfortably in
memory as raw pixel arrays, so at most one window's worth is ever held at a time, and only the
small per-frame records (index, bbox, score, landmarks) that ``clip.json`` needs are kept for the
whole clip. One :class:`~dfwb.preprocess.face.track.Tracker` is driven across every window in
turn, so a face is followed exactly as it would be if the whole clip had been decoded at once.

Output never appears half-written: every frame is written straight into a private ``.tmp-<pid>``
sibling of the final directory, and a video whose result is not ``ok``, or whose processing raises
(an interrupt included), has that sibling removed rather than kept or renamed. A successful result
replaces any previous ``out_dir`` through
:func:`~dfwb.preprocess.face.store.recover_video_dir`'s two-step swap (the previous directory is
renamed aside before the new one takes its place, and only deleted once that has succeeded), so a
process killed mid-swap never leaves ``out_dir`` missing -- the next call for the same video
restores it before doing any new work.

Numpy and cv2 are only needed once real frames exist, so, matching the rest of this package, they
are imported inside the functions that use them rather than at module scope.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import os
import shutil
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dfwb.core.errors import ConfigError, InstallationError
from dfwb.core.hashing import canonical_json
from dfwb.core.records.local import InventoryRecord, ProcessedRecord, ProcessingProfile, TrackStats
from dfwb.preprocess.face.backends import FaceBackend
from dfwb.preprocess.face.crop import crop_face, map_landmarks
from dfwb.preprocess.face.decode import DecodeError, VideoSource, open_source, require_library
from dfwb.preprocess.face.identity import cluster_subject
from dfwb.preprocess.face.sampling import sample_indices
from dfwb.preprocess.face.store import portable_reason, recover_video_dir, video_relpath
from dfwb.preprocess.face.track import Tracker, check_strategy
from dfwb.preprocess.face.types import Face

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["check_backend", "check_profile", "process_video"]

_log = logging.getLogger(__name__)

_SUBJECT_SEARCH_FRAMES = 8

# How many sampled frames are decoded, detected and (for identity-cluster) embedded together in
# one batch. Bounds peak memory for a clip of any length to roughly one window's worth of raw
# frames, rather than the whole clip's.
_WINDOW = 32

_EXTRA_HINT = 'pip install "deepfake-workbench[preprocess]"'

# Backend instances that have already been warned once about missing head pose during
# identity-cluster tracking. A WeakSet so a backend that is garbage-collected (a short-lived
# instance in a test, say) does not keep a phantom entry alive, and so that a later, unrelated
# object can never be mistaken for an already-warned backend the way a plain id() cache could be
# after the original object is gone.
_pose_warned: weakref.WeakSet[Any] = weakref.WeakSet()


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError:
        raise InstallationError("processing a video needs OpenCV", hint=_EXTRA_HINT) from None
    return cv2


_FailureStatus = Literal["no_face", "decode_error", "too_short"]


def _failure(record: InventoryRecord, *, status: _FailureStatus, reason: str) -> ProcessedRecord:
    """A failed video's row. ``reason`` often quotes a decoder's message, which names the file by
    where it sits on this machine; only the file's name is kept (see
    :func:`~dfwb.preprocess.face.store.portable_reason`)."""
    return ProcessedRecord(
        key=record.key,
        compression=record.compression,
        status=status,
        n_frames=0,
        frame_indices=[],
        relpath=video_relpath(record.key, record.compression),
        track=None,
        reason=portable_reason(reason),
    )


def _warn_missing_pose_once(backend: FaceBackend) -> None:
    if getattr(backend, "has_pose", False):
        return
    try:
        if backend in _pose_warned:
            return
        _pose_warned.add(backend)
    except TypeError:
        pass  # not weakly referenceable; warn every time rather than never warning at all
    _log.warning(
        "%s: head pose is unavailable, so faces are not weighted by yaw when finding "
        "the clip's subject",
        backend.name,
    )


def _decode_all(source: VideoSource, indices: list[int]) -> list[tuple[int, npt.NDArray[np.uint8]]]:
    """Decode ``indices`` and hold every frame in memory at once.

    Only for the identity-cluster subject search, which is fixed at
    :data:`_SUBJECT_SEARCH_FRAMES` frames -- small enough that streaming it in windows would add
    complexity for no benefit. The clip's own sampled frames go through :func:`_windows` instead.
    """
    return list(source.read(indices))


def _windows[T](iterator: Iterator[T], size: int) -> Iterator[list[T]]:
    """``iterator``'s items, grouped into lists of at most ``size``, the last one possibly
    shorter. Never buffers more than one group at a time."""
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _batch_detect(
    backend: FaceBackend, frames: list[tuple[int, npt.NDArray[np.uint8]]]
) -> list[tuple[int, list[Face]]]:
    if not frames:
        return []
    import numpy as np

    stacked = np.stack([frame for _, frame in frames])
    detected = backend.detect(stacked)
    return [(index, faces) for (index, _), faces in zip(frames, detected, strict=True)]


def _embed_all(
    embed: Any,
    frames: list[tuple[int, npt.NDArray[np.uint8]]],
    per_frame: list[tuple[int, list[Face]]],
) -> list[tuple[int, list[Face]]]:
    by_index = dict(frames)
    embedded = []
    for index, faces in per_frame:
        frame = by_index[index]
        embedded.append(
            (index, [dataclasses.replace(face, embedding=embed(frame, face)) for face in faces])
        )
    return embedded


def check_backend(profile: ProcessingProfile, backend: FaceBackend) -> None:
    """Refuse a ``profile`` that needs something ``backend`` does not provide.

    :func:`process_video` checks this itself before decoding anything; a caller about to process
    many videos checks it once, first, so that a mismatch stops everything before any work
    rather than failing every video in turn.

    Raises:
        ConfigError: the profile's track strategy is ``"identity-cluster"`` but ``backend`` has no
            ``embed`` method.
    """
    if profile.track.strategy == "identity-cluster" and getattr(backend, "embed", None) is None:
        raise ConfigError(
            f"track strategy 'identity-cluster' needs face embeddings, but backend "
            f"{backend.name!r} does not provide embed()",
            hint="use a backend that implements embed(), or choose track strategy "
            "'largest-then-iou'",
        )


def check_profile(profile: ProcessingProfile) -> None:
    """Refuse a ``profile`` this release or this installation cannot run, whatever the backend.

    Every video would fail the same way, so a caller about to process many videos checks this
    once, first (with :func:`check_backend`), rather than recording the same failure for each.

    Raises:
        ConfigError: the profile's track strategy is not one this release implements.
        InstallationError: the profile's decode library is not installed, or OpenCV (which
            writes every frame, whichever library decodes it) is not.
    """
    check_strategy(profile.track.strategy)
    require_library(profile.decode.library)
    _require_cv2()


def process_video(
    source_path: Path,
    record: InventoryRecord,
    profile: ProcessingProfile,
    backend: FaceBackend,
    out_dir: Path,
) -> ProcessedRecord:
    """Decode, track, crop and write one video's chosen frames into ``out_dir``.

    Writes every kept frame straight into ``out_dir``'s private ``.tmp-<pid>`` sibling as it is
    produced, at most :data:`_WINDOW` decoded frames held in memory at a time, and only swaps that
    sibling into ``out_dir``'s place once the whole clip has been processed and the result is
    ``ok``; for any other result, and when anything raises along the way, the sibling is removed
    and ``out_dir`` is left untouched. Any ``.tmp-*``/``.old-*`` sibling a previous attempt at
    this same video left behind (one killed outright, with no chance to clean up) is resolved
    first (see :func:`~dfwb.preprocess.face.store.recover_video_dir`).

    Args:
        source_path: The video file, or a directory of already-extracted frame images.
        record: The inventory row this video comes from (its ``key`` and ``compression`` name the
            output directory).
        profile: The processing recipe: sampling, tracking, cropping, decoding and extras.
        backend: A built face-detection backend.
        out_dir: Where the video's frames and ``clip.json`` are written on success.

    Returns:
        A :class:`ProcessedRecord` describing the outcome: ``status`` is ``"too_short"`` when the
        source reports no frames at all, ``"decode_error"`` when it cannot be opened or read (at
        open, or partway through decoding), ``"no_face"`` when frames were decoded but none
        produced a usable, croppable face, and ``"ok"`` otherwise (with ``reason`` set to
        ``"short: N of M"`` when fewer frames were written than the profile asked for).

    Raises:
        ConfigError: the profile's track strategy is ``"identity-cluster"`` but ``backend`` has no
            ``embed`` method. Raised before any decoding happens.
    """
    recover_video_dir(out_dir)

    check_backend(profile, backend)
    identity_cluster = profile.track.strategy == "identity-cluster"
    embed = getattr(backend, "embed", None)

    try:
        source = open_source(source_path, library=profile.decode.library)
    except DecodeError as exc:
        return _failure(record, status="decode_error", reason=str(exc))

    try:
        total_frames = source.total_frames
        if total_frames == 0:
            return _failure(record, status="too_short", reason="the source reports 0 frames")

        subject = None
        if identity_cluster:
            _warn_missing_pose_once(backend)
            subject_indices = sample_indices("uniform", total_frames, frames=_SUBJECT_SEARCH_FRAMES)
            subject_frames = _decode_all(source, subject_indices)
            subject_faces = _batch_detect(backend, subject_frames)
            subject_faces = _embed_all(embed, subject_frames, subject_faces)
            subject = cluster_subject([face for _, faces in subject_faces for face in faces])
            # ``read`` decodes forward once and cannot be replayed, so the main sampling pass
            # needs a fresh source.
            source = open_source(source_path, library=profile.decode.library)

        requested = sample_indices(
            profile.sampling.mode,
            total_frames,
            frames=profile.sampling.frames,
            stride=profile.sampling.stride,
        )
    except DecodeError as exc:
        return _failure(record, status="decode_error", reason=str(exc))

    min_score = getattr(profile.backend, "min_score", 0.0)
    tracker = Tracker(profile.track, min_score=min_score, subject=subject)

    tmp_dir = out_dir.with_name(out_dir.name + f".tmp-{os.getpid()}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    frames_meta: list[dict[str, Any]] = []
    failed_frames: list[dict[str, Any]] = []
    cv2 = None
    # Whatever ends this block -- a result other than ok, or an exception of any kind, an
    # interrupt included -- the private directory goes with it; once the swap below has moved it
    # into out_dir's place there is nothing left to remove.
    try:
        for window in _windows(source.read(requested), _WINDOW):
            per_frame = _batch_detect(backend, window)
            if identity_cluster:
                per_frame = _embed_all(embed, window, per_frame)
            frame_by_index = dict(window)
            for index, faces in per_frame:
                face, reason = tracker.step(index, faces)
                if face is None:
                    failed_frames.append({"index": index, "reason": reason})
                    continue
                crop_result = crop_face(
                    frame_by_index[index],
                    face.bbox,
                    scale=profile.crop.scale,
                    size=profile.crop.size,
                )
                if crop_result is None:
                    failed_frames.append({"index": index, "reason": "crop-empty"})
                    continue
                landmarks = None
                if profile.extras.landmarks and face.landmarks5 is not None:
                    landmarks = [
                        list(point) for point in map_landmarks(face.landmarks5, crop_result)
                    ]
                cv2 = cv2 or _require_cv2()
                bgr = cv2.cvtColor(crop_result.image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(tmp_dir / f"frame_{index:06d}.png"), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6]
                )
                frames_meta.append(
                    {
                        "index": index,
                        "bbox": list(crop_result.box),
                        "score": float(face.score),
                        "landmarks5": landmarks,
                    }
                )

        if not frames_meta:
            return _failure(record, status="no_face", reason="no frame produced a usable face")

        decoder = "frames" if Path(source_path).is_dir() else profile.decode.library
        clip = {
            "key": record.key,
            "compression": record.compression,
            "source": {
                "total_frames": total_frames,
                "fps": source.fps,
                "width": source.width,
                "height": source.height,
                "decoder": decoder,
            },
            "frames": frames_meta,
            "failed_frames": failed_frames,
            "track": {
                "strategy": profile.track.strategy,
                "identity_switch": tracker.identity_switch,
            },
        }
        (tmp_dir / "clip.json").write_text(canonical_json(clip) + "\n", encoding="utf-8")

        old_dir = out_dir.with_name(f"{out_dir.name}.old-{os.getpid()}")
        if out_dir.exists():
            out_dir.rename(old_dir)
        tmp_dir.rename(out_dir)
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)
    except DecodeError as exc:
        return _failure(record, status="decode_error", reason=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    n_written = len(frames_meta)
    total_requested = (
        profile.sampling.frames if profile.sampling.frames is not None else len(requested)
    )
    reason = None if n_written >= total_requested else f"short: {n_written} of {total_requested}"
    mean_confidence = sum(meta["score"] for meta in frames_meta) / n_written

    return ProcessedRecord(
        key=record.key,
        compression=record.compression,
        status="ok",
        n_frames=n_written,
        frame_indices=[meta["index"] for meta in frames_meta],
        relpath=video_relpath(record.key, record.compression),
        track=TrackStats(mean_confidence=mean_confidence, identity_switch=tracker.identity_switch),
        reason=reason,
    )
