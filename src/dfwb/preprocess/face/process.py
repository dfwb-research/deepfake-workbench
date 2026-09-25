"""Processing one video into a lossless output directory of frames plus ``clip.json``.

:func:`process_video` is the per-video unit the runner (not part of this module) schedules across
many videos and workers: it decodes the frames a profile asks for, tracks one face across them,
crops and writes each one as a lossless PNG, and describes the whole clip in ``clip.json``. Output
never appears half-written: everything is written into a private ``.tmp-<pid>`` sibling of the
final directory and only renamed into place once every frame and ``clip.json`` are on disk, so a
process killed partway through leaves nothing for a reader to see, and the very next call for the
same video clears away whatever that crash left behind before it starts its own attempt.

Numpy and cv2 are only needed once real frames exist, so, matching the rest of this package, they
are imported inside the functions that use them rather than at module scope.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dfwb.core.errors import ConfigError, InstallationError
from dfwb.core.hashing import canonical_json
from dfwb.core.records.local import InventoryRecord, ProcessedRecord, ProcessingProfile, TrackStats
from dfwb.preprocess.face.backends import FaceBackend
from dfwb.preprocess.face.crop import crop_face, map_landmarks
from dfwb.preprocess.face.decode import DecodeError, VideoSource, open_source
from dfwb.preprocess.face.identity import cluster_subject
from dfwb.preprocess.face.sampling import sample_indices
from dfwb.preprocess.face.track import select_track
from dfwb.preprocess.face.types import Face

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["process_video"]

_log = logging.getLogger(__name__)

_SUBJECT_SEARCH_FRAMES = 8
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


def _relpath(record: InventoryRecord) -> str:
    """The video's directory relative to the store root: ``<key>/<compression or "_">``."""
    return f"{record.key}/{record.compression or '_'}"


_FailureStatus = Literal["no_face", "decode_error", "too_short"]


def _failure(record: InventoryRecord, *, status: _FailureStatus, reason: str) -> ProcessedRecord:
    return ProcessedRecord(
        key=record.key,
        compression=record.compression,
        status=status,
        n_frames=0,
        frame_indices=[],
        relpath=_relpath(record),
        track=None,
        reason=reason,
    )


def _clear_stale_tmp(out_dir: Path) -> None:
    """Remove any ``<out_dir>.tmp-*`` sibling a previous, crashed attempt left behind."""
    parent = out_dir.parent
    if not parent.is_dir():
        return
    for candidate in parent.glob(f"{out_dir.name}.tmp-*"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)


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


def _decode(source: VideoSource, indices: list[int]) -> list[tuple[int, npt.NDArray[np.uint8]]]:
    return list(source.read(indices))


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


def process_video(
    source_path: Path,
    record: InventoryRecord,
    profile: ProcessingProfile,
    backend: FaceBackend,
    out_dir: Path,
) -> ProcessedRecord:
    """Decode, track, crop and write one video's chosen frames into ``out_dir``.

    Writes into ``out_dir.tmp-<pid>`` and only renames it to ``out_dir`` once every frame and
    ``clip.json`` are written; any previous ``out_dir`` is replaced, and any ``.tmp-*`` sibling
    left by a crashed earlier attempt at this same video is cleared first. Nothing is written to
    ``out_dir`` at all when the result is not ``ok``.

    Args:
        source_path: The video file, or a directory of already-extracted frame images.
        record: The inventory row this video comes from (its ``key`` and ``compression`` name the
            output directory).
        profile: The processing recipe: sampling, tracking, cropping, decoding and extras.
        backend: A built face-detection backend.
        out_dir: Where the video's frames and ``clip.json`` are written on success.

    Returns:
        A :class:`ProcessedRecord` describing the outcome: ``status`` is ``"too_short"`` when the
        source reports no frames at all, ``"decode_error"`` when it cannot be opened or read,
        ``"no_face"`` when frames were decoded but none produced a usable, croppable face, and
        ``"ok"`` otherwise (with ``reason`` set to ``"short: N of M"`` when fewer frames were
        written than the profile asked for).

    Raises:
        ConfigError: the profile's track strategy is ``"identity-cluster"`` but ``backend`` has no
            ``embed`` method. Raised before any decoding happens.
    """
    _clear_stale_tmp(out_dir)

    identity_cluster = profile.track.strategy == "identity-cluster"
    embed = getattr(backend, "embed", None)
    if identity_cluster and embed is None:
        raise ConfigError(
            f"track strategy 'identity-cluster' needs face embeddings, but backend "
            f"{backend.name!r} does not provide embed()",
            hint="use a backend that implements embed(), or choose track strategy "
            "'largest-then-iou'",
        )

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
            subject_frames = _decode(source, subject_indices)
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
        decoded = _decode(source, requested)
        per_frame = _batch_detect(backend, decoded)
        if identity_cluster:
            per_frame = _embed_all(embed, decoded, per_frame)
    except DecodeError as exc:
        return _failure(record, status="decode_error", reason=str(exc))

    min_score = getattr(profile.backend, "min_score", 0.0)
    track_result = select_track(per_frame, profile.track, min_score=min_score, subject=subject)

    frame_by_index = dict(decoded)
    failed_frames: list[dict[str, Any]] = []
    crops: list[tuple[int, npt.NDArray[np.uint8], tuple[int, int, int, int], float, Any]] = []
    for index, face, reason in track_result.frames:
        if face is None:
            failed_frames.append({"index": index, "reason": reason})
            continue
        crop_result = crop_face(
            frame_by_index[index], face.bbox, scale=profile.crop.scale, size=profile.crop.size
        )
        if crop_result is None:
            failed_frames.append({"index": index, "reason": "crop-empty"})
            continue
        landmarks = None
        if profile.extras.landmarks and face.landmarks5 is not None:
            landmarks = [list(point) for point in map_landmarks(face.landmarks5, crop_result)]
        crops.append((index, crop_result.image, crop_result.box, float(face.score), landmarks))

    if not crops:
        return _failure(record, status="no_face", reason="no frame produced a usable face")

    decoder = "frames" if Path(source_path).is_dir() else profile.decode.library
    frames_meta = [
        {"index": index, "bbox": list(box), "score": score, "landmarks5": landmarks}
        for index, _, box, score, landmarks in crops
    ]
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
            "identity_switch": track_result.identity_switch,
        },
    }

    cv2 = _require_cv2()
    tmp_dir = out_dir.with_name(out_dir.name + f".tmp-{os.getpid()}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for index, image_rgb, _, _, _ in crops:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(tmp_dir / f"frame_{index:06d}.png"), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    (tmp_dir / "clip.json").write_text(canonical_json(clip) + "\n", encoding="utf-8")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)

    n_written = len(crops)
    total_requested = (
        profile.sampling.frames if profile.sampling.frames is not None else len(requested)
    )
    reason = None if n_written >= total_requested else f"short: {n_written} of {total_requested}"
    mean_confidence = sum(score for _, _, _, score, _ in crops) / n_written

    return ProcessedRecord(
        key=record.key,
        compression=record.compression,
        status="ok",
        n_frames=n_written,
        frame_indices=[index for index, _, _, _, _ in crops],
        relpath=_relpath(record),
        track=TrackStats(
            mean_confidence=mean_confidence, identity_switch=track_result.identity_switch
        ),
        reason=reason,
    )
