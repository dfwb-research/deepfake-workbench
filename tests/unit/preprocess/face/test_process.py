"""Processing one video into a lossless output directory of frames plus ``clip.json``.

The synthetic clips here are lossless FFV1 AVI files written with OpenCV in ``tmp_path``: a small
plain background with a filled circle sliding across it, frame by frame. Content never matters to
these tests (the ``center`` backend ignores it, and the fakes below invent their own faces); only
the frame count and pixel values do, since the point is to pin down how one video is turned into a
store directory, not detection quality.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import weakref
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pytest

from dfwb.core.errors import ConfigError
from dfwb.core.records import assert_no_absolute_paths
from dfwb.core.records.local import (
    BackendSpec,
    BuilderRef,
    CropSpec,
    DecodeSpec,
    ExtrasSpec,
    InventoryRecord,
    ProcessingProfile,
    SamplingSpec,
    TrackSpec,
)
from dfwb.preprocess.face.backends.center import CenterBackend
from dfwb.preprocess.face.crop import crop_face
from dfwb.preprocess.face.decode import open_source
from dfwb.preprocess.face.process import process_video
from dfwb.preprocess.face.types import Face

# ------------------------------------------------------------------------------------- fixtures


def _write_clip(path: Path, n_frames: int, *, width: int = 48, height: int = 32) -> None:
    """A lossless FFV1 clip: a filled circle sliding left to right on a plain background."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (width, height))
    assert writer.isOpened(), path
    for index in range(n_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        center_x = int((index + 0.5) * width / n_frames)
        cv2.circle(frame, (center_x, height // 2), min(width, height) // 4, (60, 180, 220), -1)
        writer.write(frame)
    writer.release()


def _truncate(path: Path, keep_fraction: float) -> None:
    """Cut the tail off an already-encoded file, so its header still claims the old frame count."""
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * keep_fraction)])


def _decodable_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def _profile(
    *,
    profile_id: str = "toy-test",
    backend: BackendSpec | None = None,
    track: TrackSpec | None = None,
    crop: CropSpec | None = None,
    sampling: SamplingSpec | None = None,
    decode: DecodeSpec | None = None,
    extras: ExtrasSpec | None = None,
) -> ProcessingProfile:
    return ProcessingProfile(
        id=profile_id,
        backend=backend or BackendSpec(name="center"),
        track=track or TrackSpec(iou=0.5, strategy="largest-then-iou"),
        crop=crop or CropSpec(scale=1.0, size=16, square=True, align="none"),
        sampling=sampling or SamplingSpec(mode="uniform", frames=4),
        decode=decode or DecodeSpec(library="opencv", color="rgb"),
        extras=extras or ExtrasSpec(landmarks=False, mesh=False, masks=False),
    )


def _inventory_record(
    key: str = "toy/vid001", *, compression: str | None = None
) -> InventoryRecord:
    return InventoryRecord(
        key=key,
        compression=compression,
        label_key="real",
        method="pristine",
        relpath=f"clips/{key.rpartition('/')[2]}.avi",
        builder=BuilderRef(id="test", version="1"),
    )


class _EmptyFaceBackend:
    """Detects nothing, ever: every frame fails with ``no-face``."""

    name = "empty-fake"
    version = "1"
    license = "MIT"
    has_pose = False
    meta: ClassVar[dict[str, Any]] = {}

    def detect(self, frames: np.ndarray) -> list[list[Face]]:
        return [[] for _ in frames]


class _WholeFrameBackend:
    """One face per frame, the whole frame, except a chosen position gets a degenerate box."""

    name = "whole-frame-fake"
    version = "1"
    license = "MIT"
    has_pose = False
    meta: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, degenerate_at: int | None = None) -> None:
        self._degenerate_at = degenerate_at

    def detect(self, frames: np.ndarray) -> list[list[Face]]:
        out: list[list[Face]] = []
        for position, frame in enumerate(frames):
            height, width = frame.shape[:2]
            if position == self._degenerate_at:
                out.append([Face(bbox=(5.0, 5.0, 5.0, 5.0), score=1.0)])
            else:
                out.append([Face(bbox=(0.0, 0.0, float(width), float(height)), score=1.0)])
        return out


_SUBJECT_EMBEDDING = np.array([1.0, 0.0], dtype=np.float32)
_DISTRACTOR_EMBEDDING = np.array([0.0, 1.0], dtype=np.float32)


class _TwoFaceBackend:
    """Two faces per frame: a large, high-scoring subject and a small, low-scoring distractor.

    ``embed`` tags each by score, so identity clustering should settle on the subject's embedding
    and every frame should then pick the subject face by similarity, never the distractor.
    """

    name = "two-face-fake"
    version = "1"
    license = "MIT"
    meta: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, has_pose: bool = False) -> None:
        self.has_pose = has_pose

    def detect(self, frames: np.ndarray) -> list[list[Face]]:
        out: list[list[Face]] = []
        for frame in frames:
            height, width = frame.shape[:2]
            subject = Face(bbox=(0.0, 0.0, float(width), float(height)), score=1.0)
            distractor = Face(bbox=(0.0, 0.0, width / 4, height / 4), score=0.4)
            out.append([subject, distractor])
        return out

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        return (_SUBJECT_EMBEDDING if face.score >= 0.9 else _DISTRACTOR_EMBEDDING).copy()


class _LandmarkBackend:
    """One face per frame, always with five landmarks, so the landmarks flag can be exercised."""

    name = "landmark-fake"
    version = "1"
    license = "MIT"
    has_pose = False
    meta: ClassVar[dict[str, Any]] = {}

    def detect(self, frames: np.ndarray) -> list[list[Face]]:
        out: list[list[Face]] = []
        for frame in frames:
            height, width = frame.shape[:2]
            points = (
                (width * 0.3, height * 0.3),
                (width * 0.7, height * 0.3),
                (width * 0.5, height * 0.5),
                (width * 0.35, height * 0.7),
                (width * 0.65, height * 0.7),
            )
            face = Face(bbox=(0.0, 0.0, float(width), float(height)), score=1.0, landmarks5=points)
            out.append([face])
        return out


# ------------------------------------------------------------------------------- lossless frames


def test_ok_status_writes_lossless_frames_that_read_back_exactly(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=4))
    out_dir = tmp_path / "out"

    result = process_video(video, _inventory_record(), profile, CenterBackend(), out_dir)

    assert result.status == "ok"
    assert result.reason is None
    assert result.n_frames == len(result.frame_indices) == 4
    assert out_dir.is_dir()

    source = open_source(video, library="opencv")
    decoded = dict(source.read(result.frame_indices))
    backend = CenterBackend()
    for index in result.frame_indices:
        frame_rgb = decoded[index]
        face = backend.detect(np.stack([frame_rgb]))[0][0]
        expected = crop_face(frame_rgb, face.bbox, scale=1.0, size=16)
        assert expected is not None
        frame_path = out_dir / f"frame_{index:06d}.png"
        assert frame_path.is_file()
        written_bgr = cv2.imread(str(frame_path))
        written_rgb = cv2.cvtColor(written_bgr, cv2.COLOR_BGR2RGB)
        assert np.array_equal(written_rgb, expected.image)


def test_clip_json_has_the_documented_schema(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=4))
    out_dir = tmp_path / "out"

    process_video(video, _inventory_record(), profile, CenterBackend(), out_dir)

    clip = json.loads((out_dir / "clip.json").read_text("utf-8"))
    assert set(clip) == {"key", "compression", "source", "frames", "failed_frames", "track"}
    assert clip["key"] == "toy/vid001"
    assert clip["compression"] is None
    assert set(clip["source"]) == {"total_frames", "fps", "width", "height", "decoder"}
    assert clip["source"]["total_frames"] == 8
    assert clip["source"]["width"] == 48
    assert clip["source"]["height"] == 32
    assert clip["source"]["decoder"] == "opencv"
    assert clip["failed_frames"] == []
    assert set(clip["track"]) == {"strategy", "identity_switch"}
    assert clip["track"]["strategy"] == "largest-then-iou"
    assert clip["track"]["identity_switch"] is False
    for frame in clip["frames"]:
        assert set(frame) == {"index", "bbox", "score", "landmarks5"}
        assert frame["landmarks5"] is None
        assert len(frame["bbox"]) == 4
        assert frame["score"] == 1.0


def test_two_runs_produce_byte_identical_output(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=4))
    record = _inventory_record()

    result_a = process_video(video, record, profile, CenterBackend(), tmp_path / "out-a")
    result_b = process_video(video, record, profile, CenterBackend(), tmp_path / "out-b")

    assert result_a == result_b
    assert (tmp_path / "out-a" / "clip.json").read_bytes() == (
        tmp_path / "out-b" / "clip.json"
    ).read_bytes()
    for index in result_a.frame_indices:
        name = f"frame_{index:06d}.png"
        assert (tmp_path / "out-a" / name).read_bytes() == (tmp_path / "out-b" / name).read_bytes()


# ------------------------------------------------------------------------------------- statuses


def test_status_too_short_when_the_container_reports_zero_frames(tmp_path):
    video = tmp_path / "empty.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (16, 16))
    writer.release()
    out_dir = tmp_path / "out"

    result = process_video(video, _inventory_record(), _profile(), CenterBackend(), out_dir)

    assert result.status == "too_short"
    assert result.n_frames == 0
    assert result.frame_indices == []
    assert result.track is None
    assert result.reason
    assert not out_dir.exists()
    assert list(tmp_path.glob("out.tmp-*")) == []


@pytest.mark.parametrize("library", ["opencv", "pyav"])
def test_status_decode_error_for_a_corrupt_file(tmp_path, library):
    video = tmp_path / "corrupt.avi"
    video.write_bytes(b"not a real video file" * 20)
    out_dir = tmp_path / "out"
    profile = _profile(decode=DecodeSpec(library=library, color="rgb"))

    result = process_video(video, _inventory_record(), profile, CenterBackend(), out_dir)

    assert result.status == "decode_error"
    assert result.n_frames == 0
    # The reason names the file, never where it sits on this machine, so the index row is the
    # same wherever the video was processed.
    assert result.reason is not None
    assert result.reason.startswith("corrupt.avi: ")
    assert str(tmp_path) not in result.reason
    assert_no_absolute_paths(result)
    assert not out_dir.exists()


def test_status_no_face_when_the_backend_finds_nothing(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 6)
    out_dir = tmp_path / "out"

    result = process_video(video, _inventory_record(), _profile(), _EmptyFaceBackend(), out_dir)

    assert result.status == "no_face"
    assert result.n_frames == 0
    assert result.frame_indices == []
    assert result.track is None
    assert result.reason
    assert not out_dir.exists()


def test_short_decode_is_ok_with_reason(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 20)
    claimed_before = _decodable_frame_count(video)
    assert claimed_before == 20
    _truncate(video, 0.6)
    real_frames = _decodable_frame_count(video)
    assert 0 < real_frames < 20

    profile = _profile(sampling=SamplingSpec(mode="all"))
    out_dir = tmp_path / "out"

    result = process_video(video, _inventory_record(), profile, CenterBackend(), out_dir)

    assert result.status == "ok"
    assert result.n_frames == real_frames
    assert result.frame_indices == list(range(real_frames))
    assert result.reason == f"short: {real_frames} of 20"

    clip = json.loads((out_dir / "clip.json").read_text("utf-8"))
    assert clip["source"]["total_frames"] == 20


def test_reason_uses_profile_sampling_frames_as_m_when_set(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 6)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=8))
    out_dir = tmp_path / "out"

    result = process_video(video, _inventory_record(), profile, CenterBackend(), out_dir)

    assert result.status == "ok"
    assert result.n_frames == 6
    assert result.reason == "short: 6 of 8"


def test_reason_is_none_when_every_requested_frame_was_written(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=4))
    out_dir = tmp_path / "out"

    result = process_video(video, _inventory_record(), profile, CenterBackend(), out_dir)

    assert result.status == "ok"
    assert result.reason is None


# --------------------------------------------------------------------------------- atomic output


def test_resume_after_interrupt_leaves_no_partial_output(tmp_path, monkeypatch):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=4))
    record = _inventory_record()
    out_dir = tmp_path / "out"

    real_imwrite = cv2.imwrite
    calls = {"n": 0}

    def flaky_imwrite(filename: str, image: np.ndarray, params: list[int] | None = None) -> bool:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-write")
        return bool(real_imwrite(filename, image, params))

    monkeypatch.setattr(cv2, "imwrite", flaky_imwrite)
    with pytest.raises(RuntimeError, match="simulated crash"):
        process_video(video, record, profile, CenterBackend(), out_dir)
    monkeypatch.undo()

    leftovers = list(tmp_path.glob("out.tmp-*"))
    assert len(leftovers) == 1
    assert not out_dir.exists()

    result = process_video(video, record, profile, CenterBackend(), out_dir)

    assert result.status == "ok"
    assert out_dir.is_dir()
    assert list(tmp_path.glob("out.tmp-*")) == []
    assert len(list(out_dir.glob("frame_*.png"))) == 4


def test_a_crash_between_the_two_renames_of_a_redo_is_recovered_on_the_next_call(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=4))
    record = _inventory_record()
    out_dir = tmp_path / "out"

    first = process_video(video, record, profile, CenterBackend(), out_dir)
    assert first.status == "ok"
    original_frame_names = {p.name for p in out_dir.glob("frame_*.png")}

    # simulate a crash between the two renames of a redo's atomic swap: the old, good directory
    # has already been moved aside, but a new (here, abandoned) attempt was never swapped in
    old_dir = out_dir.with_name(f"{out_dir.name}.old-{os.getpid()}")
    out_dir.rename(old_dir)
    abandoned_tmp = out_dir.with_name(f"{out_dir.name}.tmp-{os.getpid()}")
    abandoned_tmp.mkdir()
    (abandoned_tmp / "frame_000000.png").write_bytes(b"an abandoned, unfinished attempt")

    # a source that is guaranteed to fail, so this call cannot itself write a fresh out_dir --
    # whatever ends up at out_dir afterwards can only be the recovery step's doing
    corrupt_video = tmp_path / "corrupt.avi"
    corrupt_video.write_bytes(b"not a real video file" * 20)
    second = process_video(corrupt_video, record, profile, CenterBackend(), out_dir)

    assert second.status == "decode_error"
    assert not old_dir.exists()
    assert not abandoned_tmp.exists()
    assert out_dir.is_dir()
    assert {p.name for p in out_dir.glob("frame_*.png")} == original_frame_names


# ----------------------------------------------------------------------------- windowed streaming


def test_backend_detect_never_receives_more_than_one_window(tmp_path):
    video = tmp_path / "video.avi"
    n_frames = 100  # more than three windows of 32
    _write_clip(video, n_frames)
    profile = _profile(sampling=SamplingSpec(mode="all"))
    batch_sizes: list[int] = []

    class _CountingBackend(CenterBackend):
        def detect(self, frames: np.ndarray) -> list[list[Face]]:
            batch_sizes.append(len(frames))
            return super().detect(frames)

    result = process_video(
        video, _inventory_record(), profile, _CountingBackend(), tmp_path / "out"
    )

    assert result.status == "ok"
    assert result.n_frames == n_frames
    assert batch_sizes  # detect was actually called
    assert max(batch_sizes) <= 32
    assert len(batch_sizes) >= 4  # ceil(100 / 32)
    assert sum(batch_sizes) == n_frames


def test_a_clip_spanning_several_windows_writes_every_frame_in_order(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 70)  # more than two windows of 32
    profile = _profile(sampling=SamplingSpec(mode="all"))
    record = _inventory_record()

    result = process_video(video, record, profile, CenterBackend(), tmp_path / "out")

    assert result.status == "ok"
    assert result.n_frames == 70
    assert result.frame_indices == list(range(70))
    clip = json.loads((tmp_path / "out" / "clip.json").read_text("utf-8"))
    assert [frame["index"] for frame in clip["frames"]] == list(range(70))
    assert len(list((tmp_path / "out").glob("frame_*.png"))) == 70


# ------------------------------------------------------------------------------------ crop-empty


def test_a_degenerate_box_is_recorded_as_crop_empty_without_failing_the_whole_clip(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=4))
    # sample_indices("uniform", 8, frames=4) is [0, 2, 4, 7]; position 1 in the batch is index 2.
    backend = _WholeFrameBackend(degenerate_at=1)
    out_dir = tmp_path / "out"

    result = process_video(video, _inventory_record(), profile, backend, out_dir)

    assert result.status == "ok"
    assert result.n_frames == 3
    assert result.frame_indices == [0, 4, 7]
    assert result.reason == "short: 3 of 4"

    clip = json.loads((out_dir / "clip.json").read_text("utf-8"))
    assert clip["failed_frames"] == [{"index": 2, "reason": "crop-empty"}]
    assert [frame["index"] for frame in clip["frames"]] == [0, 4, 7]


# -------------------------------------------------------------------------------- landmarks flag


def test_landmarks_are_mapped_into_crop_pixels_only_when_extras_landmarks_is_set(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 4)
    backend = _LandmarkBackend()

    on_profile = _profile(
        sampling=SamplingSpec(mode="uniform", frames=2),
        extras=ExtrasSpec(landmarks=True, mesh=False, masks=False),
    )
    off_profile = _profile(
        profile_id="toy-test-off",
        sampling=SamplingSpec(mode="uniform", frames=2),
        extras=ExtrasSpec(landmarks=False, mesh=False, masks=False),
    )

    process_video(video, _inventory_record(), on_profile, backend, tmp_path / "on")
    process_video(video, _inventory_record(), off_profile, backend, tmp_path / "off")

    on_clip = json.loads((tmp_path / "on" / "clip.json").read_text("utf-8"))
    off_clip = json.loads((tmp_path / "off" / "clip.json").read_text("utf-8"))

    assert all(
        frame["landmarks5"] is not None and len(frame["landmarks5"]) == 5
        for frame in on_clip["frames"]
    )
    assert all(frame["landmarks5"] is None for frame in off_clip["frames"])


# --------------------------------------------------------------------------------- identity path


def test_identity_cluster_tracks_the_subject_and_ignores_the_distractor(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 12)
    profile = _profile(
        track=TrackSpec(iou=0.3, strategy="identity-cluster"),
        sampling=SamplingSpec(mode="uniform", frames=4),
    )
    backend = _TwoFaceBackend(has_pose=True)  # avoid the pose warning muddying this test
    out_dir = tmp_path / "out"

    result = process_video(video, _inventory_record(), profile, backend, out_dir)

    assert result.status == "ok"
    assert result.n_frames == 4
    assert result.track is not None
    assert result.track.identity_switch is False

    clip = json.loads((out_dir / "clip.json").read_text("utf-8"))
    assert clip["track"]["strategy"] == "identity-cluster"
    assert clip["track"]["identity_switch"] is False
    for frame in clip["frames"]:
        # the subject's box is the whole frame; the distractor's is a quarter of it
        x1, y1, x2, y2 = frame["bbox"]
        assert (x2 - x1) > 40
        assert (y2 - y1) > 20
        assert frame["score"] == 1.0


def test_identity_cluster_without_embed_raises_config_error_before_any_work(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(track=TrackSpec(iou=0.3, strategy="identity-cluster"))
    out_dir = tmp_path / "out"

    with pytest.raises(ConfigError, match="embed"):
        process_video(video, _inventory_record(), profile, CenterBackend(), out_dir)

    assert not out_dir.exists()
    assert list(tmp_path.glob("out.tmp-*")) == []


def test_identity_cluster_warns_about_missing_pose_once_per_backend_instance(tmp_path, caplog):
    video = tmp_path / "video.avi"
    _write_clip(video, 12)
    profile = _profile(
        track=TrackSpec(iou=0.3, strategy="identity-cluster"),
        sampling=SamplingSpec(mode="uniform", frames=4),
    )
    backend = _TwoFaceBackend(has_pose=False)

    with caplog.at_level(logging.WARNING):
        process_video(video, _inventory_record("toy/vid001"), profile, backend, tmp_path / "a")
        process_video(video, _inventory_record("toy/vid002"), profile, backend, tmp_path / "b")

    pose_warnings = [
        message
        for message in caplog.messages
        if "pose" in message.lower() or "yaw" in message.lower()
    ]
    assert len(pose_warnings) == 1


# --------------------------------------------------------------------------------- edge cases


def test_missing_cv2_raises_installation_error(tmp_path, monkeypatch):
    video = tmp_path / "video.avi"
    _write_clip(video, 4)
    out_dir = tmp_path / "out"

    saved = {
        name: module for name, module in sys.modules.items() if name.partition(".")[0] == "cv2"
    }
    for name in saved:
        monkeypatch.delitem(sys.modules, name)

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] == "cv2":
                raise ModuleNotFoundError(f"blocked for this test: {fullname!r}")
            return None

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        from dfwb.core.errors import InstallationError

        with pytest.raises(InstallationError, match="OpenCV"):
            process_video(video, _inventory_record(), _profile(), CenterBackend(), out_dir)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_process_video_creates_the_output_directorys_missing_parents(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    out_dir = tmp_path / "brand-new" / "nested" / "out"

    result = process_video(video, _inventory_record(), _profile(), CenterBackend(), out_dir)

    assert result.status == "ok"
    assert out_dir.is_dir()


def _truncate_until_nothing_decodes(path: Path) -> None:
    """Shrink an encoded clip step by step until its header still opens and claims frames, but
    nothing actually decodes -- the container-overstates-its-length case pushed to the extreme."""
    data = path.read_bytes()
    for fraction in (0.5, 0.48, 0.46, 0.44, 0.42, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1):
        path.write_bytes(data[: int(len(data) * fraction)])
        capture = cv2.VideoCapture(str(path))
        opened = capture.isOpened()
        claimed = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        if opened and claimed > 0 and _decodable_frame_count(path) == 0:
            return
    raise AssertionError(f"{path}: no truncation fraction gave zero decodable frames")


def test_a_completely_undecodable_video_with_a_nonzero_reported_count_is_no_face(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 20)
    _truncate_until_nothing_decodes(video)

    profile = _profile(sampling=SamplingSpec(mode="all"))
    out_dir = tmp_path / "out"

    result = process_video(video, _inventory_record(), profile, CenterBackend(), out_dir)

    assert result.status == "no_face"
    assert not out_dir.exists()


def test_a_second_ok_run_replaces_the_first_output_directory(tmp_path):
    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=4))
    record = _inventory_record()
    out_dir = tmp_path / "out"

    process_video(video, record, profile, CenterBackend(), out_dir)
    (out_dir / "stray-leftover.txt").write_text("from an earlier run")

    result = process_video(video, record, profile, CenterBackend(), out_dir)

    assert result.status == "ok"
    assert not (out_dir / "stray-leftover.txt").exists()
    assert len(list(out_dir.glob("frame_*.png"))) == 4


def test_a_mid_stream_decode_error_during_the_windowed_pass_removes_the_tmp_dir(
    tmp_path, monkeypatch
):
    import dfwb.preprocess.face.process as process_module
    from dfwb.preprocess.face.decode import DecodeError

    video = tmp_path / "video.avi"
    _write_clip(video, 8)
    profile = _profile(sampling=SamplingSpec(mode="uniform", frames=4))
    out_dir = tmp_path / "out"

    real_source = process_module.open_source(video, library="opencv")

    class _FlakySource:
        total_frames = real_source.total_frames
        fps = real_source.fps
        width = real_source.width
        height = real_source.height

        def read(self, indices: list[int]):
            for seen, item in enumerate(real_source.read(indices), start=1):
                yield item
                if seen == 2:
                    raise DecodeError("simulated mid-stream failure")

    monkeypatch.setattr(process_module, "open_source", lambda path, *, library: _FlakySource())

    result = process_video(video, _inventory_record(), profile, CenterBackend(), out_dir)

    assert result.status == "decode_error"
    assert "simulated mid-stream failure" in (result.reason or "")
    assert not out_dir.exists()
    assert list(tmp_path.glob("out.tmp-*")) == []


def test_a_decode_error_while_reopening_for_identity_cluster_is_reported(tmp_path, monkeypatch):
    video = tmp_path / "video.avi"
    _write_clip(video, 12)
    profile = _profile(
        track=TrackSpec(iou=0.3, strategy="identity-cluster"),
        sampling=SamplingSpec(mode="uniform", frames=4),
    )
    backend = _TwoFaceBackend(has_pose=True)
    out_dir = tmp_path / "out"

    import dfwb.preprocess.face.process as process_module
    from dfwb.preprocess.face.decode import DecodeError

    real_open_source = process_module.open_source
    calls = {"n": 0}

    def flaky_open_source(path, *, library):
        calls["n"] += 1
        if calls["n"] == 2:
            raise DecodeError("simulated failure reopening for the main sampling pass")
        return real_open_source(path, library=library)

    monkeypatch.setattr(process_module, "open_source", flaky_open_source)

    result = process_video(video, _inventory_record(), profile, backend, out_dir)

    assert result.status == "decode_error"
    assert "simulated failure" in (result.reason or "")
    assert not out_dir.exists()


def test_a_backend_that_cannot_be_weakly_referenced_still_gets_the_pose_warning(tmp_path, caplog):
    video = tmp_path / "video.avi"
    _write_clip(video, 12)
    profile = _profile(
        track=TrackSpec(iou=0.3, strategy="identity-cluster"),
        sampling=SamplingSpec(mode="uniform", frames=4),
    )

    class _SlottedBackend:
        __slots__ = ()
        name = "slotted-fake"
        version = "1"
        license = "MIT"
        has_pose = False
        meta: ClassVar[dict[str, Any]] = {}

        def detect(self, frames: np.ndarray) -> list[list[Face]]:
            out = []
            for frame in frames:
                height, width = frame.shape[:2]
                out.append([Face(bbox=(0.0, 0.0, float(width), float(height)), score=1.0)])
            return out

        def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    backend = _SlottedBackend()
    # confirms this fake really cannot be weakly referenced
    with pytest.raises(TypeError):
        weakref.ref(backend)

    with caplog.at_level(logging.WARNING):
        result = process_video(video, _inventory_record(), profile, backend, tmp_path / "out")

    assert result.status == "ok"
    assert any("pose" in message.lower() for message in caplog.messages)
