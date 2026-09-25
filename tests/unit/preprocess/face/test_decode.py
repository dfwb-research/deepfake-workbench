"""Sequential, no-seeking decoding of chosen frames from a video or a directory of frame images.

``overstated_count.avi`` is a real, hand-truncated file: a 20-frame lossless FFV1 clip with its
tail cut off after encoding, so its container header still reports 20 frames (both OpenCV's
``CAP_PROP_FRAME_COUNT`` and PyAV's ``stream.frames`` read that header value) while only the first
12 frames actually decode. That reproduces, in miniature, hand-me-down video files whose recorded
frame count outlives their readable content.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from dfwb.core.errors import InstallationError
from dfwb.preprocess.face.decode import DecodeError, open_source
from dfwb.preprocess.face.sampling import sample_indices

FIXTURES = Path(__file__).parent / "fixtures"
CLEAN_AVI = FIXTURES / "clean12.avi"  # 12 real frames, accurate container count
CLEAN_MKV = FIXTURES / "clean12.mkv"  # same 12 frames; matroska carries no frame count in-header
OVERSTATED_AVI = FIXTURES / "overstated_count.avi"  # header says 20, only 12 actually decode
# a raw elementary stream, with no container around it: OpenCV opens it but cannot report a
# frame count for it up front, exercising the count-by-decoding-everything fallback.
COUNTLESS_H264 = FIXTURES / "frame_count_unknown.h264"
# has an audio stream and no video stream at all.
AUDIO_ONLY = FIXTURES / "audio_only.mp3"


@contextmanager
def _import_blocked(name: str) -> Iterator[None]:
    """Make ``import <name>`` fail for the duration of the ``with`` block, even though it is
    actually installed: removes any cached module first, since an import already in ``sys.modules``
    would otherwise short-circuit the block."""

    class _Blocker:
        def find_spec(self, fullname: str, path: object, target: object = None) -> None:
            if fullname.partition(".")[0] == name:
                raise ModuleNotFoundError(f"blocked for this test: {fullname!r}")
            return None

    saved = {key: value for key, value in sys.modules.items() if key.partition(".")[0] == name}
    for key in saved:
        del sys.modules[key]
    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_missing_opencv_raises_installation_error(tmp_path):
    with (
        _import_blocked("cv2"),
        pytest.raises(InstallationError, match="OpenCV") as excinfo,
    ):
        open_source(tmp_path / "clip.mkv", library="opencv")
    assert excinfo.value.hint == 'pip install "deepfake-workbench[preprocess]"'


def test_missing_pyav_raises_installation_error(tmp_path):
    with (
        _import_blocked("av"),
        pytest.raises(InstallationError, match="PyAV") as excinfo,
    ):
        open_source(tmp_path / "clip.mkv", library="pyav")
    assert excinfo.value.hint == 'pip install "deepfake-workbench[preprocess]"'


@pytest.mark.parametrize("library", ["opencv", "pyav"])
def test_open_source_reports_container_metadata(library):
    pytest.importorskip("cv2" if library == "opencv" else "av")

    source = open_source(CLEAN_AVI, library=library)

    assert source.total_frames == 12
    assert source.fps == 10.0
    assert source.width == 32
    assert source.height == 24


def test_pyav_total_frames_falls_back_to_a_full_decode_count_when_the_container_has_none():
    pytest.importorskip("av")

    # matroska stores no frame count in its header, unlike the avi fixtures used elsewhere here.
    source = open_source(CLEAN_MKV, library="pyav")

    assert source.total_frames == 12
    decoded = list(source.read(sample_indices("all", source.total_frames)))
    assert [index for index, _ in decoded] == list(range(12))


def test_pyav_reopen_failure_after_a_count_fallback_raises_decode_error(monkeypatch):
    av = pytest.importorskip("av")
    real_open = av.open
    calls = {"n": 0}

    def flaky_open(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_open(path, *args, **kwargs)
        raise av.error.InvalidDataError(
            1094995529, "Invalid data found when processing input", str(path)
        )

    monkeypatch.setattr(av, "open", flaky_open)

    # CLEAN_MKV has no in-header frame count, so opening it always takes the count-by-decoding
    # fallback, which is exactly the path that reopens the container a second time.
    with pytest.raises(DecodeError, match="reopen") as excinfo:
        open_source(CLEAN_MKV, library="pyav")
    assert not isinstance(excinfo.value, av.error.FFmpegError)
    assert calls["n"] == 2


def test_pyav_reopen_onto_a_file_with_no_video_stream_raises_decode_error(monkeypatch):
    av = pytest.importorskip("av")
    real_open = av.open
    calls = {"n": 0}

    def flaky_open(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_open(path, *args, **kwargs)
        return real_open(str(AUDIO_ONLY), *args, **kwargs)

    monkeypatch.setattr(av, "open", flaky_open)

    with pytest.raises(DecodeError, match="no video stream"):
        open_source(CLEAN_MKV, library="pyav")
    assert calls["n"] == 2


@pytest.mark.parametrize("library", ["opencv", "pyav"])
def test_decoder_stops_at_real_end(library):
    pytest.importorskip("cv2" if library == "opencv" else "av")

    source = open_source(OVERSTATED_AVI, library=library)
    assert source.total_frames == 20  # the container's own (wrong) claim

    wanted = sample_indices("all", source.total_frames)
    decoded = list(source.read(wanted))

    # the tail indices (12..19) never decode; they are simply absent, not an error.
    assert [index for index, _ in decoded] == list(range(12))


def test_opencv_and_pyav_give_identical_rgb_frames_on_a_lossless_fixture():
    pytest.importorskip("cv2")
    pytest.importorskip("av")

    opencv_source = open_source(CLEAN_AVI, library="opencv")
    pyav_source = open_source(CLEAN_AVI, library="pyav")
    indices = sample_indices("all", 12)

    opencv_frames = list(opencv_source.read(indices))
    pyav_frames = list(pyav_source.read(indices))

    assert [i for i, _ in opencv_frames] == [i for i, _ in pyav_frames] == indices
    for (_, cv_frame), (_, av_frame) in zip(opencv_frames, pyav_frames, strict=True):
        assert cv_frame.dtype == av_frame.dtype == np.uint8
        assert np.array_equal(cv_frame, av_frame)


@pytest.mark.parametrize("library", ["opencv", "pyav"])
def test_a_uniform_sample_only_yields_the_requested_indices(library):
    pytest.importorskip("cv2" if library == "opencv" else "av")

    source = open_source(CLEAN_AVI, library=library)
    wanted = sample_indices("uniform", source.total_frames, frames=4)

    decoded = list(source.read(wanted))

    assert [index for index, _ in decoded] == wanted
    for _, frame in decoded:
        assert frame.shape == (24, 32, 3)
        assert frame.dtype == np.uint8


@pytest.mark.parametrize("library", ["opencv", "pyav"])
def test_read_with_unsorted_duplicate_indices_yields_each_once_in_ascending_order(library):
    pytest.importorskip("cv2" if library == "opencv" else "av")

    source = open_source(CLEAN_AVI, library=library)
    decoded = list(source.read([5, 0, 5, 2, 11, 2, 0]))

    assert [index for index, _ in decoded] == [0, 2, 5, 11]


@pytest.mark.parametrize("library", ["opencv", "pyav"])
def test_a_corrupt_file_raises_decode_error(library, tmp_path):
    pytest.importorskip("cv2" if library == "opencv" else "av")
    corrupt = tmp_path / "corrupt.avi"
    corrupt.write_bytes(b"not a real video file" * 20)

    with pytest.raises(DecodeError, match=r"corrupt\.avi"):
        open_source(corrupt, library=library)


@pytest.mark.parametrize("library", ["opencv", "pyav"])
def test_a_missing_file_raises_decode_error(library, tmp_path):
    pytest.importorskip("cv2" if library == "opencv" else "av")

    with pytest.raises(DecodeError):
        open_source(tmp_path / "does-not-exist.avi", library=library)


def test_frame_directory_source_lists_images_sorted_by_name_and_ignores_other_files(tmp_path):
    cv2 = pytest.importorskip("cv2")
    directory = tmp_path / "frames"
    directory.mkdir()
    colors_bgr = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
    for index, color in enumerate(colors_bgr):
        image = np.full((4, 6, 3), color, dtype=np.uint8)
        cv2.imwrite(str(directory / f"frame_{index:03d}.png"), image)
    (directory / "frame_003.PNG").write_bytes((directory / "frame_002.png").read_bytes())
    (directory / "notes.txt").write_text("not an image")

    source = open_source(directory, library="opencv")

    assert source.total_frames == 4
    assert source.fps == 0.0
    assert source.width == 6
    assert source.height == 4

    decoded = list(source.read([0, 1, 2, 3]))
    assert [index for index, _ in decoded] == [0, 1, 2, 3]
    for (_, frame), color in zip(decoded[:3], colors_bgr, strict=True):
        expected_rgb = np.full((4, 6, 3), color[::-1], dtype=np.uint8)
        assert np.array_equal(frame, expected_rgb)


def test_frame_directory_source_stops_when_a_requested_index_is_out_of_range(tmp_path):
    cv2 = pytest.importorskip("cv2")
    directory = tmp_path / "frames"
    directory.mkdir()
    cv2.imwrite(str(directory / "frame_000.png"), np.zeros((2, 2, 3), dtype=np.uint8))

    source = open_source(directory, library="opencv")

    decoded = list(source.read([0, 5]))

    assert [index for index, _ in decoded] == [0]


def test_frame_directory_source_with_no_images_raises_decode_error(tmp_path):
    pytest.importorskip("cv2")
    directory = tmp_path / "empty"
    directory.mkdir()
    (directory / "notes.txt").write_text("not an image")

    with pytest.raises(DecodeError):
        open_source(directory, library="opencv")


def test_frame_directory_source_with_a_corrupt_first_image_raises_decode_error(tmp_path):
    pytest.importorskip("cv2")
    directory = tmp_path / "frames"
    directory.mkdir()
    (directory / "frame_000.png").write_bytes(b"not an image")

    with pytest.raises(DecodeError, match=r"frame_000\.png"):
        open_source(directory, library="opencv")


def test_frame_directory_source_stops_at_a_corrupt_image_later_in_the_sequence(tmp_path):
    cv2 = pytest.importorskip("cv2")
    directory = tmp_path / "frames"
    directory.mkdir()
    cv2.imwrite(str(directory / "frame_000.png"), np.zeros((2, 2, 3), dtype=np.uint8))
    (directory / "frame_001.png").write_bytes(b"not an image")

    source = open_source(directory, library="opencv")
    decoded = list(source.read([0, 1]))

    assert [index for index, _ in decoded] == [0]


def test_opencv_counts_frames_by_decoding_when_the_container_reports_none():
    pytest.importorskip("cv2")

    # a raw elementary stream has no header frame count for OpenCV to read.
    source = open_source(COUNTLESS_H264, library="opencv")

    decoded = list(source.read(sample_indices("all", source.total_frames)))
    assert source.total_frames == len(decoded) > 0


def test_pyav_raises_decode_error_for_a_file_with_no_video_stream():
    pytest.importorskip("av")

    with pytest.raises(DecodeError, match="no video stream"):
        open_source(AUDIO_ONLY, library="pyav")


def test_pyav_read_with_no_requested_indices_yields_nothing():
    pytest.importorskip("av")

    source = open_source(CLEAN_AVI, library="pyav")

    assert list(source.read([])) == []
