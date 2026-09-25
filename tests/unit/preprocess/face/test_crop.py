"""Cropping a detected face into a fixed-size square image, and mapping points into it.

The golden test builds its expected image with the same formulas ``crop_face`` uses, called
directly with cv2 rather than by calling ``crop_face`` itself, and pins a few of its pixels
literally: if the arithmetic in ``crop.py`` and in this test ever drifted in the same wrong
direction, the array comparison alone would not catch it, but a hard-coded pixel value would.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from dfwb.core.errors import InstallationError
from dfwb.preprocess.face.crop import CropResult, crop_face, map_landmarks


@contextmanager
def _import_blocked(name: str) -> Iterator[None]:
    """Make ``import <name>`` fail for the duration of the ``with`` block, even though it is
    actually installed: removes any cached module first, since an import already in
    ``sys.modules`` would otherwise short-circuit the block."""

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


def _gradient_frame(width: int, height: int) -> np.ndarray:
    """A deterministic, non-uniform frame: every pixel encodes its own coordinates, so a wrong
    crop offset or a transposed axis changes the sampled values rather than being masked out by
    a flat colour."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:height, 0:width]
    frame[..., 0] = xs % 256
    frame[..., 1] = ys % 256
    frame[..., 2] = (xs + ys) % 256
    return frame


# ---------------------------------------------------------------------------
# Golden crop
# ---------------------------------------------------------------------------


def test_golden_crop_matches_manual_computation():
    import cv2

    frame = _gradient_frame(100, 80)
    bbox = (30.6, 20.2, 60.9, 55.5)
    scale, size = 1.3, 16

    # Expected, computed independently of crop_face's internals but with the same arithmetic and
    # cv2 calls it is documented to use.
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    crop_size = max(x2 - x1, y2 - y1) * scale
    nx1 = int(cx - crop_size / 2)
    ny1 = int(cy - crop_size / 2)
    nx2 = int(cx + crop_size / 2)
    ny2 = int(cy + crop_size / 2)
    assert (nx1, ny1, nx2, ny2) == (22, 14, 68, 60)  # sanity-check the hand-computed box
    raw_crop = frame[ny1:ny2, nx1:nx2]
    assert raw_crop.shape[0] > size  # this box takes the INTER_AREA branch
    expected = cv2.resize(raw_crop, (size, size), interpolation=cv2.INTER_AREA)

    result = crop_face(frame, bbox, scale=scale, size=size)

    assert result is not None
    np.testing.assert_array_equal(result.image, expected)
    assert result.box == (22, 14, 68, 60)
    assert result.crop_size == pytest.approx(45.89)
    assert result.image.shape == (16, 16, 3)
    assert result.image.dtype == np.uint8
    # Pixel values pinned literally, independent of the array-equality check above.
    assert tuple(int(v) for v in result.image[0, 0]) == tuple(int(v) for v in expected[0, 0])
    assert tuple(int(v) for v in result.image[15, 15]) == tuple(int(v) for v in expected[15, 15])
    assert tuple(int(v) for v in result.image[8, 8]) == tuple(int(v) for v in expected[8, 8])


# ---------------------------------------------------------------------------
# Replicate padding
# ---------------------------------------------------------------------------


def test_crop_pads_with_replicate_outside_frame():
    frame = _gradient_frame(100, 80)
    # A tiny box hard against the top-left corner: the square crop around it extends well past
    # both the top and the left edges of the frame.
    bbox = (2.0, 1.0, 6.0, 5.0)
    scale, size = 3.0, 8

    result = crop_face(frame, bbox, scale=scale, size=size)

    assert result is not None
    # The computed box's top-left corner is negative: this frame's replicate padding is real.
    nx1, ny1, _, _ = result.box
    assert nx1 < 0
    assert ny1 < 0
    # The output's very first pixel corresponds to the replicated corner: it must equal the
    # original frame's own top-left pixel, not some extrapolated or black value.
    assert tuple(int(v) for v in result.image[0, 0]) == tuple(int(v) for v in frame[0, 0])


def test_crop_pads_with_replicate_past_bottom_right():
    frame = _gradient_frame(40, 30)
    bbox = (32.0, 24.0, 39.0, 29.0)
    scale, size = 3.0, 8

    result = crop_face(frame, bbox, scale=scale, size=size)

    assert result is not None
    _, _, nx2, ny2 = result.box
    assert nx2 > frame.shape[1]
    assert ny2 > frame.shape[0]
    assert tuple(int(v) for v in result.image[-1, -1]) == tuple(int(v) for v in frame[-1, -1])


# ---------------------------------------------------------------------------
# Interpolation selection
# ---------------------------------------------------------------------------


def _spy_on_resize_interpolation(monkeypatch) -> list[int]:
    """Wrap ``cv2.resize`` to record the ``interpolation`` flag each call receives, while still
    delegating to the real implementation -- so the recorded flag and the pixels produced both
    come from one real cv2 call, not a hand-rolled substitute."""
    import cv2

    calls: list[int] = []
    original = cv2.resize

    def _wrapped(*args, **kwargs):
        calls.append(kwargs["interpolation"])
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "resize", _wrapped)
    return calls


def test_downscale_uses_inter_area(monkeypatch):
    import cv2

    calls = _spy_on_resize_interpolation(monkeypatch)
    frame = _gradient_frame(200, 200)
    bbox = (20.0, 20.0, 100.0, 100.0)  # crop_size = 80, size = 16 -> shrinking

    result = crop_face(frame, bbox, scale=1.0, size=16)

    assert result is not None
    assert calls == [cv2.INTER_AREA]


def test_upscale_uses_inter_linear(monkeypatch):
    import cv2

    calls = _spy_on_resize_interpolation(monkeypatch)
    frame = _gradient_frame(200, 200)
    bbox = (50.0, 50.0, 58.0, 58.0)  # crop_size = 8, size = 32 -> enlarging

    result = crop_face(frame, bbox, scale=1.0, size=32)

    assert result is not None
    assert calls == [cv2.INTER_LINEAR]


def test_crop_exactly_equal_to_size_uses_inter_linear_not_area(monkeypatch):
    # The old formula's branch is a strict ">"; a crop exactly `size` pixels tall must take the
    # else (INTER_LINEAR) branch, not INTER_AREA.
    import cv2

    calls = _spy_on_resize_interpolation(monkeypatch)
    frame = _gradient_frame(64, 64)
    size = 16
    bbox = (24.0, 24.0, 24.0 + size, 24.0 + size)

    result = crop_face(frame, bbox, scale=1.0, size=size)

    assert result is not None
    _, ny1, _, ny2 = result.box
    assert ny2 - ny1 == size
    assert calls == [cv2.INTER_LINEAR]


# ---------------------------------------------------------------------------
# Truncation, not rounding
# ---------------------------------------------------------------------------


def test_box_edges_truncate_toward_zero_not_round():
    # With scale == 1.0, crop_size == box width, so nx1 == int(x1) exactly: a square bbox whose
    # left edge is X.9 makes the edge truncate down to X, where round() would instead go to X+1.
    assert int(5.9) == 5
    assert round(5.9) == 6

    frame = _gradient_frame(200, 200)
    bbox = (5.9, 5.9, 15.9, 15.9)

    result = crop_face(frame, bbox, scale=1.0, size=8)

    assert result is not None
    assert result.box[0] == 5
    assert result.box[0] == int(5.9)
    assert result.box[0] != round(5.9)


def test_negative_edge_truncates_toward_zero():
    # Same scale == 1.0 trick, with a left edge that is negative: int() truncates toward zero
    # (-3.7 -> -3), where floor() would instead go to -4.
    assert int(-3.7) == -3

    frame = _gradient_frame(50, 50)
    bbox = (-3.7, -3.7, 6.3, 6.3)

    result = crop_face(frame, bbox, scale=1.0, size=4)

    assert result is not None
    assert result.box[0] == -3
    assert result.box[0] == int(-3.7)
    assert result.box[0] != math.floor(-3.7)


# ---------------------------------------------------------------------------
# Empty crop -> None
# ---------------------------------------------------------------------------


def test_zero_size_bbox_returns_none():
    frame = _gradient_frame(50, 50)
    bbox = (10.0, 10.0, 10.0, 10.0)  # width and height both zero

    result = crop_face(frame, bbox, scale=1.0, size=16)

    assert result is None


def test_negative_size_bbox_returns_none():
    frame = _gradient_frame(50, 50)
    bbox = (20.0, 20.0, 10.0, 10.0)  # x2 < x1 and y2 < y1

    result = crop_face(frame, bbox, scale=1.0, size=16)

    assert result is None


# ---------------------------------------------------------------------------
# Resize failure -> None, but only for the same error the old code guarded against
# ---------------------------------------------------------------------------


def test_resize_failure_returns_none(monkeypatch):
    import cv2

    frame = _gradient_frame(50, 50)

    def _raise(*args, **kwargs):
        raise cv2.error("synthetic OpenCV failure")

    monkeypatch.setattr(cv2, "resize", _raise)

    result = crop_face(frame, (10.0, 10.0, 30.0, 30.0), scale=1.0, size=16)

    assert result is None


def test_a_genuine_bug_in_resize_is_not_swallowed(monkeypatch):
    import cv2

    frame = _gradient_frame(50, 50)

    def _raise(*args, **kwargs):
        raise ValueError("not an OpenCV failure")

    monkeypatch.setattr(cv2, "resize", _raise)

    with pytest.raises(ValueError, match="not an OpenCV failure"):
        crop_face(frame, (10.0, 10.0, 30.0, 30.0), scale=1.0, size=16)


# ---------------------------------------------------------------------------
# Landmark mapping
# ---------------------------------------------------------------------------


def test_map_landmarks_bbox_centre_lands_at_image_centre():
    frame = _gradient_frame(200, 200)
    bbox = (40.0, 40.0, 80.0, 80.0)
    scale, size = 1.0, 32
    result = crop_face(frame, bbox, scale=scale, size=size)
    assert result is not None

    cx, cy = (40.0 + 80.0) / 2, (40.0 + 80.0) / 2
    mapped = map_landmarks([(cx, cy)], result)

    assert len(mapped) == 1
    mx, my = mapped[0]
    assert mx == pytest.approx(size / 2)
    assert my == pytest.approx(size / 2)


def test_map_landmarks_box_corner_lands_at_image_origin():
    frame = _gradient_frame(200, 200)
    bbox = (40.0, 40.0, 80.0, 80.0)
    scale, size = 1.0, 32
    result = crop_face(frame, bbox, scale=scale, size=size)
    assert result is not None

    nx1, ny1, _, _ = result.box
    mapped = map_landmarks([(float(nx1), float(ny1))], result)

    assert mapped[0] == pytest.approx((0.0, 0.0))


def test_map_landmarks_uses_size_over_crop_size_scale_factor():
    frame = _gradient_frame(200, 200)
    bbox = (10.0, 10.0, 50.0, 50.0)
    scale, size = 1.3, 16
    result = crop_face(frame, bbox, scale=scale, size=size)
    assert result is not None

    nx1, ny1, _, _ = result.box
    s = size / result.crop_size
    point = (nx1 + 5.0, ny1 + 7.0)

    mapped = map_landmarks([point], result)

    assert mapped[0] == pytest.approx((5.0 * s, 7.0 * s))


def test_map_landmarks_returns_a_list_of_tuples_for_multiple_points():
    frame = _gradient_frame(200, 200)
    bbox = (10.0, 10.0, 50.0, 50.0)
    result = crop_face(frame, bbox, scale=1.3, size=16)
    assert result is not None

    mapped = map_landmarks([(10.0, 10.0), (50.0, 50.0), (30.0, 30.0)], result)

    assert isinstance(mapped, list)
    assert len(mapped) == 3


# ---------------------------------------------------------------------------
# CropResult
# ---------------------------------------------------------------------------


def test_crop_result_is_frozen():
    frame = _gradient_frame(50, 50)
    result = crop_face(frame, (10.0, 10.0, 30.0, 30.0), scale=1.0, size=16)
    assert isinstance(result, CropResult)

    with pytest.raises(AttributeError):
        result.crop_size = 99.0


# ---------------------------------------------------------------------------
# Missing OpenCV
# ---------------------------------------------------------------------------


def test_missing_opencv_raises_installation_error():
    frame = _gradient_frame(10, 10)
    with (
        _import_blocked("cv2"),
        pytest.raises(InstallationError, match="OpenCV") as excinfo,
    ):
        crop_face(frame, (1.0, 1.0, 5.0, 5.0), scale=1.0, size=4)
    assert excinfo.value.hint == 'pip install "deepfake-workbench[preprocess]"'
