"""Cropping a detected face into a fixed-size square image.

This is the same arithmetic the earlier face pipeline used to turn a detected bounding box into a
square crop: a square centred on the box, sized relative to it, replicate-padded where it runs off
the frame, then resized to a fixed output size. Reusing that arithmetic unchanged means a store
built with this framework matches an old one pixel for pixel, given the same frame and bounding
box.

``cv2`` is an optional extra and is only imported inside the functions that need it, so this module
itself always imports cleanly without it (matching ``decode.py``). Numpy is likewise only named
here in type annotations; the arrays this module produces come from cv2 itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dfwb.core.errors import InstallationError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["CropResult", "crop_face", "map_landmarks"]

_EXTRA_HINT = 'pip install "deepfake-workbench[preprocess]"'


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError:
        raise InstallationError("cropping a face needs OpenCV", hint=_EXTRA_HINT) from None
    return cv2


@dataclass(frozen=True)
class CropResult:
    """The result of cropping and resizing one detected face.

    Attributes:
        image: The resized square crop, ``[size, size, 3]``, ``uint8``, in whatever colour order
            the input frame used.
        box: The crop square ``(x1, y1, x2, y2)`` in the *original* frame's coordinates, before any
            padding shift. It may have negative coordinates, or extend past the frame's width or
            height, when the face sat near an edge; ``map_landmarks`` relies on it being in this
            un-shifted coordinate system.
        crop_size: The square's side length before resizing: ``max(box width, box height) *
            scale``.
    """

    image: npt.NDArray[np.uint8]
    box: tuple[int, int, int, int]
    crop_size: float


def crop_face(
    frame_rgb: npt.NDArray[np.uint8],
    bbox: tuple[float, float, float, float],
    *,
    scale: float,
    size: int,
) -> CropResult | None:
    """Crop the square region around ``bbox`` out of ``frame_rgb`` and resize it to ``size``.

    The square is centred on ``bbox`` and sized to ``max(width, height) * scale``; its edges are
    truncated to integers (``int()``, which rounds toward zero, not to the nearest integer). Any
    part of that square that falls outside ``frame_rgb`` is filled by replicating the nearest edge
    pixel (``cv2.copyMakeBorder`` with ``BORDER_REPLICATE``) rather than cropped away, so the
    result always covers a full ``size`` x ``size`` square. The resize uses ``INTER_AREA`` when the
    square is larger than ``size`` (shrinking) and ``INTER_LINEAR`` otherwise (enlarging, or no
    change).

    Returns:
        ``None`` when the computed square has zero or negative width or height, so there is
        nothing to crop (this happens for a degenerate or inverted ``bbox``), and also when
        ``cv2.resize`` itself raises the OpenCV error it raises for an image it cannot resize. In
        both cases nothing was produced for this frame; the caller is expected to record it as a
        failed frame rather than treat ``None`` as an error. Any other exception is not caught
        here and propagates, since that would be a genuine bug rather than the kind of malformed
        input the old pipeline was built to tolerate.

    Raises:
        InstallationError: OpenCV is not installed.
    """
    cv2 = _require_cv2()
    frame = frame_rgb
    height, width = frame.shape[0], frame.shape[1]
    x1, y1, x2, y2 = bbox
    box_width, box_height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    crop_size = max(box_width, box_height) * scale

    nx1 = int(center_x - crop_size / 2)
    ny1 = int(center_y - crop_size / 2)
    nx2 = int(center_x + crop_size / 2)
    ny2 = int(center_y + crop_size / 2)
    original_box = (nx1, ny1, nx2, ny2)

    pad_left = max(0, -nx1)
    pad_top = max(0, -ny1)
    pad_right = max(0, nx2 - width)
    pad_bottom = max(0, ny2 - height)

    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
        nx1, ny1, nx2, ny2 = nx1 + pad_left, ny1 + pad_top, nx2 + pad_left, ny2 + pad_top

    crop = frame[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None

    try:
        interpolation = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
        image = cv2.resize(crop, (size, size), interpolation=interpolation)
    except cv2.error:
        return None

    return CropResult(image=image, box=original_box, crop_size=crop_size)


def map_landmarks(
    points: Sequence[tuple[float, float]], result: CropResult
) -> list[tuple[float, float]]:
    """Map ``points``, given in the original frame's coordinates, into ``result``'s output image.

    Uses the same box ``crop_face`` computed and a single scale factor ``s = size / crop_size``:
    each point maps to ``((x - x1) * s, (y - y1) * s)``, where ``(x1, y1)`` is the top-left corner
    of ``result.box`` -- the original, un-padded coordinates ``crop_face`` recorded.
    """
    x1, y1, _, _ = result.box
    scale = result.image.shape[0] / result.crop_size
    return [((x - x1) * scale, (y - y1) * scale) for x, y in points]
