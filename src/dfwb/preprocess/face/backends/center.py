"""The ``center`` face backend: no detection, just the centred square of every frame.

For data whose frames are already face crops (WildDeepfake, for instance) and for smoke tests
that must not depend on a detector. Each frame yields one face: the largest square that fits,
centred, with score 1.0 and no landmarks. Cropping it at scale 1.0 therefore gives exactly that
centre square. It needs nothing beyond numpy, which dfwb always installs.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from dfwb.preprocess.face.backends._common import check_frames, parse_device
from dfwb.preprocess.face.types import Face

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["CenterBackend"]


class CenterBackend:
    """One face per frame: the centred square of side ``min(height, width)``.

    Args:
        device: Accepted like every backend's, and unused: there is nothing to run.
    """

    name = "center"
    version = "1"
    license = "MIT"
    has_pose = False

    def __init__(self, *, device: str = "cpu") -> None:
        parse_device(device)

    @property
    def meta(self) -> Mapping[str, Any]:
        """Nothing to record: no model, no settings."""
        return MappingProxyType({})

    def detect(self, frames: npt.NDArray[np.uint8]) -> list[list[Face]]:
        """The centred square of each frame, as ``(x1, y1, x2, y2)`` with ``x1 = (W - s) / 2``,
        ``y1 = (H - s) / 2`` and ``s = min(H, W)``."""
        check_frames(frames)
        count, height, width = frames.shape[:3]
        side = min(height, width)
        x1 = (width - side) / 2
        y1 = (height - side) / 2
        face = Face(bbox=(x1, y1, x1 + side, y1 + side), score=1.0)
        return [[face] for _ in range(count)]
