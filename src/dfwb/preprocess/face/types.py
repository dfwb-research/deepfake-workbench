"""The record a face-detection backend returns for each face it finds in a frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Face"]


@dataclass(frozen=True)
class Face:
    """One detected face in one frame.

    Attributes:
        bbox: The face box ``(x1, y1, x2, y2)`` in the source frame's pixel coordinates.
        score: The detector's confidence that this is a face.
        landmarks5: Five facial landmarks (eyes, nose tip, mouth corners) as ``(x, y)`` pairs in
            the same coordinates as ``bbox``, when the backend provides them.
        embedding: An identity embedding for the face, when the backend computes one; only
            identity-guided subject selection uses it. It takes no part in comparing or hashing
            faces: ``==`` on two numpy arrays gives an array rather than a bool, which would make
            comparing two faces raise.
        yaw: The head's left-right rotation in degrees (0 is facing the camera), when the backend
            estimates head pose.
    """

    bbox: tuple[float, float, float, float]
    score: float
    landmarks5: tuple[tuple[float, float], ...] | None = None
    embedding: np.ndarray | None = field(default=None, compare=False)
    yaw: float | None = None
