"""Face-detection backends: the contract every backend meets, and the backends dfwb ships.

A backend turns a batch of frames into the faces found on each. The pipeline builds one from a
profile's ``backend`` section through the ``face_backends`` registry,
``get_registry("face_backends").build(name, **params, device=...)``, so every backend's
constructor takes keyword parameters plus ``device`` (``"cpu"`` or ``"cuda:<index>"``).

The shipped backends are ``center`` (no detection: the centred square of each frame, for
pre-cropped data and smoke tests), ``insightface`` (insightface's ``buffalo_l`` models through
onnxruntime; its weights are for non-commercial research use only and need a one-time licence
acknowledgement) and ``mediapipe`` (Google's BlazeFace, Apache-2.0). Each lives in its own module
and imports its libraries only when it first detects, so this package imports without any of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dfwb.preprocess.face.types import Face

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["Face", "FaceBackend"]


@runtime_checkable
class FaceBackend(Protocol):
    """What the pipeline needs from a face-detection backend.

    Attributes:
        name: The backend's registry key, e.g. ``"insightface"``.
        version: The version of the backend's own code, recorded with every store it builds.
        license: The SPDX identifier of the backend's code. Model weights under other terms say
            so in ``meta``.
        meta: Whatever else a store should record about how its faces were found: model files
            and their sha256s, thresholds, the execution providers used, the weights' licence.

    Two members are optional, and a caller checks for them with ``getattr``:

    - ``embed(frame, face) -> np.ndarray``: the identity embedding of ``face`` in the ``[H, W,
      3]`` RGB ``frame``, scaled to unit length (see ``Face.embedding``). Identity-guided subject
      selection needs it.
    - ``has_pose: bool``: whether ``detect`` fills in ``Face.yaw``. Missing means ``False``.
    """

    name: str
    version: str
    license: str

    @property
    def meta(self) -> Mapping[str, Any]:
        """Provenance details to record with a store; see the class docstring."""
        ...

    def detect(self, frames: npt.NDArray[np.uint8]) -> list[list[Face]]:
        """The faces on each of ``frames``, an ``[N, H, W, 3]`` ``uint8`` RGB array: one list per
        frame, in frame order, each face's coordinates in that frame's pixels."""
        ...
