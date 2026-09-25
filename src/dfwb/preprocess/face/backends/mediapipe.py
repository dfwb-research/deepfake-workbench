"""The ``mediapipe`` face backend: Google MediaPipe's BlazeFace detector.

Both the code and the model weights are Apache-2.0, which makes this the backend without a
licence gate. It uses MediaPipe's face detector task in image mode, each frame on its own.

The detector model, a ``.tflite`` file, does not come with the mediapipe package. It is looked
for in ``<cache root>/models/mediapipe/`` and otherwise downloaded there from Google's model
store, and it is checked against its known sha256 either way. BlazeFace comes in two variants:
``short-range``, for faces within about two metres of the camera, and ``full-range``, for faces up
to about five metres away.

BlazeFace's six keypoints (eyes, nose tip, mouth centre and ear tragions) are not the five
landmarks ``Face.landmarks5`` holds (eyes, nose tip and mouth corners), so none are reported, and
there is no head pose (``has_pose`` is ``False``). It runs on the CPU whatever device is asked
for. mediapipe and numpy are imported on first use, not when this module is imported.

Call :meth:`MediaPipeBackend.close` when done with the backend. mediapipe releases a detector
through worker threads that are already gone once the interpreter has started shutting down, so
a detector still open at exit fails to close with a (harmless, but noisy) printed traceback.

Installing: mediapipe depends on ``opencv-contrib-python``, so the ``face-mediapipe`` extra puts a
second OpenCV distribution next to the ``opencv-python-headless`` that dfwb's ``preprocess``
extra installs. Both provide the same ``cv2`` package, so uninstalling either one breaks ``cv2``
for the other, which then has to be reinstalled (``pip install --force-reinstall``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from dfwb.core.errors import ConfigError, InstallationError, did_you_mean
from dfwb.core.fetch import fetch
from dfwb.preprocess.face.backends._common import (
    cache_models_dir,
    check_frames,
    find_verified,
    parse_device,
)
from dfwb.preprocess.face.types import Face

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["MODELS", "WEIGHTS_LICENSE", "MediaPipeBackend", "models_dir"]

_log = logging.getLogger(__name__)

_STORE = "https://storage.googleapis.com/mediapipe-models/face_detector"
# model name -> (file name, download URL, sha256); version 1 of each float16 model.
MODELS: dict[str, tuple[str, str, str]] = {
    "short-range": (
        "blaze_face_short_range.tflite",
        f"{_STORE}/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    ),
    "full-range": (
        "blaze_face_full_range.tflite",
        f"{_STORE}/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
        "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b",
    ),
}
WEIGHTS_LICENSE = "Apache-2.0"
_INSTALL_HINT = 'pip install "deepfake-workbench[face-mediapipe]"'


def models_dir() -> Path:
    """Where the BlazeFace model files are kept: ``<cache root>/models/mediapipe``."""
    return cache_models_dir() / "mediapipe"


def _require_mediapipe() -> Any:
    try:
        import mediapipe
    except ImportError:
        raise InstallationError(
            "the mediapipe backend needs mediapipe", hint=_INSTALL_HINT
        ) from None
    return mediapipe


def _to_face(detection: Any) -> Face:
    box = detection.bounding_box
    x1, y1 = float(box.origin_x), float(box.origin_y)
    return Face(
        bbox=(x1, y1, x1 + float(box.width), y1 + float(box.height)),
        score=float(detection.categories[0].score),
    )


class MediaPipeBackend:
    """BlazeFace face detection with MediaPipe.

    Building the backend only checks its parameters; the model is found (or downloaded) and
    loaded on the first :meth:`detect`, and released by :meth:`close`.

    Args:
        min_score: Faces scoring below this are not returned.
        model: ``"short-range"`` or ``"full-range"``.
        device: Accepted like every backend's; a CUDA device gets a warning, since this backend
            runs on the CPU.

    Raises:
        ConfigError: A parameter is out of range.
    """

    name = "mediapipe"
    version = "1"
    license = "Apache-2.0"
    has_pose = False

    def __init__(
        self, *, min_score: float = 0.5, model: str = "short-range", device: str = "cpu"
    ) -> None:
        if model not in MODELS:
            raise ConfigError(
                f"mediapipe backend: unknown model {model!r}{did_you_mean(model, MODELS)}",
                hint="use model: short-range or model: full-range",
            )
        if not 0.0 <= min_score <= 1.0:
            raise ConfigError(
                f"mediapipe backend: min_score must be between 0 and 1, got {min_score}",
                hint="MediaPipe's default is 0.5",
            )
        if parse_device(device) is not None:
            _log.warning("the mediapipe backend runs on the CPU; device %s is not used", device)
        self._model = model
        self._min_score = min_score
        self._detector: Any = None

    @property
    def meta(self) -> Mapping[str, Any]:
        """The model variant, its file and sha256, the weights' licence and the threshold."""
        file_name, _, sha256 = MODELS[self._model]
        return MappingProxyType(
            {
                "model": self._model,
                "file": file_name,
                "sha256": sha256,
                "weights_license": WEIGHTS_LICENSE,
                "min_score": self._min_score,
            }
        )

    def detect(self, frames: npt.NDArray[np.uint8]) -> list[list[Face]]:
        """The faces on each RGB frame, as MediaPipe orders them, without landmarks."""
        import numpy as np

        check_frames(frames)
        if len(frames) == 0:
            return []
        mediapipe = _require_mediapipe()
        detector = self._load(mediapipe)
        faces = []
        for frame in frames:
            image = mediapipe.Image(
                image_format=mediapipe.ImageFormat.SRGB, data=np.ascontiguousarray(frame)
            )
            result = detector.detect(image)
            faces.append([_to_face(detection) for detection in result.detections])
        return faces

    def close(self) -> None:
        """Release the detector. Calling it again, or before any detection, does nothing; a
        later :meth:`detect` opens a new detector."""
        detector, self._detector = self._detector, None
        if detector is not None:
            detector.close()

    def _load(self, mediapipe: Any) -> Any:
        if self._detector is None:
            file_name, url, sha256 = MODELS[self._model]
            directory = models_dir()
            path = find_verified(file_name, sha256, [directory])
            if path is None:
                _log.info("downloading the mediapipe %s face model from %s", self._model, url)
                path = fetch(url, sha256, directory / file_name)
            vision = mediapipe.tasks.vision
            options = vision.FaceDetectorOptions(
                base_options=mediapipe.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=self._min_score,
            )
            self._detector = vision.FaceDetector.create_from_options(options)
        return self._detector
