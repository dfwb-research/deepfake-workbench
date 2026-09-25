"""The ``insightface`` face backend: insightface's ``buffalo_l`` models, run with onnxruntime.

Faces are detected with SCRFD (``det_10g.onnx``) and, when asked, embedded with ArcFace
(``w600k_r50.onnx``), through a line-for-line port of insightface 0.7.3's pre- and
post-processing (:mod:`dfwb.preprocess.face._insightface_port`), without the insightface package.
Detections (boxes, scores and landmarks) are bit-identical to those of insightface 0.7.3's
``FaceAnalysis`` on the same frame. Embeddings agree to within a small tolerance (at most about
0.002 per value, cosine similarity at least 0.9999): OpenCV 5's ``warpAffine``, which aligns the
face before it is embedded, rounds slightly differently from the OpenCV 4 insightface ran with.
Only identity-guided subject selection uses embeddings, and a difference that small does not
change which faces it groups together.

The code is MIT-licensed, but the ``buffalo_l`` weights are for non-commercial research use only.
The backend therefore refuses to be built, and :func:`model_file` refuses to find or fetch a
model, until that licence has been acknowledged once on the machine (``--accept-license``). Only
then, on the backend's first ``detect``, are the models looked for:
first in ``<cache root>/models/buffalo_l/``, then in ``~/.insightface/models/buffalo_l/``, where
insightface itself keeps them. If neither has them, it downloads the ``buffalo_l`` release archive
into ``<cache root>/models/`` under a name private to the process (so that parallel workers
never delete each other's download), unpacks just the two models it uses and deletes the archive.
Every model file is checked against its known sha256 before it is used.

Head pose is not estimated (insightface's pose model is not part of the port), so ``Face.yaw``
stays ``None`` and ``has_pose`` is ``False``. onnxruntime is imported when the backend is built
(to settle which execution providers it will use), OpenCV and numpy on first use; none of them
when this module is imported.
"""

from __future__ import annotations

import hashlib
import logging
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from dfwb.core.errors import ConfigError, ContractError, InstallationError
from dfwb.core.fetch import fetch
from dfwb.preprocess.face.backends._common import (
    cache_models_dir,
    check_frame,
    check_frames,
    find_verified,
    parse_device,
    require_accepted,
)
from dfwb.preprocess.face.types import Face

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from dfwb.preprocess.face._insightface_port.arcface import ArcFace
    from dfwb.preprocess.face._insightface_port.scrfd import SCRFD

__all__ = [
    "GATE",
    "MODEL_FILES",
    "PACK_SHA256",
    "PACK_URL",
    "WEIGHTS_LICENSE",
    "InsightFaceBackend",
    "model_dirs",
    "model_file",
]

_log = logging.getLogger(__name__)

# The name the licence acknowledgement is recorded under, and the weights' terms.
GATE = "insightface-buffalo_l"
WEIGHTS_LICENSE = "non-commercial research use only"
_TERMS = "the insightface buffalo_l model weights are for non-commercial research use only"

PACK = "buffalo_l"
DETECTOR = "det_10g.onnx"
RECOGNIZER = "w600k_r50.onnx"
# The two models this backend uses, and their sha256s.
MODEL_FILES: dict[str, str] = {
    DETECTOR: "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    RECOGNIZER: "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
# The release archive holding them (288,621,354 bytes, five models in all).
PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
PACK_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"

UPSTREAM = "insightface 0.7.3 (SCRFD and ArcFace pre- and post-processing, ported)"
_INSTALL_HINT = 'pip install "deepfake-workbench[face-insightface]"'
_CHUNK = 1 << 20

# The detector's feature maps have strides of up to 32 pixels, so its square input must be a
# whole number of 32-pixel cells for the network's outputs to line up with the anchor grid.
_DET_SIZE_STEP = 32


def model_dirs() -> list[Path]:
    """Where the ``buffalo_l`` models are looked for, in order."""
    return [cache_models_dir() / PACK, Path.home() / ".insightface" / "models" / PACK]


def model_file(name: str) -> Path:
    """The verified model file ``name`` (a key of ``MODEL_FILES``), downloading it if needed.

    Raises:
        InstallationError: The ``buffalo_l`` licence has not been acknowledged; or the file is on
            disk nowhere and cannot be downloaded (``DFWB_OFFLINE`` is set, or the request
            failed).
        ContractError: The download, or a model inside it, does not hash as expected.
    """
    require_accepted(GATE, terms=_TERMS)
    directories = model_dirs()
    found = find_verified(name, MODEL_FILES[name], directories)
    if found is not None:
        return found
    _download_pack(directories[0])
    return directories[0] / name


def _download_pack(target: Path) -> None:
    """Download the release archive next to ``target``, unpack the models this backend uses into
    ``target`` (each checked against its sha256) and delete the archive.

    The archive's name includes the process id: workers starting in parallel may each download it,
    and a shared name would let one worker's clean-up delete the archive another is still
    unpacking. Unpacking itself is safe to race, since each model is written to a private
    temporary file and then renamed into place.
    """
    archive_path = target.parent / f".{PACK}.{os.getpid()}.zip"
    _log.info("downloading the insightface %s models (289 MB) from %s", PACK, PACK_URL)
    fetch(PACK_URL, PACK_SHA256, archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for name, sha256 in MODEL_FILES.items():
                _extract(archive, name, sha256, target)
    finally:
        archive_path.unlink(missing_ok=True)


def _extract(archive: zipfile.ZipFile, name: str, sha256: str, target: Path) -> None:
    try:
        member = archive.getinfo(name)
    except KeyError:
        raise ContractError(
            f"{PACK_URL}: the archive holds no {name}",
            hint="the release may have changed; re-check the expected archive and its sha256",
        ) from None
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / f".{name}.tmp-{os.getpid()}"
    try:
        digest = hashlib.sha256()
        with archive.open(member) as source, tmp.open("wb") as sink:
            for chunk in iter(lambda: source.read(_CHUNK), b""):
                sink.write(chunk)
                digest.update(chunk)
        got = digest.hexdigest()
        if got != sha256:
            raise ContractError(
                f"{name} from {PACK_URL}: sha256 is {got}, expected {sha256}",
                hint="the release may have changed; re-check the expected model sha256s",
            )
        tmp.replace(target / name)
    finally:
        tmp.unlink(missing_ok=True)


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ImportError:
        raise InstallationError(
            "the insightface backend needs onnxruntime", hint=_INSTALL_HINT
        ) from None
    return onnxruntime


def _require_cv2() -> None:
    try:
        import cv2  # noqa: F401 - only checking that it is there
    except ImportError:
        raise InstallationError(
            "the insightface backend needs OpenCV", hint=_INSTALL_HINT
        ) from None


def _to_face(row: npt.NDArray[Any], points: npt.NDArray[Any] | None) -> Face:
    """A ``Face`` from one detection row ``[x1, y1, x2, y2, score]`` and its landmarks."""
    bbox = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
    landmarks = None if points is None else tuple((float(x), float(y)) for x, y in points)
    return Face(bbox=bbox, score=float(row[4]), landmarks5=landmarks)


class InsightFaceBackend:
    """SCRFD detection and ArcFace embeddings from insightface's ``buffalo_l`` models.

    Building the backend only checks its parameters and the licence acknowledgement; the models
    are found (or downloaded) and loaded on the first :meth:`detect`, and the recognition model
    on the first :meth:`embed`.

    Args:
        model: The model pack; ``"buffalo_l"`` is the only one supported.
        det_size: The side of the square detector input, in pixels (a multiple of 32).
            insightface's default is 640.
        min_score: Faces scoring below this are not returned.
        nms: The overlap above which the lower-scoring of two faces is dropped.
        device: ``"cpu"``, or ``"cuda:<index>"`` to run on that GPU when onnxruntime has CUDA
            support (falling back to the CPU, with a warning when the backend is built, when it
            does not).

    Raises:
        ConfigError: A parameter is out of range.
        InstallationError: The ``buffalo_l`` licence has not been acknowledged (exit code 5), or
            onnxruntime is not installed.
    """

    name = "insightface"
    version = "1"
    license = "MIT"
    has_pose = False
    license_gate = GATE

    def __init__(
        self,
        *,
        model: str = PACK,
        det_size: int = 640,
        min_score: float = 0.5,
        nms: float = 0.4,
        device: str = "cpu",
    ) -> None:
        if model != PACK:
            raise ConfigError(
                f"insightface backend: unknown model {model!r}",
                hint=f"the only model this backend supports is {PACK!r}",
            )
        if det_size <= 0 or det_size % _DET_SIZE_STEP:
            raise ConfigError(
                f"insightface backend: det_size must be a positive multiple of "
                f"{_DET_SIZE_STEP}, got {det_size}",
                hint="insightface's default is 640; the shipped profiles use 256",
            )
        if not 0.0 <= min_score <= 1.0:
            raise ConfigError(
                f"insightface backend: min_score must be between 0 and 1, got {min_score}",
                hint="insightface's default is 0.5",
            )
        if not 0.0 < nms <= 1.0:
            raise ConfigError(
                f"insightface backend: nms must be above 0 and at most 1, got {nms}",
                hint="insightface's default is 0.4",
            )
        self._cuda_device = parse_device(device)
        require_accepted(GATE, terms=_TERMS)
        self._device = device
        self._requested = self._resolve_providers(_require_onnxruntime())
        self._model = model
        self._det_size = det_size
        self._min_score = min_score
        self._nms = nms
        self._detector: SCRFD | None = None
        self._recognizer: ArcFace | None = None
        self._providers: list[str] | None = None

    @property
    def meta(self) -> Mapping[str, Any]:
        """The model pack, its files and sha256s, the weights' licence, the settings, and the
        onnxruntime execution providers: those the models run on once loaded, and before that the
        ones settled on when the backend was built (the CPU alone when CUDA was asked for but
        this onnxruntime cannot provide it)."""
        if self._providers is not None:
            providers = list(self._providers)
        else:
            providers = [p if isinstance(p, str) else p[0] for p in self._requested]
        return MappingProxyType(
            {
                "model": self._model,
                "files": dict(MODEL_FILES),
                "weights_license": WEIGHTS_LICENSE,
                "det_size": self._det_size,
                "min_score": self._min_score,
                "nms": self._nms,
                "device": self._device,
                "providers": providers,
                "upstream": UPSTREAM,
            }
        )

    def detect(self, frames: npt.NDArray[np.uint8]) -> list[list[Face]]:
        """The faces on each RGB frame, best score first, each with its five landmarks (eyes,
        nose tip, mouth corners)."""
        check_frames(frames)
        if len(frames) == 0:
            return []
        detector = self._load_detector()
        faces = []
        for frame in frames:
            det, kpss = detector.detect(frame)
            faces.append(
                [_to_face(det[i], None if kpss is None else kpss[i]) for i in range(len(det))]
            )
        return faces

    def embed(self, frame: npt.NDArray[np.uint8], face: Face) -> npt.NDArray[np.float32]:
        """The identity embedding of ``face`` in the RGB ``frame``: 512 ``float32`` values scaled
        to unit length, as insightface's ``normed_embedding``.

        It matches insightface 0.7.3's embedding of the same face to within about 0.002 per value
        (cosine similarity at least 0.9999), not bit for bit: the face is aligned with OpenCV's
        ``warpAffine``, and OpenCV 5 rounds that warp slightly differently from OpenCV 4.

        Raises:
            ValueError: ``face`` has no landmarks, or ``frame`` is not one ``[H, W, 3]`` image.
        """
        import numpy as np

        if face.landmarks5 is None:
            raise ValueError("embedding a face needs its five landmarks, and this face has none")
        check_frame(frame)
        recognizer = self._load_recognizer()
        kps = np.asarray(face.landmarks5, dtype=np.float32)
        embedding = recognizer.get(frame, kps)
        normed: npt.NDArray[np.float32] = embedding / np.linalg.norm(embedding)
        return normed.astype(np.float32, copy=False)

    def _load_detector(self) -> SCRFD:
        if self._detector is None:
            from dfwb.preprocess.face._insightface_port.scrfd import SCRFD

            runtime = _require_onnxruntime()
            _require_cv2()
            session = self._session(runtime, model_file(DETECTOR))
            self._detector = SCRFD(
                session,
                input_size=(self._det_size, self._det_size),
                det_thresh=self._min_score,
                nms_thresh=self._nms,
            )
        return self._detector

    def _load_recognizer(self) -> ArcFace:
        if self._recognizer is None:
            from dfwb.preprocess.face._insightface_port.arcface import ArcFace

            runtime = _require_onnxruntime()
            _require_cv2()
            self._recognizer = ArcFace(self._session(runtime, model_file(RECOGNIZER)))
        return self._recognizer

    def _session(self, runtime: Any, path: Path) -> Any:
        options = runtime.SessionOptions()
        options.log_severity_level = 3  # errors only; insightface set this for the whole process
        session = runtime.InferenceSession(
            str(path), sess_options=options, providers=self._requested
        )
        self._providers = list(session.get_providers())
        return session

    def _resolve_providers(self, runtime: Any) -> list[Any]:
        """The execution providers to ask onnxruntime for, given the device."""
        if self._cuda_device is None:
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in runtime.get_available_providers():
            return [
                ("CUDAExecutionProvider", {"device_id": self._cuda_device}),
                "CPUExecutionProvider",
            ]
        _log.warning(
            "device %s was asked for, but this onnxruntime has no CUDA support; running on the CPU",
            self._device,
        )
        return ["CPUExecutionProvider"]
