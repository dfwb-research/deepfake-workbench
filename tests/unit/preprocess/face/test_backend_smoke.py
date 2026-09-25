"""Smoke tests of the insightface and mediapipe backends against their real libraries and models.

They run only where the library is installed and its model files are already on this machine, and
they never download anything (offline mode is on). They check the shape of what comes back and
that the calls work end to end, not detection quality: the input is
``fixtures/synthetic_face_64.png``, a 64 x 64 cartoon face drawn with OpenCV (an ellipse for the
head, two eyes, a nose and a mouth), which depicts no real person.

The insightface models are looked up where the backend itself looks (the dfwb cache root, then
``~/.insightface/models/buffalo_l``) and linked into a private cache for the test; the mediapipe
model is looked up in the dfwb cache root.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from dfwb.core import licenses
from dfwb.core.plugins import get_registry
from dfwb.preprocess.face.backends import Face, FaceBackend, _common
from dfwb.preprocess.face.backends import insightface as insightface_module
from dfwb.preprocess.face.backends import mediapipe as mediapipe_module

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_face_64.png"


def _synthetic_face() -> np.ndarray:
    image = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
    assert image is not None, FIXTURE
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _check_faces(faces: list[list[Face]], frames: int, min_score: float) -> None:
    assert isinstance(faces, list)
    assert len(faces) == frames
    for per_frame in faces:
        assert isinstance(per_frame, list)
        for face in per_frame:
            assert isinstance(face, Face)
            x1, y1, x2, y2 = face.bbox
            assert all(type(value) is float for value in face.bbox)
            assert x1 < x2
            assert y1 < y2
            assert type(face.score) is float
            assert min_score <= face.score <= 1.0
            assert face.embedding is None
            assert face.yaw is None


@pytest.fixture
def buffalo_l(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not _installed("onnxruntime"):
        pytest.skip("needs onnxruntime (the face-insightface extra)")
    found = {}
    for name, sha256 in insightface_module.MODEL_FILES.items():
        path = _common.find_verified(name, sha256, insightface_module.model_dirs())
        if path is None:
            pytest.skip(f"needs the buffalo_l model {name} on this machine")
        found[name] = path
    private = tmp_path / "cache" / "models" / "buffalo_l"
    private.mkdir(parents=True)
    for name, path in found.items():
        (private / name).symlink_to(path)
    monkeypatch.setenv("DFWB_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("DFWB_OFFLINE", "1")
    licenses.accept(insightface_module.GATE, license="non-commercial research use only")


@pytest.mark.usefixtures("buffalo_l")
def test_insightface_detects_and_embeds_with_the_real_models():
    frame = _synthetic_face()
    frames = np.stack([frame, np.ascontiguousarray(frame[:, ::-1])])
    backend = get_registry("face_backends").build(
        "insightface", model="buffalo_l", det_size=256, min_score=0.5, device="cpu"
    )
    assert isinstance(backend, FaceBackend)

    faces = backend.detect(frames)
    _check_faces(faces, 2, 0.5)
    # At this input size the cartoon is found in both frames, scoring about 0.75.
    assert all(len(per_frame) >= 1 for per_frame in faces)
    for per_frame in faces:
        for face in per_frame:
            assert face.landmarks5 is not None
            assert len(face.landmarks5) == 5
    assert backend.detect(frames) == faces  # the same frames give the same faces

    embedding = backend.embed(frame, faces[0][0])
    assert embedding.dtype == np.float32
    assert embedding.shape == (512,)
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-5)
    assert backend.meta["providers"] == ["CPUExecutionProvider"]


@pytest.fixture
def blazeface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not _installed("mediapipe"):
        pytest.skip("needs mediapipe (the face-mediapipe extra)")
    name, _, sha256 = mediapipe_module.MODELS["short-range"]
    path = _common.find_verified(name, sha256, [mediapipe_module.models_dir()])
    if path is None:
        pytest.skip(f"needs the mediapipe model {name} in the dfwb cache root")
    private = tmp_path / "cache" / "models" / "mediapipe"
    private.mkdir(parents=True)
    (private / name).symlink_to(path)
    monkeypatch.setenv("DFWB_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("DFWB_OFFLINE", "1")


@pytest.mark.usefixtures("blazeface")
def test_mediapipe_detects_with_the_real_library_and_model():
    frame = _synthetic_face()
    frames = np.stack([frame, np.ascontiguousarray(frame[:, ::-1])])
    backend = get_registry("face_backends").build("mediapipe", min_score=0.5, device="cpu")
    assert isinstance(backend, FaceBackend)
    faces = backend.detect(frames)
    _check_faces(faces, 2, 0.5)
    assert all(face.landmarks5 is None for per_frame in faces for face in per_frame)
    backend.close()
    backend.close()
