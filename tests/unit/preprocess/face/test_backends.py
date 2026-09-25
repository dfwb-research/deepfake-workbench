"""The face backend contract and the three backends dfwb ships: center, insightface, mediapipe.

``center`` needs nothing but numpy and is tested directly. The other two are tested against
stand-ins for onnxruntime and mediapipe, so these tests run without either library and without
any model file; ``test_backend_smoke.py`` runs the real libraries and models when they are there.
Model downloads go to a local HTTP server, never to the real release URLs, and every test runs
with its own licence store, cache root and home directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.server
import importlib.machinery
import io
import logging
import os
import sys
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from dfwb.core import licenses
from dfwb.core.errors import ConfigError, ContractError, InstallationError
from dfwb.core.plugins import get_registry
from dfwb.preprocess.face import types as face_types
from dfwb.preprocess.face.backends import Face, FaceBackend, _common
from dfwb.preprocess.face.backends import insightface as insightface_module
from dfwb.preprocess.face.backends import mediapipe as mediapipe_module
from dfwb.preprocess.face.backends.center import CenterBackend
from dfwb.preprocess.face.backends.insightface import InsightFaceBackend
from dfwb.preprocess.face.backends.mediapipe import MediaPipeBackend
from dfwb.preprocess.face.crop import crop_face

# ---------------------------------------------------------------------------------- fixtures


@dataclasses.dataclass
class Isolated:
    cache: Path
    home: Path
    state: Path


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Isolated:
    """A private licence store, cache root and home directory, and offline mode on, so no test
    can reach a real download URL by accident (the local server fixture turns it off)."""
    paths = Isolated(cache=tmp_path / "cache", home=tmp_path / "home", state=tmp_path / "state")
    paths.home.mkdir()
    monkeypatch.setenv("DFWB_CACHE_ROOT", str(paths.cache))
    monkeypatch.setenv("HOME", str(paths.home))
    monkeypatch.setenv("DFWB_STATE_DIR", str(paths.state))
    monkeypatch.setenv("DFWB_OFFLINE", "1")
    return paths


class _Handler(http.server.BaseHTTPRequestHandler):
    routes: dict[str, bytes] = {}
    requests: list[str] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        type(self).requests.append(self.path)
        body = type(self).routes.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@dataclasses.dataclass
class Server:
    url: str
    routes: dict[str, bytes]
    requests: list[str]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Server]:
    monkeypatch.delenv("DFWB_OFFLINE", raising=False)
    routes: dict[str, bytes] = {}
    requests: list[str] = []
    handler = type("Handler", (_Handler,), {"routes": routes, "requests": requests})
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield Server(f"http://127.0.0.1:{httpd.server_port}", routes, requests)
    finally:
        httpd.shutdown()
        thread.join()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frames(n: int, height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------------- contract


def test_face_is_re_exported_from_the_backends_package():
    assert Face is face_types.Face


def test_center_meets_the_backend_contract():
    backend = CenterBackend()
    assert isinstance(backend, FaceBackend)
    assert (backend.name, backend.version, backend.license) == ("center", "1", "MIT")
    assert dict(backend.meta) == {}
    assert backend.has_pose is False


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", None), ("cuda:0", 0), ("cuda:3", 3), ("cuda:12", 12)],
)
def test_device_names_are_cpu_or_a_numbered_cuda_device(device, expected):
    assert _common.parse_device(device) == expected


@pytest.mark.parametrize("device", ["gpu", "cuda", "cuda:", "cuda:x", "cuda:-1", "CPU", ""])
def test_other_device_names_are_a_config_error(device):
    with pytest.raises(ConfigError, match="device") as caught:
        _common.parse_device(device)
    assert "cuda:<index>" in caught.value.hint


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((4, 4, 3), np.uint8),
        np.zeros((1, 4, 4, 4), np.uint8),
        np.zeros((1, 4, 4, 3), np.float32),
        [[[[0, 0, 0]]]],
    ],
)
def test_frames_must_be_a_batch_of_rgb_uint8_images(frames):
    with pytest.raises(ValueError, match=r"\[N, H, W, 3\]"):
        _common.check_frames(frames)


# ---------------------------------------------------------------------------------- center


def test_center_returns_the_centred_square_once_per_frame():
    faces = CenterBackend().detect(np.zeros((3, 48, 80, 3), np.uint8))
    assert faces == [[Face(bbox=(16.0, 0.0, 64.0, 48.0), score=1.0)]] * 3
    (face,) = faces[0]
    assert face.landmarks5 is None
    assert all(type(value) is float for value in face.bbox)


def test_center_keeps_half_pixels_when_the_margin_is_odd():
    ((face,),) = CenterBackend().detect(np.zeros((1, 51, 50, 3), np.uint8))
    assert face.bbox == (0.0, 0.5, 50.0, 50.5)
    ((face,),) = CenterBackend().detect(np.zeros((1, 50, 53, 3), np.uint8))
    assert face.bbox == (1.5, 0.0, 51.5, 50.0)


def test_center_on_a_portrait_frame():
    (faces,) = CenterBackend().detect(np.zeros((1, 90, 50, 3), np.uint8))
    assert faces == [Face(bbox=(0.0, 20.0, 50.0, 70.0), score=1.0)]


def test_center_on_no_frames_returns_nothing():
    assert CenterBackend().detect(np.zeros((0, 10, 10, 3), np.uint8)) == []


@pytest.mark.parametrize(
    ("height", "width"), [(48, 80), (90, 50), (51, 50), (50, 51), (64, 64), (7, 12), (8, 13)]
)
def test_a_one_times_crop_of_the_center_face_is_exactly_the_centre_square(height, width):
    frames = _frames(1, height, width)
    ((face,),) = CenterBackend().detect(frames)
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    result = crop_face(frames[0], face.bbox, scale=1.0, size=side)
    assert result is not None
    assert np.array_equal(result.image, frames[0][top : top + side, left : left + side])


def test_center_accepts_any_device_and_rejects_a_malformed_one():
    CenterBackend(device="cuda:1")
    with pytest.raises(ConfigError):
        CenterBackend(device="gpu")


# ---------------------------------------------------------------------------------- licence gate


class FakeGatedBackend:
    """A backend whose weights need a licence acknowledgement, gated like the shipped ones."""

    name = "fake-gated"
    version = "1"
    license = "MIT"
    meta: dict[str, Any] = {}

    def __init__(self, *, device: str = "cpu") -> None:
        _common.require_accepted("fake-weights", terms="the fake weights are for testing only")
        self.device = device

    def detect(self, frames: np.ndarray) -> list[list[Face]]:
        return [[] for _ in frames]


def test_a_gated_backend_refuses_to_build_until_its_licence_is_acknowledged():
    registry = get_registry("face_backends")
    registry.register("fake-gated", summary="a gated stand-in")(FakeGatedBackend)
    with pytest.raises(InstallationError) as caught:
        registry.build("fake-gated", device="cpu")
    assert caught.value.exit_code == 5
    assert "fake-weights" in caught.value.message
    assert "--accept-license" in caught.value.hint
    assert "the fake weights are for testing only" in caught.value.hint

    licenses.accept("fake-weights", license="testing only")
    backend = registry.build("fake-gated", device="cpu")
    assert isinstance(backend, FakeGatedBackend)
    assert isinstance(backend, FaceBackend)


@pytest.mark.usefixtures("stand_in_models")
def test_insightface_is_gated_before_it_looks_for_any_model(isolated, served_pack):
    with pytest.raises(InstallationError) as caught:
        InsightFaceBackend()
    assert caught.value.exit_code == 5
    assert "insightface-buffalo_l" in caught.value.message
    assert "--accept-license" in caught.value.hint
    assert "non-commercial research" in caught.value.hint
    assert not isolated.cache.exists()
    assert served_pack.requests == []


# ---------------------------------------------------------------------------------- insightface

DET_BYTES = b"stand-in detector model"
REC_BYTES = b"stand-in recognition model"
OTHER_BYTES = b"a model the backend never uses"


@pytest.fixture
def accepted() -> None:
    licenses.accept(insightface_module.GATE, license="non-commercial research use only")


@pytest.fixture
def stand_in_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the backend expect the stand-in model files' hashes instead of the real ones."""
    monkeypatch.setattr(
        insightface_module,
        "MODEL_FILES",
        {"det_10g.onnx": _sha(DET_BYTES), "w600k_r50.onnx": _sha(REC_BYTES)},
    )


def _put(directory: Path, name: str, data: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(data)
    return path


def _pack(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def served_pack(server: Server, monkeypatch: pytest.MonkeyPatch) -> Server:
    """The model pack, served locally, with a third model inside that must not be extracted."""
    pack = _pack(
        {"genderage.onnx": OTHER_BYTES, "det_10g.onnx": DET_BYTES, "w600k_r50.onnx": REC_BYTES}
    )
    server.routes["/buffalo_l.zip"] = pack
    monkeypatch.setattr(insightface_module, "PACK_URL", f"{server.url}/buffalo_l.zip")
    monkeypatch.setattr(insightface_module, "PACK_SHA256", _sha(pack))
    return server


def _cache_pack(isolated: Isolated) -> Path:
    return isolated.cache / "models" / "buffalo_l"


def _home_pack(isolated: Isolated) -> Path:
    return isolated.home / ".insightface" / "models" / "buffalo_l"


@pytest.mark.usefixtures("stand_in_models", "accepted")
def test_the_cache_root_is_searched_before_the_insightface_directory(isolated):
    cached = _put(_cache_pack(isolated), "det_10g.onnx", DET_BYTES)
    _put(_home_pack(isolated), "det_10g.onnx", DET_BYTES)
    assert insightface_module.model_file("det_10g.onnx") == cached


@pytest.mark.usefixtures("stand_in_models", "accepted")
def test_the_insightface_directory_is_used_when_the_cache_lacks_the_file(isolated):
    home = _put(_home_pack(isolated), "det_10g.onnx", DET_BYTES)
    assert insightface_module.model_file("det_10g.onnx") == home
    assert not isolated.cache.exists()


@pytest.mark.usefixtures("stand_in_models", "accepted")
def test_a_file_with_the_wrong_hash_is_skipped_with_a_warning(isolated, caplog):
    _put(_cache_pack(isolated), "det_10g.onnx", b"something else")
    home = _put(_home_pack(isolated), "det_10g.onnx", DET_BYTES)
    with caplog.at_level(logging.WARNING):
        assert insightface_module.model_file("det_10g.onnx") == home
    assert "sha256" in caplog.text
    assert "det_10g.onnx" in caplog.text


@pytest.mark.usefixtures("stand_in_models", "accepted")
def test_a_missing_model_downloads_the_pack_and_extracts_only_the_two_models(isolated, served_pack):
    path = insightface_module.model_file("det_10g.onnx")
    target = _cache_pack(isolated)
    assert path == target / "det_10g.onnx"
    assert path.read_bytes() == DET_BYTES
    assert (target / "w600k_r50.onnx").read_bytes() == REC_BYTES
    assert sorted(p.name for p in target.iterdir()) == ["det_10g.onnx", "w600k_r50.onnx"]
    assert sorted(p.name for p in target.parent.iterdir()) == ["buffalo_l"]  # zip removed
    assert served_pack.requests == ["/buffalo_l.zip"]
    # Both models are now in the cache, so asking again downloads nothing.
    assert insightface_module.model_file("w600k_r50.onnx") == target / "w600k_r50.onnx"
    assert served_pack.requests == ["/buffalo_l.zip"]


@pytest.mark.usefixtures("stand_in_models", "accepted")
def test_a_pack_member_with_the_wrong_hash_is_refused_and_nothing_is_kept(
    isolated, served_pack, monkeypatch
):
    monkeypatch.setitem(insightface_module.MODEL_FILES, "w600k_r50.onnx", "0" * 64)
    with pytest.raises(ContractError, match=r"w600k_r50\.onnx"):
        insightface_module.model_file("det_10g.onnx")
    target = _cache_pack(isolated)
    leftovers = sorted(p.name for p in target.iterdir()) if target.exists() else []
    assert "w600k_r50.onnx" not in leftovers
    assert all(not name.startswith(".") for name in leftovers)
    assert not list(target.parent.glob(".buffalo_l.*.zip"))


@pytest.mark.usefixtures("stand_in_models", "accepted")
def test_a_pack_missing_a_model_is_refused(isolated, server, monkeypatch):
    pack = _pack({"det_10g.onnx": DET_BYTES})
    server.routes["/buffalo_l.zip"] = pack
    monkeypatch.setattr(insightface_module, "PACK_URL", f"{server.url}/buffalo_l.zip")
    monkeypatch.setattr(insightface_module, "PACK_SHA256", _sha(pack))
    with pytest.raises(ContractError, match=r"w600k_r50\.onnx"):
        insightface_module.model_file("det_10g.onnx")


@pytest.mark.usefixtures("stand_in_models", "accepted")
def test_offline_with_no_model_on_disk_is_an_installation_error(served_pack, monkeypatch):
    monkeypatch.setenv("DFWB_OFFLINE", "1")
    with pytest.raises(InstallationError, match="DFWB_OFFLINE"):
        insightface_module.model_file("det_10g.onnx")
    assert served_pack.requests == []


@pytest.mark.usefixtures("stand_in_models")
def test_model_file_itself_needs_the_licence_acknowledgement(isolated, served_pack):
    _put(_home_pack(isolated), "det_10g.onnx", DET_BYTES)
    with pytest.raises(InstallationError) as caught:
        insightface_module.model_file("det_10g.onnx")
    assert caught.value.exit_code == 5
    assert "--accept-license" in caught.value.hint
    assert served_pack.requests == []
    assert not isolated.cache.exists()


@pytest.mark.usefixtures("stand_in_models", "accepted")
def test_each_process_downloads_the_pack_under_its_own_name(isolated, served_pack, monkeypatch):
    # Another worker's download, next to ours, must survive our clean-up.
    models = isolated.cache / "models"
    other = _put(models, ".buffalo_l.99999999.zip", b"another worker's download")
    destinations: list[Path] = []
    real_fetch = insightface_module.fetch

    def spy(url: str, sha256: str, dest: Path) -> Path:
        destinations.append(Path(dest))
        return real_fetch(url, sha256, dest)

    monkeypatch.setattr(insightface_module, "fetch", spy)
    insightface_module.model_file("det_10g.onnx")
    assert destinations == [models / f".buffalo_l.{os.getpid()}.zip"]
    assert other.read_bytes() == b"another worker's download"
    assert sorted(p.name for p in models.iterdir()) == [".buffalo_l.99999999.zip", "buffalo_l"]


def test_the_real_pack_and_model_hashes_are_pinned():
    assert insightface_module.PACK_URL == (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    )
    assert insightface_module.PACK_SHA256 == (
        "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
    )
    assert insightface_module.MODEL_FILES == {
        "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    }


# --- a stand-in onnxruntime ---------------------------------------------------------------------

EMBEDDING = np.zeros(512, np.float32)
EMBEDDING[:2] = [3.0, 4.0]


# The faces the stand-in detector reports: (x, y) of a stride-8 grid cell and a score. Each box
# reaches 8 px to each side of its cell's anchor centre, with all five landmarks at the centre.
ONE_FACE = [((2, 2), 0.9)]


def _detector_outputs(blob: np.ndarray, faces: list[tuple[tuple[int, int], float]]) -> list[Any]:
    size = blob.shape[2]
    rows = [(size // stride) ** 2 * 2 for stride in (8, 16, 32)]
    scores = [np.zeros((n, 1), np.float32) for n in rows]
    boxes = [np.zeros((n, 4), np.float32) for n in rows]
    points = [np.zeros((n, 10), np.float32) for n in rows]
    for (x, y), score in faces:
        row = (y * (size // 8) + x) * 2
        scores[0][row] = score
        boxes[0][row] = [1, 1, 1, 1]
    return [*scores, *boxes, *points]


class FakeOnnxRuntime(ModuleType):
    """Just enough of onnxruntime for the backend: sessions that record how they were made and
    what they were fed."""

    def __init__(self, *, cuda: bool = False) -> None:
        super().__init__("onnxruntime")
        self.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None)
        self.cuda = cuda
        self.faces = ONE_FACE
        self.sessions: list[Any] = []
        runtime = self

        class SessionOptions:
            def __init__(self) -> None:
                self.log_severity_level = 2

        class InferenceSession:
            def __init__(self, path: str, sess_options: Any = None, providers: Any = None) -> None:
                self.path = Path(path)
                self.options = sess_options
                self.providers = providers
                self.blobs: list[np.ndarray] = []
                self.detector = self.path.name == "det_10g.onnx"
                runtime.sessions.append(self)

            def get_inputs(self) -> list[SimpleNamespace]:
                shape = [1, 3, "?", "?"] if self.detector else ["None", 3, 112, 112]
                return [SimpleNamespace(name="input.1", shape=shape)]

            def get_outputs(self) -> list[SimpleNamespace]:
                count = 9 if self.detector else 1
                return [SimpleNamespace(name=f"out{i}", shape=[]) for i in range(count)]

            def get_providers(self) -> list[str]:
                return [p if isinstance(p, str) else p[0] for p in self.providers]

            def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
                blob = feeds["input.1"]
                self.blobs.append(blob)
                if self.detector:
                    return _detector_outputs(blob, runtime.faces)
                return [np.repeat(EMBEDDING[None], blob.shape[0], axis=0)]

        self.SessionOptions = SessionOptions
        self.InferenceSession = InferenceSession

    def get_available_providers(self) -> list[str]:
        base = ["AzureExecutionProvider", "CPUExecutionProvider"]
        return ["CUDAExecutionProvider", *base] if self.cuda else base


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch, isolated: Isolated, accepted: None) -> FakeOnnxRuntime:
    """A stand-in onnxruntime, stand-in model files in the cache and the licence accepted."""
    fake = FakeOnnxRuntime()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(
        insightface_module,
        "MODEL_FILES",
        {"det_10g.onnx": _sha(DET_BYTES), "w600k_r50.onnx": _sha(REC_BYTES)},
    )
    _put(_cache_pack(isolated), "det_10g.onnx", DET_BYTES)
    _put(_cache_pack(isolated), "w600k_r50.onnx", REC_BYTES)
    return fake


def test_building_insightface_loads_nothing(runtime):
    backend = InsightFaceBackend()
    assert runtime.sessions == []
    assert isinstance(backend, FaceBackend)
    assert (backend.name, backend.version, backend.license) == ("insightface", "1", "MIT")
    assert backend.has_pose is False


def test_the_first_detect_opens_the_detector_once_on_the_cpu(runtime, isolated):
    backend = InsightFaceBackend(det_size=64)
    backend.detect(_frames(1, 64, 64))
    backend.detect(_frames(2, 64, 64))
    (session,) = runtime.sessions
    assert session.path == _cache_pack(isolated) / "det_10g.onnx"
    assert session.providers == ["CPUExecutionProvider"]
    assert session.options.log_severity_level == 3
    assert [blob.shape for blob in session.blobs] == [(1, 3, 64, 64)] * 3


def test_detect_returns_faces_in_frame_coordinates(runtime):
    backend = InsightFaceBackend(det_size=64)
    faces = backend.detect(_frames(2, 64, 128))  # halved into the 64 x 64 canvas
    assert len(faces) == 2
    for per_frame in faces:
        (face,) = per_frame
        # anchor centre (16, 16) in the canvas, 8 px each way, doubled back to the frame
        assert face.bbox == pytest.approx((16.0, 16.0, 48.0, 48.0))
        assert face.score == pytest.approx(0.9)
        np.testing.assert_allclose(face.landmarks5, [(32.0, 32.0)] * 5)
        assert face.embedding is None
        assert face.yaw is None
        assert all(type(value) is float for value in (*face.bbox, face.score))
        assert face.landmarks5 is not None
        assert all(type(value) is float for point in face.landmarks5 for value in point)


def test_detect_on_no_frames_opens_nothing(runtime):
    assert InsightFaceBackend().detect(np.zeros((0, 8, 8, 3), np.uint8)) == []
    assert runtime.sessions == []


def test_the_detector_sees_what_insightface_gave_it_from_the_frame_in_bgr(runtime):
    frames = _frames(1, 45, 80, seed=3)
    InsightFaceBackend(det_size=64).detect(frames)
    (session,) = runtime.sessions
    bgr = np.ascontiguousarray(frames[0][..., ::-1])
    canvas = np.zeros((64, 64, 3), np.uint8)
    canvas[:36, :64] = cv2.resize(bgr, (64, 36))  # 45 x 80 -> 36 x 64, top-left
    expected = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (64, 64), (127.5,) * 3, swapRB=True)
    assert np.array_equal(session.blobs[0], expected)


def test_the_detection_threshold_comes_from_the_parameters(runtime):
    backend = InsightFaceBackend(det_size=64, min_score=0.95, nms=0.25)
    assert backend.detect(_frames(1, 64, 64)) == [[]]  # the one face scores 0.9


def test_the_suppression_threshold_comes_from_the_parameters(runtime):
    # Two faces one cell (8 px) apart: 17 x 17 pixel boxes sharing 9 x 17 pixels, an overlap of
    # 153 / 425 = 0.36, kept at insightface's 0.4 but suppressed at 0.3.
    runtime.faces = [((2, 2), 0.9), ((3, 2), 0.8)]
    frames = _frames(1, 64, 64)
    assert len(InsightFaceBackend(det_size=64).detect(frames)[0]) == 2
    assert len(InsightFaceBackend(det_size=64, nms=0.3).detect(frames)[0]) == 1


def test_embed_opens_the_recognition_model_only_when_asked(runtime, isolated):
    backend = InsightFaceBackend(det_size=64)
    frames = _frames(1, 64, 64)
    ((face,),) = backend.detect(frames)
    assert len(runtime.sessions) == 1
    embedding = backend.embed(frames[0], face)
    assert [s.path.name for s in runtime.sessions] == ["det_10g.onnx", "w600k_r50.onnx"]
    assert embedding.dtype == np.float32
    assert embedding.shape == (512,)
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0)
    np.testing.assert_allclose(embedding[:3], [0.6, 0.8, 0.0])
    backend.embed(frames[0], face)
    assert len(runtime.sessions) == 2
    assert runtime.sessions[1].blobs[0].shape == (1, 3, 112, 112)


def test_embed_needs_the_five_landmarks(runtime):
    backend = InsightFaceBackend()
    with pytest.raises(ValueError, match="landmarks"):
        backend.embed(_frames(1, 64, 64)[0], Face(bbox=(0, 0, 10, 10), score=0.9))


def test_embed_takes_one_rgb_frame(runtime):
    backend = InsightFaceBackend()
    face = Face(bbox=(0, 0, 10, 10), score=0.9, landmarks5=((1.0, 1.0),) * 5)
    with pytest.raises(ValueError, match=r"\[H, W, 3\]"):
        backend.embed(_frames(1, 64, 64), face)


def test_a_cuda_device_asks_onnxruntime_for_that_gpu(runtime):
    runtime.cuda = True
    backend = InsightFaceBackend(det_size=64, device="cuda:1")
    backend.detect(_frames(1, 64, 64))
    (session,) = runtime.sessions
    assert session.providers == [
        ("CUDAExecutionProvider", {"device_id": 1}),
        "CPUExecutionProvider",
    ]
    assert backend.meta["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_a_cuda_device_without_cuda_support_falls_back_to_the_cpu_with_a_warning(runtime, caplog):
    with caplog.at_level(logging.WARNING):
        backend = InsightFaceBackend(det_size=64, device="cuda:0")
        assert "cuda:0" in caplog.text
        assert "CPU" in caplog.text
        backend.detect(_frames(1, 64, 64))
        backend.embed(_frames(1, 64, 64)[0], backend.detect(_frames(1, 64, 64))[0][0])
    assert [s.providers for s in runtime.sessions] == [["CPUExecutionProvider"]] * 2
    assert caplog.text.count("cuda:0") == 1  # warned once, when the backend was built


def test_meta_records_the_models_their_hashes_and_the_settings(runtime):
    backend = InsightFaceBackend(det_size=256, min_score=0.6, nms=0.35)
    meta = backend.meta
    assert meta["model"] == "buffalo_l"
    assert meta["files"] == insightface_module.MODEL_FILES
    assert meta["weights_license"] == "non-commercial research use only"
    assert (meta["det_size"], meta["min_score"], meta["nms"]) == (256, 0.6, 0.35)
    assert meta["device"] == "cpu"
    assert meta["providers"] == ["CPUExecutionProvider"]
    assert "insightface 0.7.3" in meta["upstream"]
    backend.detect(_frames(1, 64, 64))
    assert backend.meta["providers"] == ["CPUExecutionProvider"]


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"model": "antelopev2"}, "model"),
        ({"det_size": 100}, "det_size"),
        ({"det_size": 0}, "det_size"),
        ({"min_score": 1.5}, "min_score"),
        ({"min_score": -0.1}, "min_score"),
        ({"nms": 0.0}, "nms"),
        ({"nms": 1.2}, "nms"),
    ],
)
def test_insightface_parameters_are_checked(accepted, params, field):
    with pytest.raises(ConfigError, match=field):
        InsightFaceBackend(**params)


def test_detect_without_opencv_names_the_extra(runtime, monkeypatch):
    backend = InsightFaceBackend()
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(InstallationError) as caught:
        backend.detect(_frames(1, 8, 8))
    assert "OpenCV" in caught.value.message
    assert "deepfake-workbench[face-insightface]" in caught.value.hint
    assert runtime.sessions == []


def test_meta_before_loading_reports_cuda_when_onnxruntime_has_it(runtime):
    runtime.cuda = True
    backend = InsightFaceBackend(device="cuda:2")
    assert backend.meta["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert backend.meta["device"] == "cuda:2"
    assert runtime.sessions == []


def test_meta_before_loading_reports_the_cpu_when_onnxruntime_has_no_cuda(runtime):
    backend = InsightFaceBackend(device="cuda:2")
    assert backend.meta["providers"] == ["CPUExecutionProvider"]
    assert backend.meta["device"] == "cuda:2"
    assert runtime.sessions == []


def test_building_without_onnxruntime_names_the_extra(accepted, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(InstallationError) as caught:
        InsightFaceBackend()
    assert "deepfake-workbench[face-insightface]" in caught.value.hint


# ---------------------------------------------------------------------------------- mediapipe

SHORT_BYTES = b"stand-in short-range model"
FULL_BYTES = b"stand-in full-range model"


class FakeMediaPipe(ModuleType):
    """Just enough of mediapipe's tasks API: detectors that record their options and images."""

    def __init__(self, detections: list[Any]) -> None:
        super().__init__("mediapipe")
        self.__spec__ = importlib.machinery.ModuleSpec("mediapipe", None)
        self.detectors: list[Any] = []
        runtime = self

        class Image:
            def __init__(self, *, image_format: str, data: np.ndarray) -> None:
                self.image_format = image_format
                self.data = data

        class BaseOptions:
            def __init__(self, *, model_asset_path: str) -> None:
                self.model_asset_path = model_asset_path

        class FaceDetectorOptions:
            def __init__(self, **options: Any) -> None:
                self.__dict__.update(options)

        class FaceDetector:
            @classmethod
            def create_from_options(cls, options: Any) -> Any:
                detector = cls()
                detector.options = options
                detector.images = []
                detector.close_calls = 0
                runtime.detectors.append(detector)
                return detector

            def detect(self, image: Any) -> SimpleNamespace:
                self.images.append(image)
                return SimpleNamespace(detections=detections)

            def close(self) -> None:
                self.close_calls += 1

        self.Image = Image
        self.ImageFormat = SimpleNamespace(SRGB="srgb")
        self.tasks = SimpleNamespace(
            BaseOptions=BaseOptions,
            vision=SimpleNamespace(
                FaceDetector=FaceDetector,
                FaceDetectorOptions=FaceDetectorOptions,
                RunningMode=SimpleNamespace(IMAGE="image"),
            ),
        )


def _detection(x: int, y: int, width: int, height: int, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        bounding_box=SimpleNamespace(origin_x=x, origin_y=y, width=width, height=height),
        categories=[SimpleNamespace(score=score, index=0)],
        keypoints=[SimpleNamespace(x=0.5, y=0.5)] * 6,
    )


@pytest.fixture
def mediapipe_stub(monkeypatch: pytest.MonkeyPatch, server: Server) -> FakeMediaPipe:
    fake = FakeMediaPipe([_detection(10, 20, 30, 40, 0.75), _detection(1, 2, 3, 4, 0.5)])
    monkeypatch.setitem(sys.modules, "mediapipe", fake)
    server.routes["/short.tflite"] = SHORT_BYTES
    server.routes["/full.tflite"] = FULL_BYTES
    monkeypatch.setattr(
        mediapipe_module,
        "MODELS",
        {
            "short-range": (
                "blaze_face_short_range.tflite",
                f"{server.url}/short.tflite",
                _sha(SHORT_BYTES),
            ),
            "full-range": (
                "blaze_face_full_range.tflite",
                f"{server.url}/full.tflite",
                _sha(FULL_BYTES),
            ),
        },
    )
    return fake


def test_building_mediapipe_loads_nothing(mediapipe_stub, server):
    backend = MediaPipeBackend()
    assert mediapipe_stub.detectors == []
    assert server.requests == []
    assert isinstance(backend, FaceBackend)
    assert (backend.name, backend.version, backend.license) == ("mediapipe", "1", "Apache-2.0")
    assert backend.has_pose is False


def test_mediapipe_downloads_its_model_once_and_converts_detections(
    mediapipe_stub, server, isolated
):
    backend = MediaPipeBackend(min_score=0.6)
    frames = _frames(2, 48, 64)
    faces = backend.detect(frames)
    backend.detect(frames[:1])

    model = isolated.cache / "models" / "mediapipe" / "blaze_face_short_range.tflite"
    assert model.read_bytes() == SHORT_BYTES
    assert server.requests == ["/short.tflite"]
    (detector,) = mediapipe_stub.detectors
    assert detector.options.base_options.model_asset_path == str(model)
    assert detector.options.running_mode == "image"
    assert detector.options.min_detection_confidence == 0.6

    assert (
        faces
        == [
            [
                Face(bbox=(10.0, 20.0, 40.0, 60.0), score=0.75),
                Face(bbox=(1.0, 2.0, 4.0, 6.0), score=0.5),
            ]
        ]
        * 2
    )
    assert all(type(value) is float for value in faces[0][0].bbox)
    assert faces[0][0].landmarks5 is None
    assert [image.image_format for image in detector.images] == ["srgb"] * 3
    assert np.array_equal(detector.images[0].data, frames[0])  # RGB, as it came
    assert detector.images[0].data.flags["C_CONTIGUOUS"]


def test_mediapipe_close_releases_the_detector_once(mediapipe_stub):
    backend = MediaPipeBackend()
    backend.close()  # nothing loaded yet: nothing to release
    backend.detect(_frames(1, 8, 8))
    (detector,) = mediapipe_stub.detectors
    backend.close()
    backend.close()
    assert detector.close_calls == 1
    backend.detect(_frames(1, 8, 8))  # a closed backend opens a fresh detector if used again
    assert len(mediapipe_stub.detectors) == 2
    assert mediapipe_stub.detectors[1].close_calls == 0


def test_mediapipe_full_range_uses_the_other_model(mediapipe_stub, server, isolated):
    MediaPipeBackend(model="full-range").detect(_frames(1, 8, 8))
    assert server.requests == ["/full.tflite"]
    (detector,) = mediapipe_stub.detectors
    assert detector.options.base_options.model_asset_path.endswith("blaze_face_full_range.tflite")


def test_mediapipe_uses_a_verified_model_already_in_the_cache(mediapipe_stub, server, isolated):
    _put(isolated.cache / "models" / "mediapipe", "blaze_face_short_range.tflite", SHORT_BYTES)
    MediaPipeBackend().detect(_frames(1, 8, 8))
    assert server.requests == []


def test_mediapipe_on_no_frames_loads_nothing(mediapipe_stub, server):
    assert MediaPipeBackend().detect(np.zeros((0, 8, 8, 3), np.uint8)) == []
    assert mediapipe_stub.detectors == []
    assert server.requests == []


def test_mediapipe_meta_names_the_model_file_and_its_hash(mediapipe_stub):
    meta = MediaPipeBackend(model="full-range", min_score=0.4).meta
    assert meta["model"] == "full-range"
    assert meta["file"] == "blaze_face_full_range.tflite"
    assert meta["sha256"] == _sha(FULL_BYTES)
    assert meta["min_score"] == 0.4
    assert meta["weights_license"] == "Apache-2.0"


def test_the_real_mediapipe_models_are_pinned():
    base = "https://storage.googleapis.com/mediapipe-models/face_detector"
    expected = {
        "short-range": (
            "blaze_face_short_range.tflite",
            f"{base}/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
        ),
        "full-range": (
            "blaze_face_full_range.tflite",
            f"{base}/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
            "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b",
        ),
    }
    assert expected == mediapipe_module.MODELS


@pytest.mark.parametrize(
    ("params", "field"),
    [({"model": "short"}, "model"), ({"min_score": 2.0}, "min_score")],
)
def test_mediapipe_parameters_are_checked(params, field):
    with pytest.raises(ConfigError, match=field):
        MediaPipeBackend(**params)


def test_a_misspelt_mediapipe_model_gets_a_suggestion():
    with pytest.raises(ConfigError, match="short-range"):
        MediaPipeBackend(model="short-rang")


def test_mediapipe_on_a_cuda_device_warns_that_it_runs_on_the_cpu(mediapipe_stub, caplog):
    with caplog.at_level(logging.WARNING):
        MediaPipeBackend(device="cuda:0")
    assert "cuda:0" in caplog.text
    assert "CPU" in caplog.text


def test_mediapipe_without_the_library_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    with pytest.raises(InstallationError) as caught:
        MediaPipeBackend().detect(_frames(1, 8, 8))
    assert "deepfake-workbench[face-mediapipe]" in caught.value.hint


# ---------------------------------------------------------------------------------- registry


def test_the_three_backends_are_registered_with_what_they_need():
    registry = get_registry("face_backends")
    entries = {entry.key: entry for entry in registry.entries()}
    assert entries["center"].target == "dfwb.preprocess.face.backends.center:CenterBackend"
    assert entries["center"].requires == ()
    assert entries["insightface"].target == (
        "dfwb.preprocess.face.backends.insightface:InsightFaceBackend"
    )
    assert entries["insightface"].requires == ("onnxruntime", "cv2")
    assert entries["mediapipe"].target == (
        "dfwb.preprocess.face.backends.mediapipe:MediaPipeBackend"
    )
    assert entries["mediapipe"].requires == ("mediapipe",)


def test_the_registry_builds_center_with_a_device():
    backend = get_registry("face_backends").build("center", device="cpu")
    assert isinstance(backend, CenterBackend)


def test_the_registry_builds_insightface_from_profile_parameters(runtime):
    backend = get_registry("face_backends").build(
        "insightface", model="buffalo_l", det_size=256, min_score=0.5, device="cpu"
    )
    assert isinstance(backend, InsightFaceBackend)
    assert backend.meta["det_size"] == 256


@pytest.mark.parametrize(
    ("key", "module", "extra"),
    [
        ("insightface", "onnxruntime", "face-insightface"),
        ("mediapipe", "mediapipe", "face-mediapipe"),
    ],
)
def test_building_a_backend_without_its_extra_names_the_extra(monkeypatch, key, module, extra):
    from dfwb.core import registry as registry_module

    real = registry_module._is_installed
    monkeypatch.setattr(
        registry_module, "_is_installed", lambda name: name != module and real(name)
    )
    with pytest.raises(InstallationError) as caught:
        get_registry("face_backends").build(key, device="cpu")
    assert f"deepfake-workbench[{extra}]" in caught.value.hint
