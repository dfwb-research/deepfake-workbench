"""Registers the framework's built-in components through the same plugin API as any plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from dfwb.core.plugins import PluginAPI

# The package every built-in inventory builder lives in.
_BUILDERS_PACKAGE: Final = "dfwb.preprocess.inventory.builders"

# Built-in inventory builders, one row each:
#   (dataset id, "<module>:<Class>" inside dfwb.preprocess.inventory.builders,
#    display name, the dataset's expected folder under a datasets root)
# The folder is stored as registry metadata, so `dfwb datasets list` and `dfwb doctor` can
# locate every dataset without importing a single builder module.
INVENTORY_BUILDERS: tuple[tuple[str, str, str, str], ...] = (
    ("ffpp", "ffpp:FaceForensicsBuilder", "FaceForensics++", "FaceForensics++"),
    ("dfd", "dfd:DeepFakeDetectionBuilder", "DeepFakeDetection", "FaceForensics++"),
    ("uadfv", "uadfv:UADFVBuilder", "UADFV", "UADFV"),
)

# The package every built-in face backend lives in.
_FACE_BACKENDS_PACKAGE: Final = "dfwb.preprocess.face.backends"

# Built-in face backends, one row each:
#   (key, "<module>:<Class>" inside dfwb.preprocess.face.backends, summary,
#    the modules it needs, which name the extra to install when one is missing)
FACE_BACKENDS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "center",
        "center:CenterBackend",
        "No detection: the centred square of each frame (pre-cropped data, smoke tests)",
        (),
    ),
    (
        "insightface",
        "insightface:InsightFaceBackend",
        "insightface buffalo_l detection and embeddings via onnxruntime "
        "(weights: non-commercial research only)",
        ("onnxruntime", "cv2"),
    ),
    (
        "mediapipe",
        "mediapipe:MediaPipeBackend",
        "MediaPipe BlazeFace detection (Apache-2.0)",
        ("mediapipe",),
    ),
)


def register(api: PluginAPI) -> None:
    """Register built-in components.

    Built-ins are added here as their layers are implemented; each one is an import-path target,
    so registering stays cheap and imports nothing heavy.
    """
    for dataset_id, target, name, folder in INVENTORY_BUILDERS:
        api.inventory_builders.add(
            dataset_id, target=f"{_BUILDERS_PACKAGE}.{target}", summary=name, folder=folder
        )
    for key, target, summary, requires in FACE_BACKENDS:
        api.face_backends.add(
            key, target=f"{_FACE_BACKENDS_PACKAGE}.{target}", summary=summary, requires=requires
        )
