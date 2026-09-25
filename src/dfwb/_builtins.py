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
    ("celebdf-v1", "celebdf:CelebDFv1Builder", "Celeb-DF v1", "Celeb-DF-v1"),
    ("celebdf-v2", "celebdf:CelebDFv2Builder", "Celeb-DF v2", "Celeb-DF-v2"),
    ("celebdf-v3", "celebdf:CelebDFv3Builder", "Celeb-DF v3", "Celeb-DF-v3"),
    ("dfdc", "dfdc:DFDCBuilder", "DFDC", "DFDC"),
    ("dfdc-p", "dfdcp:DFDCPreviewBuilder", "DFDC Preview", "DFDC-P"),
    (
        "deeperforensics",
        "deeperforensics:DeeperForensicsBuilder",
        "DeeperForensics-1.0",
        "DeeperForensics-1.0",
    ),
    ("wilddeepfake", "wilddeepfake:WildDeepfakeBuilder", "WildDeepfake", "WildDeepfake"),
    ("ffiw10k", "ffiw10k:FFIW10KBuilder", "FFIW-10K", "FFIW10K"),
    ("kodf", "kodf:KoDFBuilder", "KoDF", "KoDF"),
    ("dfdm", "dfdm:DFDMBuilder", "DFDM", "DFDM"),
    (
        "fakeavceleb",
        "fakeavceleb:FakeAVCelebBuilder",
        "FakeAVCeleb v1.2",
        "FakeAVCeleb-v1_2",
    ),
    ("polyglotfake", "polyglotfake:PolyGlotFakeBuilder", "PolyGlotFake", "PolyGlotFake"),
    ("deepspeak-v1", "deepspeak:DeepSpeakV1Builder", "DeepSpeak v1", "DeepSpeak-v1"),
    ("deepspeak-v2", "deepspeak:DeepSpeakV2Builder", "DeepSpeak v2", "DeepSpeak-v2"),
    ("idforge-v1", "idforge:IDForgeV1Builder", "IDForge-v1", "IDForge-v1"),
    ("lav-df", "lavdf:LAVDFBuilder", "LAV-DF", "LAV-DF"),
    (
        "av-deepfake1m-pp",
        "avdeepfake1mpp:AVDeepfake1MPPBuilder",
        "AV-Deepfake1M++",
        "AV-Deepfake1M++",
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
