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
