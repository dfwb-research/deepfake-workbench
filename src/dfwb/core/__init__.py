"""dfwb.core: contracts, registries, plugins, config, paths and run metadata.

Attributes are imported on first use, so ``import dfwb.core.errors`` (for example) stays cheap.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dfwb.core import plugins, registry
    from dfwb.core.config import load_config
    from dfwb.core.detector import ClipBatch, Detector, DetectorMeta, DetectorOutput, InputSpec
    from dfwb.core.plugins import PLUGIN_API_VERSION

__all__ = [
    "PLUGIN_API_VERSION",
    "ClipBatch",
    "Detector",
    "DetectorMeta",
    "DetectorOutput",
    "InputSpec",
    "load_config",
    "plugins",
    "registry",
]

_LAZY = {
    "PLUGIN_API_VERSION": "dfwb.core.plugins",
    "load_config": "dfwb.core.config",
    "ClipBatch": "dfwb.core.detector",
    "Detector": "dfwb.core.detector",
    "DetectorMeta": "dfwb.core.detector",
    "DetectorOutput": "dfwb.core.detector",
    "InputSpec": "dfwb.core.detector",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    if name in ("plugins", "registry"):
        return importlib.import_module(f"dfwb.core.{name}")
    raise AttributeError(f"module 'dfwb.core' has no attribute {name!r}")
