"""Protocol packs (contract C3a): loading, querying, verification, materializing and pack building.

Every name below is resolved lazily on first attribute access (``__getattr__``), so
``import dfwb.protocols`` never imports pydantic models, YAML or any pack's code -- it stays as
cheap as ``import dfwb.protocols.refs`` alone.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dfwb.protocols.materialize import materialize
    from dfwb.protocols.protocol import LabelMapping, Protocol, ProtocolInfo, load
    from dfwb.protocols.protocol import list_protocols as list
    from dfwb.protocols.refs import ProtocolRef, parse_ref
    from dfwb.protocols.verify import CoverageReport, verify, write_report

__all__ = [
    "CoverageReport",
    "LabelMapping",
    "Protocol",
    "ProtocolInfo",
    "ProtocolRef",
    "list",
    "load",
    "materialize",
    "parse_ref",
    "verify",
    "write_report",
]

# name -> the module holding it, for every export whose attribute name matches its own name there.
_LAZY = {
    "Protocol": "dfwb.protocols.protocol",
    "ProtocolInfo": "dfwb.protocols.protocol",
    "LabelMapping": "dfwb.protocols.protocol",
    "load": "dfwb.protocols.protocol",
    "materialize": "dfwb.protocols.materialize",
    "ProtocolRef": "dfwb.protocols.refs",
    "parse_ref": "dfwb.protocols.refs",
    "CoverageReport": "dfwb.protocols.verify",
    "verify": "dfwb.protocols.verify",
    "write_report": "dfwb.protocols.verify",
}
# name -> (module, attribute), for exports whose public name differs from the underlying one
# (``list`` would otherwise shadow the builtin inside ``protocol.py`` itself).
_RENAMED = {"list": ("dfwb.protocols.protocol", "list_protocols")}


def __getattr__(name: str) -> Any:
    if name in _RENAMED:
        module_name, attribute = _RENAMED[name]
        return getattr(importlib.import_module(module_name), attribute)
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'dfwb.protocols' has no attribute {name!r}")
