"""Resolves a detector URI (``<scheme>:<rest>``) into a C4 :class:`~dfwb.core.detector.Detector`,
through the ``detector_sources`` registry.

A scheme is a registry key (``run``, later ``zoo``, ``py``, ``hf``); ``<rest>`` is passed verbatim
to whatever that scheme's loader expects. This module never imports a source's own package
(``dfwb.models`` for ``run:``, and so on) -- the registry loads it lazily, by import path, exactly
as it does for any other pluggable component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dfwb.core.errors import UnknownKeyError
from dfwb.core.plugins import get_registry

if TYPE_CHECKING:
    from dfwb.core.detector import Detector

__all__ = ["resolve_detector"]


def resolve_detector(uri: str) -> Detector:
    """Resolve ``<scheme>:<rest>`` to a :class:`~dfwb.core.detector.Detector`.

    Raises:
        UnknownKeyError: ``uri`` has no ``<scheme>:`` prefix, or names a scheme that is not
            registered under ``detector_sources``; the error lists every known scheme.
    """
    registry = get_registry("detector_sources")
    scheme, sep, rest = uri.partition(":")
    known = registry.keys()
    if not sep or scheme not in registry:
        raise UnknownKeyError(
            f"detector: {uri!r} does not start with a known <scheme>:, got scheme {scheme!r}",
            hint="known schemes: " + (", ".join(known) or "(none registered)"),
        )
    loader = registry.load(scheme)
    detector: Detector = loader(rest)
    return detector
