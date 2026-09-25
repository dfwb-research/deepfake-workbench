"""Resolves a detector URI (``<scheme>:<rest>``) into a C4 :class:`~dfwb.core.detector.Detector`,
through the ``detector_sources`` registry.

A scheme is a registry key (``run``, ``py``, later ``zoo``, ``hf``); ``<rest>`` is passed verbatim
to whatever that scheme's loader expects. This module never imports a source's own package
(``dfwb.models`` for ``run:``, and so on) -- the registry loads it lazily, by import path, exactly
as it does for any other pluggable component.

A loader may set plain attributes on the ``Detector`` it returns, beyond contract C4: a
``checkpoint_sha256`` (``str``, the sha256 of the exact weights file scored) and a
``training_seed`` (``int``, the seed it was trained with) -- ``dfwb.models.source.load_run`` sets
both, from the checkpoint it loads. :func:`load_py`, below, sets a third, ``fingerprint_extra``
(``str``). None of the three is required; :mod:`dfwb.score.harness` reads them duck-typed
(``getattr(detector, "checkpoint_sha256", None)``), so a source that has nothing to say about one
simply leaves it unset.

``py:<module>:<factory>`` (:func:`load_py`) imports the user's own module -- explicit user code,
run with whatever trust anything else importable on this interpreter's path already has -- and
calls the named factory, which must return a C4 ``Detector``. Because the factory's own
``meta.source`` is not a reliable cache identity (it may be left ``None``, or never change across
edits to the user's code), ``load_py`` instead sets ``fingerprint_extra`` to
``<module>:<factory>:<sha256 of the module's source file>``, so an edit to that file changes a
detector's cache key even when its declared ``meta.source`` does not. When the module has no
readable source file (a compiled extension, a namespace package, ...), the sha256 is replaced by a
fresh value on every load -- logged when it happens -- so a detector whose identity cannot be
pinned down is never trusted from a cache either.
"""

from __future__ import annotations

import importlib
import logging
import uuid
from typing import TYPE_CHECKING, Any, Final

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError
from dfwb.core.hashing import sha256_file
from dfwb.core.plugins import get_registry

if TYPE_CHECKING:
    from types import ModuleType

    from dfwb.core.detector import Detector

__all__ = ["load_py", "resolve_detector"]

_log = logging.getLogger(__name__)

# What contract C4 requires of anything a source hands back to the harness.
_DETECTOR_ATTRS: Final = ("meta", "predict", "to")


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


def _require_detector_contract(obj: Any) -> None:
    missing = [name for name in _DETECTOR_ATTRS if not hasattr(obj, name)]
    if missing:
        raise ContractError(
            f"py: factory returned a {type(obj).__name__!r} object, missing {missing} required "
            "by the detector contract",
            hint="the factory must return an object with meta, predict() and to()",
        )


def _module_source_sha256(module: ModuleType) -> str | None:
    """sha256 of ``module``'s own source file, or ``None`` when it has none readable (a compiled
    extension, a namespace package, or a file that has since gone missing or unreadable)."""
    path = getattr(module, "__file__", None)
    if not path:
        return None
    try:
        return sha256_file(path)
    except OSError:
        return None


def _py_fingerprint(module_name: str, factory_name: str, module: ModuleType) -> str:
    file_sha256 = _module_source_sha256(module)
    if file_sha256 is not None:
        return f"{module_name}:{factory_name}:{file_sha256}"
    _log.warning(
        "py: %s: module source file is not readable; caching is disabled for this detector run",
        module_name,
    )
    return f"{module_name}:{factory_name}:{uuid.uuid4().hex}"


def load_py(ref: str) -> Detector:
    """The ``py:`` detector source: ``py:<module>:<factory>`` imports ``<module>`` and calls its
    module-level ``<factory>()`` with no arguments, which must return a C4 ``Detector``.

    This runs the user's own code: whatever ``<module>`` does at import time, and whatever
    ``<factory>`` does when called, executes exactly as if the caller had imported and called it
    directly.

    Sets ``fingerprint_extra`` on the returned detector (see the module docstring); never sets
    ``checkpoint_sha256`` (a module's source sha is not a checkpoint sha) or ``training_seed``.

    Raises:
        ConfigError: ``ref`` is not ``<module>:<factory>``, ``<module>`` cannot be imported, or it
            has no callable ``<factory>``.
        ContractError: the factory's return value is missing ``meta``, ``predict`` or ``to``.
    """
    module_name, sep, factory_name = ref.partition(":")
    if not sep or not module_name or not factory_name:
        raise ConfigError(
            f"py: {ref!r} must be <module>:<factory>",
            hint="example: py:my_pkg.detectors:make_detector",
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(
            f"py: cannot import {module_name!r}: {exc}",
            hint="the module must be importable on this interpreter's path",
        ) from None
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ConfigError(
            f"py: {module_name!r} has no callable {factory_name!r}",
            hint="the factory is a module-level function (or class) that returns a detector",
        )
    detector: Detector = factory()
    _require_detector_contract(detector)
    # Duck-typed, like checkpoint_sha256 (see the module docstring): not part of contract C4, so
    # not on the Detector protocol itself -- set dynamically rather than fought past with mypy.
    fingerprint_extra = _py_fingerprint(module_name, factory_name, module)
    setattr(detector, "fingerprint_extra", fingerprint_extra)  # noqa: B010
    return detector
