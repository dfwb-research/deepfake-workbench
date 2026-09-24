"""Plugin discovery (contract C1).

Every distribution that declares an entry point in the ``dfwb.plugins`` group names a function
``register(api)``. The framework calls it once with :data:`api`, the :class:`PluginAPI` that holds
all registries. The framework's own built-ins load the same way through ``dfwb.builtins``.

Loading is lazy (the first read of any registry triggers it), idempotent and isolated: a plugin
whose ``register()`` raises is recorded as ``failed``, its partial registrations are rolled back,
and every other plugin still loads.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from typing import Any, Final

from dfwb.core.registry import (
    BUILTIN_PROVIDER,
    Entry,
    Registry,
    canonical_name,
    catalogue_requirement,
    providing,
)

__all__ = [
    "BUILTINS_GROUP",
    "DATA_REGISTRIES",
    "ENTRY_POINT_GROUP",
    "PLUGIN_API_VERSION",
    "REGISTRY_NAMES",
    "PluginAPI",
    "PluginRecord",
    "PluginReport",
    "PluginStatus",
    "api",
    "api_compatible",
    "entries_by_registry",
    "get_registry",
    "load_plugins",
    "reset",
]

PLUGIN_API_VERSION: Final = (1, 0)
ENTRY_POINT_GROUP: Final = "dfwb.plugins"
BUILTINS_GROUP: Final = "dfwb.builtins"

REGISTRY_NAMES: Final = (
    "layers",
    "transforms",
    "backbones",
    "temporal_pools",
    "heads",
    "losses",
    "metrics",
    "eval_suites",
    "face_backends",
    "inventory_builders",
    "protocol_packs",
    "detectors",
    "detector_sources",
    "callbacks",
)
DATA_REGISTRIES: Final = frozenset({"eval_suites", "protocol_packs"})

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginAPI:
    """The only view of the framework a plugin gets. Registries may be added, never removed."""

    version: tuple[int, int]
    layers: Registry[Any]
    transforms: Registry[Any]
    backbones: Registry[Any]
    temporal_pools: Registry[Any]
    heads: Registry[Any]
    losses: Registry[Any]
    metrics: Registry[Any]
    eval_suites: Registry[Any]
    face_backends: Registry[Any]
    inventory_builders: Registry[Any]
    protocol_packs: Registry[Any]
    detectors: Registry[Any]
    detector_sources: Registry[Any]
    callbacks: Registry[Any]


class PluginStatus(StrEnum):
    """Outcome of loading one entry point."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"


@dataclass(frozen=True)
class PluginRecord:
    """What happened to one entry point."""

    name: str
    group: str
    provider: str
    version: str | None
    target: str
    status: PluginStatus
    reason: str | None = None
    entries: tuple[str, ...] = ()  # "registry/key" registered (or rolled back, when failed)


@dataclass(frozen=True)
class PluginReport:
    """The result of :func:`load_plugins`."""

    records: tuple[PluginRecord, ...]

    def with_status(self, status: PluginStatus) -> tuple[PluginRecord, ...]:
        """Records with the given status, in load order."""
        return tuple(r for r in self.records if r.status is status)


_lock = threading.RLock()
_report: PluginReport | None = None
_loading_thread: int | None = None  # the thread running load_plugins(), while it runs


def _ensure_loaded() -> None:
    # A register() reading a registry must not re-enter loading; any other thread waits for it.
    if _report is None and _loading_thread != threading.get_ident():
        load_plugins()


def _unknown_key_hint(registry_name: str, key: str) -> str | None:
    report = _report
    failed = report.with_status(PluginStatus.FAILED) if report else ()
    for record in failed:
        if f"{registry_name}/{key}" in record.entries:
            return (
                f"{registry_name} {key!r} comes from plugin {record.name!r} ({record.provider}), "
                f"which failed to load ({record.reason}); run `dfwb plugins list --all`"
            )
    requirement = catalogue_requirement(registry_name, key)
    if requirement is not None:
        provider = canonical_name(re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0])
        for record in report.records if report else ():
            if record.provider == provider and record.status is not PluginStatus.OK:
                return (
                    f"{registry_name} {key!r} is provided by {provider}, which is installed but "
                    f"{record.status.value} ({record.reason}); run `dfwb plugins list --all`"
                )
        return f'{registry_name} "{key}" is provided by: pip install {requirement}'
    if failed:
        return (
            f"{len(failed)} plugin(s) failed to load and may provide it; "
            "run `dfwb plugins list --all`"
        )
    return None


def _make_api() -> PluginAPI:
    registries = {
        name: Registry[Any](
            name,
            kind="data" if name in DATA_REGISTRIES else "code",
            on_access=_ensure_loaded,
            unknown_hint=_unknown_key_hint,
        )
        for name in REGISTRY_NAMES
    }
    return PluginAPI(version=PLUGIN_API_VERSION, **registries)


api: Final = _make_api()


def get_registry(name: str) -> Registry[Any]:
    """Look up a registry by name (``temporal_pools`` or ``temporal-pools``)."""
    from dfwb.core.errors import UnknownKeyError, did_you_mean

    normalised = name.replace("-", "_").lower()
    if normalised not in REGISTRY_NAMES:
        raise UnknownKeyError(
            f"unknown registry {name!r}{did_you_mean(normalised, REGISTRY_NAMES)}",
            hint="registries: " + ", ".join(REGISTRY_NAMES),
        )
    reg: Registry[Any] = getattr(api, normalised)
    return reg


def _entry_points(group: str) -> list[metadata.EntryPoint]:
    return list(metadata.entry_points(group=group))


_CLAUSE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(\d+)(?:\.(\d+))?\s*$")


def api_compatible(specifier: str, version: tuple[int, int] = PLUGIN_API_VERSION) -> bool:
    """Whether ``version`` satisfies a plugin's ``DFWB_PLUGIN_API`` specifier, e.g. ``">=1.0,<2"``.

    Raises:
        ValueError: The specifier cannot be parsed.
    """
    clauses = [c for c in specifier.split(",") if c.strip()]
    if not clauses:
        raise ValueError("empty specifier")
    for clause in clauses:
        match = _CLAUSE.match(clause)
        if match is None:
            raise ValueError(f"cannot parse {clause.strip()!r}")
        op, major, minor = match.group(1), int(match.group(2)), int(match.group(3) or 0)
        other = (major, minor)
        ok = {
            ">=": version >= other,
            "<=": version <= other,
            ">": version > other,
            "<": version < other,
            "==": version == other,
            "!=": version != other,
        }[op]
        if not ok:
            return False
    return True


def _describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text.splitlines()[0]}" if text else type(exc).__name__


def _provider_of(ep: metadata.EntryPoint, group: str) -> tuple[str, str | None]:
    if group == BUILTINS_GROUP:
        return BUILTIN_PROVIDER, _dist_version(ep)
    if ep.dist is not None:
        return canonical_name(ep.dist.name), ep.dist.version
    return canonical_name(ep.value.partition(":")[0].partition(".")[0]), None


def _dist_version(ep: metadata.EntryPoint) -> str | None:
    return ep.dist.version if ep.dist is not None else None


def _disabled_names() -> set[str]:
    raw = os.environ.get("DFWB_PLUGINS_DISABLE", "")
    return {canonical_name(part.strip()) for part in raw.split(",") if part.strip()}


def _load_group(group: str) -> list[PluginRecord]:
    records: list[PluginRecord] = []
    all_off = os.environ.get("DFWB_PLUGINS", "").strip().lower() == "none"
    disabled = _disabled_names()
    points = sorted(_entry_points(group), key=lambda ep: (_provider_of(ep, group)[0], ep.name))
    for ep in points:
        provider, version = _provider_of(ep, group)
        base: dict[str, Any] = {
            "name": ep.name,
            "group": group,
            "provider": provider,
            "version": version,
            "target": ep.value,
        }
        if group == ENTRY_POINT_GROUP:
            if all_off:
                records.append(
                    PluginRecord(**base, status=PluginStatus.DISABLED, reason="DFWB_PLUGINS=none")
                )
                continue
            if canonical_name(ep.name) in disabled or provider in disabled:
                records.append(
                    PluginRecord(
                        **base, status=PluginStatus.DISABLED, reason="DFWB_PLUGINS_DISABLE"
                    )
                )
                continue
        try:
            register = ep.load()
        except Exception as exc:
            records.append(PluginRecord(**base, status=PluginStatus.FAILED, reason=_describe(exc)))
            continue
        declared = None
        for module_name in (
            ep.value.partition(":")[0].strip(),
            getattr(register, "__module__", ""),
        ):
            declared = getattr(sys.modules.get(module_name or ""), "DFWB_PLUGIN_API", None)
            if declared is not None:
                break
        if declared is not None:
            try:
                compatible = api_compatible(str(declared))
            except ValueError as exc:
                reason = f"invalid DFWB_PLUGIN_API {declared!r}: {exc}"
                records.append(PluginRecord(**base, status=PluginStatus.FAILED, reason=reason))
                continue
            if not compatible:
                have = ".".join(map(str, PLUGIN_API_VERSION))
                reason = f"needs plugin API {declared}, this dfwb provides {have}"
                _log.warning("skipping plugin %r (%s): %s", ep.name, provider, reason)
                records.append(PluginRecord(**base, status=PluginStatus.SKIPPED, reason=reason))
                continue
        with providing(provider) as added:
            try:
                register(api)
            except Exception as exc:
                keys = tuple(f"{reg.name}/{entry.key}" for reg, entry in added)
                for reg, entry in added:
                    reg._remove(entry)
                records.append(
                    PluginRecord(
                        **base, status=PluginStatus.FAILED, reason=_describe(exc), entries=keys
                    )
                )
                continue
        keys = tuple(f"{reg.name}/{entry.key}" for reg, entry in added)
        records.append(PluginRecord(**base, status=PluginStatus.OK, entries=keys))
    return records


def load_plugins() -> PluginReport:
    """Discover and register all plugins once; later calls return the same report."""
    global _report, _loading_thread
    with _lock:
        if _report is not None:
            return _report
        _loading_thread = threading.get_ident()
        try:
            records = _load_group(BUILTINS_GROUP) + _load_group(ENTRY_POINT_GROUP)
            _report = PluginReport(tuple(records))
        finally:
            _loading_thread = None
        return _report


def reset() -> None:
    """Forget every registration and the load report. For tests only."""
    global _report
    with _lock:
        for name in REGISTRY_NAMES:
            reg: Registry[Any] = getattr(api, name)
            reg._clear()
        _report = None


def entries_by_registry() -> dict[str, list[Entry]]:
    """All entries of every registry, loading plugins first."""
    load_plugins()
    return {name: get_registry(name).entries() for name in REGISTRY_NAMES}
