"""Loading configs: YAML files, ``extends`` chains, overrides, interpolation, validation.

``extends`` references are resolved relative to the file that contains them, or as
``<package>://<path>`` resources inside an installed package (``dfwb://templates/x.yaml`` is the
shipped template ``x.yaml``). Packages are located without being imported, so a config can never
run code.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dfwb.core.config._yaml import load_yaml
from dfwb.core.config.interp import interpolate
from dfwb.core.config.merge import deep_merge
from dfwb.core.config.overrides import apply_overrides
from dfwb.core.config.schema import ConfigModel, check_components, validate_config
from dfwb.core.errors import ConfigError
from dfwb.core.hashing import fingerprint
from dfwb.core.paths import absolute

__all__ = ["LoadedConfig", "compose", "config_fingerprint", "dump_yaml", "load_config"]

_PACKAGE_REF = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)://(.+)$")


@dataclass(frozen=True)
class _Source:
    key: str  # identity for cycle detection
    path: Path  # file to read
    display: str  # how the user wrote it


@dataclass(frozen=True)
class LoadedConfig:
    """A fully resolved config."""

    model: ConfigModel
    data: dict[str, Any]  # resolved, with defaults filled in (what config.resolved.yaml holds)
    sources: tuple[str, ...]  # files merged, base first
    fingerprint: str

    @property
    def schema(self) -> str:
        return str(self.data["schema"])


def _source_for(ref: str, base_dir: Path) -> _Source:
    match = _PACKAGE_REF.match(ref)
    if match is None:
        path = Path(ref).expanduser()
        path = path if path.is_absolute() else base_dir / path
        return _Source(str(absolute(path)), path, ref)
    package, relative = match.group(1).replace("-", "_"), match.group(2)
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        spec = None
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    if not locations:
        raise ConfigError(
            f"extends: {ref!r}: package {package!r} is not installed",
            hint="install the package that provides this config, or fix the reference",
        )
    root = Path(locations[0]).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ConfigError(
            f"extends: {ref!r} points outside package {package!r}", hint="remove '..' from the path"
        )
    hint = (
        "run `dfwb config templates`" if package == "dfwb" else "check the path inside the package"
    )
    if not path.is_file():
        raise ConfigError(f"extends: {ref!r} does not exist", hint=hint)
    return _Source(f"{package}://{relative}", path, ref)


def _read(source: _Source) -> dict[str, Any]:
    try:
        text = source.path.read_text("utf-8")
    except FileNotFoundError:
        raise ConfigError(
            f"config file not found: {source.display}", hint="check the path"
        ) from None
    try:
        document = load_yaml(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark is not None else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(
            f"{source.display}: invalid YAML{where}: {problem}",
            hint="check indentation and quoting",
        ) from None
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ConfigError(
            f"{source.display}: the top level must be a mapping",
            hint="start the file with 'schema: ...'",
        )
    return document


def _linearise(
    source: _Source,
    stack: tuple[str, ...],
    order: list[tuple[_Source, dict[str, Any]]],
    seen: set[str],
) -> None:
    """Collect the chain depth-first, left to right, parents before children, each file once."""
    document = _read(source)
    extends = document.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list) or not all(isinstance(e, str) for e in extends):
        raise ConfigError(
            f"{source.display}: 'extends' must be a list of references", hint="extends: [base.yaml]"
        )
    for ref in extends:
        parent = _source_for(ref, source.path.parent)
        if parent.key in stack:
            chain = " -> ".join([*stack, parent.key])
            raise ConfigError(
                f"extends cycle: {chain}", hint="remove one of the extends references"
            )
        if parent.key not in seen:
            _linearise(parent, (*stack, parent.key), order, seen)
    if source.key not in seen:
        seen.add(source.key)
        order.append((source, document))


def compose(path: str | os.PathLike[str]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Read ``path`` and merge its ``extends`` chain; returns the data and the files, base first.

    The chain is linearised first (depth-first, left to right, parents before children, each file
    once) and the files are then merged in that order. So in a diamond the shared base is applied
    once, and a later file's ``+key`` appends to what earlier files set.
    """
    source = _Source(str(absolute(path)), Path(path), str(path))
    order: list[tuple[_Source, dict[str, Any]]] = []
    _linearise(source, (source.key,), order, set())
    schemas = {str(doc["schema"]) for _, doc in order if "schema" in doc}
    if len(schemas) > 1:
        raise ConfigError(
            f"{path}: the extends chain mixes schemas: {', '.join(sorted(schemas))}",
            hint="every file in a chain must use the same schema",
        )
    data: dict[str, Any] = {}
    for _, document in order:
        data = deep_merge(data, document)
    return data, tuple(item.display for item, _ in order)


def config_fingerprint(model: ConfigModel, data: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``data`` without the model's run-only fields."""
    trimmed = copy.deepcopy(dict(data))
    for path in getattr(type(model), "fingerprint_exclude", ()):
        node: Any = trimmed
        for part in path[:-1]:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return fingerprint(trimmed)


def load_config(
    path: str | os.PathLike[str],
    overrides: Sequence[str] = (),
    *,
    env: Mapping[str, str] | None = None,
    check_registries: bool = False,
) -> LoadedConfig:
    """Compose, override, interpolate and validate a config file.

    Args:
        path: The config file.
        overrides: ``dotted.path=value`` strings, applied after composition, before interpolation.
        env: Environment for ``${env:…}`` (defaults to ``os.environ``).
        check_registries: Also check every component against the installed plugins.
    """
    data, sources = compose(path)
    data = apply_overrides(data, overrides)
    data = interpolate(data, env=os.environ if env is None else env)
    model = validate_config(data, source=str(path))
    if check_registries:
        from dfwb.core.plugins import REGISTRY_NAMES, get_registry

        check_components(model, {name: get_registry(name) for name in REGISTRY_NAMES})
    resolved = model.model_dump(mode="json", by_alias=True)
    return LoadedConfig(model, resolved, sources, config_fingerprint(model, resolved))


def dump_yaml(data: Mapping[str, Any]) -> str:
    """YAML text for a resolved config (key order preserved)."""
    return yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True, default_flow_style=False)
