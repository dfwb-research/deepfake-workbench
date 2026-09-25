"""Checkpoints: ``model.safetensors`` (weights only) plus ``detector.json`` (meta, the resolved
``model:`` config, and the registry providers that built it). Nothing here executes arbitrary
code as it loads: only plain tensors and JSON ever reach disk or come back off it.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from dfwb.core.config.schema import ModelSection
from dfwb.core.errors import InstallationError
from dfwb.core.plugins import PluginStatus, get_registry, load_plugins
from dfwb.core.registry import BUILTIN_PROVIDER, LOCAL_PROVIDER
from dfwb.models.detector import AssembledDetector, build_detector

__all__ = ["load", "save", "weights_path"]

_MODEL_FILE = "model.safetensors"
_META_FILE = "detector.json"
_FRAMEWORK_DISTRIBUTION = "deepfake-workbench"

# (registry name, ModelSection field name) for the components every detector always has.
_COMPONENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("backbones", "backbone"),
    ("temporal_pools", "temporal_pool"),
    ("heads", "head"),
)


def _component_entry(
    registry_name: str, key: str, versions: Mapping[str, str | None]
) -> dict[str, Any]:
    entry = get_registry(registry_name).entry(key)
    return {
        "registry": registry_name,
        "key": entry.key,
        "provider": entry.provider,
        "version": versions.get(entry.provider),
    }


def _components(section: ModelSection) -> list[dict[str, Any]]:
    """``{registry, key, provider, version}`` for every registry key this config uses."""
    report = load_plugins()
    versions = {r.provider: r.version for r in report.records if r.status is PluginStatus.OK}
    components = [
        _component_entry(registry_name, getattr(section, field_name).name, versions)
        for registry_name, field_name in _COMPONENT_FIELDS
    ]
    if section.stem is not None:
        components.append(_component_entry("layers", section.stem.name, versions))
    return components


def weights_path(directory: str | Path) -> Path:
    """The ``model.safetensors`` file inside a checkpoint ``directory`` -- for a caller (the
    ``run:`` detector source) that wants to hash or otherwise inspect the exact weights file
    without knowing this module's own layout."""
    return Path(directory) / _MODEL_FILE


def save(
    directory: str | Path, detector: AssembledDetector, model_cfg: ModelSection | Mapping[str, Any]
) -> None:
    """Write ``model.safetensors`` and ``detector.json`` to ``directory`` (created if needed)."""
    section = (
        model_cfg if isinstance(model_cfg, ModelSection) else ModelSection.model_validate(model_cfg)
    )
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    # .clone() breaks any aliasing (e.g. tied weights) safetensors refuses to save in place.
    state = {name: tensor.detach().cpu().clone() for name, tensor in detector.state_dict().items()}
    save_file(state, str(out / _MODEL_FILE))
    payload = {
        "meta": dataclasses.asdict(detector.meta),
        "model": section.model_dump(),
        "components": _components(section),
    }
    (out / _META_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def _installed_version(provider: str) -> tuple[bool, str | None]:
    """``(installed, version)`` for ``provider``.

    A ``local`` registration (added directly at runtime, not through an installed plugin) has no
    distribution to check at all, so it always counts as installed, with no version to compare.
    """
    if provider == LOCAL_PROVIDER:
        return True, None
    name = _FRAMEWORK_DISTRIBUTION if provider == BUILTIN_PROVIDER else provider
    try:
        return True, metadata.version(name)
    except metadata.PackageNotFoundError:
        return False, None


def _major(version: str) -> str:
    return version.split(".", 1)[0]


def _check_providers(components: list[dict[str, Any]]) -> None:
    for component in components:
        registry_name, key, provider, recorded = (
            component["registry"],
            component["key"],
            component["provider"],
            component["version"],
        )
        where = f"{registry_name}/{key}"
        installed, version = _installed_version(provider)
        if not installed:
            raise InstallationError(
                f"{where}: provider {provider!r} is not installed",
                hint=f"install the plugin that provides {provider!r} (pip install {provider})",
            )
        if recorded is not None and version is not None and _major(version) != _major(recorded):
            raise InstallationError(
                f"{where}: provider {provider!r} is installed at {version}, but this checkpoint "
                f"was saved with major version {_major(recorded)}",
                hint=f'pip install "{provider}=={_major(recorded)}.*"',
            )


def _restore_input_spec(data: Mapping[str, Any]) -> dict[str, Any]:
    def _tuple_or_none(value: Any) -> tuple[Any, ...] | None:
        return None if value is None else tuple(value)

    return {
        **data,
        "size": tuple(data["size"]),
        "value_range": tuple(data["value_range"]),
        "mean": _tuple_or_none(data.get("mean")),
        "std": _tuple_or_none(data.get("std")),
    }


def load(directory: str | Path) -> AssembledDetector:
    """Rebuild the detector recorded in ``directory``'s ``detector.json``, load its weights, and
    return it in eval mode.

    Raises:
        InstallationError: A recorded component's provider is missing, or installed at an
            incompatible major version.
    """
    src = Path(directory)
    payload = json.loads((src / _META_FILE).read_text("utf-8"))
    _check_providers(payload["components"])
    section = ModelSection.model_validate(payload["model"])
    overrides = _restore_input_spec(payload["meta"]["input"])
    detector = build_detector(
        section, input_spec_overrides=overrides, source=payload["meta"]["source"]
    )
    state = load_file(str(src / _MODEL_FILE))
    detector.load_state_dict(state, strict=True)
    detector.eval()
    return detector
