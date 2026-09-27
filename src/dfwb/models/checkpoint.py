"""Checkpoints: ``model.safetensors`` (weights only) plus ``detector.json`` (meta, the resolved
``model:`` config, the registry providers that built it, and whatever the backbone needs to
rebuild its architecture offline). Nothing here executes arbitrary code as it loads: only plain
tensors and JSON ever reach disk or come back off it.

Loading never downloads anything: the backbone is rebuilt with ``pretrained`` off (its weights
come from ``model.safetensors`` a moment later) through :meth:`Backbone.from_checkpoint
<dfwb.models.backbone.Backbone.from_checkpoint>`, and the saved ``DetectorMeta`` is restored as
it was saved -- name, version, licence and training data included -- not regenerated from the
software installed now.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from dfwb.core.config.schema import ComponentSpec, ModelSection
from dfwb.core.detector import DetectorMeta, InputSpec
from dfwb.core.errors import InstallationError
from dfwb.core.plugins import PluginStatus, get_registry, load_plugins
from dfwb.core.registry import BUILTIN_PROVIDER, LOCAL_PROVIDER
from dfwb.models.backbone import Backbone
from dfwb.models.detector import AssembledDetector, assemble_detector

__all__ = ["load", "save"]

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
        "backbone_state": _backbone_state(detector.backbone),
    }
    (out / _META_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def _backbone_state(backbone: Any) -> dict[str, Any]:
    """What a backbone records to rebuild itself offline; ``{}`` for one that says nothing.

    A plugin backbone need not subclass :class:`~dfwb.models.backbone.Backbone`, so
    ``checkpoint_state()`` is optional.
    """
    state_of = getattr(backbone, "checkpoint_state", None)
    return dict(state_of()) if callable(state_of) else {}


def _installed_version(provider: str) -> tuple[bool, str | None]:
    """``(installed, version)`` for ``provider``.

    A ``local`` registration (added directly at runtime, not through an installed plugin) has no
    distribution to check at all, so it always counts as installed, with no version to compare.
    """
    if provider == LOCAL_PROVIDER:
        return True, None
    try:
        return True, metadata.version(_distribution(provider))
    except metadata.PackageNotFoundError:
        return False, None


_RELEASE = re.compile(r"(\d+)(?:\.(\d+))?")


def _series(version: str) -> str:
    """The release series a compatible version must share: the major version, or -- below 1.0,
    where a minor release may break things -- ``0.<minor>``."""
    match = _RELEASE.match(version)
    if match is None:
        return version
    major, minor = match.group(1), match.group(2) or "0"
    return f"{major}.{minor}" if major == "0" else major


def _distribution(provider: str) -> str:
    return _FRAMEWORK_DISTRIBUTION if provider == BUILTIN_PROVIDER else provider


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
        if recorded is not None and version is not None and _series(version) != _series(recorded):
            raise InstallationError(
                f"{where}: provider {provider!r} is installed at {version}, but this checkpoint "
                f"was saved with {recorded} (release series {_series(recorded)})",
                hint=f'pip install "{_distribution(provider)}=={_series(recorded)}.*"',
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


def _restore_meta(data: Mapping[str, Any]) -> DetectorMeta:
    return DetectorMeta(
        name=data["name"],
        version=data["version"],
        contract_version=tuple(data["contract_version"]),
        input=InputSpec(**_restore_input_spec(data["input"])),
        license=data["license"],
        weights_license=data.get("weights_license"),
        citation=data.get("citation"),
        source=data.get("source"),
        training_data=tuple(data.get("training_data", ())),
    )


def _accepts(target: Any, name: str) -> bool:
    try:
        return name in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False


def _rebuild_backbone(spec: ComponentSpec, state: Mapping[str, Any]) -> Backbone:
    """The saved backbone's architecture, built without fetching anything: ``pretrained`` is
    switched off wherever the backbone takes it (its weights come from the checkpoint), and a
    :class:`Backbone` subclass rebuilds through its own :meth:`Backbone.from_checkpoint`."""
    registry = get_registry("backbones")
    target = registry.load(spec.name)
    params = dict(spec.params)
    if _accepts(target, "pretrained"):
        params["pretrained"] = False
    kwargs = registry.validate(spec.name, **params)
    if isinstance(target, type) and issubclass(target, Backbone):
        return target.from_checkpoint(kwargs, state)
    backbone: Backbone = target(**kwargs)
    return backbone


def load(directory: str | Path) -> AssembledDetector:
    """Rebuild the detector recorded in ``directory``'s ``detector.json``, load its weights, and
    return it in eval mode, with the ``DetectorMeta`` it was saved with.

    Nothing is downloaded: the backbone is rebuilt with ``pretrained`` off, from the checkpoint's
    own record of its architecture where it keeps one (see the module docstring).

    Raises:
        InstallationError: A recorded component's provider is missing, or installed at an
            incompatible version (another major version, or below 1.0 another minor one).
    """
    src = Path(directory)
    payload = json.loads((src / _META_FILE).read_text("utf-8"))
    _check_providers(payload["components"])
    section = ModelSection.model_validate(payload["model"])
    backbone = _rebuild_backbone(section.backbone, payload.get("backbone_state", {}))
    detector = assemble_detector(section, backbone, _restore_meta(payload["meta"]))
    state = load_file(str(src / _MODEL_FILE))
    detector.load_state_dict(state, strict=True)
    detector.eval()
    return detector
