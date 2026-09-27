"""Optimisers (``adamw``, ``sgd``) over a detector's named parameter groups.

Optimisers are not a plugin registry: they are built from the resolved ``AssembledDetector``
itself, since group names come from the backbone's own ``param_groups()`` plus ``head``, ``pool``
and ``stem``, and a config typo in a group name can only be caught once that detector exists.

``groups`` may also name ``backbone``: it applies to every group of the backbone's own (``embed``,
``blocks.<i>``, ``norm``, ...), so one entry configures a backbone whatever its groups are called.
An entry for one of those groups by name replaces the ``backbone`` entry for that group, whole.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from torch import nn

from dfwb.core.config.schema import ComponentSpec
from dfwb.core.errors import ConfigError, did_you_mean, model_fields_at, validation_messages
from dfwb.models.detector import AssembledDetector

__all__ = ["build_optimizer"]

_SECTION = "optim"
_OPTIMIZER_NAMES = ("adamw", "sgd")
_BACKBONE = "backbone"  # every group of the backbone's own


class _GroupSpec(BaseModel):
    """A named group's overrides: ``lr_scale`` multiplies the base LR; ``weight_decay`` replaces
    it (``None`` keeps the top-level value)."""

    model_config = ConfigDict(extra="forbid")

    lr_scale: float = 1.0
    weight_decay: float | None = None


class _CommonOptimParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float
    weight_decay: float = 0.0
    groups: dict[str, dict[str, Any]] = Field(default_factory=dict)
    layer_decay: float | None = None


class _AdamWParams(_CommonOptimParams):
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)


class _SGDParams(_CommonOptimParams):
    weight_decay: float = 0.0
    momentum: float = 0.9
    nesterov: bool = False


def _validate[M: _CommonOptimParams](spec: ComponentSpec, model_cls: type[M]) -> M:
    try:
        return model_cls.model_validate(spec.params)
    except ValidationError as exc:
        lines = validation_messages(
            exc, prefix=(_SECTION,), fields_at=lambda loc: model_fields_at(model_cls, loc)
        )
        raise ConfigError(
            f"{_SECTION}: {len(lines)} invalid value(s)\n  " + "\n  ".join(lines),
            hint=f"see the {model_cls.__name__.removesuffix('Params').lower()} parameters",
        ) from None


def _validate_groups(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, _GroupSpec]:
    result: dict[str, _GroupSpec] = {}
    for name, value in raw.items():
        try:
            result[name] = _GroupSpec.model_validate(value)
        except ValidationError as exc:
            lines = validation_messages(
                exc,
                prefix=(_SECTION, "groups", name),
                fields_at=lambda loc: model_fields_at(_GroupSpec, loc),
            )
            raise ConfigError(
                f"{_SECTION}: {len(lines)} invalid value(s)\n  " + "\n  ".join(lines),
                hint="group parameters: lr_scale, weight_decay",
            ) from None
    return result


def _group_sources(detector: AssembledDetector) -> dict[str, list[nn.Parameter]]:
    sources: dict[str, list[nn.Parameter]] = dict(detector.backbone.param_groups())
    sources["head"] = list(detector.head.parameters())
    sources["pool"] = list(detector.pool.parameters()) if detector.pool is not None else []
    sources["stem"] = list(detector.stem.parameters()) if detector.stem is not None else []
    return sources


def _layer_multiplier(name: str, layer_decay: float | None, num_blocks: int) -> float:
    if layer_decay is None:
        return 1.0
    if name.startswith("blocks."):
        index = int(name.removeprefix("blocks."))
        return layer_decay ** (num_blocks - index)
    if name == "embed":
        return layer_decay ** (num_blocks + 1)
    return 1.0


def _param_groups(detector: AssembledDetector, params: _CommonOptimParams) -> list[dict[str, Any]]:
    group_overrides = _validate_groups(params.groups)
    sources = _group_sources(detector)
    backbone_groups = set(detector.backbone.param_groups())
    valid = [*sources, _BACKBONE]
    for name in group_overrides:
        if name not in sources and name != _BACKBONE:
            raise ConfigError(
                f"{_SECTION}.groups.{name}: unknown group{did_you_mean(name, valid)}",
                hint="valid groups: " + ", ".join(sorted(valid)) + f" ({_BACKBONE}: all of the "
                "backbone's own)",
            )

    num_blocks = sum(1 for name in sources if name.startswith("blocks."))
    torch_groups: list[dict[str, Any]] = []
    for name, source_params in sources.items():
        trainable = [p for p in source_params if p.requires_grad]
        if not trainable:
            continue
        override = group_overrides.get(name)
        if override is None and name in backbone_groups:
            override = group_overrides.get(_BACKBONE)
        if override is None:
            override = _GroupSpec()
        weight_decay = (
            override.weight_decay if override.weight_decay is not None else params.weight_decay
        )
        multiplier = override.lr_scale * _layer_multiplier(name, params.layer_decay, num_blocks)
        torch_groups.append(
            {
                "params": trainable,
                "lr": params.lr * multiplier,
                "weight_decay": weight_decay,
                "name": name,
            }
        )
    if not torch_groups:
        raise ConfigError(
            f"{_SECTION}: the detector has no trainable parameters",
            hint="check the backbone's freeze settings",
        )
    return torch_groups


def build_optimizer(spec: ComponentSpec, detector: AssembledDetector) -> torch.optim.Optimizer:
    """Build ``adamw`` or ``sgd`` over ``detector``'s named parameter groups.

    Frozen (``requires_grad=False``) parameters are excluded, and a named group left with no
    trainable parameters is dropped rather than passed to the optimiser empty.

    Raises:
        ConfigError: An unknown optimiser name, an unknown parameter, or a ``groups`` key that
            names no parameter group of this detector.
    """
    if spec.name == "adamw":
        adamw_params = _validate(spec, _AdamWParams)
        groups = _param_groups(detector, adamw_params)
        return torch.optim.AdamW(groups, lr=adamw_params.lr, betas=adamw_params.betas)
    if spec.name == "sgd":
        sgd_params = _validate(spec, _SGDParams)
        groups = _param_groups(detector, sgd_params)
        return torch.optim.SGD(
            groups,
            lr=sgd_params.lr,
            momentum=sgd_params.momentum,
            nesterov=sgd_params.nesterov,
        )
    raise ConfigError(
        f"{_SECTION}.name: {spec.name!r} is not one of {list(_OPTIMIZER_NAMES)}"
        f"{did_you_mean(spec.name, _OPTIMIZER_NAMES)}",
        hint=f"known optimisers: {', '.join(_OPTIMIZER_NAMES)}",
    )
