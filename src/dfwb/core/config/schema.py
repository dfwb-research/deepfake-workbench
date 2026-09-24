"""Config schemas (contract C2), keyed by the top-level ``schema:`` value.

Every section forbids unknown keys. Pluggable components are written ``{name: <key>, …params}``
(:class:`ComponentSpec`); their params are checked against the component's own signature or params
model by :func:`check_components`, not here, because they depend on installed plugins.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationError

from dfwb.core.errors import (
    ConfigError,
    DFWBError,
    InstallationError,
    Loc,
    did_you_mean,
    format_loc,
    model_fields_at,
    validation_messages,
)

__all__ = [
    "SCHEMAS",
    "ComponentOf",
    "ComponentSpec",
    "ConfigModel",
    "TrainConfig",
    "check_components",
    "validate_config",
]


class ConfigModel(BaseModel):
    """Base for config sections: unknown keys are errors; numbers are accepted for strings."""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)


@dataclass(frozen=True)
class ComponentOf:
    """Marks a :class:`ComponentSpec` field as naming a key of the given registry."""

    registry: str


class ComponentSpec(BaseModel):
    """A pluggable component: ``{name: <registry key>, **params}``."""

    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

    name: str = Field(min_length=1)

    @property
    def params(self) -> dict[str, Any]:
        """Everything except ``name``."""
        return dict(self.model_extra or {})


WhereValue = str | int | float | bool | None


class DataSource(ConfigModel):
    protocol: str
    split: Literal["train", "val", "test"]
    where: dict[str, WhereValue | list[WhereValue]] = Field(default_factory=dict)


class SuiteRef(ConfigModel):
    suite: str


class ClipsPerVideo(ConfigModel):
    train: int = Field(ge=1)
    eval: int = Field(ge=1)


class ClipSection(ConfigModel):
    frames: int = Field(ge=1)
    sampling: Literal["uniform", "consecutive", "random-window"]
    clips_per_video: ClipsPerVideo


class TransformsSection(ConfigModel):
    train: list[Annotated[ComponentSpec, ComponentOf("transforms")]] = Field(default_factory=list)


class LoaderSection(ConfigModel):
    batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    balance: str | None = None


def _suite_or_list(value: Any) -> str:
    # Pick the union branch from the input's shape, so errors name one real path.
    return "@suite" if isinstance(value, (dict, SuiteRef)) else "@list"


class DataSection(ConfigModel):
    processing: str
    clip: ClipSection
    labels: str
    train: list[DataSource] = Field(min_length=1)
    val: list[DataSource] = Field(default_factory=list)
    test: Annotated[
        Annotated[SuiteRef, Tag("@suite")] | Annotated[list[DataSource], Tag("@list")],
        Discriminator(_suite_or_list),
    ] = Field(default_factory=list)
    transforms: TransformsSection = Field(default_factory=TransformsSection)
    loader: LoaderSection


class ModelSection(ConfigModel):
    backbone: Annotated[ComponentSpec, ComponentOf("backbones")]
    temporal_pool: Annotated[ComponentSpec, ComponentOf("temporal_pools")]
    head: Annotated[ComponentSpec, ComponentOf("heads")]


class RunSection(ConfigModel):
    name: str = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    output_root: str | None = None


class TrainSection(ConfigModel):
    max_epochs: int = Field(ge=1)
    precision: str
    devices: int = Field(ge=1)
    monitor: str
    mode: Literal["max", "min"]


class EvalSection(ConfigModel):
    metrics: list[str] = Field(min_length=1)
    aggregate: str


class TrainConfig(ConfigModel):
    """``schema: dfwb.train/1``: one training experiment."""

    fingerprint_exclude: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("run", "name"),
        ("run", "output_root"),
    )

    schema_: Literal["dfwb.train/1"] = Field(alias="schema")
    run: RunSection
    data: DataSection
    model: ModelSection
    loss: Annotated[ComponentSpec, ComponentOf("losses")]
    optim: ComponentSpec  # optimisers are built by the train layer, not a plugin registry
    schedule: ComponentSpec  # schedules likewise
    train: TrainSection
    eval: EvalSection


SCHEMAS: dict[str, type[ConfigModel]] = {"dfwb.train/1": TrainConfig}


def validate_config(data: Mapping[str, Any], *, source: str) -> ConfigModel:
    """Validate a composed config against the model named by its ``schema:`` key."""
    schema = data.get("schema")
    known = ", ".join(SCHEMAS)
    if schema is None:
        raise ConfigError(
            f"{source}: missing top-level 'schema:' key",
            hint=f"add e.g. 'schema: dfwb.train/1' (known: {known})",
        )
    model = SCHEMAS.get(str(schema))
    if model is None:
        raise ConfigError(
            f"{source}: unknown schema {schema!r}{did_you_mean(str(schema), SCHEMAS)}",
            hint=f"known schemas: {known}",
        )
    try:
        return model.model_validate(dict(data))
    except ValidationError as exc:
        lines = validation_messages(exc, fields_at=lambda loc: model_fields_at(model, loc))
        raise ConfigError(
            f"{source}: {len(lines)} invalid value(s)\n  " + "\n  ".join(lines),
            hint="see `dfwb schema export c2` for every key and type",
        ) from None


def _component_registry(info_annotation: Any, metadata: list[Any]) -> str | None:
    for item in metadata:
        if isinstance(item, ComponentOf):
            return item.registry
    if typing.get_origin(info_annotation) is Annotated:
        for item in typing.get_args(info_annotation)[1:]:
            if isinstance(item, ComponentOf):
                return item.registry
    return None


def _walk_components(
    value: Any, loc: tuple[str | int, ...]
) -> list[tuple[tuple[str | int, ...], str, ComponentSpec]]:
    found: list[tuple[tuple[str | int, ...], str, ComponentSpec]] = []
    if isinstance(value, BaseModel) and not isinstance(value, ComponentSpec):
        for name, info in type(value).model_fields.items():
            child = getattr(value, name)
            registry = _component_registry(info.annotation, info.metadata)
            if registry is None and typing.get_origin(info.annotation) is list:
                (inner,) = typing.get_args(info.annotation) or (None,)
                if typing.get_origin(inner) is Annotated:
                    registry = _component_registry(inner, [])
            key = info.alias or name
            if registry is not None and isinstance(child, ComponentSpec):
                found.append(((*loc, key), registry, child))
            elif registry is not None and isinstance(child, list):
                found.extend(((*loc, key, i), registry, c) for i, c in enumerate(child))
            else:
                found.extend(_walk_components(child, (*loc, key)))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(_walk_components(item, (*loc, i)))
    return found


def check_components(config: ConfigModel, registries: Mapping[str, Any]) -> None:
    """Check every component's ``name`` and params against the installed registries.

    Raises one error listing every problem, each at its full config path (for example
    ``model.backbone.freeze.mode: …``). When the problems have different hints, each problem keeps
    its own ``hint:`` line, so a hint naming a failed plugin is never lost. It is an
    :class:`InstallationError` when every problem is a missing installation, and a
    :class:`ConfigError` otherwise.
    """
    found: list[tuple[list[str], str, bool]] = []  # (problem lines, hint, missing install)
    for loc, registry_name, spec in _walk_components(config, ()):
        try:
            registries[registry_name].validate(spec.name, **spec.params)
        except ConfigError as exc:
            pairs: tuple[tuple[Loc, str], ...] = exc.problems or (((), exc.message),)
            lines = [f"{format_loc((*loc, *where))}: {text}" for where, text in pairs]
            found.append((lines, exc.hint, False))
        except DFWBError as exc:
            found.append(
                (
                    [f"{format_loc(loc)}: {exc.message}"],
                    exc.hint,
                    isinstance(exc, InstallationError),
                )
            )
    if not found:
        return
    hints = {hint for _, hint, _ in found}
    body: list[str] = []
    for lines, hint, _ in found:
        body.extend(f"  {line}" for line in lines)
        if len(hints) > 1:
            body.append(f"    hint: {hint}")
    count = sum(len(lines) for lines, _, _ in found)
    kind = InstallationError if all(missing for _, _, missing in found) else ConfigError
    raise kind(
        f"{count} invalid component value(s)\n" + "\n".join(body),
        hint=found[0][1]
        if len(hints) == 1
        else "fix each problem above; `dfwb plugins list --all` shows installed and failed plugins",
    )
