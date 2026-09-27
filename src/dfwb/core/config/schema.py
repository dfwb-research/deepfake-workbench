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

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SerializerFunctionWrapHandler,
    Tag,
    ValidationError,
    field_validator,
    model_serializer,
)
from pydantic_core import PydanticCustomError

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
    "EarlyStopSection",
    "InputOverrides",
    "TrainConfig",
    "check_components",
    "validate_config",
]


class ConfigModel(BaseModel):
    """Base for config sections: unknown keys are errors; numbers are accepted for strings."""

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)


@dataclass(frozen=True)
class ComponentOf:
    """Marks a :class:`ComponentSpec` field as naming a key of the given registry.

    ``supplied`` holds ``(parameter, where its value comes from)`` for each parameter the
    framework always passes itself when it builds the component: a pool's and a head's ``dim``
    is the backbone's output size. A config leaves these out, and setting one is an error (the
    component would get it twice). ``defaulted`` names parameters the framework passes only when
    the config does not: a stem's ``in_channels`` is the image's 3 channels unless the config
    sets its own, which then wins. Either way, a config that leaves them out is complete.
    """

    registry: str
    supplied: tuple[tuple[str, str], ...] = ()
    defaulted: tuple[str, ...] = ()


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
    #: A training source's share of the draws under ``data.loader.balance: source``, relative to
    #: the other training sources' weights.
    weight: float = Field(default=1.0, gt=0)


class SuiteRef(ConfigModel):
    suite: str


class ClipsPerVideo(ConfigModel):
    train: int = Field(ge=1)
    eval: int = Field(ge=1)


class ClipSection(ConfigModel):
    frames: int = Field(ge=1)
    sampling: Literal["uniform", "consecutive", "random-window"]
    clips_per_video: ClipsPerVideo
    stride: int = Field(default=1, ge=1)


class TransformsSection(ConfigModel):
    train: list[Annotated[ComponentSpec, ComponentOf("transforms")]] = Field(default_factory=list)


class LoaderSection(ConfigModel):
    batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    #: ``none`` (a seeded shuffle), ``video-label`` (real and fake videos equally likely) or
    #: ``source`` (every training source weighted equally); unset means ``none``.
    balance: Literal["none", "video-label", "source"] | None = None


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
    #: Train on real/fake pairs from each training protocol's pair list.
    pairs: bool = False
    #: Let input adaptation proceed past a store that cannot serve the detector's input (a
    #: crop-kind mismatch, or a narrower crop than the detector asks for); every source records
    #: what it had to accept.
    allow_input_mismatch: bool = False


# InputSpec fields that must hold a value; the others may be null on purpose.
_INPUT_NEEDS_VALUE = ("modality", "crop", "size", "frames", "sampling", "color", "value_range")


class InputOverrides(ConfigModel):
    """``model.input``: overrides of the backbone's own input spec (the detector contract's
    ``InputSpec``).

    Only the keys written here override; every other one keeps the backbone's value. So the
    resolved config keeps exactly the keys that were written, and ``crop_scale: null`` (any crop
    scale the store kept, e.g. for a full-frame input) stays distinct from leaving it out.
    """

    modality: Literal["frames", "audio", "audiovisual"] | None = None
    crop: Literal["face", "full-frame"] | None = None
    crop_scale: float | None = Field(default=None, gt=0)
    size: tuple[int, int] | None = None
    frames: int | None = Field(default=None, ge=1)
    sampling: Literal["uniform", "consecutive", "any"] | None = None
    color: Literal["rgb", "bgr"] | None = None
    value_range: tuple[float, float] | None = None
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    preferred_profile: str | None = None

    @field_validator(*_INPUT_NEEDS_VALUE)
    @classmethod
    def _needs_a_value(cls, value: Any) -> Any:
        if value is None:
            raise PydanticCustomError("not_null", "may not be null")
        return value

    @model_serializer(mode="wrap")
    def _written_keys_only(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        return {key: value for key, value in data.items() if key in self.model_fields_set}

    def overrides(self) -> dict[str, Any]:
        """The written keys, as ``InputSpec`` keyword arguments."""
        return {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name in self.model_fields_set
        }


_BACKBONE_DIM = (("dim", "the backbone's output size"),)


class ModelSection(ConfigModel):
    backbone: Annotated[ComponentSpec, ComponentOf("backbones")]
    temporal_pool: Annotated[ComponentSpec, ComponentOf("temporal_pools", supplied=_BACKBONE_DIM)]
    head: Annotated[ComponentSpec, ComponentOf("heads", supplied=_BACKBONE_DIM)]
    stem: Annotated[ComponentSpec | None, ComponentOf("layers", defaulted=("in_channels",))] = None
    input: InputOverrides | None = None


class RunSection(ConfigModel):
    name: str = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    output_root: str | None = None


_ONE_PROCESS = (
    "training runs in a single process (one GPU, or the CPU): validation scores, the score dump "
    "and the checkpoints are all produced by that one process"
)

# Lightning Trainer arguments ``train.lightning`` may not set, and why.
_LIGHTNING_REFUSED: dict[str, str] = {
    "enable_checkpointing": "checkpoints are written as safetensors by the run itself; "
    "Lightning's own checkpointing pickles, so it stays off",
    "strategy": _ONE_PROCESS,
    "num_nodes": _ONE_PROCESS,
    "callbacks": "the run directory's checkpoints, score dumps and guards come from the "
    "framework's own callbacks; add a plugin's callbacks with train.callbacks",
    "logger": "the run directory's logs come from train.loggers",
    "default_root_dir": "everything a run writes goes under its own run directory",
    "max_epochs": "set train.max_epochs instead",
    "precision": "set train.precision instead",
}


def _one_device(value: Any) -> bool:
    if isinstance(value, list):
        return len(value) == 1
    return bool(value == 1 or value == "1")


LoggerName = Literal["csv", "tensorboard", "wandb"]


def _default_loggers() -> list[LoggerName]:
    return ["csv", "tensorboard"]


class EarlyStopSection(ConfigModel):
    """Stop once the monitor has gone ``patience`` validations without improving by more than
    ``min_delta``."""

    patience: int = Field(ge=1)
    min_delta: float = Field(default=0.0, ge=0)


#: ``train.precision``: ``auto`` picks per device (``bf16-mixed`` on a CUDA GPU that supports
#: bfloat16, else ``16-mixed`` on CUDA, else ``32-true``); the others are Lightning's own.
Precision = Literal["auto", "32-true", "bf16-mixed", "16-mixed"]


class TrainSection(ConfigModel):
    max_epochs: int = Field(ge=1)
    precision: Precision = "auto"
    devices: int = Field(ge=1)
    monitor: str = "val/video_auc"
    mode: Literal["max", "min"] = "max"
    early_stop: EarlyStopSection | None = None
    #: Stop once the loss has not been finite for this many steps in a row.
    nan_tolerance: int = Field(default=3, ge=1)
    #: Rewrite the run's heartbeat file every this many training steps.
    heartbeat_steps: int = Field(default=50, ge=1)
    #: The CSV log is always written; TensorBoard when it is installed; W&B only when listed.
    loggers: list[LoggerName] = Field(default_factory=_default_loggers)
    #: Extra keyword arguments for Lightning's ``Trainer`` (``deterministic``, ``accelerator``,
    #: ``log_every_n_steps``, ...), except the ones the run itself must control.
    lightning: dict[str, Any] = Field(default_factory=dict)
    #: Callbacks from the ``callbacks`` registry (a plugin's), run beside the framework's own;
    #: their ``state_dict()`` is saved with the run's resume state.
    callbacks: list[Annotated[ComponentSpec, ComponentOf("callbacks")]] = Field(
        default_factory=list
    )

    @field_validator("devices")
    @classmethod
    def _single_device(cls, value: int) -> int:
        if value != 1:
            raise PydanticCustomError("one_device", f"{_ONE_PROCESS}; set devices: 1")
        return value

    @field_validator("lightning")
    @classmethod
    def _passthrough(cls, value: dict[str, Any]) -> dict[str, Any]:
        for key, option in value.items():
            if key in _LIGHTNING_REFUSED and not (key == "num_nodes" and option == 1):
                reason = _LIGHTNING_REFUSED[key]
                raise PydanticCustomError(
                    "refused_trainer_argument", f"{key!r} is not allowed here: {reason}"
                )
            if key == "devices" and not _one_device(option):
                raise PydanticCustomError(
                    "refused_trainer_argument",
                    f"'devices' is not allowed here: {_ONE_PROCESS} (devices: 1, or --device)",
                )
        return value


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


def _component_marker(info_annotation: Any, metadata: list[Any]) -> ComponentOf | None:
    for item in metadata:
        if isinstance(item, ComponentOf):
            return item
    if typing.get_origin(info_annotation) is Annotated:
        for item in typing.get_args(info_annotation)[1:]:
            if isinstance(item, ComponentOf):
                return item
    return None


_Found = tuple[tuple[str | int, ...], ComponentOf, ComponentSpec]


def _walk_components(value: Any, loc: tuple[str | int, ...]) -> list[_Found]:
    found: list[_Found] = []
    if isinstance(value, BaseModel) and not isinstance(value, ComponentSpec):
        for name, info in type(value).model_fields.items():
            child = getattr(value, name)
            marker = _component_marker(info.annotation, info.metadata)
            if marker is None and typing.get_origin(info.annotation) is list:
                (inner,) = typing.get_args(info.annotation) or (None,)
                if typing.get_origin(inner) is Annotated:
                    marker = _component_marker(inner, [])
            key = info.alias or name
            if marker is not None and isinstance(child, ComponentSpec):
                found.append(((*loc, key), marker, child))
            elif marker is not None and isinstance(child, list):
                found.extend(((*loc, key, i), marker, c) for i, c in enumerate(child))
            else:
                found.extend(_walk_components(child, (*loc, key)))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(_walk_components(item, (*loc, i)))
    return found


def _unsupplied(
    pairs: tuple[tuple[Loc, str], ...], marker: ComponentOf, spec: ComponentSpec
) -> tuple[tuple[Loc, str], ...]:
    """``pairs`` without the problems about a parameter the framework passes itself (which the
    config rightly left out)."""
    names = [name for name, _ in marker.supplied] + list(marker.defaulted)
    skipped = {(name,) for name in names if name not in spec.params}
    return tuple((where, text) for where, text in pairs if tuple(where) not in skipped)


def _set_but_supplied(marker: ComponentOf, spec: ComponentSpec) -> list[tuple[Loc, str]]:
    return [
        ((name,), f"set from {source}; leave it out")
        for name, source in marker.supplied
        if name in spec.params
    ]


def check_components(config: ConfigModel, registries: Mapping[str, Any]) -> None:
    """Check every component's ``name`` and params against the installed registries.

    Raises one error listing every problem, each at its full config path (for example
    ``model.backbone.freeze.mode: …``). When the problems have different hints, each problem keeps
    its own ``hint:`` line, so a hint naming a failed plugin is never lost. It is an
    :class:`InstallationError` when every problem is a missing installation, and a
    :class:`ConfigError` otherwise.
    """
    found: list[tuple[list[str], str, bool]] = []  # (problem lines, hint, missing install)
    for loc, marker, spec in _walk_components(config, ()):
        supplied = _set_but_supplied(marker, spec)
        if supplied:
            lines = [f"{format_loc((*loc, *where))}: {text}" for where, text in supplied]
            found.append((lines, "the framework passes these when it builds the detector", False))
        try:
            registries[marker.registry].validate(spec.name, **spec.params)
        except ConfigError as exc:
            pairs: tuple[tuple[Loc, str], ...] = exc.problems or (((), exc.message),)
            pairs = _unsupplied(pairs, marker, spec)
            if not pairs:
                continue
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
