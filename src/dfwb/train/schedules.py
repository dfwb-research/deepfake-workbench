"""Learning-rate schedules (``constant``, ``cosine``, ``step``), each with linear warmup from 0.

Every schedule steps once per optimiser step, never per epoch. ``build_schedule`` takes
``steps_per_epoch`` and ``epochs`` so warmup and decay lengths are configured in epochs but applied
in steps. Each optimiser parameter group keeps the base LR :func:`dfwb.train.optim.build_optimizer`
gave it; the schedule multiplies every group by the same per-step factor, so a group's
``lr_scale``/``layer_decay`` survives the schedule unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from dfwb.core.config.schema import ComponentSpec
from dfwb.core.errors import ConfigError, did_you_mean, model_fields_at, validation_messages

__all__ = ["build_schedule"]

_SECTION = "schedule"
_SCHEDULE_NAMES = ("constant", "cosine", "step")

_LRLambda = Callable[[int], float]


class _CommonScheduleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warmup_epochs: float = 0.0


class _ConstantParams(_CommonScheduleParams):
    pass


class _CosineParams(_CommonScheduleParams):
    min_lr: float = 0.0


class _StepParams(_CommonScheduleParams):
    step_epochs: float = Field(gt=0)
    gamma: float = 0.1


def _validate[S: _CommonScheduleParams](spec: ComponentSpec, model_cls: type[S]) -> S:
    try:
        return model_cls.model_validate(spec.params)
    except ValidationError as exc:
        lines = validation_messages(
            exc, prefix=(_SECTION,), fields_at=lambda loc: model_fields_at(model_cls, loc)
        )
        raise ConfigError(
            f"{_SECTION}: {len(lines)} invalid value(s)\n  " + "\n  ".join(lines),
            hint=f"see the {spec.name!r} schedule's accepted parameters",
        ) from None


def _warmup_steps(params: _CommonScheduleParams, steps_per_epoch: int) -> int:
    return round(params.warmup_epochs * steps_per_epoch)


def _warmup_factor(step: int, warmup_steps: int) -> float | None:
    """The linear warmup ramp (0 at step 0) while inside warmup, else ``None``."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    return None


def _constant_lambda(*, warmup_steps: int) -> _LRLambda:
    def lr_lambda(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        return warm if warm is not None else 1.0

    return lr_lambda


def _cosine_lambda(
    params: _CosineParams, *, warmup_steps: int, total_steps: int, base_lr: float
) -> _LRLambda:
    min_ratio = params.min_lr / base_lr if base_lr else 0.0
    remaining = total_steps - warmup_steps

    def lr_lambda(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        if warm is not None:
            return warm
        progress = 1.0 if remaining <= 0 else min((step - warmup_steps) / remaining, 1.0)
        cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
        return min_ratio + (1 - min_ratio) * cosine_term

    return lr_lambda


def _step_lambda(params: _StepParams, *, warmup_steps: int, steps_per_epoch: int) -> _LRLambda:
    period = max(1, round(params.step_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        if warm is not None:
            return warm
        drops = (step - warmup_steps) // period
        return params.gamma**drops

    return lr_lambda


def build_schedule(
    spec: ComponentSpec,
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    epochs: int,
) -> LRScheduler:
    """Build ``constant``, ``cosine`` or ``step`` over ``optimizer``, stepped once per step.

    ``optimizer.defaults["lr"]`` is the reference base LR that ``cosine``'s ``min_lr`` decays
    towards, as a ratio; every parameter group is multiplied by that same per-step ratio, so a
    group already scaled by ``build_optimizer`` (``lr_scale``, ``layer_decay``) keeps its relative
    scale for the whole run.

    Raises:
        ConfigError: An unknown schedule name, or an unknown or ill-typed parameter.
    """
    total_steps = steps_per_epoch * epochs
    base_lr = float(optimizer.defaults.get("lr", 0.0))

    lr_lambda: _LRLambda
    if spec.name == "constant":
        constant_params = _validate(spec, _ConstantParams)
        warmup_steps = _warmup_steps(constant_params, steps_per_epoch)
        lr_lambda = _constant_lambda(warmup_steps=warmup_steps)
    elif spec.name == "cosine":
        cosine_params = _validate(spec, _CosineParams)
        warmup_steps = _warmup_steps(cosine_params, steps_per_epoch)
        lr_lambda = _cosine_lambda(
            cosine_params, warmup_steps=warmup_steps, total_steps=total_steps, base_lr=base_lr
        )
    elif spec.name == "step":
        step_params = _validate(spec, _StepParams)
        warmup_steps = _warmup_steps(step_params, steps_per_epoch)
        lr_lambda = _step_lambda(
            step_params, warmup_steps=warmup_steps, steps_per_epoch=steps_per_epoch
        )
    else:
        raise ConfigError(
            f"{_SECTION}.name: {spec.name!r} is not one of {list(_SCHEDULE_NAMES)}"
            f"{did_you_mean(spec.name, _SCHEDULE_NAMES)}",
            hint=f"known schedules: {', '.join(_SCHEDULE_NAMES)}",
        )
    return LambdaLR(optimizer, lr_lambda=lr_lambda)
