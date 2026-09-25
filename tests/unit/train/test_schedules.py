"""``constant``, ``cosine`` and ``step`` schedules, each with linear warmup from 0.

Schedules step once per optimiser step. The hand-computed sequences below are built with plain
``math``, independently of ``dfwb.train.schedules``, never by calling the code under test.
"""

from __future__ import annotations

import math

import pytest
import torch
from tests.unit.train.conftest import make_detector

from dfwb.core.config.schema import ComponentSpec
from dfwb.core.errors import ConfigError
from dfwb.train.optim import build_optimizer
from dfwb.train.schedules import build_schedule


def _sgd(lr: float = 1.0) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([{"params": [param], "lr": lr, "name": "g"}], lr=lr)


def _run(schedule: torch.optim.lr_scheduler.LRScheduler, optimizer, steps: int) -> list[float]:
    lrs = [schedule.get_last_lr()[0]]
    for _ in range(steps):
        optimizer.step()
        schedule.step()
        lrs.append(schedule.get_last_lr()[0])
    return lrs


# --------------------------------------------------------------------------- constant


def test_constant_without_warmup_keeps_the_lr_fixed():
    optimizer = _sgd(lr=0.5)
    schedule = build_schedule(
        ComponentSpec(name="constant"), optimizer, steps_per_epoch=4, epochs=3
    )
    lrs = _run(schedule, optimizer, 12)
    assert lrs == [pytest.approx(0.5)] * 13


def test_constant_with_warmup_ramps_linearly_from_zero_then_holds():
    optimizer = _sgd(lr=1.0)
    schedule = build_schedule(
        ComponentSpec(name="constant", warmup_epochs=1), optimizer, steps_per_epoch=4, epochs=3
    )
    lrs = _run(schedule, optimizer, 12)
    expected = [0.0, 0.25, 0.5, 0.75] + [1.0] * 9
    for a, e in zip(lrs, expected, strict=True):
        assert a == pytest.approx(e)


# --------------------------------------------------------------------------- cosine (hand-computed)


def test_cosine_hand_computed_lr_sequence_with_5_epoch_warmup_10_steps_each():
    steps_per_epoch = 10
    epochs = 10
    warmup_steps = 50  # 5 warmup epochs * 10 steps/epoch
    total_steps = steps_per_epoch * epochs
    min_lr = 0.1  # base lr is 1.0, so this is also the min ratio

    optimizer = _sgd(lr=1.0)
    schedule = build_schedule(
        ComponentSpec(name="cosine", warmup_epochs=5, min_lr=min_lr),
        optimizer,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )

    def expected(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        remaining = total_steps - warmup_steps
        progress = min((step - warmup_steps) / remaining, 1.0)
        cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (1 - min_lr) * cosine_term

    actual = _run(schedule, optimizer, total_steps)
    for step, lr in enumerate(actual):
        assert lr == pytest.approx(expected(step), abs=1e-9), step
    assert actual[0] == 0.0  # step 0 has LR 0
    assert actual[warmup_steps] == pytest.approx(1.0)  # warmup ends at the full base LR
    assert actual[-1] == pytest.approx(min_lr, abs=1e-9)  # fully decayed by the last step


def test_cosine_min_lr_defaults_to_zero():
    optimizer = _sgd(lr=1.0)
    schedule = build_schedule(ComponentSpec(name="cosine"), optimizer, steps_per_epoch=2, epochs=1)
    lrs = _run(schedule, optimizer, 2)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- step


def test_step_schedule_multiplies_by_gamma_every_step_epochs_after_warmup():
    steps_per_epoch = 4
    warmup_steps = 4  # 1 warmup epoch
    period = 8  # step_epochs(2) * steps_per_epoch(4)

    optimizer = _sgd(lr=1.0)
    schedule = build_schedule(
        ComponentSpec(name="step", warmup_epochs=1, step_epochs=2, gamma=0.1),
        optimizer,
        steps_per_epoch=steps_per_epoch,
        epochs=5,
    )

    def expected(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        drops = (step - warmup_steps) // period
        return 0.1**drops

    actual = _run(schedule, optimizer, steps_per_epoch * 5)
    for step, lr in enumerate(actual):
        assert lr == pytest.approx(expected(step)), step


def test_step_defaults_gamma_to_0_1_and_no_warmup():
    optimizer = _sgd(lr=1.0)
    schedule = build_schedule(
        ComponentSpec(name="step", step_epochs=1), optimizer, steps_per_epoch=1, epochs=3
    )
    lrs = _run(schedule, optimizer, 3)
    assert lrs == [pytest.approx(v) for v in (1.0, 0.1, 0.1**2, 0.1**3)]


# --------------------------------------------------------------------------- group multipliers


def test_group_multipliers_survive_the_schedule():
    detector = make_detector()
    optimizer = build_optimizer(
        ComponentSpec(
            name="adamw",
            lr=1.0,
            groups={"blocks.0": {"lr_scale": 0.1}, "blocks.2": {"lr_scale": 1.0}},
        ),
        detector,
    )
    schedule = build_schedule(
        ComponentSpec(name="cosine", warmup_epochs=0, min_lr=0.2),
        optimizer,
        steps_per_epoch=5,
        epochs=4,
    )
    by_name = {g["name"]: g for g in optimizer.param_groups}
    ratio_before = by_name["blocks.0"]["lr"] / by_name["blocks.2"]["lr"]
    assert ratio_before == pytest.approx(0.1)
    for _ in range(20):
        optimizer.step()
        schedule.step()
    ratio_after = by_name["blocks.0"]["lr"] / by_name["blocks.2"]["lr"]
    assert ratio_after == pytest.approx(0.1)


# --------------------------------------------------------------------------- config errors


def test_unknown_schedule_name_is_a_config_error_with_did_you_mean():
    optimizer = _sgd()
    with pytest.raises(ConfigError, match=r"schedule\.name") as info:
        build_schedule(ComponentSpec(name="cosin"), optimizer, steps_per_epoch=1, epochs=1)
    assert "cosine" in info.value.message


def test_unknown_param_is_a_config_error():
    optimizer = _sgd()
    with pytest.raises(ConfigError) as info:
        build_schedule(
            ComponentSpec(name="constant", foo=1), optimizer, steps_per_epoch=1, epochs=1
        )
    assert "schedule" in info.value.message
    assert "foo" in info.value.message


def test_step_requires_step_epochs():
    optimizer = _sgd()
    with pytest.raises(ConfigError, match="step_epochs"):
        build_schedule(ComponentSpec(name="step"), optimizer, steps_per_epoch=1, epochs=1)


def test_step_epochs_must_be_positive():
    optimizer = _sgd()
    with pytest.raises(ConfigError, match="step_epochs"):
        build_schedule(
            ComponentSpec(name="step", step_epochs=0), optimizer, steps_per_epoch=1, epochs=1
        )


def test_cosine_rejects_a_param_that_belongs_to_step():
    optimizer = _sgd()
    with pytest.raises(ConfigError, match="step_epochs"):
        build_schedule(
            ComponentSpec(name="cosine", step_epochs=2), optimizer, steps_per_epoch=1, epochs=1
        )
