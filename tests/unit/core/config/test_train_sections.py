"""The training-run keys of ``dfwb.train/1``: input-spec overrides, paired data, the loader's
balance, and the ``train:`` section's monitor, early stop, guards, loggers and Lightning
passthrough."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from dfwb.core.config.loader import config_fingerprint
from dfwb.core.config.schema import TrainConfig, check_components, validate_config
from dfwb.core.errors import ConfigError
from dfwb.core.registry import Registry

BASE: dict[str, Any] = {
    "schema": "dfwb.train/1",
    "run": {"name": "r", "seeds": [1]},
    "data": {
        "processing": "p",
        "clip": {"frames": 1, "sampling": "uniform", "clips_per_video": {"train": 1, "eval": 1}},
        "labels": "binary",
        "train": [{"protocol": "toyfake/official", "split": "train"}],
        "loader": {"batch_size": 4, "num_workers": 0},
    },
    "model": {
        "backbone": {"name": "tiny-cnn"},
        "temporal_pool": {"name": "mean"},
        "head": {"name": "linear"},
    },
    "loss": {"name": "bce"},
    "optim": {"name": "adamw", "lr": 0.001},
    "schedule": {"name": "constant"},
    "train": {"max_epochs": 1, "precision": "32-true", "devices": 1},
    "eval": {"metrics": ["auc"], "aggregate": "mean-prob"},
}


def _config(**sections: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(BASE)
    for name, values in sections.items():
        data[name] = {**data[name], **values}
    return data


def _problems(data: dict[str, Any]) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        validate_config(data, source="exp.yaml")
    return caught.value.message.splitlines()[1:]


# ------------------------------------------------------------------------------ defaults


def test_new_keys_have_defaults():
    config = validate_config(BASE, source="exp.yaml")
    assert isinstance(config, TrainConfig)
    assert config.model.input is None
    assert (config.data.pairs, config.data.allow_input_mismatch) == (False, False)
    assert config.data.loader.balance is None
    train = config.train
    assert (train.monitor, train.mode) == ("val/video_auc", "max")
    assert train.early_stop is None
    assert (train.nan_tolerance, train.heartbeat_steps) == (3, 50)
    assert train.loggers == ["csv", "tensorboard"]
    assert train.lightning == {}


# ------------------------------------------------------------------------------ model.input


def test_input_overrides_keep_exactly_the_keys_written():
    data = _config(model={"input": {"crop": "full-frame", "crop_scale": None, "size": [32, 32]}})
    config = validate_config(data, source="exp.yaml")
    assert isinstance(config, TrainConfig)
    assert config.model.input is not None
    assert config.model.input.overrides() == {
        "crop": "full-frame",
        "crop_scale": None,
        "size": (32, 32),
    }
    dumped = config.model_dump(mode="json", by_alias=True)
    assert dumped["model"]["input"] == {"crop": "full-frame", "crop_scale": None, "size": [32, 32]}
    # the resolved form validates back to the same config, and so to the same fingerprint
    again = validate_config(dumped, source="config.resolved.yaml")
    assert again.model_dump(mode="json", by_alias=True) == dumped
    assert config_fingerprint(again, dumped) == config_fingerprint(config, dumped)


def test_input_overrides_refuse_null_where_the_spec_needs_a_value():
    lines = _problems(_config(model={"input": {"size": None, "crop": None}}))
    assert "  model.input.crop: may not be null" in lines
    assert "  model.input.size: may not be null" in lines


def test_input_overrides_name_unknown_keys_and_bad_values():
    lines = _problems(_config(model={"input": {"crop_scal": 1.3, "crop": "ful-frame"}}))
    assert any(
        line.startswith("  model.input.crop_scal: unknown key (did you mean 'crop_scale'")
        for line in lines
    )
    assert (
        "  model.input.crop: 'ful-frame' is not one of ['face', 'full-frame'] "
        "(did you mean 'full-frame'?)"
    ) in lines


# ------------------------------------------------------------------------------ data


def test_balance_is_one_of_three_modes():
    lines = _problems(_config(data={"loader": {"batch_size": 4, "num_workers": 0, "balance": "x"}}))
    assert lines == ["  data.loader.balance: 'x' is not one of ['none', 'video-label', 'source']"]
    typo = {"batch_size": 4, "num_workers": 0, "balance": "video-labl"}
    assert "(did you mean 'video-label'?)" in _problems(_config(data={"loader": typo}))[0]


def test_pairs_and_allow_input_mismatch_are_booleans():
    config = validate_config(
        _config(data={"pairs": True, "allow_input_mismatch": True}), source="exp.yaml"
    )
    assert isinstance(config, TrainConfig)
    assert (config.data.pairs, config.data.allow_input_mismatch) == (True, True)


# ------------------------------------------------------------------------------ train


def test_early_stop_needs_a_positive_patience():
    config = validate_config(_config(train={"early_stop": {"patience": 2}}), source="x")
    assert isinstance(config, TrainConfig)
    assert config.train.early_stop is not None
    assert (config.train.early_stop.patience, config.train.early_stop.min_delta) == (2, 0.0)
    lines = _problems(_config(train={"early_stop": {"patience": 0, "min_delt": 0.1}}))
    assert any(line.startswith("  train.early_stop.patience: ") for line in lines)
    assert "  train.early_stop.min_delt: unknown key (did you mean 'min_delta'?)" in lines


def test_guards_must_be_positive():
    lines = _problems(_config(train={"nan_tolerance": 0, "heartbeat_steps": 0}))
    assert any(line.startswith("  train.nan_tolerance: ") for line in lines)
    assert any(line.startswith("  train.heartbeat_steps: ") for line in lines)


def test_loggers_are_named():
    lines = _problems(_config(train={"loggers": ["csv", "tensorbord"]}))
    assert lines == [
        "  train.loggers[1]: 'tensorbord' is not one of ['csv', 'tensorboard', 'wandb'] "
        "(did you mean 'tensorboard'?)"
    ]


def test_training_runs_on_one_device():
    lines = _problems(_config(train={"devices": 2}))
    assert len(lines) == 1
    assert lines[0].startswith("  train.devices: training runs in a single process")


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("enable_checkpointing", True, "safetensors"),
        ("strategy", "ddp", "single process"),
        ("devices", 2, "single process"),
        ("devices", "auto", "single process"),
        ("devices", [0, 1], "single process"),
        ("num_nodes", 2, "single process"),
        ("callbacks", [], "run directory"),
        ("logger", False, "run directory"),
        ("default_root_dir", "/tmp/x", "run directory"),
        ("max_epochs", 3, "train.max_epochs"),
        ("precision", "16-mixed", "train.precision"),
    ],
)
def test_lightning_passthrough_refuses_keys_that_break_the_run(key, value, reason):
    (line,) = _problems(_config(train={"lightning": {key: value}}))
    assert line.startswith(f"  train.lightning: {key!r} ")
    assert reason in line


def test_lightning_passthrough_keeps_everything_else():
    options = {"deterministic": True, "accelerator": "cpu", "devices": 1, "log_every_n_steps": 1}
    config = validate_config(_config(train={"lightning": options}), source="x")
    assert isinstance(config, TrainConfig)
    assert config.train.lightning == options
    one_gpu = validate_config(_config(train={"lightning": {"devices": [1]}}), source="x")
    assert isinstance(one_gpu, TrainConfig)
    assert one_gpu.train.lightning == {"devices": [1]}


# ------------------------------------------------------------------------------ components


def _supplying_registries() -> dict[str, Registry]:
    registries = {
        name: Registry(name)
        for name in ("backbones", "temporal_pools", "heads", "losses", "layers")
    }
    target = "tests.unit.core._targets"
    registries["backbones"].add("tiny-cnn", f"{target}:open_kwargs", summary="x")
    registries["temporal_pools"].add("mean", f"{target}:make_head", summary="x")
    registries["heads"].add("linear", f"{target}:make_head", summary="x")
    registries["losses"].add("bce", f"{target}:open_kwargs", summary="x")
    registries["layers"].add("srm", f"{target}:make_stem", summary="x")
    return registries


def test_values_the_framework_supplies_are_not_reported_missing():
    # a pool and a head get `dim` from the backbone when the detector is built, and a stem gets
    # `in_channels`: a config that leaves them out is complete.
    registries = _supplying_registries()
    data = _config(model={"stem": {"name": "srm"}})
    check_components(validate_config(data, source="x"), registries)

    # ...but anything else about them still is.
    data["model"]["head"] = {"name": "linear", "dropot": 0.1}
    with pytest.raises(ConfigError) as caught:
        check_components(validate_config(data, source="x"), registries)
    assert caught.value.message.splitlines() == [
        "1 invalid component value(s)",
        "  model.head.dropot: unknown parameter (did you mean 'dropout'?)",
    ]


def test_a_dim_the_framework_supplies_may_not_be_set():
    # the detector is built with the backbone's output size as `dim`; a second one would clash.
    data = _config(
        model={
            "temporal_pool": {"name": "mean", "dim": 32},
            "head": {"name": "linear", "dim": 64},
        }
    )
    with pytest.raises(ConfigError) as caught:
        check_components(validate_config(data, source="x"), _supplying_registries())
    lines = caught.value.message.splitlines()
    assert lines[0] == "2 invalid component value(s)"
    assert "  model.temporal_pool.dim: set from the backbone's output size; leave it out" in lines
    assert "  model.head.dim: set from the backbone's output size; leave it out" in lines


def test_a_stem_may_set_its_own_in_channels():
    # a stem's in_channels defaults to the 3 image channels, but a config may choose another.
    data = _config(model={"stem": {"name": "srm", "in_channels": 1}})
    check_components(validate_config(data, source="x"), _supplying_registries())
