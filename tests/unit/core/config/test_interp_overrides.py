import pytest

from dfwb.core.config.interp import interpolate
from dfwb.core.config.overrides import apply_overrides, parse_override
from dfwb.core.errors import ConfigError


def test_env_with_and_without_default():
    data = {"a": "${env:HOME_DIR}", "b": "${env:MISSING,./runs}", "c": "x-${env:HOME_DIR}-y"}
    assert interpolate(data, env={"HOME_DIR": "/h"}) == {"a": "/h", "b": "./runs", "c": "x-/h-y"}


def test_missing_env_without_default():
    with pytest.raises(ConfigError) as info:
        interpolate({"run": {"root": "${env:NOPE}"}}, env={})
    assert info.value.message == "run.root: environment variable NOPE is not set"
    assert "${env:NOPE,<default>}" in info.value.hint


def test_ref_keeps_type_when_whole_value_and_resolves_chains():
    data = {
        "a": [1, 2],
        "b": "${ref:a}",
        "c": "${ref:b}",
        "d": "n=${ref:e.f}",
        "e": {"f": 3, "g": True},
        "h": "flag=${ref:e.g}",
        "i": "${ref:list[1].x}",
        "list": [{"x": 0}, {"x": "${env:X}"}],
    }
    out = interpolate(data, env={"X": "ten"})
    assert out["b"] == [1, 2]
    assert out["c"] == [1, 2]
    assert out["d"] == "n=3"
    assert out["h"] == "flag=true"
    assert out["i"] == "ten"
    assert data["b"] == "${ref:a}"  # input untouched


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"a": "${ref:b}", "b": "${ref:a}"}, "reference cycle"),
        ({"a": "${ref:a}"}, "reference cycle"),
        ({"a": "${ref:nope}"}, "points to a missing key"),
        ({"a": "${foo:bar}"}, "unknown interpolation 'foo'"),
        ({"a": "${env:X"}, "malformed interpolation"),
        ({"a": "x${ref:b}", "b": [1]}, "cannot embed a list"),
    ],
)
def test_interpolation_errors(data, message):
    with pytest.raises(ConfigError, match=message):
        interpolate(data, env={})


@pytest.mark.parametrize(
    ("text", "path", "value"),
    [
        ("optim.lr=3e-4", ("optim", "lr"), 3e-4),  # a float, as in a config file
        ("train.max_epochs=5", ("train", "max_epochs"), 5),
        ("data.train[0].split=train", ("data", "train", 0, "split"), "train"),
        ("eval.metrics=[auc, eer]", ("eval", "metrics"), ["auc", "eer"]),
        ("run.name=a=b", ("run", "name"), "a=b"),
        ("run.name=", ("run", "name"), ""),
        ("x={broken", ("x",), "{broken"),
    ],
)
def test_parse_override(text, path, value):
    assert parse_override(text) == (path, value)


@pytest.mark.parametrize("text", ["novalue", "=3", " =3"])
def test_parse_override_rejects(text):
    with pytest.raises(ConfigError, match="invalid override"):
        parse_override(text)


def test_apply_overrides_in_order_on_a_copy():
    data = {"optim": {"lr": 1}, "data": {"train": [{"split": "val"}]}}
    out = apply_overrides(data, ["optim.lr=2", "optim.lr=3", "data.train[0].split=train"])
    assert out == {"optim": {"lr": 3}, "data": {"train": [{"split": "train"}]}}
    assert data["optim"]["lr"] == 1
