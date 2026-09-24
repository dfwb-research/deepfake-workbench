import pytest

from dfwb.core.config.dotted import get_path, parse_path, set_path
from dfwb.core.config.merge import deep_merge
from dfwb.core.errors import ConfigError


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a", ("a",)),
        ("data.train[0].split", ("data", "train", 0, "split")),
        ("a[1][2].b", ("a", 1, 2, "b")),
        ("eval.metrics[3]", ("eval", "metrics", 3)),
    ],
)
def test_parse_path(text, expected):
    assert parse_path(text) == expected


@pytest.mark.parametrize(
    "text", ["", ".a", "a.", "a..b", "[0]", "a[x]", "a[1", "a]b", "a.[0]", "a[0]b"]
)
def test_parse_path_rejects(text):
    with pytest.raises(ConfigError):
        parse_path(text)


def test_get_and_set_path():
    data = {"data": {"train": [{"split": "train"}]}}
    assert get_path(data, ("data", "train", 0, "split")) == "train"
    with pytest.raises(KeyError):
        get_path(data, ("data", "val"))
    set_path(data, ("data", "train", 0, "split"), "val")
    set_path(data, ("optim", "groups", "backbone", "lr_scale"), 0.1)
    assert data["data"]["train"][0]["split"] == "val"
    assert data["optim"] == {"groups": {"backbone": {"lr_scale": 0.1}}}


def test_set_path_errors():
    data = {"a": [1], "b": 3}
    with pytest.raises(ConfigError, match="out of range"):
        set_path(data, ("a", 5), 1)
    with pytest.raises(ConfigError, match="not a list"):
        set_path(data, ("b", 0), 1)
    with pytest.raises(ConfigError, match="not a mapping"):
        set_path(data, ("b", "c"), 1)


def test_maps_merge_lists_and_scalars_replace():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1, "keep": True}
    over = {"a": {"y": [9], "z": 3}, "b": {"now": "a map"}}
    assert deep_merge(base, over) == {
        "a": {"x": 1, "y": [9], "z": 3},
        "b": {"now": "a map"},
        "keep": True,
    }
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1, "keep": True}  # inputs untouched


def test_plus_prefix_appends():
    base = {"eval": {"metrics": ["auc"]}}
    assert deep_merge(base, {"eval": {"+metrics": ["eer"]}}) == {
        "eval": {"metrics": ["auc", "eer"]}
    }
    assert deep_merge({}, {"eval": {"+metrics": ["eer"]}}) == {"eval": {"metrics": ["eer"]}}


@pytest.mark.parametrize(
    ("base", "over", "message"),
    [
        ({"m": 1}, {"+m": [1]}, "cannot append to a int"),
        ({}, {"+m": 1}, "must be a list"),
        ({}, {"m": [1], "+m": [2]}, r"both 'm' and '\+m'"),
        ({}, {"+": [1]}, "needs a key name"),
    ],
)
def test_plus_prefix_errors(base, over, message):
    with pytest.raises(ConfigError, match=message):
        deep_merge(base, over)


def test_merge_returns_copies():
    base = {"a": [{"b": 1}]}
    merged = deep_merge(base, {})
    merged["a"][0]["b"] = 2
    assert base["a"][0]["b"] == 1
