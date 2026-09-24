import json
from types import SimpleNamespace

import pytest

from dfwb.core import plugins


class EP:
    def __init__(self, name, register, dist):
        self.name, self.value, self.group = name, f"{name}_mod:register", "dfwb.plugins"
        self._register = register
        self.dist = SimpleNamespace(name=dist, version="0.9.0")

    def load(self):
        return self._register


@pytest.fixture
def installed(monkeypatch):
    def good(api):
        api.layers.add(
            "stem",
            target="tests.unit.core._targets:Stem",
            summary="A test stem",
            aliases=("st",),
            requires=("math",),
            paper="x",
        )

    def bad(api):
        raise RuntimeError("broken plugin")

    eps = {
        "dfwb.plugins": [EP("good", good, "good-dist"), EP("bad", bad, "bad-dist")],
        "dfwb.builtins": [],
    }
    monkeypatch.setattr(plugins, "_entry_points", lambda group: eps[group])


def test_list_human_and_failure_note(run, installed):
    result = run("plugins", "list")
    assert result.code == 0
    assert "layers/stem  good-dist  A test stem" in result.out
    assert "note: 1 plugin(s) failed or were skipped" in result.err
    result = run("plugins", "list", "--all")
    assert "bad     bad-dist   0.9.0    failed  RuntimeError: broken plugin" in result.out


def test_list_json(run, installed):
    data = json.loads(run("plugins", "list", "--all", "--json").out)
    assert data["entries"] == [
        {"registry": "layers", "key": "stem", "provider": "good-dist", "summary": "A test stem"}
    ]
    assert {p["name"]: p["status"] for p in data["plugins"]} == {"bad": "failed", "good": "ok"}
    assert [p["name"] for p in json.loads(run("plugins", "list", "--json").out)["plugins"]] == [
        "good"
    ]


def test_list_empty(run, monkeypatch):
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [])
    assert run("plugins", "list").out == "no components registered\n"


def test_info(run, installed):
    result = run("plugins", "info", "layers/ST")
    assert result.code == 0
    assert "qualified  good-dist:stem" in result.out
    assert "meta       paper=x" in result.out
    data = json.loads(run("plugins", "info", "layers/stem", "--json").out)
    assert data["requires"] == ["math"]


@pytest.mark.parametrize(
    ("ref", "code", "text"),
    [
        ("layers", 2, "expected REGISTRY/KEY"),
        ("layer/stem", 2, "did you mean 'layers'"),
        ("layers/stme", 2, "did you mean 'stem'"),
        ("heads/x", 2, "1 plugin(s) failed to load"),
    ],
)
def test_info_errors_have_hints(run, installed, ref, code, text):
    result = run("plugins", "info", ref)
    assert result.code == code
    assert text in result.err
    assert "hint: " in result.err


def test_group_without_a_subcommand_shows_its_help(run):
    result = run("plugins")
    assert result.code == 0
    assert "Commands:" in result.out
