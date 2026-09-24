"""Tests for ``dfwb.protocols.newpack``: scaffolding a new protocol pack distribution."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError
from dfwb.protocols.lint import lint_pack
from dfwb.protocols.newpack import new_pack

_EXPECTED_FILES = {
    Path("pyproject.toml"),
    Path("README.md"),
    Path("NOTICE.md"),
    Path("LICENSE-DATA"),
    Path("src/my_pack/__init__.py"),
    Path("src/my_pack/packs/pack.yaml"),
}


def test_scaffold_writes_the_expected_files(tmp_path):
    directory = tmp_path / "my-pack"

    written = new_pack(directory, name="my-pack", author="Ada Lovelace")

    assert written == sorted(written)
    assert {p.relative_to(directory) for p in written} == _EXPECTED_FILES
    assert all(p.is_file() for p in written)


def test_scaffold_creates_the_directory_when_it_does_not_exist(tmp_path):
    directory = tmp_path / "does" / "not" / "exist-yet"

    new_pack(directory, name="my-pack")

    assert directory.is_dir()


def test_scaffolded_pack_lints_with_no_issues(tmp_path):
    directory = tmp_path / "my-pack"

    new_pack(directory, name="my-pack")

    assert lint_pack(directory / "src" / "my_pack" / "packs") == []


def test_pyproject_parses_and_the_entry_point_targets_register(tmp_path):
    directory = tmp_path / "my-pack"

    new_pack(directory, name="my-pack", author="Ada Lovelace")

    data = tomllib.loads((directory / "pyproject.toml").read_text("utf-8"))

    assert data["build-system"]["requires"] == ["hatchling>=1.27"]
    assert data["project"]["name"] == "my-pack"
    assert data["project"]["license"] == "MIT AND CC-BY-4.0"
    assert data["project"]["license-files"] == ["LICENSE", "LICENSE-DATA"]
    assert data["project"]["dependencies"] == []
    assert data["project"]["entry-points"]["dfwb.plugins"] == {"my-pack": "my_pack:register"}


class _RecordingRegistry:
    """A ten-line fake standing in for one registry of the real ``PluginAPI``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def add(self, key: str, *, target: str, summary: str, **meta: object) -> None:
        self.calls.append({"key": key, "target": target, "summary": summary, **meta})


class _RecordingAPI:
    def __init__(self) -> None:
        self.protocol_packs = _RecordingRegistry()


def _load_generated_module(directory: Path, import_name: str):
    init_path = directory / "src" / import_name / "__init__.py"
    spec = importlib.util.spec_from_file_location(import_name, init_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_register_adds_exactly_one_protocol_pack(tmp_path):
    directory = tmp_path / "my-pack"
    new_pack(directory, name="my-pack")

    module = _load_generated_module(directory, "my_pack")
    assert module.DFWB_PLUGIN_API == ">=1.0,<2"

    api = _RecordingAPI()
    module.register(api)

    assert len(api.protocol_packs.calls) == 1
    (call,) = api.protocol_packs.calls
    assert call["key"] == "my-pack"
    assert call["target"] == "my_pack:packs"
    assert isinstance(call["summary"], str)
    assert call["summary"]


def test_refuses_a_non_empty_directory(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "keep.txt").write_text("hi")

    with pytest.raises(ConfigError):
        new_pack(directory, name="my-pack")


def test_refuses_a_directory_that_is_only_a_stray_dotfile(tmp_path):
    # "non-empty" means any entry at all, hidden or not.
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / ".keep").write_text("")

    with pytest.raises(ConfigError):
        new_pack(directory, name="my-pack")


@pytest.mark.parametrize(
    "name", ["My_Pack", "My Pack", "-leading", "trailing-", "", "a_b", "a--b", "déjà-vu"]
)
def test_refuses_a_name_that_is_not_lower_kebab(tmp_path, name):
    with pytest.raises(ConfigError):
        new_pack(tmp_path / "out", name=name)
