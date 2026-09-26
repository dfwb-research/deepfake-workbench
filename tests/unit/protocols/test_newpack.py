"""Tests for ``dfwb.protocols.newpack``: scaffolding a new protocol pack distribution."""

from __future__ import annotations

import importlib.util
import subprocess
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError
from dfwb.protocols.lint import lint_pack
from dfwb.protocols.newpack import new_pack

_EXPECTED_FILES = {
    Path("pyproject.toml"),
    Path("README.md"),
    Path("NOTICE.md"),
    Path("LICENSE"),
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


def test_license_is_mit_with_the_author_and_current_year(tmp_path):
    directory = tmp_path / "my-pack"

    new_pack(directory, name="my-pack", author="Ada Lovelace")

    text = (directory / "LICENSE").read_text("utf-8")
    year = datetime.now(UTC).year
    assert text.startswith("MIT License")
    assert f"Copyright (c) {year} Ada Lovelace" in text


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


@pytest.mark.parametrize(
    "author",
    [
        "",
        "   ",
        "line one\nline two",
        "x" * (100 + 1),
        'Quote"Mark',
        "Back\\Slash",
        "Back`Tick",
    ],
)
def test_refuses_a_bad_author(tmp_path, author):
    directory = tmp_path / "out"

    with pytest.raises(ConfigError):
        new_pack(directory, name="my-pack", author=author)

    assert not directory.exists()


def test_refuses_a_bad_name_before_writing_anything(tmp_path):
    directory = tmp_path / "out"

    with pytest.raises(ConfigError):
        new_pack(directory, name="Not Kebab")

    assert not directory.exists()


def test_declared_license_files_all_exist(tmp_path):
    # Fast, hermetic regression cover for the LICENSE/license-files mismatch: no network or build
    # tooling needed, just that every path pyproject.toml's ``license-files`` names is really on
    # disk in the scaffold -- the exact thing that was wrong before ``LICENSE.tmpl`` was added.
    directory = tmp_path / "my-pack"
    new_pack(directory, name="my-pack")

    data = tomllib.loads((directory / "pyproject.toml").read_text("utf-8"))

    license_files = data["project"]["license-files"]
    assert license_files  # not empty -- an empty list would vacuously "pass" the check below
    for relative in license_files:
        assert (directory / relative).is_file(), f"declared license-files entry missing: {relative}"


@pytest.mark.network
def test_scaffolded_package_builds_and_ships_its_pack_yaml(tmp_path):
    # An end-to-end check that the generated pyproject.toml is not just well-formed but actually
    # buildable, and that hatchling's default packaging really does carry the pack's data (the
    # generated package has no ``.py``-only assumption to lean on -- ``pack.yaml`` is the only file
    # under ``packs/``). Marked ``network``: on a cold ``uv`` cache / offline machine, the isolated
    # build environment ``uv build`` creates for the generated package must resolve ``hatchling``
    # (its own build-system requirement) from PyPI, which is unavailable in CI.
    directory = tmp_path / "my-pack"
    new_pack(directory, name="my-pack")
    out_dir = tmp_path / "dist"

    result = subprocess.run(
        ["uv", "build", str(directory), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
    assert any(name.endswith("my_pack/packs/pack.yaml") for name in names), names


def test_the_readme_describes_list_and_recipe_datasets_and_the_dfwb_they_need(tmp_path):
    directory = tmp_path / "my-pack"

    new_pack(directory, name="my-pack")

    readme = " ".join((directory / "README.md").read_text("utf-8").split())
    assert "`distribution: list`" in readme
    assert "`distribution: recipe`" in readme
    assert "dfwb protocols materialize <dataset_id>" in readme
    assert "dfwb protocols lint --release" in readme
    assert "a dfwb that reads the recipe hash fields: 0.1.0b3 or later, once released" in readme
    assert "its `pyproject.toml` pins no dfwb version" in readme
    # The scaffold stays dependency-free; the requirement lives in the README instead.
    pyproject = tomllib.loads((directory / "pyproject.toml").read_text("utf-8"))
    assert pyproject["project"]["dependencies"] == []
