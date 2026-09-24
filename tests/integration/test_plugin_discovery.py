"""Installed plugin distributions show up in `dfwb plugins list`; a failing one is isolated."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
DFWB = Path(sys.executable).parent / "dfwb"


def _install_as_distribution(site: Path, project: Path) -> None:
    """Make `project` importable and discoverable from `site`, exactly as an installed wheel is:
    the package directory plus a .dist-info folder holding METADATA and entry_points.txt."""
    import tomllib

    meta = tomllib.loads((project / "pyproject.toml").read_text())["project"]
    for package in (project / "src").iterdir():
        shutil.copytree(package, site / package.name)
    info = site / f"{meta['name'].replace('-', '_')}-{meta['version']}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {meta['name']}\nVersion: {meta['version']}\n"
    )
    lines = [
        "[dfwb.plugins]",
        *(f"{k} = {v}" for k, v in meta["entry-points"]["dfwb.plugins"].items()),
    ]
    (info / "entry_points.txt").write_text("\n".join(lines) + "\n")


def _dfwb(*args: str, exe: Path = DFWB, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(exe), *args], capture_output=True, text=True, env={**os.environ, **env}, check=False
    )


def _check(exe: Path, **env: str) -> None:
    done = _dfwb("plugins", "list", "--all", "--json", exe=exe, **env)
    assert done.returncode == 0, done.stderr
    data = json.loads(done.stdout)
    # Only the fake distributions' own entries are asserted here: the framework's own builtins
    # (provider "dfwb") grow as more of the framework is implemented, and are not this test's
    # concern -- it is about entry-point discovery of installed distributions.
    components = {
        f"{e['registry']}/{e['key']}": e["provider"]
        for e in data["entries"]
        if e["provider"] != "dfwb"
    }
    assert components == {
        "layers/fake-stem": "dfwb-fake-plugin-ok",
        "losses/fake-loss": "dfwb-fake-plugin-ok",
    }
    status = {p["name"]: (p["status"], p["reason"]) for p in data["plugins"]}
    assert status["dfwb"] == ("ok", None)
    assert status["fake-ok"] == ("ok", None)
    assert status["fake-broken"] == ("failed", "RuntimeError: simulated failure inside register()")

    human = _dfwb("plugins", "list", exe=exe, **env)
    assert "layers/fake-stem  dfwb-fake-plugin-ok  Fake stem layer" in human.stdout
    assert "1 plugin(s) failed or were skipped" in human.stderr

    lookup = _dfwb("plugins", "info", "heads/half-registered", exe=exe, **env)
    assert lookup.returncode == 2
    assert (
        "which failed to load (RuntimeError: simulated failure inside register())" in lookup.stderr
    )


def test_fake_distributions_on_the_path(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    for name in ("fake-plugin-ok", "fake-plugin-broken"):
        _install_as_distribution(site, FIXTURES / name)
    pythonpath = os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")])
    _check(DFWB, PYTHONPATH=pythonpath)

    off = json.loads(
        _dfwb(
            "plugins", "list", "--all", "--json", PYTHONPATH=pythonpath, DFWB_PLUGINS="none"
        ).stdout
    )
    # DFWB_PLUGINS=none disables discovered entry-point distributions, not the framework's own
    # builtins (provider "dfwb"), so only the fake distributions' entries must be gone.
    assert {e["provider"] for e in off["entries"]} <= {"dfwb"}
    assert {p["name"]: p["status"] for p in off["plugins"]} == {
        "dfwb": "ok",
        "fake-broken": "disabled",
        "fake-ok": "disabled",
    }
    one = json.loads(
        _dfwb(
            "plugins",
            "list",
            "--all",
            "--json",
            PYTHONPATH=pythonpath,
            DFWB_PLUGINS_DISABLE="dfwb-fake-plugin-broken",
        ).stdout
    )
    assert {p["name"]: p["status"] for p in one["plugins"]}["fake-broken"] == "disabled"


@pytest.mark.slow
@pytest.mark.network
def test_fake_distributions_installed_in_a_fresh_venv(tmp_path):
    """The same check against real wheels installed by uv into a brand-new virtual environment."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    venv = tmp_path / "venv"
    subprocess.run([uv, "venv", "--quiet", "--python", sys.executable, str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--quiet",
            "--python",
            str(python),
            str(REPO),
            str(FIXTURES / "fake-plugin-ok"),
            str(FIXTURES / "fake-plugin-broken"),
        ],
        check=True,
    )
    _check(venv / "bin" / "dfwb")
