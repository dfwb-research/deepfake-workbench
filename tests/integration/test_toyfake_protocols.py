"""toyfake from nothing to a verified protocol, through the ``dfwb`` command, with torch blocked.

``datasets synth --no-media`` writes the tree, ``inventory build`` scans it, and the built-in
toyfake pack then covers every video: ``protocols verify`` exits 0 and ``protocols list`` shows
each toyfake scheme.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DFWB = Path(sys.executable).parent / "dfwb"

_BLOCKER = """\
import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] == "torch":
            raise ModuleNotFoundError(f"No module named {fullname!r} (blocked)", name=fullname)
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _runner(tmp_path: Path):
    site = tmp_path / "blocker"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_BLOCKER)
    home = tmp_path / "home"
    home.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("DFWB_")}
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]),
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "DFWB_WORK_ROOT": str(tmp_path / "work"),
        }
    )

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(DFWB), *args], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
        )

    return run


def test_synth_inventory_verify_and_list_with_torch_blocked(tmp_path):
    dfwb = _runner(tmp_path)
    out = tmp_path / "datasets"

    synth = dfwb("datasets", "synth", "toyfake", "--out", str(out), "--no-media", "--json")
    assert synth.returncode == 0, synth.stderr
    root = Path(json.loads(synth.stdout)["root"])
    assert root == out / "toyfake"

    built = dfwb("inventory", "build", "toyfake", "--root", str(root), "--json")
    assert built.returncode == 0, built.stderr
    assert json.loads(built.stdout)["by_task"] == {"REAL": 80, "BLEND_A": 60, "BLEND_B": 60}

    verified = dfwb("protocols", "verify", "toyfake/official", "--json")
    assert verified.returncode == 0, verified.stderr + verified.stdout
    report = json.loads(verified.stdout)
    assert report["pack"] == "toyfake"
    assert report["counts"]["have"] == 200
    assert report["counts"]["missing"] == report["counts"]["extra"] == 0

    listed = dfwb("protocols", "list", "--json")
    assert listed.returncode == 0, listed.stderr
    schemes = {
        row["scheme"]: row["default"]
        for row in json.loads(listed.stdout)
        if row["dataset_id"] == "toyfake"
    }
    assert schemes == {
        "official": True,
        "ident-72-14-14": False,
        "all-test": False,
        "benchmark": False,
    }
    human = dfwb("protocols", "list")
    assert human.returncode == 0, human.stderr
    assert "toyfake/official*" in human.stdout

    for step in (synth, built, verified, listed, human):
        assert "blocked" not in step.stderr
