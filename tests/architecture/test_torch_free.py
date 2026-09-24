"""Torch-free layers import, and torch-free commands run, with torch blocked."""

import json
import sys
from pathlib import Path

import pytest

DFWB = Path(sys.executable).parent / "dfwb"  # the console script of this environment

TORCH_FREE = ("dfwb.core", "dfwb.protocols", "dfwb.eval", "dfwb.preprocess", "dfwb.cli")

IMPORT_ALL = """
import importlib, json, pkgutil, sys
imported = []
for name in {packages!r}:
    package = importlib.import_module(name)
    imported.append(name)
    for info in pkgutil.walk_packages(package.__path__, name + "."):
        importlib.import_module(info.name)
        imported.append(info.name)
loaded = sorted(m for m in {watch!r} if m in sys.modules)
print(json.dumps({{"imported": imported, "loaded": loaded}}))
"""


def test_torch_free_layers_import_with_torch_blocked(blocked):
    code = IMPORT_ALL.format(packages=TORCH_FREE, watch=("torch",))
    done = blocked([sys.executable, "-c", code], block=("torch",))
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert set(TORCH_FREE) <= set(result["imported"])
    assert result["loaded"] == []


def test_core_imports_with_torch_and_numpy_blocked(blocked):
    code = IMPORT_ALL.format(packages=("dfwb.core",), watch=("torch", "numpy"))
    done = blocked([sys.executable, "-c", code], block=("torch", "numpy"))
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["loaded"] == []


# Every torch-free command; each CLI task adds its own lines.
TORCH_FREE_COMMANDS = [
    ["--help"],
    ["--version"],
    ["completion", "bash"],
    ["doctor"],
    ["doctor", "--json"],
]


@pytest.mark.parametrize("args", TORCH_FREE_COMMANDS, ids=" ".join)
def test_cli_commands_run_with_torch_blocked(blocked, args):
    done = blocked([str(DFWB), *args], block=("torch",))
    assert done.returncode == 0, done.stderr
