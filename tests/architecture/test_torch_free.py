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
    ["plugins", "list", "--all"],
    ["protocols", "list"],
    ["config", "templates"],
    ["schema", "export", "c1"],
    ["schema", "export", "c2"],
    ["schema", "export", "c3"],
    ["schema", "export", "c4"],
    ["schema", "export", "c5"],
]


@pytest.mark.parametrize("args", TORCH_FREE_COMMANDS, ids=" ".join)
def test_cli_commands_run_with_torch_blocked(blocked, args):
    done = blocked([str(DFWB), *args], block=("torch",))
    assert done.returncode == 0, done.stderr


def test_config_and_lookup_commands_run_with_torch_blocked(blocked, tmp_path):
    steps = [
        (["config", "init", "--out", "exp.yaml"], 0),
        (["config", "show", "-c", "exp.yaml"], 0),
        (["config", "validate", "-c", "exp.yaml"], 2),  # no components are installed
        (["plugins", "info", "layers/srm"], 2),  # unknown key, with an install hint
    ]
    for args, code in steps:
        done = blocked([str(DFWB), *args], block=("torch",), cwd=tmp_path)
        assert done.returncode == code, (args, done.stderr)
        assert "blocked by dfwb tests" not in done.stderr, (args, done.stderr)
        if code:
            assert "hint: " in done.stderr


def test_protocols_verify_runs_with_torch_blocked(blocked, tmp_path):
    # No roots configured and no pack installed: still exercises verify's own code path (root
    # resolution, then load()) without importing torch.
    done = blocked([str(DFWB), "protocols", "verify", "nope"], block=("torch",), cwd=tmp_path)
    assert done.returncode == 2, done.stderr
    assert "hint: " in done.stderr
