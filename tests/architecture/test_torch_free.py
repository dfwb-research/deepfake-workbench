"""Torch-free layers import, and torch-free commands run, with torch blocked."""

import json
import sys
from pathlib import Path

import pytest
from tests.unit.cli._runs import FP_A, fake_run
from tests.unit.eval.conftest import make_meta, make_rows

from dfwb.core.records import write_scores

DFWB = Path(sys.executable).parent / "dfwb"  # the console script of this environment
_REPO_ROOT = Path(__file__).resolve().parents[2]

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


def test_eval_imports_with_torch_scipy_and_matplotlib_blocked(blocked):
    watch = ("torch", "scipy", "matplotlib", "pyarrow")
    code = IMPORT_ALL.format(packages=("dfwb.eval",), watch=watch)
    done = blocked([sys.executable, "-c", code], block=watch)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert "dfwb.eval" in result["imported"]
    assert result["loaded"] == []


def test_core_imports_with_torch_and_numpy_blocked(blocked):
    code = IMPORT_ALL.format(packages=("dfwb.core",), watch=("torch", "numpy"))
    done = blocked([sys.executable, "-c", code], block=("torch", "numpy"))
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["loaded"] == []


def test_inventory_and_protocols_import_with_torch_and_numpy_blocked(blocked):
    packages = ("dfwb.preprocess.inventory", "dfwb.protocols")
    code = IMPORT_ALL.format(packages=packages, watch=("torch", "numpy"))
    done = blocked([sys.executable, "-c", code], block=("torch", "numpy"))
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert set(packages) <= set(result["imported"])
    assert "dfwb.preprocess.inventory.runner" in result["imported"]
    assert result["loaded"] == []


# Every torch-free command; each CLI task adds its own lines.
TORCH_FREE_COMMANDS = [
    ["--help"],
    ["--version"],
    ["completion", "bash"],
    ["doctor"],
    ["doctor", "--json"],
    ["plugins", "list", "--all"],
    ["preprocess", "--help"],
    ["preprocess", "profiles"],
    ["preprocess", "profiles", "--json"],
    ["protocols", "list"],
    ["datasets", "list"],
    ["datasets", "list", "--json"],
    ["config", "templates"],
    ["schema", "export", "c1"],
    ["schema", "export", "c2"],
    ["schema", "export", "c3"],
    ["schema", "export", "c4"],
    ["schema", "export", "c5"],
    ["train", "--help"],
    ["runs", "--help"],
    ["runs", "list"],
    ["runs", "list", "--json"],
]


@pytest.mark.parametrize("args", TORCH_FREE_COMMANDS, ids=" ".join)
def test_cli_commands_run_with_torch_blocked(blocked, args):
    done = blocked([str(DFWB), *args], block=("torch",))
    assert done.returncode == 0, done.stderr


def test_config_and_lookup_commands_run_with_torch_blocked(blocked, tmp_path):
    steps = [
        (["config", "init", "--out", "exp.yaml"], 0),
        (["config", "show", "-c", "exp.yaml"], 0),
        (["config", "validate", "-c", "exp.yaml"], 5),  # every component needs torch
        (["plugins", "info", "layers/srm"], 2),  # unknown key, with an install hint
        (["datasets", "info", "nope"], 2),  # no such builder
        (["inventory", "build", "nope"], 2),
        (["inventory", "show", "nope"], 2),  # no work root, or no inventory
    ]
    for args, code in steps:
        done = blocked([str(DFWB), *args], block=("torch",), cwd=tmp_path)
        assert done.returncode == code, (args, done.stderr)
        assert "blocked by dfwb tests" not in done.stderr, (args, done.stderr)
        assert "No such command" not in done.stderr, (args, done.stderr)
        if code:
            assert "hint: " in done.stderr


def test_runs_commands_read_runs_with_torch_blocked(blocked, tmp_path, monkeypatch):
    run_dir = fake_run(tmp_path / "runs", "toy", "20260101-000000", 0, FP_A, latest=True)
    monkeypatch.setenv("DFWB_RUNS_ROOT", str(tmp_path / "runs"))
    for args in (
        ["runs", "list", "--json"],
        ["runs", "show", "toy"],
        ["runs", "show", str(run_dir)],
    ):
        done = blocked([str(DFWB), *args], block=("torch",), cwd=tmp_path)
        assert done.returncode == 0, (args, done.stderr)
        assert FP_A[:12] in done.stdout
    # training itself needs torch, and says how to get it
    done = blocked([str(DFWB), "train", "--resume", str(run_dir)], block=("torch",), cwd=tmp_path)
    assert done.returncode == 5, done.stderr
    assert "blocked by dfwb tests" not in done.stderr
    assert "deepfake-workbench[train]" in done.stderr


def test_protocols_verify_runs_with_torch_blocked(blocked, tmp_path):
    # No roots configured and no pack installed: still exercises verify's own code path (root
    # resolution, then load()) without importing torch.
    done = blocked([str(DFWB), "protocols", "verify", "nope"], block=("torch",), cwd=tmp_path)
    assert done.returncode == 2, done.stderr
    assert "hint: " in done.stderr


def test_eval_runs_with_torch_blocked(blocked, tmp_path):
    rows = make_rows(15, 15)
    meta = make_meta(coverage={"expected": 30, "ok": 30, "missing": 0, "error": 0})
    path, _ = write_scores(tmp_path / "a.scores.csv", rows, meta)
    done = blocked(
        [str(DFWB), "eval", str(path), "--metrics", "auc", "--bootstrap", "10", "--json"],
        block=("torch",),
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    data = json.loads(done.stdout)
    assert data["exit_code"] == 0


def test_pytest_collection_has_no_errors_with_torch_blocked(blocked):
    """Pins the torch-free CI job's own outcome: collecting the whole suite with torch
    unimportable must produce zero collection errors. A test module that needs torch (or one of
    timm, transformers, peft, lightning) is expected to skip at collection, the way
    ``pytest.importorskip`` at its top does; it must never blow up the run instead."""
    done = blocked(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        block=("torch",),
        cwd=_REPO_ROOT,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "errors during collection" not in done.stdout, done.stdout
    assert "\nERROR " not in done.stdout, done.stdout
