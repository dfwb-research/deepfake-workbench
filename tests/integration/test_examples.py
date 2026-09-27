"""``examples/`` scripts actually run: each one, in its own temporary directory and roots.

Every example is invoked exactly as `uv run python examples/<name>.py`, never with the real
datasets root, `~/.cache/dfwb` or `~/.local/state/dfwb` -- `DFWB_DATASETS_ROOT`, `DFWB_WORK_ROOT`
and `DFWB_RUNS_ROOT` all point inside `tmp_path`, and `HOME` is redirected too, so nothing escapes
it. Each script also creates and works inside its own `dfwb-example-*` directory for the loose
files it writes (a config, a detector module, a CSV, an output directory) that are not covered by
those three roots, so running an example from a real clone leaves exactly one new, easily
`.gitignore`d directory behind -- never files scattered across the clone root.

`import_foreign_scores.py` needs only the base install plus `eval` (no `torch`) and runs in well
under a second, so it is part of the default (fast) test selection. The other two need `preprocess`
and `train` and each run a short training loop, so they are `slow` and skipped by default; select
them with `pytest -m slow`.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("DFWB_")}
    env.update(
        {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "DFWB_DATASETS_ROOT": str(tmp_path / "datasets"),
            "DFWB_WORK_ROOT": str(tmp_path / "work"),
            "DFWB_RUNS_ROOT": str(tmp_path / "runs"),
        }
    )
    return env


def _run_example(name: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        cwd=tmp_path,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_import_foreign_scores_needs_no_torch(tmp_path):
    result = _run_example("import_foreign_scores.py", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    example_root = tmp_path / "dfwb-example-import"
    assert not (tmp_path / "imported").exists(), "must not leak into the caller's own directory"
    assert not (tmp_path / "my_scores.csv").exists(), "must not leak into the caller's directory"
    assert (example_root / "imported").is_dir()
    assert list((example_root / "imported").glob("*.scores.csv"))


@pytest.mark.slow
@pytest.mark.skipif(
    not (_installed("av") and _installed("cv2") and _installed("lightning")),
    reason="needs the preprocess and train extras",
)
def test_quickstart_toyfake(tmp_path):
    result = _run_example("quickstart_toyfake.py", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "runs" / "toy-cpu" / "latest").exists()
    assert list((tmp_path / "runs" / "scores").rglob("*.scores.csv"))
    assert not (tmp_path / "toy-cpu.yaml").exists(), "must not leak into the caller's directory"
    assert (tmp_path / "dfwb-example-toyfake" / "toy-cpu.yaml").exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not (_installed("av") and _installed("cv2") and _installed("torch")),
    reason="needs the preprocess and train extras",
)
def test_score_custom_detector(tmp_path):
    result = _run_example("score_custom_detector.py", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list((tmp_path / "runs" / "scores").rglob("*.scores.csv"))
    assert not (tmp_path / "my_detector.py").exists(), "must not leak into the caller's directory"
    assert (tmp_path / "dfwb-example-detector" / "my_detector.py").exists()
