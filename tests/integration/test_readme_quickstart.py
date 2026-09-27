"""The README's Quickstart, run verbatim in a scratch copy of the clone after ``setup.sh``.

A new user clones, runs ``./scripts/setup.sh`` (which writes ``.env``) and pastes the Quickstart's
command blocks in order. Those blocks name where the run and its score files end up
(``runs/toy-cpu/...``, ``runs/scores/...``), so they only work if ``.env`` puts the runs root
there too. This test does exactly what that user does -- ``setup.sh --dry-run`` for the local
part of the setup (``.env`` and the directories; the Quickstart's own ``uv run`` commands build
the environment), then every ``bash`` block of the Quickstart section, byte for byte -- in an
isolated environment (see :mod:`tests.integration._scratch_clone`). Marked ``slow``.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest
from tests.integration._scratch_clone import REPO, copy_clone, isolated_env

README = REPO / "README.md"


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_installed("av") and _installed("cv2") and _installed("lightning")),
        reason="needs the preprocess and train extras",
    ),
]


def _quickstart_blocks() -> list[str]:
    text = README.read_text("utf-8")
    heading = "\n## Quickstart\n"
    assert heading in text, "the README's Quickstart heading moved or was reworded"
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"```bash\n(.*?)```", section, re.DOTALL)
    assert len(blocks) == 3, "expected the Quickstart's three bash blocks"
    return blocks


def test_the_readme_quickstart_runs_verbatim_after_setup(tmp_path):
    clone = tmp_path / "clone"
    copy_clone(clone)
    assert not (clone / ".env").exists()
    env = isolated_env(tmp_path)

    setup = subprocess.run(
        ["bash", "scripts/setup.sh", "--dry-run"],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert setup.returncode == 0, setup.stdout + setup.stderr

    script = clone / "_readme_quickstart.sh"
    script.write_text("set -euo pipefail\n" + "\n".join(_quickstart_blocks()), encoding="utf-8")
    result = subprocess.run(
        ["bash", str(script)],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # Everything landed where the README says, inside the clone and the scratch home.
    assert (Path(env["HOME"]) / "datasets" / "toyfake").is_dir()
    assert (clone / "runs" / "toy-cpu" / "latest").exists()
    scores = clone / "runs" / "scores" / "tiny-cnn-mean-linear"
    assert list((scores / "toyfake-official").glob("*.scores.csv"))
    assert len(list(scores.glob("toyfake-*/*.scores.csv"))) == 2
    assert (clone / "report" / "report.md").is_file()
    assert not (clone / "data" / "runs").exists()
