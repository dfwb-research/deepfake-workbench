"""``docs/guides/reproduce-a-benchmark.md``'s toyfake command block, run verbatim.

The guide is prose someone reads and copies commands from -- a unit test that calls the CLI
directly with the right roots already set up can pass even when the *documented* command line is
wrong (e.g. a shell variable, such as ``$DFWB_DATASETS_ROOT``, that only ever exists inside the
`.env` file `dfwb` loads itself, never in the shell that runs the command). So this test extracts
the guide's own toyfake code block, byte for byte, and runs it -- through ``bash``, in a fresh
rsync copy of the clone, with no roots pre-set beyond what `./scripts/setup.sh` and its `.env`
give it -- exactly as a new user pasting the block would.

Needs the ``preprocess`` and ``train`` extras (``uv sync`` inside the scratch clone installs them
for real; no test in this file mocks or skips that step). Marked ``slow``.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs" / "guides" / "reproduce-a-benchmark.md"


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


def _toyfake_block() -> str:
    text = GUIDE.read_text("utf-8")
    heading = "## The toyfake benchmark: nothing to download"
    assert heading in text, "the guide's toyfake section heading moved or was reworded"
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]
    match = re.search(r"```bash\n(.*?)```", section, re.DOTALL)
    assert match, "no ```bash block found under the guide's toyfake section"
    return match.group(1)


def _clone(dest: Path) -> None:
    dest.mkdir()
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude=/.venv",
            "--exclude=/.git",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=/.pytest_cache",
            "--exclude=/.mypy_cache",
            "--exclude=/.ruff_cache",
            "--exclude=/.import_linter_cache",
            "--exclude=/.hypothesis",
            "--exclude=.coverage*",
            "--exclude=/runs",
            "--exclude=/data",
            "--exclude=/site",
            f"{REPO}/",
            f"{dest}/",
        ],
        check=True,
    )


def test_the_guides_toyfake_block_runs_verbatim_in_a_fresh_clone(tmp_path):
    block = _toyfake_block()
    clone = tmp_path / "clone"
    _clone(clone)

    script = clone / "_guide_toyfake_block.sh"
    script.write_text("set -euo pipefail\n" + block, encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script)],
        cwd=clone,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # the run and its scores landed exactly where the guide (and .env's own defaults) say they do
    assert (clone / "data" / "runs" / "toy-cpu" / "latest").exists()
    assert list((clone / "data" / "runs" / "scores").rglob("*.scores.csv"))
