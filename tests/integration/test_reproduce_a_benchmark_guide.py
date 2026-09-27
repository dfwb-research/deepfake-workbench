"""``docs/guides/reproduce-a-benchmark.md``'s toyfake command block, run verbatim.

The guide is prose someone reads and copies commands from -- a unit test that calls the CLI
directly with the right roots already set up can pass even when the *documented* command line is
wrong (e.g. a shell variable, such as ``$DFWB_DATASETS_ROOT``, that only ever exists inside the
`.env` file `dfwb` loads itself, never in the shell that runs the command). So this test extracts
the guide's own toyfake code block, byte for byte, and runs it -- through ``bash``, in a scratch
copy of the clone holding only what a fresh clone would, with no roots pre-set beyond what
`./scripts/setup.sh` and its `.env` give it, no ``DFWB_*`` variable from the shell running the
tests, and a scratch home and XDG directories (see :mod:`tests.integration._scratch_clone`) --
exactly as a new user pasting the block would.

It then runs the part of the real benchmark's steps 6 and 7 that toyfake can: the step-6
``score --suite`` command and the step-7 ``eval --suite`` command, with the run, suite and
device names swapped for toyfake's, so step 7's glob is checked against the files ``score
--suite`` actually writes.

Needs the ``preprocess`` and ``train`` extras (``uv sync`` inside the scratch clone installs them
for real; no test in this file mocks or skips that step). Marked ``slow``.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess

import pytest
from tests.integration._scratch_clone import REPO, copy_clone, isolated_env

GUIDE = REPO / "docs" / "guides" / "reproduce-a-benchmark.md"

# The real benchmark's names, and toyfake's in their place.
_TOYFAKE_FOR_REAL = {
    "ffpp-c23-vit-b16": "toy-cpu",
    "cross-dataset-v1": "toyfake",
    "--device cuda:0": "--device cpu",
}


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


def _section(heading: str) -> str:
    text = GUIDE.read_text("utf-8")
    assert heading in text, f"the guide's {heading!r} heading moved or was reworded"
    return text.split(heading, 1)[1].split("\n#", 1)[0]


def _toyfake_block() -> str:
    section = _section("## The toyfake benchmark: nothing to download")
    match = re.search(r"```bash\n(.*?)```", section, re.DOTALL)
    assert match, "no ```bash block found under the guide's toyfake section"
    return match.group(1)


def _first_command(heading: str) -> str:
    """The first command of the first ``bash`` block under ``heading``, as toyfake runs it."""
    match = re.search(r"```bash\n(.*?)```", _section(heading), re.DOTALL)
    assert match, f"no ```bash block found under {heading!r}"
    command = next(line for line in match.group(1).splitlines() if line.startswith("uv "))
    for real, toyfake in _TOYFAKE_FOR_REAL.items():
        command = command.replace(real, toyfake)
    return command


def test_the_guides_toyfake_block_runs_verbatim_in_a_fresh_clone(tmp_path):
    block = _toyfake_block()
    clone = tmp_path / "clone"
    copy_clone(clone)
    assert not (clone / ".env").exists()
    env = isolated_env(tmp_path)

    script = clone / "_guide_toyfake_block.sh"
    script.write_text("set -euo pipefail\n" + block, encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script)],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # the run and its scores landed exactly where the guide (and .env's own defaults) say they do
    assert (clone / "runs" / "toy-cpu" / "latest").exists()
    assert list((clone / "runs" / "scores").rglob("*.scores.csv"))
    assert not (clone / "data" / "runs").exists()

    # Steps 6 and 7 of the real benchmark, on toyfake: score the suite into its own --out, then
    # evaluate it with step 7's own glob.
    score = _first_command("### 6. Score the suite")
    evaluate = _first_command("### 7. Eval")
    assert "--suite toyfake" in score, score
    assert "--out runs/scores/toy-cpu" in score, score
    assert "--suite toyfake" in evaluate, evaluate
    assert "runs/scores/toy-cpu/" in evaluate, evaluate
    steps = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{score}\n{evaluate}"],
        cwd=clone,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert steps.returncode == 0, steps.stdout + steps.stderr
    written = sorted((clone / "runs" / "scores" / "toy-cpu").glob("*/*/*.scores.csv"))
    assert [path.parent.name for path in written] == ["toyfake-ident-72-14-14", "toyfake-official"]
    # the suite's aggregate rows, one per group, follow the per-file table
    assert re.search(r"^in-domain\s+auc\s+mean\s", steps.stdout, re.MULTILINE), steps.stdout
    assert re.search(r"^cross\s+auc\s+mean\s", steps.stdout, re.MULTILINE), steps.stdout
