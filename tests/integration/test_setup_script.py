"""``scripts/setup.sh``, run in a temporary copy of the clone.

No test here touches the network: real ``uv sync`` and ``dfwb doctor`` calls only ever happen
under ``--dry-run``, which prints them instead of running them (see the script's own header for
why the flag still performs the local, offline parts of setup -- copying ``.env`` and creating
the data directories -- for real, since those are exactly what idempotence and the
never-overwrite rule need to be checked against).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SETUP_SH = REPO / "scripts" / "setup.sh"
ENV_EXAMPLE = REPO / ".env.example"
INSTALL_MD = REPO / "docs" / "install.md"
QUICKSTART_MD = REPO / "docs" / "quickstart.md"


def _clone(tmp_path: Path) -> Path:
    """A minimal copy of the clone: just what ``setup.sh`` looks at or writes next to itself."""
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SETUP_SH, tmp_path / "scripts" / "setup.sh")
    shutil.copy2(ENV_EXAMPLE, tmp_path / ".env.example")
    return tmp_path


def _run(clone: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/setup.sh", *args],
        cwd=clone,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_setup_sh_is_executable_and_has_a_bash_shebang():
    assert SETUP_SH.stat().st_mode & 0o111, "scripts/setup.sh must be executable"
    first_line = SETUP_SH.read_text("utf-8").splitlines()[0]
    assert first_line == "#!/usr/bin/env bash"


def test_dry_run_never_touches_the_network_or_the_venv(tmp_path):
    clone = _clone(tmp_path)
    result = _run(clone, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "+ uv sync" in result.stdout
    assert "+ uv run dfwb doctor" in result.stdout
    assert not (clone / ".venv").exists()


def test_dry_run_sets_up_env_and_the_data_directories_for_real(tmp_path):
    clone = _clone(tmp_path)
    result = _run(clone, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert (clone / ".env").is_file()
    assert (clone / ".env").read_text("utf-8") == ENV_EXAMPLE.read_text("utf-8")
    for name in ("datasets", "work", "runs", "cache"):
        assert (clone / "data" / name).is_dir()


def test_dry_run_is_idempotent_and_never_overwrites_an_existing_env(tmp_path):
    clone = _clone(tmp_path)
    (clone / ".env").write_text("DFWB_CUSTOM=keep-me\n", encoding="utf-8")

    first = _run(clone, "--dry-run")
    assert first.returncode == 0, first.stderr
    assert "leaving it alone" in first.stdout
    assert (clone / ".env").read_text("utf-8") == "DFWB_CUSTOM=keep-me\n"

    second = _run(clone, "--dry-run")
    assert second.returncode == 0, second.stderr
    assert (clone / ".env").read_text("utf-8") == "DFWB_CUSTOM=keep-me\n"


def test_default_extras_match_the_quickstart_cpu_recipe(tmp_path):
    quickstart = QUICKSTART_MD.read_text("utf-8")
    assert "uv sync --extra train --extra preprocess" in quickstart

    clone = _clone(tmp_path)
    no_flags = _run(clone, "--dry-run")
    cpu_flag = _run(clone, "--dry-run", "--cpu")
    assert no_flags.returncode == cpu_flag.returncode == 0
    for result in (no_flags, cpu_flag):
        assert "+ uv sync --locked --extra train --extra preprocess" in result.stdout


def test_preprocess_and_train_flags_add_only_their_own_extra(tmp_path):
    clone = _clone(tmp_path)
    preprocess_only = _run(clone, "--dry-run", "--preprocess")
    train_only = _run(clone, "--dry-run", "--train")
    assert preprocess_only.returncode == train_only.returncode == 0
    assert "+ uv sync --locked --extra preprocess" in preprocess_only.stdout
    assert "--extra train" not in preprocess_only.stdout.split("uv sync")[1].split("\n")[0]
    assert "+ uv sync --locked --extra train" in train_only.stdout
    assert "--extra preprocess" not in train_only.stdout.split("uv sync")[1].split("\n")[0]


def test_gpu_flag_reproduces_installs_own_cuda_recipe_exactly(tmp_path):
    install_md = INSTALL_MD.read_text("utf-8")
    match = re.search(r"To train on a GPU with `uv`.*?```bash\n(.*?)```", install_md, re.DOTALL)
    assert match, "docs/install.md's GPU recipe code block moved or was reworded"
    recipe_lines = [line for line in match.group(1).splitlines() if line.strip()]
    # The first line (`uv sync --extra train`) is what setup.sh actually runs (with --locked);
    # every line after it is manual and driver-specific, and must be printed verbatim.
    assert recipe_lines[0] == "uv sync --extra train"
    manual_lines = recipe_lines[1:]

    clone = _clone(tmp_path)
    result = _run(clone, "--dry-run", "--gpu")
    assert result.returncode == 0, result.stderr
    assert "+ uv sync --locked --extra train" in result.stdout
    for line in manual_lines:
        assert line in result.stdout, f"missing from --gpu output: {line!r}"


def test_unknown_flag_is_an_error(tmp_path):
    clone = _clone(tmp_path)
    result = _run(clone, "--bogus")
    assert result.returncode != 0
    assert "unknown option" in result.stderr


def test_help_flag_prints_usage_and_exits_zero(tmp_path):
    clone = _clone(tmp_path)
    result = _run(clone, "--help")
    assert result.returncode == 0
    assert "Usage: scripts/setup.sh" in result.stdout
    assert "--dry-run" in result.stdout


def test_shellcheck_is_clean_if_available():
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck is not installed")
    result = subprocess.run(
        ["shellcheck", str(SETUP_SH)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
