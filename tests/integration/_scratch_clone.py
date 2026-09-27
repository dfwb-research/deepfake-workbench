"""A scratch copy of this clone, and an environment that runs it the way a new user's shell would.

The slow tests that run documented command blocks verbatim (the README's quickstart, the
reproduce-a-benchmark guide) must not pass or fail because of the machine running them: a
maintainer's own `.env`, `data/` or `runs/` in the checkout, a `DFWB_*` variable exported in the
shell that runs pytest, or a user config under the real home directory. So the copy holds only
what a fresh `git clone` would, and the environment has no `DFWB_*` variable and a home directory
and XDG base directories of its own, inside the test's scratch directory.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

__all__ = ["REPO", "copy_clone", "isolated_env"]

REPO = Path(__file__).resolve().parents[2]

# Used only when git cannot list the checkout's files: everything a fresh clone would not hold.
_NOT_IN_A_CLONE = (
    "/.git",
    "/.venv",
    "/.venv*",
    "__pycache__",
    "*.pyc",
    "/.pytest_cache",
    "/.mypy_cache",
    "/.ruff_cache",
    "/.import_linter_cache",
    "/.hypothesis",
    ".coverage*",
    "/.env",
    "/dfwb.toml",
    "/runs",
    "/data",
    "/site",
    "/dist",
    "/build",
)

_GIT_LS_FILES = (
    "git",
    "-C",
    str(REPO),
    "ls-files",
    "-z",
    "--cached",
    "--others",
    "--exclude-standard",
)
_RSYNC_LISTED = ("rsync", "-a", "--from0", "--files-from=-", "--ignore-missing-args")

# Variables that describe the test runner's own environment, never a new user's shell.
_RUNNER_ONLY = frozenset({"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_RUN_RECURSION_DEPTH"})

# The XDG base directories, each pointed into the scratch directory.
_XDG = ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR")


def _clone_files() -> bytes | None:
    """The NUL-separated paths a fresh clone would hold: every file git tracks, plus any new
    file .gitignore does not exclude; ``None`` when git cannot say."""
    try:
        result = subprocess.run(
            [*_GIT_LS_FILES],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def copy_clone(dest: Path) -> None:
    """Copy this checkout into ``dest`` as a fresh clone would have it: never a git-ignored,
    local-only file such as ``.env``, ``data/``, ``runs/`` or a virtual environment."""
    dest.mkdir(parents=True)
    files = _clone_files()
    if files is not None:
        subprocess.run(
            [*_RSYNC_LISTED, f"{REPO}/", f"{dest}/"],
            input=files,
            check=True,
        )
        return
    excludes = [f"--exclude={pattern}" for pattern in _NOT_IN_A_CLONE]
    subprocess.run(["rsync", "-a", *excludes, f"{REPO}/", f"{dest}/"], check=True)


def _uv_dir(*args: str) -> str:
    result = subprocess.run(
        ["uv", *args, "--color", "never"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def isolated_env(scratch: Path) -> dict[str, str]:
    """``os.environ`` without any ``DFWB_*`` variable or the test runner's own virtual
    environment, with ``HOME`` and the XDG base directories inside ``scratch``.

    uv keeps using the runner's own cache and managed Pythons (asked of ``uv`` before ``HOME``
    moves), so a scratch clone installs from what is already downloaded.
    """
    uv_dirs = {
        "UV_CACHE_DIR": _uv_dir("cache", "dir"),
        "UV_PYTHON_INSTALL_DIR": _uv_dir("python", "dir"),
    }
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("DFWB_", "XDG_")) and name not in _RUNNER_ONLY
    }
    home = scratch / "home"
    home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home)
    for name in _XDG:
        directory = scratch / "xdg" / name.removeprefix("XDG_").lower()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[name] = str(directory)
    env.update(uv_dirs)
    return env
