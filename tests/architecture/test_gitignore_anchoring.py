"""``.gitignore`` patterns must be anchored to the repo root, or an unrelated same-named
directory elsewhere in the tree is silently swallowed too.

A bare ``data/`` (no leading or middle slash) is not anchored: git matches it at *any* depth, so
it once also matched ``src/dfwb/data/`` -- a real source package -- meaning a new file added
there would be silently git-ignored unless force-added. ``.gitignore`` must spell the repo's own
``data/`` (and ``runs/``) as ``/data/``/``/runs/`` instead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _is_ignored(relpath: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", relpath],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed on {relpath!r}: {result.stderr}")
    return result.returncode == 0


def test_a_new_file_under_src_dfwb_data_is_not_git_ignored():
    assert not _is_ignored("src/dfwb/data/a-new-file.py")


def test_the_clones_own_data_directory_is_still_git_ignored():
    assert _is_ignored("data/datasets/x")


def test_a_nested_runs_directory_is_not_git_ignored():
    assert not _is_ignored("src/dfwb/some_package/runs/x")


def test_the_clones_own_runs_directory_is_still_git_ignored():
    assert _is_ignored("runs/toy-cpu/x")
