"""Helpers for architecture tests: run Python or the dfwb console script with imports blocked."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

_BLOCKER = """\
import importlib.abc
import sys

_BLOCKED = {blocked!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in _BLOCKED:
            message = f"No module named {{fullname!r}} (blocked by dfwb tests)"
            raise ModuleNotFoundError(message, name=fullname)
        return None


sys.meta_path.insert(0, _Blocker())
"""

Runner = Callable[..., subprocess.CompletedProcess[str]]


@pytest.fixture
def blocked(tmp_path: Path) -> Runner:
    """Run a command in a subprocess where the named top-level modules cannot be imported."""

    def _run(
        command: Sequence[str], *, block: Sequence[str] = ("torch",), cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        site = tmp_path / "blocker"
        site.mkdir(exist_ok=True)
        (site / "sitecustomize.py").write_text(_BLOCKER.format(blocked=tuple(block)))
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]),
        }
        return subprocess.run(
            list(command), cwd=cwd or tmp_path, env=env, capture_output=True, text=True, check=False
        )

    return _run
