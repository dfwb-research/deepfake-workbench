from __future__ import annotations

import pytest
from tests._dfwb_cli import run_dfwb


@pytest.fixture
def run(capsys, monkeypatch, tmp_path):
    """Run `dfwb ARGS...` in-process through the shared helper, in an isolated directory and
    home."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "home" / ".cache"))
    for name in (
        "DFWB_DATASETS_ROOT",
        "DFWB_WORK_ROOT",
        "DFWB_CACHE_ROOT",
        "DFWB_RUNS_ROOT",
        "DFWB_DEBUG",
    ):
        monkeypatch.delenv(name, raising=False)

    def _run(*args: str):
        return run_dfwb(capsys, *args)

    return _run
