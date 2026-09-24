from __future__ import annotations

from types import SimpleNamespace

import pytest

from dfwb.cli.main import main


@pytest.fixture
def run(capsys, monkeypatch, tmp_path):
    """Run `dfwb ARGS...` in-process through main() in an isolated directory and home."""
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

    def _run(*args: str) -> SimpleNamespace:
        code = main(list(args))
        captured = capsys.readouterr()
        return SimpleNamespace(code=code, out=captured.out, err=captured.err)

    return _run
