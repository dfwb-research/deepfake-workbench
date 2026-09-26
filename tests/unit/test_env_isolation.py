"""An in-process ``dfwb`` command applies a .env to ``os.environ``; no test may leak it.

The two tests run in order: the first applies a .env through ``main()``, the second checks that
the variable it set is gone again. (Run in parallel, the second may land on another worker, where
it holds trivially; run in order, it checks the restore.)
"""

from __future__ import annotations

import os

from tests._dfwb_cli import run_dfwb

from dfwb.core import envfile

_PROBE = "DFWB_ISOLATION_PROBE"


def test_an_in_process_command_applies_its_env_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DFWB_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text(f"{_PROBE}=set\n")
    assert run_dfwb(capsys, "doctor", "--json").code == 0
    assert os.environ[_PROBE] == "set"
    assert envfile.last_applied() is not None


def test_the_next_test_starts_from_the_original_environment():
    assert _PROBE not in os.environ
    assert envfile.last_applied() is None
