"""Shared helper for every in-process ``dfwb`` invocation in the test suite.

Every test that runs a ``dfwb`` command in-process (as opposed to as a subprocess) calls
:func:`run_dfwb` instead of :func:`dfwb.cli.main.main` directly, so the suite itself enforces the
framework's own rule: a command that exits non-zero always explains itself with a ``hint: `` line
on stderr. ``tests/architecture/test_hint_on_every_failure.py`` is the guard that no test bypasses
this by importing ``main`` for itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dfwb.cli.main import main

__all__ = ["run_dfwb"]


def run_dfwb(capsys: pytest.CaptureFixture[str], *args: str) -> SimpleNamespace:
    """Run ``dfwb ARGS...`` in-process; return its exit code and captured stdout/stderr.

    Every non-zero exit is checked here, once, for a non-empty ``hint: `` line on stderr -- the
    one property every failing ``dfwb`` invocation in this suite must have -- instead of leaving
    each call site to remember to check it.
    """
    code = main(list(args))
    captured = capsys.readouterr()
    if code:
        hints = [
            line.removeprefix("hint:").strip()
            for line in captured.err.splitlines()
            if line.startswith("hint:")
        ]
        assert any(hints), (
            f"`dfwb {' '.join(args)}` exited {code} without printing a non-empty `hint:` line on "
            f"stderr:\n{captured.err}"
        )
    return SimpleNamespace(code=code, out=captured.out, err=captured.err)
