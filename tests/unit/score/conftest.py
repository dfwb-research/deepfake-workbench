"""Fixtures for ``dfwb.score`` tests: the ``scoretoy`` pack, a processed store, and roots pointed
at a throwaway work/runs tree."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")


@pytest.fixture
def scoretoy_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The ``scoretoy`` protocol pack (and the ``fake`` detector source), installed."""
    from tests.unit.score._toy import install_scoretoy_pack

    return install_scoretoy_pack(tmp_path, monkeypatch)


@pytest.fixture
def score_roots(tmp_path: Path, scoretoy_pack: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``DFWB_WORK_ROOT``/``DFWB_RUNS_ROOT`` pointed at a throwaway tree; returns the work root."""
    work_root = tmp_path / "work"
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work_root))
    monkeypatch.setenv("DFWB_RUNS_ROOT", str(tmp_path / "runs"))
    return work_root
