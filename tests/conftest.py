"""Shared fixtures: every test starts with empty registries, no plugins loaded, and the
environment it was started with."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from dfwb.core import envfile, plugins


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    """Undo whatever a test left in ``os.environ``, e.g. a .env an in-process command applied."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)
    envfile._remember(None)


@pytest.fixture(autouse=True)
def _clean_plugins(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("DFWB_PLUGINS", raising=False)
    monkeypatch.delenv("DFWB_PLUGINS_DISABLE", raising=False)
    plugins.reset()
    yield
    plugins.reset()


@pytest.fixture
def requires_torch():
    """Skip cleanly where torch is not installed; give back the ``torch`` module where it is."""
    return pytest.importorskip("torch")
