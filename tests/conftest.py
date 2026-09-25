"""Shared fixtures: every test starts with empty registries and no plugins loaded."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dfwb.core import plugins


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
