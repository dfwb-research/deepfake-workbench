"""``dfwb.eval``'s lazily-resolved public names."""

from __future__ import annotations

import pytest

import dfwb.eval as ev


def test_public_names_resolve():
    for name in ev.__all__:
        assert getattr(ev, name) is not None


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = ev.nope
