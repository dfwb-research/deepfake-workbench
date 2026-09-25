"""``dfwb.score.sources``: resolving a ``<scheme>:<rest>`` detector URI through the
``detector_sources`` registry."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dfwb.core.errors import UnknownKeyError
from dfwb.score.sources import resolve_detector


def test_resolves_a_registered_scheme(scoretoy_pack):
    detector = resolve_detector("fake:")
    assert detector.meta.name == "fake-detector"


def test_passes_the_uri_tail_to_the_scheme_loader(scoretoy_pack):
    detector = resolve_detector("fake:scale=1.6")
    assert detector.meta.input.crop_scale == 1.6


def test_unknown_scheme_lists_the_known_schemes(scoretoy_pack):
    with pytest.raises(UnknownKeyError) as info:
        resolve_detector("bogus:whatever")
    assert "bogus" in info.value.message
    assert "run" in info.value.hint
    assert "fake" in info.value.hint


def test_a_uri_with_no_scheme_is_rejected():
    with pytest.raises(UnknownKeyError):
        resolve_detector("no-colon-here")
