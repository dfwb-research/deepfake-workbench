import pytest

import dfwb.core as core


def test_public_names_resolve():
    for name in core.__all__:
        assert getattr(core, name) is not None
    assert core.registry.Registry
    assert core.plugins.load_plugins


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = core.nope


def test_plugin_api_version_is_exposed_at_the_top_of_core():
    assert core.PLUGIN_API_VERSION == (1, 0)
