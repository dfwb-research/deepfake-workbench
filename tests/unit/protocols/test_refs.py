import pytest

from dfwb.core.errors import ConfigError
from dfwb.protocols.refs import ProtocolRef, parse_ref


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ffpp", ProtocolRef(None, "ffpp", None, None)),
        ("ffpp/official", ProtocolRef(None, "ffpp", "official", None)),
        (
            "celebdf-v2/official+ident-80-20",
            ProtocolRef(None, "celebdf-v2", "official+ident-80-20", None),
        ),
        (
            "dfwb-protocols:ffpp/official@1.2.0",
            ProtocolRef("dfwb-protocols", "ffpp", "official", "1.2.0"),
        ),
        ("kodf/ident-72-14-14@3f9a1c2e", ProtocolRef(None, "kodf", "ident-72-14-14", "3f9a1c2e")),
        # PEP 440 pre-, post- and dev-release versions pin too.
        ("toyfake/official@0.1.0a2", ProtocolRef(None, "toyfake", "official", "0.1.0a2")),
        ("toyfake@1.0.0rc1", ProtocolRef(None, "toyfake", None, "1.0.0rc1")),
        ("toyfake@1.0.0.post1", ProtocolRef(None, "toyfake", None, "1.0.0.post1")),
        ("toyfake@1.0.0.dev3", ProtocolRef(None, "toyfake", None, "1.0.0.dev3")),
        ("toyfake@1.0.0b2.post1.dev4", ProtocolRef(None, "toyfake", None, "1.0.0b2.post1.dev4")),
        ("toyfake@1.0.0-rc.1", ProtocolRef(None, "toyfake", None, "1.0.0-rc.1")),
    ],
)
def test_parse_ref(text, expected):
    assert parse_ref(text) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "FFPP/official",
        "ffpp/",
        "ffpp/official@v1",
        "a:b:c/x",
        "ffpp/official@3f9",
        "ffpp@1.0",
        "ffpp@1.0.0+local",
        "ffpp@1.0.0x1",
        "ffpp@1!1.0.0",
    ],
)
def test_parse_ref_rejects(bad):
    with pytest.raises(ConfigError) as info:
        parse_ref(bad)
    assert "[<pack>:]<dataset>[/<scheme>][@<version or hash>]" in info.value.hint
