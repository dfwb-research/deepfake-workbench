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
    ],
)
def test_parse_ref(text, expected):
    assert parse_ref(text) == expected


@pytest.mark.parametrize(
    "bad", ["", "FFPP/official", "ffpp/", "ffpp/official@v1", "a:b:c/x", "ffpp/official@3f9"]
)
def test_parse_ref_rejects(bad):
    with pytest.raises(ConfigError) as info:
        parse_ref(bad)
    assert "[<pack>:]<dataset>[/<scheme>][@<version or hash>]" in info.value.hint
