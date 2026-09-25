"""Pack versions: MAJOR.MINOR.PATCH with PEP 440 pre-, post- and dev-release suffixes."""

from __future__ import annotations

import re

import pytest

from dfwb.protocols._versions import VERSION_PATTERN, PackVersion, parse_version


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.2.3", PackVersion((1, 2, 3))),
        ("0.1.0a2", PackVersion((0, 1, 0), pre=(0, 2))),
        ("1.0.0-alpha.2", PackVersion((1, 0, 0), pre=(0, 2))),
        ("1.0.0B1", PackVersion((1, 0, 0), pre=(1, 1))),
        ("1.0.0c1", PackVersion((1, 0, 0), pre=(2, 1))),
        ("1.0.0-preview3", PackVersion((1, 0, 0), pre=(2, 3))),
        ("1.0.0rc", PackVersion((1, 0, 0), pre=(2, 0))),
        ("1.0.0.post1", PackVersion((1, 0, 0), post=1)),
        ("1.0.0-1", PackVersion((1, 0, 0), post=1)),
        ("1.0.0rev2", PackVersion((1, 0, 0), post=2)),
        ("1.0.0.dev", PackVersion((1, 0, 0), dev=0)),
        ("1.0.0b2.post1.dev4", PackVersion((1, 0, 0), pre=(1, 2), post=1, dev=4)),
    ],
)
def test_parse_version(text, expected):
    assert parse_version(text) == expected
    assert re.fullmatch(VERSION_PATTERN, text)


@pytest.mark.parametrize(
    "text", ["1.0", "1.0.0.0", "v1.0.0", "1!1.0.0", "1.0.0+local", "1.0.0x1", "1.0.0-", ""]
)
def test_not_a_pack_version(text):
    assert parse_version(text) is None
    assert not re.fullmatch(VERSION_PATTERN, text)


def test_pep440_order():
    chain = [
        "1.0.0.dev1",
        "1.0.0a1.dev1",
        "1.0.0a1",
        "1.0.0a1.post1.dev1",
        "1.0.0a1.post1",
        "1.0.0b1",
        "1.0.0rc1",
        "1.0.0",
        "1.0.0.post1.dev1",
        "1.0.0.post1",
        "1.0.1.dev1",
    ]
    keys = []
    for text in chain:
        version = parse_version(text)
        assert version is not None
        keys.append(version.sort_key)
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
