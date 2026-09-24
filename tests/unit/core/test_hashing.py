import math

import pytest

from dfwb.core.hashing import canonical_json, fingerprint, sha256_bytes, sha256_file, sha256_text


def test_canonical_json_is_order_independent_and_compact():
    assert canonical_json({"b": 1, "a": [1, 2.5, "é"]}) == '{"a":[1,2.5,"é"],"b":1}'
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_float_spelling_does_not_change_fingerprint():
    assert fingerprint({"lr": 1e-4}) == fingerprint({"lr": 0.0001})


def test_nan_is_rejected():
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json({"x": math.nan})


def test_sha256_helpers_agree(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1_000_000)
    assert sha256_file(path) == sha256_bytes(b"abc" * 1_000_000)
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
