"""Canonical JSON and sha256 helpers used for fingerprints and file hashes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable

__all__ = [
    "canonical_json",
    "fingerprint",
    "sha256_bytes",
    "sha256_file",
    "sha256_stream",
    "sha256_text",
]

_CHUNK = 1 << 20


def canonical_json(obj: object) -> str:
    """Serialise ``obj`` deterministically: sorted keys, no whitespace, UTF-8, no NaN/Infinity."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hex sha256 of the UTF-8 encoding of ``text``."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_stream(chunks: Iterable[bytes]) -> str:
    """Hex sha256 over a stream of byte chunks."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hex sha256 of a file's bytes, read in 1 MiB chunks."""
    with open(path, "rb") as handle:  # noqa: PTH123 - accepts any PathLike
        return sha256_stream(iter(lambda: handle.read(_CHUNK), b""))


def fingerprint(obj: object) -> str:
    """sha256 of :func:`canonical_json` of ``obj``."""
    return sha256_text(canonical_json(obj))
