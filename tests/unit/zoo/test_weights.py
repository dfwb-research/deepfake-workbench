"""The weight manager: fetch, then a cache hit, then a sha mismatch that removes the file, then
offline mode -- all against a local HTTP server, never the real network."""

from __future__ import annotations

import hashlib

import pytest

from dfwb.core.errors import ContractError, InstallationError
from dfwb.zoo.card import WeightSpec
from dfwb.zoo.weights import cache_dir, ensure_weights

CONTENT = b"pretend these are model weights\n" * 50
SHA256 = hashlib.sha256(CONTENT).hexdigest()


def _spec(url: str, *, sha256: str = SHA256, fmt: str = "safetensors") -> WeightSpec:
    return WeightSpec(
        id="default",
        url=url,
        sha256=sha256,
        bytes=len(CONTENT),
        format=fmt,  # type: ignore[arg-type]
    )


def test_ensure_weights_downloads_and_verifies(server, isolated):
    server.routes["/w.safetensors"] = CONTENT
    spec = _spec(f"{server.url}/w.safetensors")

    path = ensure_weights("gend", spec)

    assert path.read_bytes() == CONTENT
    assert path == cache_dir("gend", SHA256) / "weights.safetensors"
    assert server.requests == ["/w.safetensors"]


def test_a_second_call_is_a_cache_hit_and_makes_no_request(server, isolated):
    server.routes["/w.safetensors"] = CONTENT
    spec = _spec(f"{server.url}/w.safetensors")
    first = ensure_weights("gend", spec)
    server.requests.clear()

    second = ensure_weights("gend", spec)

    assert second == first
    assert server.requests == []


def test_sha_mismatch_raises_and_removes_the_file(server, isolated):
    server.routes["/w.safetensors"] = CONTENT
    wrong_sha = "0" * 64
    spec = _spec(f"{server.url}/w.safetensors", sha256=wrong_sha)

    with pytest.raises(ContractError, match="sha256"):
        ensure_weights("gend", spec)

    assert not (cache_dir("gend", wrong_sha) / "weights.safetensors").exists()


def test_offline_mode_refuses_an_uncached_download(server, isolated, monkeypatch):
    monkeypatch.setenv("DFWB_OFFLINE", "1")
    spec = _spec(f"{server.url}/w.safetensors")

    with pytest.raises(InstallationError, match="DFWB_OFFLINE"):
        ensure_weights("gend", spec)


def test_offline_mode_still_serves_an_already_cached_file(server, isolated, monkeypatch):
    server.routes["/w.safetensors"] = CONTENT
    spec = _spec(f"{server.url}/w.safetensors")
    ensure_weights("gend", spec)
    server.requests.clear()

    monkeypatch.setenv("DFWB_OFFLINE", "1")
    path = ensure_weights("gend", spec)

    assert path.read_bytes() == CONTENT
    assert server.requests == []


def test_weights_filename_matches_the_declared_format(server, isolated):
    server.routes["/w.pt"] = CONTENT
    sha = hashlib.sha256(CONTENT).hexdigest()
    spec = _spec(f"{server.url}/w.pt", sha256=sha, fmt="pytorch")

    path = ensure_weights("run-adapter", spec)

    assert path.name == "weights.pt"
