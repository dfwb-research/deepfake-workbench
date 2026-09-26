"""The weight manager: fetch, then a cache hit, then a sha mismatch that removes the file, then
offline mode -- all against a local HTTP server, never the real network."""

from __future__ import annotations

import hashlib
import sys

import pytest

from dfwb.core.errors import ContractError, InstallationError
from dfwb.zoo.card import WeightSpec
from dfwb.zoo.weights import cache_dir, ensure_weights, load_weights

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


def test_a_size_mismatch_against_the_card_is_reported(server, isolated):
    server.routes["/w.safetensors"] = CONTENT
    spec = _spec(f"{server.url}/w.safetensors")
    spec.bytes = len(CONTENT) + 1  # sha256 still matches CONTENT; only the declared size is wrong

    with pytest.raises(ContractError, match="bytes"):
        ensure_weights("gend", spec)


# --------------------------------------------------------------------- a tampered cache hit


def test_a_tampered_cache_hit_is_deleted_and_refetched(server, isolated):
    server.routes["/w.safetensors"] = CONTENT
    spec = _spec(f"{server.url}/w.safetensors")
    dest = cache_dir("gend", SHA256) / "weights.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"tampered content, wrong hash entirely")
    server.requests.clear()

    path = ensure_weights("gend", spec)

    assert path == dest
    assert path.read_bytes() == CONTENT
    assert server.requests == ["/w.safetensors"]  # it really was re-downloaded


def test_a_tampered_cache_hit_offline_raises_naming_the_path(server, isolated, monkeypatch):
    spec = _spec(f"{server.url}/w.safetensors")
    dest = cache_dir("gend", SHA256) / "weights.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"tampered content, wrong hash entirely")

    monkeypatch.setenv("DFWB_OFFLINE", "1")
    with pytest.raises(ContractError) as info:
        ensure_weights("gend", spec)
    assert str(dest) in info.value.message
    assert dest.is_file()  # left in place for the user to inspect or replace by hand


# ---------------------------------------------------------------------------------- load_weights


def test_load_weights_reads_a_safetensors_file(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    path = tmp_path / "weights.safetensors"
    save_file({"w": torch.zeros(3)}, str(path))
    spec = WeightSpec(
        id="default", url="x://x", sha256="a" * 64, format="safetensors", bytes=path.stat().st_size
    )

    state = load_weights(path, spec)

    assert list(state) == ["w"]


def test_load_weights_reads_a_torch_checkpoint_under_weights_only(tmp_path):
    torch = pytest.importorskip("torch")

    path = tmp_path / "weights.pt"
    torch.save({"w": torch.zeros(3)}, path)
    spec = WeightSpec(id="default", url="x://x", sha256="a" * 64, format="pytorch")

    state = load_weights(path, spec)

    assert list(state) == ["w"]


class _NotATensor:
    """A module-level (picklable) stand-in for something a checkpoint should never hold."""


def test_load_weights_refuses_a_pickle_holding_a_non_tensor_object(tmp_path):
    torch = pytest.importorskip("torch")

    path = tmp_path / "weights.pt"
    torch.save({"w": _NotATensor()}, path)
    spec = WeightSpec(id="default", url="x://x", sha256="a" * 64, format="pytorch")

    with pytest.raises(ContractError):
        load_weights(path, spec)


def test_load_weights_reports_missing_safetensors(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)
    spec = WeightSpec(id="default", url="x://x", sha256="a" * 64, format="safetensors")

    with pytest.raises(InstallationError, match="safetensors"):
        load_weights(tmp_path / "w.safetensors", spec)


def test_load_weights_reports_missing_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    spec = WeightSpec(id="default", url="x://x", sha256="a" * 64, format="pytorch")

    with pytest.raises(InstallationError, match="torch"):
        load_weights(tmp_path / "w.pt", spec)


def test_load_weights_refuses_an_unknown_format(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"whatever")
    spec = WeightSpec(id="default", url="x://x", sha256="a" * 64, format="safetensors")
    spec.format = "onnx"  # bypasses the card's own Literal, as a defensive check would need to

    with pytest.raises(ContractError, match="format"):
        load_weights(path, spec)
