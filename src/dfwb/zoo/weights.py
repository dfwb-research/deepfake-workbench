"""The weight manager: fetch a zoo adapter's weights once, verified, and reuse the cached copy.

Weights are cached at ``$DFWB_CACHE_ROOT/zoo/<name>/<sha256>/weights<ext>``: the sha256 is already
part of the path, so a file found there is trusted on the strength of :func:`dfwb.core.fetch.fetch`
having verified it when it was written, not re-hashed on every later use. A missing file is
downloaded through that same verified, atomic download (a sha256 mismatch removes the partial file
and raises; ``DFWB_OFFLINE=1`` refuses the download outright) -- this module adds only the caching
decision on top of it.
"""

from __future__ import annotations

from pathlib import Path

from dfwb.core.fetch import fetch
from dfwb.core.paths import require_root, resolve_roots
from dfwb.zoo.card import WeightSpec

__all__ = ["cache_dir", "ensure_weights", "weights_filename"]

_EXTENSIONS: dict[str, str] = {"safetensors": ".safetensors", "pytorch": ".pt"}


def cache_dir(name: str, sha256: str) -> Path:
    """``$DFWB_CACHE_ROOT/zoo/<name>/<sha256>/``."""
    roots = resolve_roots()
    cache_root = require_root("cache", roots)
    return cache_root / "zoo" / name / sha256


def weights_filename(spec: WeightSpec) -> str:
    """The cached file's own name: content is already addressed by the directory's sha256, so
    this only needs to carry the right extension for whatever reads it back (safetensors,
    ``torch.load``)."""
    return f"weights{_EXTENSIONS[spec.format]}"


def ensure_weights(name: str, spec: WeightSpec) -> Path:
    """The local, sha256-verified path to ``spec``'s weights, downloading it once if needed.

    A file already at the cache path is returned as-is (no network, no re-hash): its content is
    addressed by the sha256 already in its own directory name, so its mere presence there is
    exactly what a prior, verified download left behind.

    Raises:
        InstallationError: the file is not cached and ``DFWB_OFFLINE`` is set, or the download
            itself failed.
        ContractError: the downloaded bytes do not hash to ``spec.sha256`` (the partial file is
            removed).
    """
    dest = cache_dir(name, spec.sha256) / weights_filename(spec)
    if dest.is_file():
        return dest
    return fetch(spec.url, spec.sha256, dest)
