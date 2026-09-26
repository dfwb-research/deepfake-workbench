"""The weight manager: fetch a zoo adapter's weights once, verified, reuse and re-verify the
cached copy, and load it back only through the two ways contract C4 allows.

Weights are cached at ``$DFWB_CACHE_ROOT/zoo/<name>/<sha256>/weights<ext>``. Unlike a typical
content-addressed cache, a file found there is not simply trusted on the strength of a past,
verified download: it is re-hashed against the card on every call, so local tampering or
corruption (a stray edit, bit rot, a mistakenly overwritten file) is caught before it is ever
handed to an adapter, not assumed away by the path alone. A missing or tampered file is downloaded
(or re-downloaded) through :func:`dfwb.core.fetch.fetch` (atomic, sha256-verified); a tampered file
is deleted first so the redownload starts clean, unless ``DFWB_OFFLINE=1`` is set, in which case
there is nothing safe to serve and this raises instead, naming the bad file rather than silently
deleting the only copy there is.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dfwb.core.errors import ContractError, InstallationError
from dfwb.core.fetch import OFFLINE_ENV, fetch
from dfwb.core.hashing import sha256_file
from dfwb.core.paths import require_root, resolve_roots
from dfwb.zoo.card import WeightSpec

__all__ = ["cache_dir", "ensure_weights", "load_weights", "verify_weights", "weights_filename"]

_EXTENSIONS: dict[str, str] = {"safetensors": ".safetensors", "pytorch": ".pt"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _offline() -> bool:
    """Whether ``DFWB_OFFLINE`` asks for offline mode -- the same truthy spelling
    :func:`dfwb.core.fetch.fetch` itself checks."""
    return os.environ.get(OFFLINE_ENV, "").lower() in _TRUE_VALUES


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


def verify_weights(path: Path, spec: WeightSpec) -> None:
    """Raise :class:`ContractError`, naming ``path``, unless it hashes and sizes to what ``spec``
    declares. Exposed (beyond :func:`ensure_weights`'s own use of it) so a caller can check an
    already-cached file without downloading anything (``dfwb zoo verify``)."""
    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec.sha256:
        raise ContractError(
            f"{path}: sha256 is {actual_sha256}, expected {spec.sha256}",
            hint="the cached file does not match the adapter card",
        )
    actual_size = path.stat().st_size
    if actual_size != spec.bytes:
        raise ContractError(
            f"{path}: is {actual_size} bytes, the adapter card declares {spec.bytes}",
            hint="check the adapter card's weights entry",
        )


def ensure_weights(name: str, spec: WeightSpec) -> Path:
    """The local, verified path to ``spec``'s weights, downloading it once if needed.

    A file already at the cache path is re-verified against ``spec`` (sha256 and size) every time,
    rather than trusted on the path alone: a mismatch means the cached
    copy has been tampered with or corrupted since it was written, and is deleted and re-fetched
    -- unless ``DFWB_OFFLINE`` is set, in which case there is no safe way to replace it, and this
    raises instead of either serving a bad file or silently deleting the only copy there is.

    Raises:
        InstallationError: the file is not cached and ``DFWB_OFFLINE`` is set, or the download
            itself failed.
        ContractError: a cached file does not match ``spec`` and ``DFWB_OFFLINE`` is set, or a
            freshly downloaded file's declared size does not match the card (its sha256 is already
            verified by :func:`~dfwb.core.fetch.fetch` itself).
    """
    dest = cache_dir(name, spec.sha256) / weights_filename(spec)
    if dest.is_file():
        try:
            verify_weights(dest, spec)
        except ContractError:
            if _offline():
                raise ContractError(
                    f"{dest}: does not match the adapter card, and {OFFLINE_ENV} is set, so it "
                    "cannot be re-downloaded",
                    hint=f"remove {dest} and re-run without {OFFLINE_ENV}, or restore a good copy "
                    "by hand",
                ) from None
            dest.unlink()
        else:
            return dest
    fetch(spec.url, spec.sha256, dest)
    verify_weights(dest, spec)
    return dest


def load_weights(path: Path, spec: WeightSpec) -> Any:
    """Load verified weights at ``path`` into memory, the only two ways contract C4 allows:
    safetensors, or a torch checkpoint via ``torch.load(weights_only=True)`` -- never an arbitrary
    pickle. The format is read off ``spec.format``, never guessed from the file's extension.

    Raises:
        InstallationError: the library ``spec.format`` needs (safetensors, or torch) is not
            installed (the ``[zoo]`` extra).
        ContractError: ``spec.format`` is neither ``"safetensors"`` nor ``"pytorch"``, or a
            ``"pytorch"`` checkpoint could not be loaded under ``weights_only=True`` (it holds
            something other than plain tensors).
    """
    if spec.format == "safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            # safetensors' own torch integration needs torch too: name whichever of the two is
            # actually missing (exc.name), not always "safetensors" when torch is the real gap.
            missing = exc.name or "safetensors"
            raise InstallationError(
                f"{path}: needs {missing}, which is not installed",
                hint='pip install "deepfake-workbench[zoo]"',
            ) from None
        return load_file(str(path))
    if spec.format == "pytorch":
        try:
            import torch
        except ImportError:
            raise InstallationError(
                f"{path}: needs torch, which is not installed",
                hint='pip install "deepfake-workbench[zoo]"',
            ) from None
        try:
            return torch.load(path, weights_only=True, map_location="cpu")
        except Exception as exc:
            raise ContractError(
                f"{path}: could not be loaded as a torch checkpoint ({type(exc).__name__}: {exc})",
                hint="weights are loaded with weights_only=True, which refuses anything that is "
                "not plain tensors; re-export the checkpoint without extra pickled objects",
            ) from exc
    raise ContractError(
        f"{path}: unknown weights format {spec.format!r}",
        hint="weights format must be 'safetensors' or 'pytorch'",
    )
