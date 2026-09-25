"""Verified downloads of model assets.

``fetch`` is the one place dfwb touches the network for a large binary such as a model weights
file: a plain stdlib download, written atomically and checked against a known sha256 before it is
kept. Callers above core -- a model zoo, a backend's weight loader -- decide when to call it and
whether a cached copy already satisfies the request; this module never caches or skips on its own.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

from dfwb.core.errors import ContractError, InstallationError

__all__ = ["OFFLINE_ENV", "fetch"]

OFFLINE_ENV = "DFWB_OFFLINE"
_CHUNK = 1 << 20
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _offline() -> bool:
    """Whether ``DFWB_OFFLINE`` asks for offline mode (same truthy spelling as ``--debug``)."""
    return os.environ.get(OFFLINE_ENV, "").lower() in _TRUE_VALUES


def _network_hint(destination: Path) -> str:
    return (
        f"check your network connection, or set {OFFLINE_ENV}=1 and place the file at "
        f"{destination} yourself"
    )


def fetch(url: str, sha256: str, dest: str | os.PathLike[str]) -> Path:
    """Download ``url`` to ``dest``, keeping it only if it hashes to ``sha256``.

    The download is atomic: bytes land in a hidden sibling of ``dest`` first, and that file
    becomes ``dest`` only once every byte has arrived and the whole thing hashes correctly.
    Anything short of that -- a connection dropped mid-transfer, a wrong hash, a failed request --
    leaves no file at ``dest`` at all.

    Args:
        url: Where to download from.
        sha256: The expected hex sha256 of the downloaded bytes.
        dest: Local path to write the verified file to; parent directories are created.

    Raises:
        InstallationError: ``DFWB_OFFLINE`` is set, so nothing is downloaded; or the request
            itself failed (a bad status such as 404, a refused connection, a timeout, ...).
        ContractError: the downloaded bytes do not hash to ``sha256``.
    """
    destination = Path(dest)
    if _offline():
        raise InstallationError(
            f"{url}: refusing to download while {OFFLINE_ENV} is set",
            hint=f"unset {OFFLINE_ENV} to allow downloads, or place the file at "
            f"{destination} yourself",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
                for chunk in iter(lambda: response.read(_CHUNK), b""):
                    handle.write(chunk)
                    digest.update(chunk)
        except urllib.error.HTTPError as exc:
            raise InstallationError(
                f"{url}: download failed ({exc.code} {exc.reason})",
                hint=_network_hint(destination),
            ) from None
        except urllib.error.URLError as exc:
            raise InstallationError(
                f"{url}: download failed ({exc.reason})",
                hint=_network_hint(destination),
            ) from None
        got = digest.hexdigest()
        if got != sha256.lower():
            raise ContractError(
                f"{url}: sha256 mismatch: downloaded file hashes to {got}, expected {sha256}",
                hint="the release may have changed, or the download may be corrupt; "
                "re-check the expected sha256",
            )
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)
    return destination
