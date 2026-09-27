"""Build tiny synthetic WildDeepfake release archives for the unpacker's tests.

Real WildDeepfake shards are never read here: every archive is built from scratch, in memory,
with the same shape the builder's card describes (``<shard>/<label>/<sequence>/<frame>.png``),
so the unpacker's own tests never touch the machine's real, read-only copy of the dataset.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

__all__ = ["write_raw_shard", "write_realistic_shard", "write_shard"]

# (sequence, frame filename, content) triples.
Frame = tuple[str, str, bytes]


def write_shard(path: Path, shard_id: str, label: str, frames: list[Frame]) -> None:
    """Write a plain (uncompressed) tar at ``path``, named like a shard despite ``.tar.gz``.

    Each frame becomes a member ``<shard_id>/<label>/<sequence>/<frame filename>`` holding
    ``content``, matching the release's own internal shape.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as tar:
        for sequence, frame_name, content in frames:
            info = tarfile.TarInfo(name=f"{shard_id}/{label}/{sequence}/{frame_name}")
            info.size = len(content)
            tar.addfile(info, fileobj=io.BytesIO(content))


def write_realistic_shard(path: Path, shard_id: str, label: str, frames: list[Frame]) -> None:
    """Write a shard the way a plain ``tar -cf`` of a real directory tree actually produces it:
    an uncompressed tar (still named ``.tar.gz``), every member prefixed ``./``, and an explicit
    directory entry for ``./``, ``./<shard_id>/``, ``./<shard_id>/<label>/`` and each sequence's
    own folder -- not just the frame files themselves.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sequences = dict.fromkeys(sequence for sequence, _, _ in frames)
    with tarfile.open(path, mode="w") as tar:
        for dirname in (
            "./",
            f"./{shard_id}/",
            f"./{shard_id}/{label}/",
            *(f"./{shard_id}/{label}/{sequence}/" for sequence in sequences),
        ):
            info = tarfile.TarInfo(name=dirname)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        for sequence, frame_name, content in frames:
            info = tarfile.TarInfo(name=f"./{shard_id}/{label}/{sequence}/{frame_name}")
            info.size = len(content)
            tar.addfile(info, fileobj=io.BytesIO(content))


def write_raw_shard(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    """Write a plain tar of exactly ``members`` (each with optional content), for safety tests.

    Lets a test add a member of any type (a symlink, a hard link, a device file, an absolute or
    ``..`` path) that :func:`write_shard` cannot express.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as tar:
        for info, content in members:
            if content is not None:
                info.size = len(content)
                tar.addfile(info, fileobj=io.BytesIO(content))
            else:
                tar.addfile(info)
