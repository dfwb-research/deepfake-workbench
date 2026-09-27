"""Fixtures for ``dfwb.data`` tests: an installed protocol pack, and a hand-built processed store.

Building a real, hash-checked protocol pack is exactly what ``tests/unit/protocols/conftest.py``
already does; this reuses its ``toyone`` builder (three tasks, two schemes, a ``binary`` mapping
that excludes ``TOYONE-FAKE_B``) rather than writing a second copy of it. The processed-store side
is hand-built here with plain core record I/O -- exactly how :mod:`dfwb.data.index` itself reads a
store -- with no dependency on the face pipeline's own ``Store``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.unit.protocols.conftest import make_pack, register_packs, write_toyone_dataset

from dfwb.core.records import ProcessedRecord, TrackStats, write_jsonl

__all__ = [
    "processed_record",
    "toyone_pack",
    "write_inventory_meta",
    "write_provenance",
    "write_store_frames",
    "write_store_index",
]


def _tiny_png(size: int = 2) -> bytes:
    return cv2.imencode(".png", np.zeros((size, size, 3), dtype=np.uint8))[1].tobytes()


_TINY_PNG = _tiny_png()


@pytest.fixture
def toyone_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the ``toyone`` fixture dataset as its own pack; returns its dataset directory."""
    root = make_pack(
        tmp_path, "toyone-pack", {"toyone": {}}, builders={"toyone": write_toyone_dataset}
    )
    register_packs(monkeypatch, {"toyone-pack": root})
    return root / "toyone"


def processed_record(
    key: str,
    *,
    compression: str | None = None,
    status: str = "ok",
    n_frames: int = 4,
) -> ProcessedRecord:
    """A minimal, valid ``ProcessedRecord``, overridable field by field."""
    return ProcessedRecord(
        key=key,
        compression=compression,
        status=status,  # type: ignore[arg-type]
        n_frames=n_frames,
        frame_indices=list(range(n_frames)),
        relpath=f"{key}/{compression or '_'}",
        track=TrackStats(mean_confidence=0.9, identity_switch=False),
        reason=None,
    )


def write_store_index(store_dir: Path, records: Sequence[ProcessedRecord]) -> None:
    """Write a store's ``index.jsonl`` from hand-built records, without going through ``Store``."""
    store_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(store_dir / "index.jsonl", records)


def write_store_frames(
    store_dir: Path, record: ProcessedRecord, n: int | None = None, *, size: int | None = None
) -> None:
    """Write ``n`` (default: ``record.n_frames``) black PNGs under ``record``'s frame dir.

    ``size`` (default: the module's tiny 2x2 placeholder) should be the store's own
    ``profile.crop.size`` whenever a real C4 detector will actually run over these frames: input
    adaptation may see the store's declared crop size already equal to what the detector wants and
    skip inserting a resize step (a real store's frames are always genuinely that size), so a
    placeholder narrower than declared silently starves a detector whose architecture needs real
    spatial extent (a fixed-size CNN, unlike a detector that only ever reduces over every pixel).
    """
    frame_dir = store_dir / record.relpath
    frame_dir.mkdir(parents=True, exist_ok=True)
    payload = _TINY_PNG if size is None else _tiny_png(size)
    for i in range(record.n_frames if n is None else n):
        (frame_dir / f"frame_{i:06d}.png").write_bytes(payload)


def write_provenance(dataset_dir: Path, *, builder_version: str) -> None:
    """A minimal ``PROVENANCE.json`` naming the inventory builder version the pack was built at."""
    payload = {
        "builder": {"id": "toyone", "version": builder_version},
        "dfwb": "0.1.0",
        "source_listing_sha256": "0" * 64,
        "rules": {"official": {"rule": "official", "params": {}}},
    }
    (dataset_dir / "PROVENANCE.json").write_text(json.dumps(payload), encoding="utf-8")


def write_inventory_meta(work_root: Path, dataset: str, *, builder_version: str) -> None:
    """A minimal ``inventory.meta.json`` naming the inventory builder version used locally."""
    meta_dir = work_root / dataset
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "builder": {"id": dataset, "version": builder_version},
        "dfwb": "0.1.0",
        "count": 0,
        "compressions": [],
        "by_task": {},
        "location_source": {},
    }
    (meta_dir / "inventory.meta.json").write_text(json.dumps(payload), encoding="utf-8")
