"""Performance budget: load and filter 250k rows in under a second.

The fixture build (writing 250,000 ``VideoRecord``s and split rows to disk) is not timed -- only
``load(...).records(...)`` is.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from tests.unit.protocols.conftest import _dump, make_pack, register_packs

from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    SchemeCard,
    SplitRow,
    VideoRecord,
    write_jsonl,
    write_split_tsv,
)
from dfwb.core.records.protocol import LicenseInfo
from dfwb.protocols.protocol import load

N = 250_000
_SPLITS = ("train", "val", "test")


def _write_big_dataset(dataset_dir: Path, dataset_id: str) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits").mkdir()
    label_key = f"{dataset_id.upper()}-REAL"

    videos = [
        VideoRecord(f"REAL/{i:06d}", None, label_key, "original", identity=f"{i:06d}")
        for i in range(N)
    ]
    write_jsonl(dataset_dir / "videos.jsonl.gz", videos)

    rows = [SplitRow(v.key, v.compression, _SPLITS[i % 3]) for i, v in enumerate(videos)]
    sha256 = write_split_tsv(dataset_dir / "splits" / "official.tsv.gz", rows)

    card = DatasetCard(
        id=dataset_id,
        name=dataset_id,
        release="1",
        license=LicenseInfo(summary="Synthetic 250k-row fixture for the performance test"),
        access="tests only",
        modalities=["video"],
        key_rule="fixture",
        schemes={"official": SchemeCard(kind="official", source="fixture", sha256=sha256)},
        default_scheme="official",
    )
    (dataset_dir / "dataset.yaml").write_text(_dump(card))
    labels = LabelVocab(vocab={label_key: {"binary": 0}}, mappings={"binary": {"from": "binary"}})
    (dataset_dir / "labels.yaml").write_text(_dump(labels))


@pytest.mark.slow
def test_records_on_250k_rows_is_fast(tmp_path, monkeypatch):
    root = make_pack(tmp_path, "big-pack", {"big": {}}, builders={"big": _write_big_dataset})
    register_packs(monkeypatch, {"big-pack": root})

    start = time.perf_counter()
    protocol = load("big")
    records = protocol.records(split="test", where={"compression": None})
    elapsed = time.perf_counter() - start

    expected = sum(1 for i in range(N) if _SPLITS[i % 3] == "test")
    assert len(records) == expected
    assert elapsed < 1.0, f"load()+records() took {elapsed:.3f}s for {N} rows (budget: 1.0s)"
