"""Fixtures shared across ``tests/unit`` subpackages (eval and CLI import tests both need one)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tests.unit.protocols.conftest import make_pack, register_packs

from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    SchemeCard,
    SplitRow,
    VideoRecord,
    write_jsonl,
    write_split_tsv,
)
from dfwb.core.records.protocol import LabelMappingSpec, LicenseInfo


def _dump(model) -> str:
    return yaml.safe_dump(model.model_dump(mode="json", by_alias=True), sort_keys=False)


def write_import_fixture_dataset(dataset_dir: Path, dataset_id: str) -> None:
    """5 ``CDF/0000N`` videos (2 real, 3 fake), all in ``test``; ``binary`` labels, no excludes.

    A small, purpose-built pack for ``dfwb eval import`` tests: real keys already look like the
    thesis convention (``<task>/<video_id>``), so a key template test can compose them exactly,
    and mismatched keys (an extension, a foreign prefix, a missing task prefix) can be built
    against it precisely.
    """
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits").mkdir()

    specs = [
        ("00001", "CDF-REAL", "original"),
        ("00002", "CDF-REAL", "original"),
        ("00003", "CDF-FAKE", "FaceSwap"),
        ("00004", "CDF-FAKE", "FaceSwap"),
        ("00005", "CDF-FAKE", "FaceSwap"),
    ]
    videos = [
        VideoRecord(f"CDF/{vid}", None, label_key, method, identity=vid)
        for vid, label_key, method in specs
    ]
    write_jsonl(dataset_dir / "videos.jsonl.gz", videos)
    rows = [SplitRow(v.key, v.compression, "test") for v in videos]
    sha256 = write_split_tsv(dataset_dir / "splits" / "official.tsv.gz", rows)

    card = DatasetCard(
        id=dataset_id,
        name="Import Fixture",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only",
        modalities=["video"],
        key_rule="fixture",
        schemes={"official": SchemeCard(kind="official", source="fixture", sha256=sha256)},
        default_scheme="official",
    )
    (dataset_dir / "dataset.yaml").write_text(_dump(card))
    vocab = {"CDF-REAL": {"binary": 0}, "CDF-FAKE": {"binary": 1}}
    labels = LabelVocab(vocab=vocab, mappings={"binary": LabelMappingSpec(from_="binary")})
    (dataset_dir / "labels.yaml").write_text(_dump(labels))


@pytest.fixture
def import_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install :func:`write_import_fixture_dataset` as pack ``import-pack``.

    Returns its dataset directory.
    """
    root = make_pack(
        tmp_path, "import-pack", {"cdf": {}}, builders={"cdf": write_import_fixture_dataset}
    )
    register_packs(monkeypatch, {"import-pack": root})
    return root / "cdf"
