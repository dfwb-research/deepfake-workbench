"""Files written under contract version 1 must stay readable by every later release.

The fixtures under fixtures/v1/ are frozen: never edit them. When a contract gains a new major
version, add fixtures/v2/ next to them and keep this test for v1 (readers support the current and
the previous major version).
"""

from pathlib import Path

import yaml

from dfwb.core.records import (
    DatasetCard,
    InventoryRecord,
    LabelVocab,
    PackCard,
    ProcessedRecord,
    ProcessingProfile,
    VideoRecord,
    read_jsonl,
    read_scores,
    read_split_tsv,
)

V1 = Path(__file__).parent / "fixtures" / "v1"


def _yaml(name):
    return yaml.safe_load((V1 / name).read_text())


def test_v1_cards_load():
    assert PackCard.model_validate(_yaml("pack.yaml")).datasets == ["toyfake"]
    assert DatasetCard.model_validate(_yaml("dataset.yaml")).default_scheme == "official"
    assert LabelVocab.model_validate(_yaml("labels.yaml")).mappings["binary"].from_ == "binary"
    assert ProcessingProfile.model_validate(_yaml("profile.yaml")).sampling.frames == 8


def test_v1_rows_load_strictly():
    assert [v.key for v in read_jsonl(V1 / "videos.jsonl", VideoRecord, strict=True)] == [
        "real/0000",
        "blend-a/0000",
    ]
    assert read_jsonl(V1 / "inventory.jsonl", InventoryRecord, strict=True)[0].probe.frames == 8
    assert read_jsonl(V1 / "index.jsonl", ProcessedRecord, strict=True)[0].status == "ok"
    assert [r.split for r in read_split_tsv(V1 / "official.tsv")] == ["test", "train"]


def test_v1_score_file_loads():
    scores = read_scores(V1 / "v1.scores.csv")
    assert scores.meta is not None
    assert scores.meta.coverage.missing == 1
    assert [r.status for r in scores.rows] == ["ok", "missing"]
    assert scores.rows[1].extras == {"x_note": "no clip"}
