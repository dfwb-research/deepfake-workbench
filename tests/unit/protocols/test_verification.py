"""Tests for ``dfwb.protocols.verification``: coverage reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError
from dfwb.core.records import BuilderRef, InventoryRecord, VideoRecord, read_jsonl, write_jsonl
from dfwb.protocols.verification import CoverageReport, verify, write_report


def _inventory_record(
    video: VideoRecord, *, label_key: str | None = None, method: str | None = None
) -> InventoryRecord:
    """An ``InventoryRecord`` matching ``video`` on disk, unless ``label_key``/``method`` differ."""
    return InventoryRecord(
        key=video.key,
        compression=video.compression,
        label_key=video.label_key if label_key is None else label_key,
        method=video.method if method is None else method,
        relpath=f"{video.key}.mp4",
        builder=BuilderRef("fixture", "1"),
        identity=video.identity,
        source_id=video.source_id,
        target_id=video.target_id,
        pair_key=video.pair_key,
        attrs=dict(video.attrs),
    )


def _write_inventory(path: Path, records: list[InventoryRecord]) -> Path:
    write_jsonl(path, records)
    return path


def test_full_coverage_exit_0(toyone_pack, tmp_path):
    videos = read_jsonl(toyone_pack / "videos.jsonl.gz", VideoRecord)
    inventory = _write_inventory(
        tmp_path / "inventory.jsonl", [_inventory_record(v) for v in videos]
    )

    report = verify("toyone/official", inventory=inventory, work_root=tmp_path / "work")

    assert isinstance(report, CoverageReport)
    assert report.dataset == "toyone"
    assert report.scheme == "official"
    assert report.pack == "toyone-pack"
    assert report.counts == {
        "have": 16,
        "missing": 0,
        "missing_requested": 0,
        "extra": 0,
        "label_mismatch": 0,
    }
    assert report.warnings == []
    assert report.exit_code == 0


def test_missing_in_requested_split_exit_3_but_other_split_only_warns(toyone_pack, tmp_path):
    videos = read_jsonl(toyone_pack / "videos.jsonl.gz", VideoRecord)
    # FAKE_A/a1 is a single row (no compression variant) assigned to "train" in "official".
    records = [_inventory_record(v) for v in videos if v.key != "FAKE_A/a1"]
    inventory = _write_inventory(tmp_path / "inventory.jsonl", records)

    report = verify(
        "toyone/official", inventory=inventory, splits=["test"], work_root=tmp_path / "work"
    )

    assert report.requested_splits == ("test",)
    assert report.counts["missing"] == 1
    assert report.counts["missing_requested"] == 0
    assert report.exit_code == 0


def test_label_mismatch_exit_4(toyone_pack, tmp_path):
    videos = read_jsonl(toyone_pack / "videos.jsonl.gz", VideoRecord)
    records = [
        _inventory_record(v, method="Wrong") if v.key == "FAKE_A/a1" else _inventory_record(v)
        for v in videos
    ]
    inventory = _write_inventory(tmp_path / "inventory.jsonl", records)

    report = verify("toyone/official", inventory=inventory, work_root=tmp_path / "work")

    assert report.counts["label_mismatch"] == 1
    assert report.counts["have"] == 15
    assert report.samples["label_mismatch"] == ["FAKE_A/a1|"]
    assert report.exit_code == 4


def test_extra_is_info(toyone_pack, tmp_path):
    videos = read_jsonl(toyone_pack / "videos.jsonl.gz", VideoRecord)
    records = [_inventory_record(v) for v in videos]
    records.append(
        InventoryRecord(
            key="FAKE_A/nope",
            compression=None,
            label_key="TOYONE-FAKE_A",
            method="FakeA",
            relpath="FAKE_A/nope.mp4",
            builder=BuilderRef("fixture", "1"),
        )
    )
    inventory = _write_inventory(tmp_path / "inventory.jsonl", records)

    report = verify("toyone/official", inventory=inventory, work_root=tmp_path / "work")

    assert report.counts["extra"] == 1
    assert report.counts["have"] == 16
    assert report.samples["extra"] == ["FAKE_A/nope|"]
    assert report.exit_code == 0


def test_release_mismatch_heuristic(release_pack, tmp_path):
    videos = read_jsonl(release_pack / "videos.jsonl.gz", VideoRecord)
    assert len(videos) == 30

    # Keep v01..v20 (20 have), drop v21..v30 (10 missing), add 10 differently-keyed extras.
    kept = [v for v in videos if int(v.key.rsplit("v", 1)[1]) <= 20]
    records = [_inventory_record(v) for v in kept]
    for i in range(1, 11):
        records.append(
            InventoryRecord(
                key=f"FAKE_A/x{i:02d}",
                compression=None,
                label_key="RELEASE-FAKE_A",
                method="FakeA",
                relpath=f"FAKE_A/x{i:02d}.mp4",
                builder=BuilderRef("fixture", "1"),
            )
        )
    inventory = _write_inventory(tmp_path / "inventory.jsonl", records)

    report = verify("release/official", inventory=inventory, work_root=tmp_path / "work")

    assert report.counts["missing"] == 10
    assert report.counts["extra"] == 10
    assert report.warnings == [
        "FAKE_A: 10 missing and 10 extra — the local copy may be a different upstream "
        "release (see the card's 'release')"
    ]


def test_unknown_split_raises_config_error_with_suggestion(toyone_pack, tmp_path):
    videos = read_jsonl(toyone_pack / "videos.jsonl.gz", VideoRecord)
    inventory = _write_inventory(
        tmp_path / "inventory.jsonl", [_inventory_record(v) for v in videos]
    )

    with pytest.raises(ConfigError) as info:
        verify("toyone/official", inventory=inventory, splits=["tset"], work_root=tmp_path / "work")

    assert "toyone/official: split 'tset' is not in this scheme" in str(info.value)
    assert "did you mean 'test'" in str(info.value)
    assert info.value.hint == "splits in this scheme: test, train, val"


def test_no_inventory_raises_config_error_with_inventory_build_hint(toyone_pack, tmp_path):
    with pytest.raises(ConfigError) as info:
        verify(
            "toyone/official",
            inventory=tmp_path / "no-such-inventory.jsonl",
            work_root=tmp_path / "work",
        )
    assert "run: dfwb inventory build toyone" in info.value.hint


def test_report_written_atomically_and_sorted(toyone_pack, tmp_path):
    videos = read_jsonl(toyone_pack / "videos.jsonl.gz", VideoRecord)
    inventory = _write_inventory(
        tmp_path / "inventory.jsonl", [_inventory_record(v) for v in videos]
    )
    work_root = tmp_path / "work"
    report = verify("toyone/official", inventory=inventory, work_root=work_root)

    path = write_report(report, work_root)

    assert path == work_root / "toyone" / "verify" / "official.json"
    assert list(path.parent.iterdir()) == [path]  # no leftover tmp file

    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data == report.to_json()
    assert json.dumps(data, indent=2, sort_keys=True) + "\n" == text
