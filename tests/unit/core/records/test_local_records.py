import json
import time

import pytest
import yaml

from dfwb.core.errors import ContractError
from dfwb.core.records import (
    BuilderRef,
    InventoryRecord,
    Probe,
    ProcessedRecord,
    ProcessingProfile,
    TrackStats,
    VideoRecord,
    read_jsonl,
    write_jsonl,
)

PROFILE = """\
id: face-256-1.3x-32f
backend: {name: insightface, model: buffalo_l, det_size: 640}
track: {iou: 0.5, strategy: largest-then-iou}
crop: {scale: 1.3, size: 256, square: true, align: none}
sampling: {mode: uniform, frames: 32}
decode: {library: pyav, color: rgb}
extras: {landmarks: true, mesh: false, masks: false}
"""


def test_processing_profile_from_contract_and_its_id():
    profile = ProcessingProfile.model_validate(yaml.safe_load(PROFILE))
    assert profile.backend.name == "insightface"
    assert profile.backend.model_extra == {"model": "buffalo_l", "det_size": 640}
    assert profile.profile_id() == f"face-256-1.3x-32f-{profile.sha256()[:8]}"
    changed = ProcessingProfile.model_validate(
        yaml.safe_load(PROFILE.replace("frames: 32", "frames: 16"))
    )
    assert changed.sha256() != profile.sha256()


def test_inventory_records_round_trip_with_nested_parts(tmp_path):
    records = [
        InventoryRecord(
            "Deepfakes/000_003",
            "c23",
            "FF-FS_DF",
            "Deepfakes",
            "manipulated_sequences/Deepfakes/c23/videos/000_003.mp4",
            BuilderRef("ffpp", "1.0.0"),
            identity="000",
            probe=Probe(frames=396, fps=29.97, has_audio=False),
        ),
        InventoryRecord(
            "000",
            "c23",
            "FF-REAL",
            "original",
            "original_sequences/youtube/c23/videos/000.mp4",
            BuilderRef("ffpp", "1.0.0"),
        ),
    ]
    path = tmp_path / "inventory.jsonl"
    write_jsonl(path, records)
    assert read_jsonl(path, InventoryRecord) == records
    assert read_jsonl(path, InventoryRecord, strict=True) == records


def test_inventory_relpath_must_be_relative(tmp_path):
    bad = InventoryRecord("k", None, "X", "m", "/abs/k.mp4", BuilderRef("b", "1"))
    with pytest.raises(ContractError, match=r"inventory.jsonl\[0\].relpath"):
        write_jsonl(tmp_path / "inventory.jsonl", [bad])


def test_processed_records_round_trip(tmp_path):
    rows = [
        ProcessedRecord(
            "000", "c23", "ok", 32, list(range(0, 320, 10)), "000", TrackStats(0.98, False)
        ),
        ProcessedRecord(
            "001", "c23", "no_face", 0, [], "001", reason="no face in 50 sampled frames"
        ),
    ]
    write_jsonl(tmp_path / "index.jsonl", rows)
    assert read_jsonl(tmp_path / "index.jsonl", ProcessedRecord, strict=True) == rows


@pytest.mark.slow
def test_reading_250k_video_records_takes_under_a_second(tmp_path):
    line = json.dumps(
        {
            "key": "Deepfakes/000_003",
            "compression": "c23",
            "label_key": "FF-FS_DF",
            "method": "Deepfakes",
            "identity": "000",
            "source_id": "003",
            "target_id": "000",
            "pair_key": "000_003",
            "attrs": {},
        },
        sort_keys=True,
    )
    path = tmp_path / "videos.jsonl"
    path.write_text((line + "\n") * 250_000)
    start = time.perf_counter()
    rows = read_jsonl(path, VideoRecord)
    elapsed = time.perf_counter() - start
    assert len(rows) == 250_000
    assert elapsed < 1.0, f"read took {elapsed:.2f}s"


def test_processing_profile_round_trips_through_yaml():
    profile = ProcessingProfile.model_validate(yaml.safe_load(PROFILE))
    dumped = yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False)
    again = ProcessingProfile.model_validate(yaml.safe_load(dumped))
    assert again == profile
    assert again.profile_id() == profile.profile_id()
