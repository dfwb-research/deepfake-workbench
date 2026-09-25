"""``dfwb.score.writer.assemble_meta``: builds a valid C5 meta with every field present."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests.unit.score._toy import PROTOCOL, load_fake, toy_profile

from dfwb.core.records.scores import SCORE_SCHEMA, ScoreRow
from dfwb.data.adapt import AdaptResult
from dfwb.protocols.protocol import load as load_protocol
from dfwb.score.writer import assemble_meta


def _rows() -> list[ScoreRow]:
    return [
        ScoreRow("scoretoy", "REAL/r00", None, 0, 0.1, "ok", n_clips=4, n_frames=4),
        ScoreRow("scoretoy", "FAKE/f00", None, 1, None, "missing"),
        ScoreRow("scoretoy", "FAKE/f01", None, 1, None, "error"),
    ]


def test_every_c5_meta_field_is_present(scoretoy_pack, tmp_path):
    protocol = load_protocol(PROTOCOL, work_root=tmp_path)
    profile = toy_profile("toy-face")
    detector = load_fake("")
    adaptation = AdaptResult(chain=lambda clip: clip, derived_crop=False, mismatch=False)

    meta = assemble_meta(
        detector_meta=detector.meta,
        checkpoint_sha256=None,
        protocol=protocol,
        split="test",
        where={"compression": "c23"},
        labels="binary",
        processing_profile=profile,
        adaptation=adaptation,
        aggregate_mode="mean-prob",
        clips_per_video=4,
        rows=_rows(),
        seed=7,
    )

    payload = meta.model_dump(mode="json", by_alias=True)
    assert payload["schema"] == SCORE_SCHEMA
    assert payload["detector"] == {
        "name": "fake-detector",
        "version": "0",
        "source": "fake:test",
        "checkpoint_sha256": None,
        "contract_version": [1, 0],
    }
    assert payload["protocol"]["id"] == protocol.ref
    assert payload["protocol"]["split"] == "test"
    assert payload["protocol"]["where"] == {"compression": "c23"}
    assert payload["protocol"]["pack"] == "scoretoy-pack"
    assert payload["protocol"]["pack_version"] == protocol.pack_version
    assert payload["protocol"]["scheme_sha256"] == protocol.sha256
    assert payload["labels"] == "binary"
    assert payload["processing_profile"] == {"id": profile.profile_id(), "sha256": profile.sha256()}
    assert payload["input_adaptation"] == {"derived_crop": False, "mismatch_override": False}
    assert payload["aggregation"] == {"clip_to_video": "mean-prob", "clips_per_video": 4}
    assert payload["coverage"] == {"expected": 3, "ok": 1, "missing": 1, "error": 1}
    assert payload["seed"] == 7
    assert payload["env"]["dfwb"]
    assert payload["env"]["python"]
    assert "git" in payload  # present (null outside a git checkout is still valid C5)
    assert "command" in payload
    assert payload["created"].endswith("Z") or "+" in payload["created"]


def test_coverage_is_recomputed_from_rows_not_trusted(scoretoy_pack, tmp_path):
    protocol = load_protocol(PROTOCOL, work_root=tmp_path)
    profile = toy_profile("toy-face")
    detector = load_fake("")
    adaptation = AdaptResult(chain=lambda clip: clip, derived_crop=False, mismatch=False)

    rows = [ScoreRow("scoretoy", "REAL/r00", None, 0, 0.1, "ok")] * 1
    meta = assemble_meta(
        detector_meta=detector.meta,
        checkpoint_sha256=None,
        protocol=protocol,
        split="test",
        where=None,
        labels="binary",
        processing_profile=profile,
        adaptation=adaptation,
        aggregate_mode="mean-prob",
        clips_per_video=4,
        rows=rows,
        seed=0,
    )
    assert meta.coverage.expected == 1
    assert meta.coverage.ok == 1
    assert meta.coverage.missing == 0
    assert meta.coverage.error == 0
