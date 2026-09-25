"""``dfwb.score.harness.score(frames=True)``: the optional per-clip/per-frame parquet dump."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from tests.unit.score._toy import PROTOCOL, toy_profile, write_toy_store

from dfwb.core.records.scores import read_scores
from dfwb.score.harness import _frame_records, score
from dfwb.score.writer import frames_path_for


def _score(tmp_path, detector_uri="fake:", **kwargs):
    kwargs.setdefault("protocol", PROTOCOL)
    kwargs.setdefault("split", "test")
    kwargs.setdefault("out", tmp_path / "out")
    return score(detector_uri, **kwargs)


# --------------------------------------------------------------------------- _frame_records itself


def _batch(clip_index, frame_indices):
    from types import SimpleNamespace

    return SimpleNamespace(
        clip_index=torch.tensor(clip_index), frame_indices=torch.tensor(frame_indices)
    )


def test_frame_records_uses_the_detectors_own_frame_scores_when_present():
    from types import SimpleNamespace

    batch = _batch([0, 1], [[0, 1], [2, 3]])
    keys = [("scoretoy", "REAL/r00", None), ("scoretoy", "FAKE/f00", None)]
    scores = [0.3, 0.7]
    output = SimpleNamespace(frame_scores=torch.tensor([[0.1, 0.5], [0.6, 0.8]]))

    records = _frame_records(batch, keys, scores, output)

    assert len(records) == 4
    by_key_frame = {(r.key, r.frame_index): r.frame_score for r in records}
    assert by_key_frame[("REAL/r00", 0)] == pytest.approx(0.1)
    assert by_key_frame[("REAL/r00", 1)] == pytest.approx(0.5)
    assert by_key_frame[("FAKE/f00", 2)] == pytest.approx(0.6)
    assert by_key_frame[("FAKE/f00", 3)] == pytest.approx(0.8)
    # frame_score is not simply the clip score broadcast, unlike the fallback below
    assert {r.frame_score for r in records} != {r.clip_score for r in records}
    for record in records:
        assert record.clip_score == scores[0 if record.key == "REAL/r00" else 1]


def test_frame_records_broadcasts_the_clip_score_when_frame_scores_is_absent():
    from types import SimpleNamespace

    batch = _batch([0], [[0, 1, 2]])
    keys = [("scoretoy", "REAL/r00", None)]
    scores = [0.42]
    output = SimpleNamespace(frame_scores=None)

    records = _frame_records(batch, keys, scores, output)

    assert len(records) == 3
    assert {r.frame_index for r in records} == {0, 1, 2}
    for record in records:
        assert record.frame_score == pytest.approx(0.42)
        assert record.clip_score == pytest.approx(0.42)


def test_frame_records_handles_a_bare_output_with_no_frame_scores_attribute_at_all():
    from types import SimpleNamespace

    batch = _batch([0], [[0]])
    output = SimpleNamespace()  # no frame_scores attribute at all (getattr fallback)

    records = _frame_records(batch, [("d", "k", None)], [0.9], output)

    assert len(records) == 1
    assert records[0].frame_score == pytest.approx(0.9)


# ------------------------------------------------------------------------------- score(frames=True)


def test_frames_false_never_writes_a_parquet_file(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path)

    assert result.frames_path is None
    assert not frames_path_for(result.csv_path).is_file()


def test_frames_true_writes_one_row_per_clip_per_frame(score_roots, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:frames=2", clips_per_video=3, frames=True)

    assert result.frames_path is not None
    assert result.frames_path == frames_path_for(result.csv_path)
    table = pq.read_table(result.frames_path)
    assert table.num_rows == 8 * 3 * 2  # 8 videos * 3 clips * 2 frames each
    assert set(table.column_names) == {
        "dataset",
        "key",
        "compression",
        "clip_index",
        "frame_index",
        "frame_score",
        "clip_score",
    }
    rows = table.to_pylist()
    for row in rows:
        assert row["frame_score"] == pytest.approx(row["clip_score"])  # FakeDetector's fallback


def test_frames_is_never_written_on_a_cache_hit(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    first = _score(tmp_path, frames=True)
    assert first.cached is False
    assert first.frames_path is not None

    second = _score(tmp_path, frames=True)

    assert second.cached is True
    assert second.frames_path is None


def test_frames_true_is_skipped_without_pyarrow(score_roots, tmp_path, monkeypatch):
    import builtins

    write_toy_store(score_roots, toy_profile("toy-face"))
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    result = _score(tmp_path, frames=True)

    assert result.frames_path is None
    assert result.cached is False
    scored = read_scores(result.csv_path)  # the main C5 file is still written normally
    assert {row.status for row in scored.rows} == {"ok"}
