"""``dfwb.score.harness.score(frames=True)``: the optional per-clip/per-frame parquet dump."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from tests.unit.score import _toy
from tests.unit.score._toy import (
    PROTOCOL,
    toy_profile,
    toy_run_profile,
    write_toy_run,
    write_toy_store,
)

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


def test_a_plain_run_then_frames_recomputes_and_writes_a_file(score_roots, tmp_path):
    """A cache hit whose own frames file does not exist yet (it was cached before ``--frames``
    was first asked for) is treated as a miss, not served without one."""
    write_toy_store(score_roots, toy_profile("toy-face"))
    plain = _score(tmp_path)
    assert plain.cached is False
    assert plain.frames_path is None

    result = _score(tmp_path, frames=True)

    assert result.cached is False
    assert result.csv_path == plain.csv_path  # same cache entry: frames never enters the key
    assert result.frames_path is not None
    assert result.frames_path.is_file()


def test_frames_twice_is_cached_with_the_frames_path_set(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    first = _score(tmp_path, frames=True)
    assert first.cached is False
    assert first.frames_path is not None

    second = _score(tmp_path, frames=True)

    assert second.cached is True
    assert second.csv_path == first.csv_path
    assert second.frames_path == first.frames_path
    assert second.frames_path is not None
    assert second.frames_path.is_file()


def test_frames_without_pyarrow_raises_before_any_predict_call(score_roots, tmp_path, monkeypatch):
    import builtins
    import uuid

    from dfwb.core.errors import InstallationError

    write_toy_store(score_roots, toy_profile("toy-face"))
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    spy_id = str(uuid.uuid4())

    with pytest.raises(InstallationError, match="pyarrow") as info:
        _score(tmp_path, f"fake:spy={spy_id}", frames=True)

    assert "deepfake-workbench[eval]" in info.value.hint
    assert spy_id not in _toy.SPY_CALLS  # predict() was never called


def test_frames_true_via_a_real_image_backbone_run_detector(score_roots, tmp_path):
    """End-to-end with a real C4 detector (``run:``, ``tiny-cnn``) that actually fills
    ``DetectorOutput.frame_scores`` -- exercises the non-fallback branch through the full
    pipeline, not just :func:`_frame_records` in isolation."""
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    write_toy_store(score_roots, toy_run_profile())
    run_dir = write_toy_run(tmp_path / "runs", name="toy", seed=0, fingerprint="fp")

    result = score(
        f"run:{run_dir}#best",
        protocol=PROTOCOL,
        split="test",
        out=tmp_path / "out",
        batch_size=4,
        frames=True,
    )

    assert result.coverage["ok"] == result.coverage["expected"]
    assert result.frames_path is not None
    table = pq.read_table(result.frames_path)
    # tiny-cnn is a frame model (InputSpec.frames == 1): ok videos * clips_per_video(4) * 1 frame.
    assert table.num_rows == result.coverage["ok"] * 4
    rows = table.to_pylist()
    for row in rows:
        assert 0.0 <= row["frame_score"] <= 1.0
        assert 0.0 <= row["clip_score"] <= 1.0


def test_frames_false_after_a_frames_run_unlinks_the_stale_frames_file(score_roots, tmp_path):
    """A fresh CSV written without a matching frames dump must never leave a stale one behind:
    its presence has to always match the CSV it sits next to."""
    write_toy_store(score_roots, toy_profile("toy-face"))
    with_frames = _score(tmp_path, frames=True)
    assert with_frames.frames_path is not None
    assert with_frames.frames_path.is_file()

    without_frames = _score(tmp_path, force=True, frames=False)

    assert without_frames.csv_path == with_frames.csv_path
    assert without_frames.frames_path is None
    assert not frames_path_for(without_frames.csv_path).is_file()


# ------------------------------------------------------------------ validating frame_scores


@pytest.mark.parametrize("bad", ["frames-shape", "frames-nan", "frames-range"])
def test_bad_frame_scores_mark_their_videos_error_and_scoring_continues(score_roots, tmp_path, bad):
    """``frame_scores`` is checked like ``score``: ``[B, T]``, finite, in ``[0, 1]``. A batch
    that fails marks its own videos ``error``; it never aborts the run."""
    pytest.importorskip("pyarrow")
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, f"fake:bad={bad}&frames=2", batch_size=4, frames=True)

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"error"}
    assert result.coverage == {"expected": 8, "ok": 0, "missing": 0, "error": 8}
    assert result.frames_path is None


def test_bad_frame_scores_do_not_matter_without_frames(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:bad=frames-nan&frames=2", batch_size=4)

    assert result.coverage["ok"] == 8


def test_valid_frame_scores_reach_the_parquet_file(score_roots, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:bad=frames-ok&frames=2", clips_per_video=1, frames=True)

    assert result.coverage["ok"] == 8
    frame_scores = sorted(
        {round(r["frame_score"], 4) for r in pq.read_table(result.frames_path).to_pylist()}
    )
    assert frame_scores == [0.1, 0.9]


def test_frame_records_refuses_frame_scores_that_are_not_a_tensor():
    from types import SimpleNamespace

    batch = _batch([0], [[0, 1]])
    output = SimpleNamespace(frame_scores=[[0.1, 0.2]])

    with pytest.raises(ValueError, match="not a Tensor"):
        _frame_records(batch, [("d", "k", None)], [0.5], output)
