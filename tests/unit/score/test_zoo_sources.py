"""``zoo:`` through the full scoring harness: a valid C5 file, and the seed reaching ``zoo:random``
end to end -- ``score(..., seed=...)`` threads through ``resolve_detector`` into the adapter, so
two seeds give two different score files and the same seed reproduces its scores exactly."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests.unit.score._toy import PROTOCOL, toy_run_profile, write_toy_store

from dfwb.core.records.scores import read_scores
from dfwb.score.harness import score


def test_zoo_chance_produces_a_valid_c5_file(score_roots, tmp_path):
    write_toy_store(score_roots, toy_run_profile())

    result = score("zoo:chance", protocol=PROTOCOL, split="test", out=tmp_path / "out")

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"ok"}
    assert all(row.score == pytest.approx(0.5) for row in scored.rows)
    assert scored.meta is not None
    assert scored.meta.detector.source == "zoo:chance"
    assert scored.meta.detector.checkpoint_sha256 is None


def test_zoo_random_produces_a_valid_c5_file(score_roots, tmp_path):
    write_toy_store(score_roots, toy_run_profile())

    result = score("zoo:random", protocol=PROTOCOL, split="test", out=tmp_path / "out", seed=0)

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"ok"}
    assert all(0.0 <= row.score <= 1.0 for row in scored.rows)
    assert scored.meta is not None
    assert scored.meta.seed == 0


def test_two_seeds_give_two_different_score_files_with_different_scores(score_roots, tmp_path):
    write_toy_store(score_roots, toy_run_profile())

    first = score("zoo:random", protocol=PROTOCOL, split="test", out=tmp_path / "out", seed=0)
    second = score("zoo:random", protocol=PROTOCOL, split="test", out=tmp_path / "out", seed=1)

    assert first.csv_path != second.csv_path
    first_scores = {(r.dataset, r.key): r.score for r in read_scores(first.csv_path).rows}
    second_scores = {(r.dataset, r.key): r.score for r in read_scores(second.csv_path).rows}
    assert first_scores != second_scores
    first_meta = read_scores(first.csv_path).meta
    second_meta = read_scores(second.csv_path).meta
    assert first_meta is not None
    assert second_meta is not None
    assert first_meta.seed == 0
    assert second_meta.seed == 1


def test_the_same_seed_reproduces_its_scores_exactly(score_roots, tmp_path):
    write_toy_store(score_roots, toy_run_profile())

    first = score("zoo:random", protocol=PROTOCOL, split="test", out=tmp_path / "out", seed=7)
    assert first.cached is False
    second = score("zoo:random", protocol=PROTOCOL, split="test", out=tmp_path / "out", seed=7)

    # Re-running with an identical configuration reuses the cached file outright...
    assert second.csv_path == first.csv_path
    assert second.cached is True
    # ...and forcing a fresh recompute with the same seed reproduces the same scores exactly.
    third = score(
        "zoo:random", protocol=PROTOCOL, split="test", out=tmp_path / "out", seed=7, force=True
    )
    first_scores = [r.score for r in read_scores(first.csv_path).rows]
    third_scores = [r.score for r in read_scores(third.csv_path).rows]
    assert first_scores == third_scores
