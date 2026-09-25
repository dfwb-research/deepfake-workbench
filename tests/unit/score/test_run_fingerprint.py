"""The ``run:`` detector fingerprint: ``checkpoint_sha256``/``training_seed`` flow into C5 and
the cache key, end to end through ``dfwb.score.harness.score``."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests.unit.score._toy import PROTOCOL, toy_run_profile, write_toy_run, write_toy_store

from dfwb.core.hashing import sha256_file
from dfwb.core.records.scores import read_scores
from dfwb.models import checkpoint
from dfwb.score.harness import score


def _score(tmp_path, run_ref, **kwargs):
    kwargs.setdefault("protocol", PROTOCOL)
    kwargs.setdefault("split", "test")
    kwargs.setdefault("out", tmp_path / "out")
    kwargs.setdefault("batch_size", 4)
    return score(f"run:{run_ref}", **kwargs)


def test_best_and_last_give_different_cache_paths(score_roots, tmp_path):
    write_toy_store(score_roots, toy_run_profile())
    run_dir = write_toy_run(
        tmp_path / "runs", name="toy", seed=0, fingerprint="shared-fp", tags=("best", "last")
    )

    best = _score(tmp_path, f"{run_dir}#best")
    last = _score(tmp_path, f"{run_dir}#last")

    assert best.csv_path != last.csv_path
    best_meta = read_scores(best.csv_path).meta
    last_meta = read_scores(last.csv_path).meta
    assert best_meta.detector.checkpoint_sha256 != last_meta.detector.checkpoint_sha256
    assert best_meta.detector.source == last_meta.detector.source == "run:shared-fp"


def test_different_seeds_give_different_paths_and_record_their_own_seeds(score_roots, tmp_path):
    write_toy_store(score_roots, toy_run_profile())
    run_a = write_toy_run(tmp_path / "runs", name="toy-a", seed=0, fingerprint="shared-fp")
    run_b = write_toy_run(tmp_path / "runs", name="toy-b", seed=1, fingerprint="shared-fp")

    result_a = _score(tmp_path, f"{run_a}#best")
    result_b = _score(tmp_path, f"{run_b}#best")

    assert result_a.csv_path != result_b.csv_path
    assert read_scores(result_a.csv_path).meta.seed == 0
    assert read_scores(result_b.csv_path).meta.seed == 1


def test_c5_checkpoint_sha256_equals_the_weights_files_sha256(score_roots, tmp_path):
    write_toy_store(score_roots, toy_run_profile())
    run_dir = write_toy_run(tmp_path / "runs", name="toy", seed=3, fingerprint="fp")

    result = _score(tmp_path, f"{run_dir}#best")

    expected = sha256_file(checkpoint.weights_path(run_dir / "checkpoints" / "best"))
    meta = read_scores(result.csv_path).meta
    assert meta.detector.checkpoint_sha256 == expected
    assert meta.detector.checkpoint_sha256 is not None


def test_seedless_run_falls_back_to_the_requested_seed(score_roots, tmp_path):
    write_toy_store(score_roots, toy_run_profile())
    # seed=None: write_toy_run never writes env.json, so training_seed is None.
    run_dir = write_toy_run(tmp_path / "runs", name="toy", seed=None, fingerprint="fp")

    result = _score(tmp_path, f"{run_dir}#best", seed=42)

    assert read_scores(result.csv_path).meta.seed == 42
