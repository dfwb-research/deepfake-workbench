"""The full toyfake journey end to end, through the real ``dfwb`` command, on CPU.

Synthesise (200 videos, the built-in toyfake pack's own reference tree) -> ``inventory build`` ->
``protocols verify`` (full coverage) -> ``preprocess run`` -> ``train`` (the shipped ``toy-cpu``
template, 2 epochs) -> ``score`` the trained run against ``toyfake/official``'s test split ->
``eval`` it (metrics with confidence intervals) -> ``score`` the ``zoo:random`` dummy detector,
both against the same official test split (to pair with the run's file) and against
``toyfake/all-test`` (200 videos: a lower-variance split for the sanity check that a detector with
no information does about as well as chance) -> ``eval compare`` the run against random.

The AUC "within 0.5 +/- 0.1" sanity floor is checked on ``all-test``, not ``official/test``: with
only 41 videos the official test split's AUC standard deviation under ``zoo:random`` is large
enough (about 0.09) that a fixed seed can land outside a +/-0.1 band by chance, where the larger,
200-video ``all-test`` split's standard deviation (about 0.04) does not. The run-vs-random
``eval compare`` stays on the official test split, so both files share the same keys.

Skipped where PyAV or OpenCV (the ``preprocess`` extra) is not installed, where Lightning (the
``train`` extra) is not installed, or where scipy (the ``eval`` extra, needed for ``eval
compare``'s DeLong test on AUC) is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("lightning")
pytest.importorskip("scipy")

from dfwb.core.hashing import sha256_file
from dfwb.core.records import read_scores

DFWB = Path(sys.executable).parent / "dfwb"

VIDEOS = 200  # the built-in toyfake pack's official scheme lists exactly this tree
OFFICIAL_TEST = 41  # toyfake/official's test split
SEED = 0
PROFILE = "toy-64-center-8f"

_CONFIG = """\
schema: dfwb.train/1
extends: ["dfwb://templates/toy-cpu.yaml"]
"""


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(
    not (_installed("av") and _installed("cv2")),
    reason="needs the preprocess extra (av, opencv) to encode and decode toyfake's videos",
)


def _env(tmp_path: Path, datasets_root: Path, work_root: Path, runs_root: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("DFWB_")}
    env.update(
        {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "DFWB_DATASETS_ROOT": str(datasets_root),
            "DFWB_WORK_ROOT": str(work_root),
            "DFWB_RUNS_ROOT": str(runs_root),
        }
    )
    return env


def _run(tmp_path: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(DFWB), "--no-env-file", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _full_coverage(n: int) -> dict[str, int]:
    return {"expected": n, "ok": n, "missing": 0, "error": 0}


def test_the_full_toyfake_journey(tmp_path):
    datasets_root = tmp_path / "datasets"
    work_root = tmp_path / "work"
    runs_root = tmp_path / "runs"
    env = _env(tmp_path, datasets_root, work_root, runs_root)

    # -------------------------------------------------------------- synth -> inventory -> verify

    synth = _run(
        tmp_path,
        env,
        "datasets",
        "synth",
        "toyfake",
        "--out",
        str(datasets_root),
        "--videos",
        str(VIDEOS),
        "--seed",
        str(SEED),
        "--json",
    )
    assert synth.returncode == 0, synth.stderr
    assert json.loads(synth.stdout)["n_videos"] == VIDEOS

    built = _run(tmp_path, env, "inventory", "build", "toyfake", "--json")
    assert built.returncode == 0, built.stderr
    assert json.loads(built.stdout)["count"] == VIDEOS

    verified = _run(tmp_path, env, "protocols", "verify", "toyfake/official", "--json")
    verified_payload = json.loads(verified.stdout)
    assert verified.returncode == 0, verified.stderr
    assert verified_payload["exit_code"] == 0
    assert verified_payload["counts"]["have"] == VIDEOS
    assert verified_payload["counts"]["missing_requested"] == 0
    assert verified_payload["counts"]["label_mismatch"] == 0

    # ------------------------------------------------------------------------------- preprocess

    processed = _run(
        tmp_path,
        env,
        "preprocess",
        "run",
        "toyfake",
        "--profile",
        PROFILE,
        "--workers",
        "0",
        "--json",
    )
    assert processed.returncode == 0, processed.stderr
    assert json.loads(processed.stdout)["counts_by_status"] == {"ok": VIDEOS}

    # ----------------------------------------------------------------------------------- train

    config = tmp_path / "train.yaml"
    config.write_text(_CONFIG, "utf-8")

    trained = _run(tmp_path, env, "train", "-c", str(config), "--device", "cpu", "--json")
    assert trained.returncode == 0, trained.stderr
    (train_result,) = json.loads(trained.stdout)
    run_dir = Path(train_result["run_dir"])
    assert train_result["metrics"]["epochs"] == 2

    checkpoint_sha256 = sha256_file(run_dir / "checkpoints" / "best" / "model.safetensors")

    # the run's own "latest" symlink (<runs root>/<run name>/latest), the way a real user would
    # point --detector at a run they just trained.
    latest_link = run_dir.parent / "latest"
    assert latest_link.is_symlink()
    assert latest_link.resolve() == run_dir.resolve()
    run_detector = f"run:{latest_link}"

    # ------------------------------------------------------------------ score the trained run

    score_run_args = (
        "score",
        "--detector",
        run_detector,
        "--protocol",
        "toyfake/official",
        "--split",
        "test",
        "--device",
        "cpu",
        "--json",
    )
    scored_run = _run(tmp_path, env, *score_run_args)
    assert scored_run.returncode == 0, scored_run.stderr
    (run_row,) = json.loads(scored_run.stdout)["results"]
    assert run_row["cached"] is False
    assert run_row["coverage"] == _full_coverage(OFFICIAL_TEST)
    run_scores_path = Path(run_row["csv"])

    run_scored = read_scores(run_scores_path)
    assert run_scored.meta is not None
    assert run_scored.meta.coverage.model_dump() == _full_coverage(OFFICIAL_TEST)
    assert run_scored.meta.detector.checkpoint_sha256 == checkpoint_sha256

    # a second, identical score call is a cache hit
    recached = _run(tmp_path, env, *score_run_args)
    assert recached.returncode == 0, recached.stderr
    (recached_row,) = json.loads(recached.stdout)["results"]
    assert recached_row["cached"] is True
    assert recached_row["csv"] == run_row["csv"]

    # -------------------------------------------------------------------------------------- eval

    evaluated = _run(tmp_path, env, "eval", str(run_scores_path), "--json")
    assert evaluated.returncode == 0, evaluated.stderr
    (run_file_row,) = json.loads(evaluated.stdout)["tables"]["files"]
    run_auc = run_file_row["metrics"]["auc"]
    assert 0.0 <= run_auc["value"] <= 1.0
    assert run_auc["ci_lo"] is not None
    assert run_auc["ci_hi"] is not None
    assert run_auc["ci_lo"] <= run_auc["value"] <= run_auc["ci_hi"]

    # ---------------------------------------------------------------- score zoo:random (x2)

    # on the official test split, to pair against the run's own file in the compare below
    scored_random_official = _run(
        tmp_path,
        env,
        "score",
        "--detector",
        "zoo:random",
        "--protocol",
        "toyfake/official",
        "--split",
        "test",
        "--device",
        "cpu",
        "--allow-input-mismatch",
        "--json",
    )
    assert scored_random_official.returncode == 0, scored_random_official.stderr
    (random_official_row,) = json.loads(scored_random_official.stdout)["results"]
    assert random_official_row["coverage"] == _full_coverage(OFFICIAL_TEST)
    random_official_path = Path(random_official_row["csv"])

    # on all-test (200 videos): the sanity floor that a detector with no information whatsoever
    # scores about as well as chance, checked where the statistics are stable enough for a fixed
    # seed not to land outside +/-0.1 by chance.
    scored_random_all = _run(
        tmp_path,
        env,
        "score",
        "--detector",
        "zoo:random",
        "--protocol",
        "toyfake/all-test",
        "--split",
        "test",
        "--device",
        "cpu",
        "--allow-input-mismatch",
        "--json",
    )
    assert scored_random_all.returncode == 0, scored_random_all.stderr
    (random_all_row,) = json.loads(scored_random_all.stdout)["results"]
    assert random_all_row["coverage"] == _full_coverage(VIDEOS)
    random_all_path = Path(random_all_row["csv"])

    random_evaluated = _run(
        tmp_path,
        env,
        "eval",
        str(random_all_path),
        "--metrics",
        "auc",
        "--bootstrap",
        "0",
        "--json",
    )
    assert random_evaluated.returncode == 0, random_evaluated.stderr
    (random_file_row,) = json.loads(random_evaluated.stdout)["tables"]["files"]
    random_auc = random_file_row["metrics"]["auc"]["value"]
    assert 0.4 <= random_auc <= 0.6

    # -------------------------------------------------------------------------------- compare

    compared = _run(
        tmp_path,
        env,
        "eval",
        "compare",
        str(run_scores_path),
        str(random_official_path),
        "--json",
    )
    assert compared.returncode == 0, compared.stderr
    (comparison,) = json.loads(compared.stdout)["comparisons"]
    assert comparison["n"] == OFFICIAL_TEST
    auc_comparison = comparison["metrics"]["auc"]
    assert auc_comparison["delta"] == pytest.approx(auc_comparison["b"] - auc_comparison["a"])
    assert auc_comparison["delong_z"] is not None
    assert auc_comparison["delong_p"] is not None
