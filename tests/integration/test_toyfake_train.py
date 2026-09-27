"""The toyfake journey end to end, through the ``dfwb`` command, on CPU.

``datasets synth`` writes 200 videos, ``inventory build`` scans them, ``preprocess run`` with the
``toy-64-center-8f`` profile processes every one of them into a store, and ``dfwb train`` on the
shipped ``toy-cpu`` template trains a tiny detector for two epochs.

``--videos 200 --seed 0`` is not an arbitrary size: it is exactly the tree the built-in ``toyfake``
protocol pack's ``official`` scheme lists (its card says so), so every video the scheme's train and
val splits ask for is actually in the store -- nothing is excluded, and both classes end up in both
splits, so the reported AUC is a real, defined number, never the "no validation" fallback.

Skipped where PyAV or OpenCV (the ``preprocess`` extra) is not installed, since ``datasets synth``
needs PyAV to encode real media and the face pipeline needs OpenCV to decode it; skipped where
Lightning (the ``train`` extra) is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("lightning")

import torch

from dfwb.core.detector import ClipBatch
from dfwb.models.source import load_run

DFWB = Path(sys.executable).parent / "dfwb"

VIDEOS = 200  # the built-in toyfake pack's official scheme lists exactly this tree
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


def _env(tmp_path: Path, datasets_root: Path, work_root: Path) -> dict[str, str]:
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
        }
    )
    return env


def _run(tmp_path: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(DFWB), *args], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )


def _predict_on_a_batch(run_dir: Path) -> None:
    detector = load_run(f"{run_dir}#best")
    size = detector.meta.input.size
    batch = ClipBatch(
        clips=torch.rand(3, 1, 3, *size),
        keys=["a", "b", "c"],
        dataset_ids=["toyfake"] * 3,
        compressions=[None] * 3,
        clip_index=torch.zeros(3, dtype=torch.long),
        frame_indices=torch.zeros(3, 1, dtype=torch.long),
    )
    with torch.inference_mode():
        prediction = detector.predict(batch)
    assert prediction.score.shape == (3,)
    assert torch.all((prediction.score >= 0) & (prediction.score <= 1))


def test_the_toyfake_journey_trains_on_cpu(tmp_path):
    datasets_root = tmp_path / "datasets"
    work_root = tmp_path / "work"
    env = _env(tmp_path, datasets_root, work_root)

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

    config = tmp_path / "train.yaml"
    config.write_text(_CONFIG, "utf-8")

    trained = _run(
        tmp_path,
        env,
        "train",
        "-c",
        str(config),
        "--device",
        "cpu",
        "--json",
    )
    assert trained.returncode == 0, trained.stderr
    (result,) = json.loads(trained.stdout)
    run_dir = Path(result["run_dir"])

    # the documented run directory layout
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "checkpoints",
        "config.resolved.yaml",
        "data.json",
        "env.json",
        "fingerprint.txt",
        "logs",
        "metrics.json",
        "report.md",
        "scores",
    ]
    checkpoint_files = sorted(
        p.relative_to(run_dir / "checkpoints").as_posix()
        for p in (run_dir / "checkpoints").rglob("*")
        if p.is_file()
    )
    assert checkpoint_files == [
        "best/detector.json",
        "best/model.safetensors",
        "last/detector.json",
        "last/model.safetensors",
    ]

    # this is exactly the tree the built-in pack's official scheme lists: nothing is excluded,
    # and both classes reach both splits.
    data = json.loads((run_dir / "data.json").read_text("utf-8"))
    (train_source,) = data["train"][0]["index"]["sources"]
    (val_source,) = data["val"][0]["index"]["sources"]
    assert train_source["counts"]["excluded"] == {}
    assert val_source["counts"]["excluded"] == {}
    for source in (train_source, val_source):
        assert source["counts"]["included"] > 0
        assert set(source["labels"]) == {"0", "1"}
        assert all(count > 0 for count in source["labels"].values())

    # the validation AUC is a defined number: not NaN, not the "no validation" fallback
    monitor = result["metrics"]["monitor"]
    assert monitor["key"] == "val/video_auc"
    assert monitor["fallback"] is False
    assert monitor["best"] is not None
    assert math.isfinite(monitor["best"])

    # the best checkpoint loads through the run: detector source and predicts on a batch
    _predict_on_a_batch(run_dir)
