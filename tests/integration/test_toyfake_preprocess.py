"""toyfake through the face pipeline, end to end, through the ``dfwb`` command.

``datasets synth`` writes 40 videos, ``inventory build`` scans them, and ``preprocess run`` with
the dependency-free ``center`` backend processes every one of them: ``status`` then reports every
video ``ok``, each with exactly 8 frames, and running the whole pipeline again into a second,
independent work root writes a byte-identical store.

Skipped where PyAV or OpenCV (the ``preprocess`` extra) is not installed: ``datasets synth`` needs
PyAV to encode toyfake's videos, and the face pipeline needs OpenCV to decode them and write
frames.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dfwb.core.records import ProcessedRecord, read_jsonl

DFWB = Path(sys.executable).parent / "dfwb"

VIDEOS = 40
PROFILE = "toy-64-center-8f"
FRAMES = 8


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


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root``, by POSIX relative path, with its bytes."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _process(tmp_path: Path, datasets_root: Path, work_root: Path) -> Path:
    """Build the inventory and run the face pipeline into ``work_root``; return the store."""
    env = _env(tmp_path, datasets_root, work_root)

    built = _run(
        tmp_path, env, "inventory", "build", "toyfake", "--root", str(datasets_root / "toyfake")
    )
    assert built.returncode == 0, built.stderr

    ran = _run(
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
    assert ran.returncode == 0, ran.stderr
    summary = json.loads(ran.stdout)
    assert summary["counts_by_status"] == {"ok": VIDEOS}
    assert summary["n_skipped"] == 0
    store = Path(summary["store"])

    status = _run(tmp_path, env, "preprocess", "status", "toyfake", "--profile", PROFILE, "--json")
    assert status.returncode == 0, status.stderr
    rows = json.loads(status.stdout)
    assert rows, "status reported no rows"
    assert all(row["status"] == "ok" for row in rows)
    assert sum(row["count"] for row in rows) == VIDEOS

    records = read_jsonl(store / "index.jsonl", ProcessedRecord)
    assert len(records) == VIDEOS
    for record in records:
        assert record.status == "ok", record
        assert record.n_frames == FRAMES, record
        assert len(record.frame_indices) == FRAMES, record
        assert record.reason is None, record

    return store


def test_toyfake_face_pipeline_is_deterministic_across_two_work_roots(tmp_path):
    datasets_root = tmp_path / "datasets"
    synth = _run(
        tmp_path,
        _env(tmp_path, datasets_root, tmp_path / "unused-work"),
        "datasets",
        "synth",
        "toyfake",
        "--out",
        str(datasets_root),
        "--videos",
        str(VIDEOS),
        "--json",
    )
    assert synth.returncode == 0, synth.stderr
    result = json.loads(synth.stdout)
    assert result["root"] == str(datasets_root / "toyfake")
    assert result["n_videos"] == VIDEOS

    store_a = _process(tmp_path, datasets_root, tmp_path / "work-a")
    store_b = _process(tmp_path, datasets_root, tmp_path / "work-b")

    tree_a, tree_b = _tree(store_a), _tree(store_b)
    assert sorted(tree_a) == sorted(tree_b)
    for name, data in tree_a.items():
        assert tree_b[name] == data, f"{name} differs between the two runs"
