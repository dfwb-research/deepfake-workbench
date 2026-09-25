"""A fabricated run directory, as training leaves it (only the files ``dfwb runs`` reads), so
the ``runs`` commands can be tested without torch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

__all__ = ["FP_A", "FP_B", "fake_run"]

FP_A = "a" * 64
FP_B = "b" * 64


def fake_run(
    root: Path,
    name: str,
    stamp: str,
    seed: int,
    fingerprint: str,
    *,
    completed: bool = True,
    latest: bool = False,
) -> Path:
    """A run directory as training leaves it (only the files `dfwb runs` reads)."""
    run_dir = root / name / f"{stamp}-s{seed}"
    run_dir.mkdir(parents=True)
    config: dict[str, Any] = {"schema": "dfwb.train/1", "run": {"name": name, "seeds": [seed]}}
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "fingerprint.txt").write_text(f"{fingerprint}\n")
    env = {
        "seed": seed,
        "created": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T00:00:00Z",
        "command": "dfwb train -c exp.yaml",
        "env": {"dfwb": "0.1.0", "python": "3.12.0", "torch": "2.6.0", "device": "cpu"},
        "git": {"commit": "0123456789abcdef", "dirty": True},
        "plugins": {},
    }
    (run_dir / "env.json").write_text(json.dumps(env))
    source = {
        "name": "toyfake-official",
        "protocol": {"ref": "toyfake/official", "split": "train"},
        "index": {"sources": [{"counts": {"in_split": 10, "included": 9, "excluded": {}}}]},
    }
    (run_dir / "data.json").write_text(json.dumps({"train": [source], "val": []}))
    if completed:
        metrics = {
            "seed": seed,
            "fingerprint": fingerprint,
            "epochs": 2,
            "global_step": 8,
            "monitor": {
                "key": "val/video_auc",
                "mode": "max",
                "fallback": False,
                "best": 0.75,
                "best_epoch": 1,
            },
            "val": {"val/loss": 0.5, "val/video_auc": 0.75},
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
    else:
        (run_dir / "resume").mkdir()
        (run_dir / "resume" / "state.json").write_text(json.dumps({"epoch": 1, "seed": seed}))
    if latest:
        (root / name / "latest").symlink_to(run_dir.name)
    return run_dir
