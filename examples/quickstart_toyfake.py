#!/usr/bin/env python3
"""The "try it in five minutes" quickstart, run as a script instead of typed at a shell.

Mirrors ``docs/quickstart.md`` exactly: synthesize toyfake, build its inventory, verify it,
process it, train a tiny model on it for two epochs, then score and evaluate the trained run.
Needs the ``train`` and ``preprocess`` extras (``uv sync --extra train --extra preprocess``).

Reads its three roots from ``DFWB_DATASETS_ROOT``, ``DFWB_WORK_ROOT`` and ``DFWB_RUNS_ROOT`` if
already set (a test points these at a temporary directory); otherwise it creates
``./dfwb-example-toyfake/{datasets,work,runs}`` next to wherever it is run from. It never touches
a real datasets root, ``~/.cache/dfwb`` or ``~/.local/state/dfwb``.

    python examples/quickstart_toyfake.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dfwb.cli.main import main

EXAMPLE_ROOT = Path("dfwb-example-toyfake").resolve()


def _root(env_var: str, subdir: str) -> Path:
    return Path(os.environ[env_var]) if env_var in os.environ else EXAMPLE_ROOT / subdir


def _run(*args: str) -> None:
    print(f"$ dfwb {' '.join(args)}")
    code = main(["--no-env-file", *args])
    if code != 0:
        raise SystemExit(f"dfwb {' '.join(args)} exited with {code}")


def main_() -> None:
    datasets_root = _root("DFWB_DATASETS_ROOT", "datasets")
    work_root = _root("DFWB_WORK_ROOT", "work")
    runs_root = _root("DFWB_RUNS_ROOT", "runs")
    os.environ["DFWB_DATASETS_ROOT"] = str(datasets_root)
    os.environ["DFWB_WORK_ROOT"] = str(work_root)
    os.environ["DFWB_RUNS_ROOT"] = str(runs_root)

    _run("datasets", "synth", "toyfake", "--out", str(datasets_root))
    _run("inventory", "build", "toyfake")
    _run("protocols", "verify", "toyfake")

    _run("preprocess", "run", "toyfake", "--profile", "toy-64-center-8f")
    _run("preprocess", "status", "toyfake", "--profile", "toy-64-center-8f")

    config = Path("toy-cpu.yaml")
    config.write_text(
        "schema: dfwb.train/1\nextends: [dfwb://templates/toy-cpu.yaml]\n", encoding="utf-8"
    )
    _run("train", "-c", str(config), "--device", "cpu")

    _run("runs", "list")
    _run("runs", "show", "toy-cpu")

    _run(
        "score",
        "--detector",
        f"run:{runs_root}/toy-cpu/latest#best",
        "--protocol",
        "toyfake/official",
        "--split",
        "test",
    )
    scores = sorted((runs_root / "scores").rglob("*.scores.csv"))
    if not scores:
        raise SystemExit("no score file was written")
    _run("eval", str(scores[-1]))


if __name__ == "__main__":
    main_()
    sys.exit(0)
