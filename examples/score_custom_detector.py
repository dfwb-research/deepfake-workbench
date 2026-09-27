#!/usr/bin/env python3
"""Score a detector of your own -- no training, no adapter card -- through a ``py:`` source.

Mirrors the "add a detector" journey: `BrightnessDetector`, a toy `Detector` (contract C4) that
scores a clip by its own mean pixel brightness, is resolved by `dfwb score` through
``py:<module>:<factory>`` and run over the toyfake dataset it generates and processes first. Needs
the ``preprocess`` and ``train`` extras (`dfwb.core.detector.ClipBatch.clips` is a `torch.Tensor`
regardless of what the detector itself does with it).

Reads its roots from ``DFWB_DATASETS_ROOT``, ``DFWB_WORK_ROOT`` and ``DFWB_RUNS_ROOT`` if already
set (a test points these at a temporary directory); otherwise it creates and works inside
``./dfwb-example-detector/`` (``{datasets,work,runs}`` plus the detector module itself), so nothing
is left loose in the caller's own directory. It never touches a real datasets root,
``~/.cache/dfwb`` or ``~/.local/state/dfwb``.

    uv run python examples/score_custom_detector.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dfwb.cli.main import main

EXAMPLE_ROOT = Path("dfwb-example-detector").resolve()
PROFILE = "toy-64-center-8f"

_DETECTOR_MODULE = '''\
"""A toy Detector: P(fake) = the clip's own mean pixel brightness."""

from __future__ import annotations

import torch

from dfwb.core.detector import DetectorMeta, DetectorOutput, InputSpec

_SPEC = InputSpec(crop="full-frame", crop_scale=None, size=(64, 64), frames=8)


class BrightnessDetector:
    def __init__(self) -> None:
        self.meta = DetectorMeta(
            name="brightness",
            version="0.1",
            contract_version=(1, 0),
            input=_SPEC,
            license="MIT",
            weights_license=None,
            citation=None,
            source="py:my_detector:make_detector",
        )

    def to(self, device: torch.device) -> "BrightnessDetector":
        return self

    def predict(self, batch) -> DetectorOutput:
        score = batch.clips.mean(dim=tuple(range(1, batch.clips.ndim))).clamp(0.0, 1.0)
        return DetectorOutput(score=score)


def make_detector() -> BrightnessDetector:
    return BrightnessDetector()
'''


def _root(env_var: str, subdir: str) -> Path:
    return Path(os.environ[env_var]) if env_var in os.environ else EXAMPLE_ROOT / subdir


def _run(*args: str) -> None:
    print(f"$ dfwb {' '.join(args)}")
    code = main(["--no-env-file", *args])
    if code != 0:
        raise SystemExit(f"dfwb {' '.join(args)} exited with {code}")


def main_() -> None:
    EXAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    datasets_root = _root("DFWB_DATASETS_ROOT", "datasets")
    work_root = _root("DFWB_WORK_ROOT", "work")
    runs_root = _root("DFWB_RUNS_ROOT", "runs")
    os.environ["DFWB_DATASETS_ROOT"] = str(datasets_root)
    os.environ["DFWB_WORK_ROOT"] = str(work_root)
    os.environ["DFWB_RUNS_ROOT"] = str(runs_root)

    (EXAMPLE_ROOT / "my_detector.py").write_text(_DETECTOR_MODULE, encoding="utf-8")
    # What `PYTHONPATH=<dir>` does for a real shell invocation.
    sys.path.insert(0, str(EXAMPLE_ROOT))

    _run("datasets", "synth", "toyfake", "--out", str(datasets_root))
    _run("inventory", "build", "toyfake")
    _run("preprocess", "run", "toyfake", "--profile", PROFILE)

    _run(
        "score",
        "--detector",
        "py:my_detector:make_detector",
        "--protocol",
        "toyfake/official",
        "--split",
        "test",
        "--profile",
        PROFILE,
    )
    scores = sorted((runs_root / "scores").rglob("*.scores.csv"))
    if not scores:
        raise SystemExit("no score file was written")
    _run("eval", str(scores[-1]), "--bootstrap", "0")


if __name__ == "__main__":
    main_()
    sys.exit(0)
