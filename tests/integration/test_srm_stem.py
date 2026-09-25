"""The ``srm`` stem, wired in front of a real backbone through the ``dfwb`` command.

``dfwb_torch_srm`` registers ``layers/srm`` (an SRM/high-pass residual filter bank) with the
``layers`` registry; ``model.stem`` puts it in front of any backbone. This trains the toyfake
journey's config for one epoch with ``model.stem: {name: srm, bank: srm30, mode: gray}`` added,
and checks that the stem the run actually built is SRM, not some other ``layers`` entry, and that
its output was adapted back to the backbone's expected channel count.

``gray`` mode outputs one channel per kernel in ``srm30`` (30), not 3, so the stem hook wraps it
in a 1x1 convolution back to 3 channels: the framework's generic adapter, not anything of
``dfwb_torch_srm``'s own.

Skipped unless ``dfwb_torch_srm`` (and so torch) is importable. It is never installed into this
project's own environment: run this file with an overlay that adds it for the invocation alone,
e.g. ``uv run --with-editable <path to the srm package> pytest tests/integration/test_srm_stem.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("dfwb_torch_srm")

import torch
from dfwb_torch_srm.modules import SRMConv2d
from torch import nn

from dfwb.models.source import load_run

DFWB = Path(sys.executable).parent / "dfwb"

VIDEOS = 40
SEED = 0
PROFILE = "toy-64-center-8f"

_CONFIG = """\
schema: dfwb.train/1
extends: ["dfwb://templates/toy-cpu.yaml"]
model:
  stem: {name: srm, bank: srm30, mode: gray}
train:
  max_epochs: 1
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


def test_the_srm_stem_trains_and_is_really_srm(tmp_path):
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

    built = _run(tmp_path, env, "inventory", "build", "toyfake", "--json")
    assert built.returncode == 0, built.stderr

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

    # only 9 training clips are left once the built-in pack's video list is joined against this
    # smaller tree (see test_toyfake_train.py), so the template's own batch size (16) is too big.
    trained = _run(
        tmp_path,
        env,
        "train",
        "-c",
        str(config),
        "data.loader.batch_size=4",
        "--device",
        "cpu",
        "--json",
    )
    assert trained.returncode == 0, trained.stderr
    (result,) = json.loads(trained.stdout)
    run_dir = Path(result["run_dir"])
    assert result["metrics"]["epochs"] == 1

    # the recorded resolved config names the stem
    detector_json = json.loads((run_dir / "checkpoints/best/detector.json").read_text("utf-8"))
    assert detector_json["model"]["stem"] == {"name": "srm", "bank": "srm30", "mode": "gray"}

    # and the built detector's stem really is SRM, adapted back to the backbone's 3 channels
    detector = load_run(f"{run_dir}#best")
    assert isinstance(detector.stem, nn.Sequential)
    srm_layer, adapter = detector.stem
    assert isinstance(srm_layer, SRMConv2d)
    assert srm_layer.bank == "srm30"
    assert srm_layer.mode == "gray"
    assert srm_layer.out_channels == 30  # one channel per srm30 kernel, in gray mode
    assert isinstance(adapter, nn.Conv2d)
    assert (adapter.in_channels, adapter.out_channels) == (30, 3)  # back to the backbone's input

    x = torch.rand(2, 3, 64, 64)
    y = detector.stem(x)
    assert y.shape == (2, 3, 64, 64)
