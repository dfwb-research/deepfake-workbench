from pathlib import Path

import pytest

EXPERIMENT = """\
schema: dfwb.train/1
extends: [dfwb://templates/binary-frame.yaml]
run: {name: vit-b16-ffpp-c23, seeds: [42], output_root: "${env:DFWB_RUNS_ROOT,./runs}"}
data:
  processing: face-256-1.3x-32f
  train: [{protocol: ffpp/official, split: train, where: {compression: c23}}]
  val:   [{protocol: ffpp/official, split: val,   where: {compression: c23}}]
  test:  {suite: cross-dataset-v1}
model:
  backbone: {name: timm, model: vit_base_patch16_224.augreg_in21k, pretrained: true,
             freeze: {mode: partial, trainable_blocks: 2}}
"""


@pytest.fixture
def write(tmp_path):
    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    return _write


@pytest.fixture
def experiment(write):
    return write("exp.yaml", EXPERIMENT)
