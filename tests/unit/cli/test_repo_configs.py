"""The runnable configs shipped at the repo root (``configs/``), the way a fresh clone sees them.

Every one of them extends a shipped template and passes ``dfwb config validate`` -- which checks
component params against the installed plugins but never touches a protocol pack or processed
store (``docs/quickstart.md``, ``dfwb config validate --help``) -- so no dataset, protocol pack or
real data is needed here, only the ``train`` extra for the backbones these configs use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO / "configs"
CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))

# Configs whose header must say they need data obtained from outside dfwb.
NEEDS_REAL_DATA = {
    "ffpp-c23-vit-b16.yaml",
    "celebdf-v2-vit-b16.yaml",
    "cross-dataset-ffpp-celebdf.yaml",
}


def test_the_expected_configs_are_shipped():
    assert {p.name for p in CONFIGS} == {
        "toyfake-cpu.yaml",
        "ffpp-c23-vit-b16.yaml",
        "celebdf-v2-vit-b16.yaml",
        "cross-dataset-ffpp-celebdf.yaml",
    }


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_extends_a_shipped_template_and_has_a_header_comment(path):
    text = path.read_text("utf-8")
    assert text.splitlines()[0].startswith("# "), "the header comment says what the config runs"
    assert "extends: [dfwb://templates/" in text


@pytest.mark.parametrize(
    "path", [p for p in CONFIGS if p.name in NEEDS_REAL_DATA], ids=lambda p: p.name
)
def test_configs_needing_real_data_say_so_in_their_header(path):
    lines = path.read_text("utf-8").splitlines()
    header = "\n".join(line for line in lines if line.startswith("#"))
    assert "needs" in header.lower()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_passes_dfwb_config_validate(path, run, requires_torch):
    result = run("config", "validate", "-c", str(path))
    assert result.code == 0, result.err
