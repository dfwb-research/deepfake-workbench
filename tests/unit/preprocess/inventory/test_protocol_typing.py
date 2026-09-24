"""The ``InventoryBuilder`` contract accepts every reasonable builder style, statically and at run
time: a ``BaseBuilder`` subclass, attributes set in ``__init__``, and plain class attributes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.unit.preprocess.inventory import _protocol_typing as styles

from dfwb.preprocess.inventory.base import InventoryBuilder

REPO = Path(__file__).resolve().parents[4]
MODULE = Path(styles.__file__).resolve().relative_to(REPO)


def test_every_builder_style_type_checks_as_the_contract():
    done = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml", str(MODULE)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_every_builder_style_passes_the_runtime_check():
    assert all(isinstance(builder, InventoryBuilder) for builder in styles.builders())
    assert not isinstance(styles.not_a_builder(), InventoryBuilder)
