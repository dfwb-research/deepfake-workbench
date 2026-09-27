"""The example settings in the README and ``.env.example`` pass the tool's own checks.

A user who copies them verbatim must not be refused: every write root (work, runs, cache)
sits outside every datasets root, because raw data is read-only.
"""

import os
import re
from pathlib import Path

import pytest

from dfwb.protocols._rawdata import check_outside_datasets_roots

ROOT = Path(__file__).resolve().parents[2]
_SETTING = re.compile(r"^#?\s*(DFWB_[A-Z0-9_]+)=(\S+)\s*$")
_WRITE_ROOTS = ("DFWB_WORK_ROOT", "DFWB_RUNS_ROOT", "DFWB_CACHE_ROOT")


def _settings(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = _SETTING.match(line.strip())
        if match:
            found.setdefault(match.group(1), match.group(2))
    return found


@pytest.mark.parametrize("name", ["README.md", ".env.example"])
def test_example_write_roots_are_outside_the_example_datasets_roots(name: str) -> None:
    settings = _settings((ROOT / name).read_text(encoding="utf-8"))
    datasets = [Path(p) for p in settings["DFWB_DATASETS_ROOT"].split(os.pathsep) if p]
    write_roots = [key for key in _WRITE_ROOTS if key in settings]
    assert write_roots, f"{name} shows no write root"
    for key in write_roots:
        check_outside_datasets_roots(Path(settings[key]), datasets, what=key)
