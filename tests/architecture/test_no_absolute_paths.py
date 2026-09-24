"""Shipped data (templates, and packs once they exist) never contains machine-specific paths."""

import re
from pathlib import Path

import dfwb

SOURCE = Path(dfwb.__file__).parent
MACHINE_PATH = re.compile(r"(?<![\w.])/(home|mnt|media|Users|scratch)/")


def test_shipped_data_has_no_absolute_paths():
    offenders = []
    for path in sorted(SOURCE.rglob("*")):
        if path.suffix in {".yaml", ".yml", ".json", ".toml", ".tsv", ".md", ".py"}:
            for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
                if MACHINE_PATH.search(line):
                    offenders.append(f"{path.relative_to(SOURCE)}:{lineno}: {line.strip()}")
    assert offenders == []


def test_the_check_catches_machine_paths():
    # built at runtime so the repository's own no-absolute-paths hook does not flag this file
    assert MACHINE_PATH.search("root: /" + "home/someone/data")
    assert MACHINE_PATH.search("x=/" + "mnt/storage")
    assert not MACHINE_PATH.search("https://example.org/" + "home/page")
    assert not MACHINE_PATH.search("dfwb://templates/binary-frame.yaml")
