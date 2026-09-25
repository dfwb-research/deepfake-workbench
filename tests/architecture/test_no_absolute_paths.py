"""Shipped data (templates, and packs once they exist) never contains machine-specific paths.

``SOURCE.rglob("*")`` already walks everything under the ``dfwb`` package -- including the
built-in protocol pack (``dfwb/_packs/**``) and the ``new-pack`` scaffolding templates
(``dfwb/protocols/_newpack/**``) -- so both are covered by suffix alone; ``.tmpl`` is in the
checked suffixes for exactly this reason, since the scaffolding templates would otherwise be
skipped.
"""

import re
from pathlib import Path

import dfwb

SOURCE = Path(dfwb.__file__).parent
MACHINE_PATH = re.compile(r"(?<![\w.])/(home|mnt|media|Users|scratch)/")
_CHECKED_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".tsv", ".md", ".py", ".tmpl"}


def test_shipped_data_has_no_absolute_paths():
    offenders = []
    for path in sorted(SOURCE.rglob("*")):
        if path.suffix in _CHECKED_SUFFIXES:
            for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
                if MACHINE_PATH.search(line):
                    offenders.append(f"{path.relative_to(SOURCE)}:{lineno}: {line.strip()}")
    assert offenders == []


def test_the_checked_suffixes_cover_the_pack_and_the_newpack_templates():
    # A regression guard for the check above: if the built-in pack or the new-pack templates ever
    # stopped existing under `SOURCE`, or their extensions stopped being checked, the absolute-path
    # scan would silently cover nothing for them.
    pack_files = list((SOURCE / "_packs").rglob("*"))
    assert any(p.is_file() and p.suffix in _CHECKED_SUFFIXES for p in pack_files)
    newpack_files = list((SOURCE / "protocols" / "_newpack").glob("*.tmpl"))
    assert newpack_files
    assert all(p.suffix in _CHECKED_SUFFIXES for p in newpack_files)


def test_the_check_catches_machine_paths():
    # built at runtime so the repository's own no-absolute-paths hook does not flag this file
    assert MACHINE_PATH.search("root: /" + "home/someone/data")
    assert MACHINE_PATH.search("x=/" + "mnt/storage")
    assert not MACHINE_PATH.search("https://example.org/" + "home/page")
    assert not MACHINE_PATH.search("dfwb://templates/binary-frame.yaml")
