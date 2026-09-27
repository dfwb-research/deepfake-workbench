"""The framework never imports a ``dfwb_torch*`` utility package: those are runtime plugins,
installed by users who want them, never a dependency of this package. Tests exercise the generic
plugin hooks (the stem registry, freeze modes, ...) with fake plugins instead.

Every module under ``src/dfwb`` is AST-scanned for ``import dfwb_torch...`` or
``from dfwb_torch... import ...``, so this holds even for an import buried inside a function body
that would not show up on a plain text grep of the module's top level.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "dfwb"


def _imported_top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.partition(".")[0])
    return names


def test_no_module_imports_a_dfwb_torch_package():
    offenders = {
        str(path.relative_to(REPO)): sorted(
            n for n in _imported_top_level_names(path) if n.startswith("dfwb_torch")
        )
        for path in SRC.rglob("*.py")
    }
    offenders = {path: names for path, names in offenders.items() if names}
    assert offenders == {}


def test_the_check_catches_a_dfwb_torch_import(tmp_path):
    (tmp_path / "offender.py").write_text("from dfwb_torch_srm.modules import SRMConv2d\n")
    names = _imported_top_level_names(tmp_path / "offender.py")
    assert "dfwb_torch_srm" in names
