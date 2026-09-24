"""The framework never imports utility (dfwb_torch_*) or protocol (dfwb_protocols) packages.

import-linter cannot express a prefix wildcard, so this scans the source instead.
"""

import ast
from pathlib import Path

import dfwb

SOURCE = Path(dfwb.__file__).parent


def _imported_names(tree: ast.AST) -> list[tuple[int, str]]:
    names: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append((node.lineno, node.module))
    return names


def test_no_imports_of_downstream_packages():
    offenders = []
    for path in sorted(SOURCE.rglob("*.py")):
        for lineno, name in _imported_names(ast.parse(path.read_text("utf-8"))):
            top = name.partition(".")[0]
            if top.startswith("dfwb_torch_") or top == "dfwb_protocols":
                offenders.append(f"{path.relative_to(SOURCE)}:{lineno} imports {name}")
    assert offenders == []
