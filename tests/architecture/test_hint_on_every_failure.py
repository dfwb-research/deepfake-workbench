"""Every in-process ``dfwb`` invocation in the test suite goes through one shared helper.

``tests/_dfwb_cli.py`` wraps :func:`dfwb.cli.main.main` and checks, for every non-zero exit, that
a non-empty ``hint: `` line was printed on stderr -- the property this framework promises for
every failing command. A test that imports ``main`` for itself could call it without that check
ever running, silently reintroducing a command that fails with no hint. This scans every
git-tracked test file's own syntax tree (not its text, so a docstring or a string of Python source
handed to a subprocess -- as ``tests/unit/cli/test_main.py`` does -- cannot trip it) for an import
that would let it do that, and requires it to be absent everywhere but the helper itself.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The one file allowed to import `dfwb.cli.main.main`, since it *is* the shared helper: every
# other test file must call through it instead.
_HELPER = "tests/_dfwb_cli.py"


def _imports_main_callable(tree: ast.Module) -> list[int]:
    """Line numbers of anything in ``tree`` that could call ``dfwb.cli.main.main`` directly:
    importing the callable by name (``from dfwb.cli.main import main``, however aliased), or
    importing the module by its dotted path (``import dfwb.cli.main``, which makes
    ``dfwb.cli.main.main(...)`` reachable)."""
    lines = []
    for node in ast.walk(tree):
        imports_the_callable = (
            isinstance(node, ast.ImportFrom)
            and node.module == "dfwb.cli.main"
            and any(alias.name == "main" for alias in node.names)
        )
        imports_the_module = isinstance(node, ast.Import) and any(
            alias.name == "dfwb.cli.main" for alias in node.names
        )
        if imports_the_callable or imports_the_module:
            lines.append(node.lineno)
    return lines


def _test_files() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files", "tests"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return [REPO / line for line in listing.splitlines() if line.endswith(".py")]


def test_no_test_imports_main_directly_except_the_shared_helper():
    offenders = []
    for path in _test_files():
        rel = str(path.relative_to(REPO))
        if rel == _HELPER:
            continue
        tree = ast.parse(path.read_text("utf-8"), filename=rel)
        offenders.extend(f"{rel}:{lineno}" for lineno in _imports_main_callable(tree))
    assert offenders == []


def test_the_check_catches_a_direct_import_of_the_callable():
    assert _imports_main_callable(ast.parse("from dfwb.cli.main import main\n")) == [1]
    assert _imports_main_callable(ast.parse("from dfwb.cli.main import main as m\n")) == [1]
    assert _imports_main_callable(ast.parse("import dfwb.cli.main\n")) == [1]


def test_the_check_spares_legitimate_neighbours():
    # Importing the *module* under its own name (to reach e.g. `main_module.LazyGroup`), not the
    # `main` callable -- `tests/unit/cli/test_main.py` does exactly this.
    module_import = ast.parse("from dfwb.cli import main as main_module\n")
    assert _imports_main_callable(module_import) == []

    # A string that merely contains the phrase, meant for a subprocess: not a real import in this
    # file's own syntax tree.
    embedded_string = ast.parse("code = 'from dfwb.cli.main import main; main([])'\n")
    assert _imports_main_callable(embedded_string) == []

    # Importing anything else from the same module is unrelated.
    other_name = ast.parse("from dfwb.cli.main import cli\n")
    assert _imports_main_callable(other_name) == []
