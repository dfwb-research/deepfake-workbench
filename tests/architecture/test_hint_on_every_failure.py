"""Every in-process ``dfwb`` invocation in the test suite goes through one shared helper.

``tests/_dfwb_cli.py`` wraps :func:`dfwb.cli.main.main` and checks, for every non-zero exit, that
a non-empty ``hint: `` line was printed on stderr -- the property this framework promises for
every failing command. A test that reaches ``dfwb.cli.main.main`` for itself, by any static route,
could call it without that check ever running, silently reintroducing a command that fails with no
hint. This scans every git-tracked test file's own syntax tree (not its text, so a docstring or a
string of Python source handed to a subprocess -- as ``tests/unit/cli/test_main.py`` does -- cannot
trip it) for any way of reaching it, and requires that to be absent everywhere but the helper
itself.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The one file allowed to reach `dfwb.cli.main.main`, since it *is* the shared helper: every other
# test file must call through it instead.
_HELPER = "tests/_dfwb_cli.py"

_TARGET = "dfwb.cli.main.main"


def _naive_import_only_check(tree: ast.Module) -> list[int]:
    """The guard's first version (fix round 0): flags an import that names ``main`` directly, but
    never looks at how an imported module or alias is later *called*. Kept only so the two tests
    below can document, and pin, the exact gap a review found in it -- a future simplification of
    the real check (:func:`_reaches_main_directly`) can't silently reopen that gap without one of
    those tests failing first."""
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


def _literal_dotted(node: ast.expr) -> str | None:
    """The dotted name exactly as written, e.g. ``"a.b.c"`` for the attribute chain ``a.b.c``;
    ``None`` if ``node`` is not a plain chain of names and attributes (a call result, a subscript,
    a string, ...), since nothing here can call ``.main`` off those without another import first."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _literal_dotted(node.value)
        return f"{base}.{node.attr}" if base is not None else None
    return None


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    """Map every name ``tree`` binds through ``import``/``from ... import ...`` to the fully
    dotted path it refers to.

    This walks the whole tree, not just module-level statements, so a lazy import inside a
    fixture or a test body counts too; it is a conservative, whole-file approximation rather than
    real scope analysis, which this guard does not need to be exact about. Reassigning a name
    after import (``m = main_module.main``) or reaching a module through
    ``importlib.import_module(...)`` is out of scope: nothing in this repository does either
    today, and both would need real alias/points-to analysis to track, not a static scan.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    bindings[alias.asname] = alias.name
                else:
                    # `import a.b.c` (no `as`) binds only the top-level name `a`, to itself --
                    # `a.b.c...` still starts from that same name.
                    top = alias.name.split(".", 1)[0]
                    bindings[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                local = alias.asname or alias.name
                bindings[local] = f"{node.module}.{alias.name}"
    return bindings


def _resolve(dotted: str, bindings: dict[str, str]) -> str:
    """``dotted`` with its leftmost component replaced by what it is bound to, if anything."""
    head, _, rest = dotted.partition(".")
    resolved_head = bindings.get(head, head)
    return f"{resolved_head}.{rest}" if rest else resolved_head


def _reaches_main_directly(tree: ast.Module) -> list[int]:
    """Line numbers of anything in ``tree`` that could call ``dfwb.cli.main.main`` directly.

    Two kinds of line are flagged:

    - A binding of a local name straight to the callable (``from dfwb.cli.main import main``,
      however aliased) -- flagged at the import, whether or not it is ever called.
    - A call whose target, once its base name is resolved through this file's own import bindings,
      is ``dfwb.cli.main.main`` -- flagged at the call. This is what catches
      ``from dfwb.cli import main as main_module`` (or with no alias at all) followed by
      ``main_module.main(...)``/``main.main(...)``, and ``import dfwb.cli`` (or
      ``import dfwb.cli.main``) followed by ``dfwb.cli.main.main(...)``: none of these import the
      callable by name, so only looking at calls catches them.
    """
    bindings = _import_bindings(tree)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                if f"{node.module}.{alias.name}" == _TARGET:
                    lines.append(node.lineno)
        elif isinstance(node, ast.Call):
            dotted = _literal_dotted(node.func)
            if dotted is not None and _resolve(dotted, bindings) == _TARGET:
                lines.append(node.lineno)
    return lines


def _test_files() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files", "tests"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return [REPO / line for line in listing.splitlines() if line.endswith(".py")]


def test_no_test_reaches_main_directly_except_the_shared_helper():
    offenders = []
    for path in _test_files():
        rel = str(path.relative_to(REPO))
        if rel == _HELPER:
            continue
        tree = ast.parse(path.read_text("utf-8"), filename=rel)
        offenders.extend(f"{rel}:{lineno}" for lineno in _reaches_main_directly(tree))
    assert offenders == []


def test_the_check_catches_a_direct_import_of_the_callable():
    assert _reaches_main_directly(ast.parse("from dfwb.cli.main import main\n")) == [1]
    assert _reaches_main_directly(ast.parse("from dfwb.cli.main import main as m\n")) == [1]
    # Not imported by name here, but calling `.main` off it two lines down is still a direct
    # reach, and is caught below alongside the other call-based shapes.
    assert _reaches_main_directly(ast.parse("import dfwb.cli.main\n")) == []


def test_the_check_catches_every_call_based_bypass_shape():
    # Each of these was reproduced against `_naive_import_only_check` (fix round 0's logic) and
    # found unflagged; every one must be caught here.
    shapes = [
        "from dfwb.cli import main\nmain.main([])\n",
        "from dfwb.cli import main as main_module\nmain_module.main([])\n",
        "import dfwb.cli\ndfwb.cli.main.main([])\n",
        "import dfwb.cli.main\ndfwb.cli.main.main([])\n",
        "from dfwb.cli.main import main\nmain([])\n",
        "from dfwb.cli.main import main as m\nm([])\n",
    ]
    for source in shapes:
        assert _reaches_main_directly(ast.parse(source)), source


def test_the_check_spares_legitimate_neighbours():
    # Importing the *module* under an alias (to reach e.g. `main_module.LazyGroup`), and actually
    # using it that way -- `tests/unit/cli/test_main.py` does exactly this -- must stay allowed,
    # even though the same import is what makes `main_module.main(...)` resolvable above.
    lazygroup_use = ast.parse(
        "from dfwb.cli import main as main_module\n"
        "group = main_module.LazyGroup(lazy_subcommands={})\n"
        "main_module.cli.lazy_subcommands['x'] = ('m:f', 'help')\n"
    )
    assert _reaches_main_directly(lazygroup_use) == []

    # A string that merely contains the phrase, meant for a subprocess: not a real import or call
    # in this file's own syntax tree.
    embedded_string = ast.parse("code = 'from dfwb.cli.main import main; main([])'\n")
    assert _reaches_main_directly(embedded_string) == []

    # Importing anything else from the same module, and calling *that*, is unrelated.
    other_name = ast.parse("from dfwb.cli.main import cli\ncli.main(['x'])\n")
    assert _reaches_main_directly(other_name) == []

    # A plain `import dfwb.cli.main` that never calls `.main` off it (e.g. only reads `COMMANDS`)
    # is not itself a violation -- only actually calling it is.
    unused_import = ast.parse("import dfwb.cli.main\nprint(dfwb.cli.main.COMMANDS)\n")
    assert _reaches_main_directly(unused_import) == []


# Fix round 0's gap, pinned so it cannot silently regress: `_naive_import_only_check` really did
# miss every call-based bypass shape, which is exactly why `_reaches_main_directly` exists.
_BYPASS_SHAPES_THE_NAIVE_CHECK_MISSED = [
    "from dfwb.cli import main\nmain.main([])\n",
    "from dfwb.cli import main as main_module\nmain_module.main([])\n",
    "import dfwb.cli\ndfwb.cli.main.main([])\n",
]


def test_the_naive_import_only_check_missed_every_call_based_bypass():
    for source in _BYPASS_SHAPES_THE_NAIVE_CHECK_MISSED:
        assert _naive_import_only_check(ast.parse(source)) == []
