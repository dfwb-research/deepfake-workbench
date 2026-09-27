"""Loading third-party weights never runs arbitrary pickled code, and vendored code stays verbatim.

A pickled checkpoint can execute anything when it is unpickled, so every ``torch.load`` in the
framework must pass ``weights_only=True`` (plain tensors and containers only). This scans the
source tree's syntax, not its text, so a docstring that merely mentions ``torch.load`` never
counts, and an aliased ``from torch import load`` is caught as well.

Upstream code vendored under ``dfwb/zoo/_vendor/<name>/`` must carry its licence, a notice and a
hash list proving every file is still the upstream original. None ships today, so the second
check passes with nothing to check until the first vendored adapter arrives -- and then fails the
moment one arrives without its paperwork.
"""

from __future__ import annotations

import ast
from pathlib import Path

import dfwb

SOURCE = Path(dfwb.__file__).parent
VENDOR = SOURCE / "zoo" / "_vendor"


def _torch_loads(source: str, name: str) -> tuple[int, list[str]]:
    """``(how many torch.load calls source makes, the ones without weights_only=True)``."""
    calls = 0
    problems: list[str] = []
    tree = ast.parse(source, filename=name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "torch":
            problems.extend(
                f"{name}:{node.lineno}: imports torch.load by name (call torch.load instead)"
                for alias in node.names
                if alias.name == "load"
            )
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            continue
        calls += 1
        safe = any(
            keyword.arg == "weights_only"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
        if not safe:
            problems.append(f"{name}:{node.lineno}: torch.load without weights_only=True")
    return calls, problems


def _unsafe_torch_loads(source: str, name: str) -> list[str]:
    return _torch_loads(source, name)[1]


def test_the_scan_flags_an_unsafe_torch_load():
    """A guard for the guard: the scan below would pass vacuously if it never matched anything."""
    unsafe = "import torch\nstate = torch.load(path)\n"
    explicit_false = "import torch\nstate = torch.load(path, weights_only=False)\n"
    aliased = "from torch import load\n"
    safe = "import torch\nstate = torch.load(path, weights_only=True, map_location='cpu')\n"

    assert _unsafe_torch_loads(unsafe, "x.py") == ["x.py:2: torch.load without weights_only=True"]
    assert _unsafe_torch_loads(explicit_false, "x.py")
    assert _unsafe_torch_loads(aliased, "x.py")
    assert _unsafe_torch_loads(safe, "x.py") == []


def test_every_torch_load_in_the_source_passes_weights_only_true():
    problems: list[str] = []
    calls = 0
    for path in sorted(SOURCE.rglob("*.py")):
        found, unsafe = _torch_loads(path.read_text("utf-8"), str(path.relative_to(SOURCE)))
        calls += found
        problems.extend(unsafe)
    assert problems == []
    assert calls > 0  # the zoo weight manager's own call is found, so the scan is not vacuous


def test_every_vendored_package_has_a_valid_layout():
    from dfwb.zoo.strategies import check_vendored_layout

    vendored = sorted(p for p in VENDOR.iterdir() if p.is_dir()) if VENDOR.is_dir() else []
    problems = {
        vendor_dir.name: check_vendored_layout(vendor_dir)
        for vendor_dir in vendored
        if vendor_dir.name != "__pycache__"
    }
    assert {name: found for name, found in problems.items() if found} == {}
