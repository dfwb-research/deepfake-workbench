"""Generate ``reference/api/``: mkdocstrings pages for each layer's public API.

Run by the ``gen-files`` MkDocs plugin at build time; nothing here is committed. "Public" here
means: a module that is not private (no path segment starting with ``_``) and defines ``__all__``.
A module with no ``__all__`` at all -- a CLI
subcommand module, or a bare re-export ``__init__.py`` -- names nothing of its own to document and
is skipped; :mod:`dfwb.cli.main` is the one CLI module that does define one.

Two groups of modules are left out on top of that general rule, both named explicitly below
(``_EXCLUDED_PREFIXES``, ``_EXCLUDED_EXACT``):

- the per-dataset inventory builder modules (``dfwb.preprocess.inventory.builders.*``, one class
  each, ~20 of them), even though each defines a one-name ``__all__``. Including them would
  dominate the ``preprocess`` page with near-identical entries; each one is already documented
  from its own data by the generated ``datasets/*.md`` pages, and the base class they all share
  (``BaseBuilder``) is documented once, in full, right here.
- ``dfwb.core.config.schema``, left out of the ``core`` page: its models are the experiment
  config, documented in full (every section, not only ``__all__``) by the dedicated
  ``reference/config.md`` (``docs/gen_config.py``) instead. Rendering it on both pages gave every
  shared class two "primary" URLs, which ``mkdocs-autorefs`` rightly refuses to pick between.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mkdocs_gen_files

SRC = Path(__file__).resolve().parent.parent / "src" / "dfwb"

# Layer -> one-line summary, in the order the framework's own layering puts them (cli at the top,
# core at the bottom); matches the layers named in pyproject.toml's import-linter contract.
_LAYERS: dict[str, str] = {
    "cli": "The `dfwb` command-line interface.",
    "train": "Losses, optimisers, schedules, the Lightning module and run directories.",
    "score": "Runs a Detector over a protocol split and writes a score file.",
    "zoo": "Adapter contract, weight fetching and verification, adapter cards.",
    "models": "Backbones, temporal pools, heads and detector assembly.",
    "data": "Torch datasets that join protocol splits with a processed store.",
    "eval": "Metrics, aggregation, uncertainty, suites and reports over score files.",
    "preprocess": "Inventory builders and the face pipeline.",
    "protocols": "Protocol packs: loading, querying, verification, materialising, building.",
    "core": "Contracts, registries, plugins, config, paths and run metadata.",
}

# The module docstring says why each group is left out. Each key is a layer name; the values are
# relative to that layer (``dfwb.<layer>.`` already stripped), matching how ``_layer_modules``
# compares them.
_EXCLUDED_PREFIXES: dict[str, tuple[str, ...]] = {
    "preprocess": ("inventory.builders.",),
}
_EXCLUDED_EXACT: dict[str, frozenset[str]] = {
    "core": frozenset({"config.schema"}),
}


def _all_of(path: Path) -> list[str] | None:
    """The literal ``__all__`` list a module defines at its top level, or ``None``."""
    tree = ast.parse(path.read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return None
            return list(value) if isinstance(value, list) else None
    return None


def _is_private(relative_parts: tuple[str, ...]) -> bool:
    return any(part.startswith("_") for part in relative_parts)


def _layer_modules(layer: str) -> list[tuple[str, list[str]]]:
    """``(dotted module name, __all__)`` for every public, documented module of ``layer``."""
    found: list[tuple[str, list[str]]] = []
    for path in sorted((SRC / layer).rglob("*.py")):
        relative = path.relative_to(SRC / layer).with_suffix("")
        parts = relative.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts or _is_private(parts):
            continue
        dotted = ".".join(("dfwb", layer, *parts))
        relative = dotted.removeprefix(f"dfwb.{layer}.")
        if relative.startswith(_EXCLUDED_PREFIXES.get(layer, ())):
            continue
        if relative in _EXCLUDED_EXACT.get(layer, frozenset()):
            continue
        names = _all_of(path)
        if names:
            found.append((dotted, names))
    return found


def _page(layer: str, summary: str, modules: list[tuple[str, list[str]]]) -> str:
    lines = [f"# `dfwb.{layer}`", "", summary, ""]
    for dotted, _names in modules:
        lines.append(f"::: {dotted}")
        lines.append("    options:")
        lines.append("      heading_level: 2")
        lines.append("")
    return "\n".join(lines)


def _index(rows: list[tuple[str, str, int]]) -> str:
    lines = [
        "# API reference",
        "",
        "One page per layer, holding every non-private module of that layer which declares "
        '`__all__` -- dfwb\'s own definition of "public": those exact names, nothing more.',
        "",
        "| Layer | Modules documented |",
        "|---|---|",
    ]
    for layer, _summary, count in rows:
        lines.append(f"| [`dfwb.{layer}`]({layer}.md) | {count} |")
    lines.append("")
    return "\n".join(lines)


def generate() -> None:
    rows: list[tuple[str, str, int]] = []
    for layer, summary in _LAYERS.items():
        modules = _layer_modules(layer)
        rows.append((layer, summary, len(modules)))
        with mkdocs_gen_files.open(f"reference/api/{layer}.md", "w") as handle:
            handle.write(_page(layer, summary, modules))
    with mkdocs_gen_files.open("reference/api/index.md", "w") as handle:
        handle.write(_index(rows))


generate()
