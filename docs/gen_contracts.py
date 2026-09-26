"""Generate ``reference/contracts/``: the C1-C5 JSON Schemas.

Run by the ``gen-files`` MkDocs plugin at build time; nothing here is committed. Reuses
:func:`dfwb.cli.schema.build_schema` -- the same function ``dfwb schema export`` and
``tests/contract/test_schema_export.py`` use -- so there is exactly one definition of each
contract's schema, and this page can never drift from what ``dfwb schema export`` actually prints.
"""

from __future__ import annotations

import json

import mkdocs_gen_files

from dfwb.cli.schema import TITLES, build_schema

_CONTRACTS = ("c1", "c2", "c3", "c4", "c5")


def _page(contract: str) -> str:
    document = build_schema(contract)
    version = document["x-dfwb-contract"]["version"]
    body = json.dumps(document, indent=2, sort_keys=True)
    return (
        f"# {contract.upper()}: {TITLES[contract]}\n\n"
        f"Schema version: `{version}`.\n\n"
        "```json\n"
        f"{body}\n"
        "```\n"
    )


def _index() -> str:
    lines = [
        "# Contracts",
        "",
        "The five JSON Schemas dfwb's file formats and plugin API commit to, generated with "
        "`dfwb schema export` (also printed by `dfwb schema export <c1..c5>` and checked "
        "byte-for-byte against these in `tests/contract/`).",
        "",
    ]
    for contract in _CONTRACTS:
        lines.append(f"- [{contract.upper()}: {TITLES[contract]}]({contract}.md)")
    lines.append("")
    return "\n".join(lines)


def generate() -> None:
    for contract in _CONTRACTS:
        with mkdocs_gen_files.open(f"reference/contracts/{contract}.md", "w") as handle:
            handle.write(_page(contract))
    with mkdocs_gen_files.open("reference/contracts/index.md", "w") as handle:
        handle.write(_index())


generate()
