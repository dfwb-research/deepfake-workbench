"""Generate ``datasets/*.md``: one page per dataset registered as an inventory builder.

Run by the ``gen-files`` MkDocs plugin at build time; nothing here is committed. Every field comes
from the builder's own ``card_info`` and ``describe_layout()`` -- both plain class-level data, so
this script never reads a datasets root and never imports torch (inventory builders are
torch-free; see ``tests/architecture/test_torch_free.py``).

Split schemes are listed only for a dataset a pack shipped *inside the framework itself* publishes
(``toyfake``, the only one). Every other dataset's schemes live in a separately-installed protocol
pack (``dfwb-protocols``), not in this repository, so this generator never claims to know them.
"""

from __future__ import annotations

import mkdocs_gen_files

from dfwb.core.plugins import get_registry
from dfwb.preprocess.inventory.base import BaseBuilder
from dfwb.protocols.packs import installed_packs

_NAV_PATH = "datasets/index.md"


def _license_line(license_info: object) -> str:
    if not isinstance(license_info, dict):
        return str(license_info or "unknown")
    spdx = license_info.get("spdx")
    summary = str(license_info.get("summary") or "")
    return f"{summary} ({spdx})" if spdx else summary


def _schemes_section(builder: BaseBuilder) -> str:
    """Scheme names published by a pack shipped *inside the framework*, or a pointer elsewhere."""
    for pack in installed_packs():
        is_builtin = pack.provider == "dfwb"
        if pack.card is not None and builder.dataset_id in pack.card.datasets and is_builtin:
            names = ", ".join(f"`{name}`" for name in sorted(builder.schemes))
            return (
                f"The `{pack.name}` pack, shipped with dfwb itself, publishes: {names} "
                f"(default `{builder.default_scheme}`)."
            )
    return (
        "Split schemes for this dataset are not shipped with dfwb itself; they live in a "
        "separately-installed protocol pack such as `dfwb-protocols`."
    )


def _page(builder: BaseBuilder) -> str:
    card = dict(builder.card_info)
    name = str(card.get("name", builder.dataset_id))
    lines = [f"# {name}", ""]
    release = card.get("release")
    if release:
        lines.append(f"*Release: {release}*")
        lines.append("")
    aliases = card.get("aliases") or []
    if aliases:
        lines.append(f"Also known as: {', '.join(map(str, aliases))}.")
        lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Dataset id | `{builder.dataset_id}` |")
    lines.append(f"| Folder | `{builder.expected_folder}` |")
    homepage = card.get("homepage")
    if homepage:
        lines.append(f"| Homepage | <{homepage}> |")
    paper = card.get("paper")
    if isinstance(paper, dict) and paper.get("title"):
        venue = f", {paper['venue']}" if paper.get("venue") else ""
        year = f" ({paper['year']})" if paper.get("year") else ""
        lines.append(f"| Paper | {paper['title']}{venue}{year} |")
    lines.append(f"| Licence | {_license_line(card.get('license'))} |")
    lines.append(f"| Access | {card.get('access', 'unknown')} |")
    lines.append(f"| Modalities | {', '.join(map(str, card.get('modalities') or []))} |")
    compressions = card.get("compressions")
    if compressions:
        lines.append(f"| Compressions | {', '.join(map(str, compressions))} |")
    lines.extend(["", "## Layout", "", "```text", builder.describe_layout(), "```", ""])
    lines.extend(["## Split schemes", "", _schemes_section(builder), ""])
    lines.extend(
        [
            "## Using it",
            "",
            "```bash",
            f"dfwb datasets info {builder.dataset_id}",
            f"dfwb inventory build {builder.dataset_id} --root /path/to/{builder.expected_folder}",
            "```",
            "",
            "See [Adding a dataset](../guides/add-a-dataset.md) to publish protocols for a dataset "
            "not listed here.",
            "",
        ]
    )
    return "\n".join(lines)


def _index(rows: list[tuple[str, str, str]]) -> str:
    lines = [
        "# Datasets",
        "",
        "One page per dataset dfwb has a built-in inventory builder for. dfwb never distributes "
        "media: you obtain each dataset from its own owner, under the owner's terms, and point "
        "`DFWB_DATASETS_ROOT` at where you kept it (`dfwb doctor` and `dfwb datasets list` show "
        "what is found).",
        "",
        "| Dataset | Folder |",
        "|---|---|",
    ]
    for dataset_id, name, expected_folder in rows:
        lines.append(f"| [{name}]({dataset_id}.md) | `{expected_folder}` |")
    lines.append("")
    return "\n".join(lines)


def generate() -> None:
    entries = get_registry("inventory_builders").entries()
    rows: list[tuple[str, str, str]] = []
    for entry in entries:
        builder = get_registry("inventory_builders").build(entry.qualified_key)
        assert isinstance(builder, BaseBuilder)
        name = str(builder.card_info.get("name", builder.dataset_id))
        rows.append((builder.dataset_id, name, builder.expected_folder))
        with mkdocs_gen_files.open(f"datasets/{builder.dataset_id}.md", "w") as handle:
            handle.write(_page(builder))
    rows.sort(key=lambda row: row[0])
    with mkdocs_gen_files.open(_NAV_PATH, "w") as handle:
        handle.write(_index(rows))


generate()
