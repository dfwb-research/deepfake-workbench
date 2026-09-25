"""``dfwb datasets``: the datasets dfwb knows, where they are on this machine, and their layout."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from dfwb.cli._output import emit_json, json_option, table

if TYPE_CHECKING:
    from dfwb.core.paths import ResolvedRoot, RootName
    from dfwb.preprocess.inventory.runner import DatasetCopy, FolderStatus

_MISSING = "—"


@click.group()
def datasets() -> None:
    """List supported datasets and show how each one is laid out."""


def _locate_context() -> tuple[Mapping[RootName, ResolvedRoot], dict[str, tuple[Path, str]]]:
    """The roots and dataset overrides; a broken config gives none (with a warning)."""
    from dfwb.core.errors import ConfigError
    from dfwb.core.paths import dataset_overrides, resolve_roots

    try:
        return resolve_roots(), dataset_overrides()
    except ConfigError as exc:
        click.echo(f"warning: cannot read the roots; no folder is located: {exc.message}", err=True)
        return {}, {}


def _pack_names() -> dict[str, list[str]]:
    """Dataset id -> the healthy installed packs that publish it. Never raises."""
    from dfwb.core.errors import DFWBError
    from dfwb.protocols.packs import installed_packs

    try:
        packs = installed_packs()
    except DFWBError as exc:
        click.echo(f"warning: cannot list protocol packs: {exc.message}", err=True)
        return {}
    found: dict[str, list[str]] = {}
    for pack in packs:
        for dataset_id in pack.card.datasets if pack.card is not None else ():
            found.setdefault(dataset_id, []).append(pack.name)
    return {dataset_id: sorted(names) for dataset_id, names in found.items()}


def _location_json(status: FolderStatus) -> dict[str, Any]:
    location = status.location
    return {
        "path": None if location is None else str(location.path),
        "source": status.source,
        "also_found": [] if location is None else [str(p) for p in location.also_found],
    }


def _copies_json(status: FolderStatus, copies: Sequence[DatasetCopy]) -> dict[str, Any]:
    """Every copy found, each naming which task x compression directories hold a video."""
    return {
        "source": status.source,
        "problem": status.problem,
        "copies": [
            {"path": str(copy.path), "source": copy.source, "video_dirs": list(copy.video_dirs)}
            for copy in copies
        ],
    }


@datasets.command("list")
@json_option
def list_(as_json: bool) -> None:
    """List every dataset with a builder or a protocol pack, and where its folder is.

    Builder details come from registry metadata alone, so no builder is imported; a folder that
    is not found shows as a dash.
    """
    from dfwb.core.plugins import get_registry
    from dfwb.preprocess.inventory.runner import folder_status

    roots, overrides = _locate_context()
    packs = _pack_names()
    entries = get_registry("inventory_builders").entries()
    repeated = {key for key, n in Counter(e.key for e in entries).items() if n > 1}

    rows: list[dict[str, Any]] = []
    for entry in entries:
        folder = entry.meta.get("folder")
        folder = folder if isinstance(folder, str) and folder else None
        status = folder_status(entry.key, folder, roots, overrides)
        rows.append(
            {
                "id": entry.qualified_key if entry.key in repeated else entry.key,
                "name": entry.summary,
                "folder": folder,
                **_location_json(status),
                "packs": packs.get(entry.key, []),
            }
        )
    for dataset_id in sorted(set(packs) - {e.key for e in entries}):
        status = folder_status(dataset_id, None, roots, overrides)  # only an override can find it
        rows.append(
            {
                "id": dataset_id,
                "name": None,
                "folder": None,
                **_location_json(status),
                "packs": packs[dataset_id],
            }
        )
    rows.sort(key=lambda row: str(row["id"]))

    if as_json:
        emit_json(rows)
        return
    if not rows:
        click.echo("no datasets: no inventory builders or protocol packs are installed")
        return
    click.echo(
        table(
            ["ID", "NAME", "FOLDER", "FOUND AT", "PACKS"],
            [
                [
                    row["id"],
                    row["name"] or _MISSING,
                    row["folder"] or _MISSING,
                    row["path"] or _MISSING,
                    ", ".join(row["packs"]) or _MISSING,
                ]
                for row in rows
            ],
        )
    )


def _pack_schemes(dataset_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The schemes installed packs publish for ``dataset_id``, and any problems reading them."""
    from dfwb.core.errors import DFWBError
    from dfwb.protocols._yaml import read_card
    from dfwb.protocols.packs import installed_packs

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    try:
        packs = installed_packs()
    except DFWBError as exc:
        return rows, [f"cannot list protocol packs: {exc.message}"]
    for pack in packs:
        if pack.card is None or dataset_id not in pack.card.datasets:
            continue
        try:
            card = read_card(pack.dataset_dir(dataset_id))
        except DFWBError as exc:
            problems.append(f"pack {pack.name!r}: {exc.message}")
            continue
        rows.extend(
            {
                "pack": pack.name,
                "scheme": name,
                "kind": scheme.kind,
                "default": name == card.default_scheme,
            }
            for name, scheme in card.schemes.items()
        )
    return rows, problems


def _card_lines(card: Mapping[str, Any]) -> list[str]:
    """The descriptive card fields as ``label: value`` lines, skipping any that are unset."""
    lines = []
    license_info = card.get("license")
    if isinstance(license_info, Mapping):
        spdx = license_info.get("spdx")
        license_text = str(license_info.get("summary", "")) + (f" ({spdx})" if spdx else "")
    else:
        license_text = str(license_info or "")
    paper = card.get("paper")
    paper_text = str(paper.get("title", "")) if isinstance(paper, Mapping) else str(paper or "")
    fields = [
        ("aliases", ", ".join(map(str, card.get("aliases") or []))),
        ("homepage", str(card.get("homepage") or "")),
        ("paper", paper_text),
        ("license", license_text),
        ("access", str(card.get("access") or "")),
        ("modalities", ", ".join(map(str, card.get("modalities") or []))),
        ("compressions", ", ".join(map(str, card.get("compressions") or []))),
        ("keys", str(card.get("key_rule") or "")),
    ]
    for label, value in fields:
        if value:
            lines.append(f"{label + ':':<14}{value}")
    return lines


@datasets.command("info")
@click.argument("dataset")
@json_option
def info(dataset: str, as_json: bool) -> None:
    """Show DATASET's details, its expected layout, its local folder and its schemes."""
    from dfwb.core.registry import catalogue_requirement
    from dfwb.preprocess.inventory.runner import dataset_copies, folder_status, get_builder

    builder = get_builder(dataset)
    dataset_id = builder.dataset_id
    roots, overrides = _locate_context()
    status = folder_status(dataset_id, builder.expected_folder, roots, overrides)
    copies = () if status.location is None else dataset_copies(builder, status.location, roots)
    schemes, problems = _pack_schemes(dataset_id)
    card = dict(builder.card_info)

    if as_json:
        emit_json(
            {
                "id": dataset_id,
                "builder": {"id": dataset_id, "version": builder.version},
                "card": card,
                "layout": builder.describe_layout(),
                "location": _copies_json(status, copies),
                "schemes": schemes,
                "problems": problems,
            }
        )
        return

    release = card.get("release")
    title = f"{card.get('name', dataset_id)} ({dataset_id})"
    click.echo(title + (f"  release {release}" if release else ""))
    for line in _card_lines(card):
        click.echo(line)
    click.echo("")
    click.echo(builder.describe_layout())
    click.echo("")
    if not copies:
        click.echo(f"folder: not found ({status.problem})")
    else:
        for copy in copies:
            videos = ", ".join(copy.video_dirs) if copy.video_dirs else "no videos found"
            click.echo(f"{copy.source}: {copy.path}")
            click.echo(f"  videos: {videos}")
    click.echo("")
    if schemes:
        click.echo("schemes in installed protocol packs:")
        rows = [
            [row["pack"], row["scheme"] + ("*" if row["default"] else ""), row["kind"]]
            for row in schemes
        ]
        click.echo(table(["PACK", "SCHEME", "KIND"], rows))
    else:
        requirement = catalogue_requirement("protocol_packs", dataset_id)
        suggestion = f" (pip install {requirement})" if requirement else ""
        click.echo(f"no installed protocol pack publishes {dataset_id}{suggestion}")
    for problem in problems:
        click.echo(f"warning: {problem}", err=True)
