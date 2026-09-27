"""``dfwb datasets``: the datasets dfwb knows, where they are on this machine, and their layout."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from dfwb.cli._output import distribution_text, emit_json, json_option, table

if TYPE_CHECKING:
    from dfwb.core.paths import ResolvedRoot, RootName
    from dfwb.preprocess.inventory.runner import DatasetCopy, FolderStatus

_MISSING = "—"
# How many mismatched-label warnings `unpack` names outright (plain text and --json alike);
# the rest are folded into the "and N more" count.
_WARNINGS_SHOWN = 3


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
    from dfwb.protocols.protocol import materialized_schemes

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
        here = materialized_schemes(pack.dataset_dir(dataset_id), dataset_id, card)
        rows.extend(
            {
                "pack": pack.name,
                "scheme": name,
                "kind": scheme.kind,
                "default": name == card.default_scheme,
                "distribution": card.distribution,
                "materialized": None if here is None else name in here,
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


def _schemes_table(schemes: list[dict[str, Any]]) -> str:
    """The plain-text table of :func:`_pack_schemes`' rows, the default scheme starred."""
    rows = [
        [
            row["pack"],
            row["scheme"] + ("*" if row["default"] else ""),
            row["kind"],
            distribution_text(row["distribution"], row["materialized"]),
        ]
        for row in schemes
    ]
    return table(["PACK", "SCHEME", "KIND", "DISTRIBUTION"], rows)


def _pack_card(dataset_id: str) -> dict[str, Any] | None:
    """The card fields of the first installed, healthy pack that publishes ``dataset_id``.

    Never raises: a pack listing that cannot be read degrades to no card, exactly like
    :func:`_pack_names` and :func:`_pack_schemes`.
    """
    from dfwb.core.errors import DFWBError
    from dfwb.protocols._yaml import read_card
    from dfwb.protocols.packs import installed_packs

    try:
        packs = installed_packs()
    except DFWBError:
        return None
    for pack in packs:
        if pack.card is not None and dataset_id in pack.card.datasets:
            try:
                card = read_card(pack.dataset_dir(dataset_id))
            except DFWBError:
                continue
            return card.model_dump(mode="json", by_alias=True)
    return None


def _info_pack_only(dataset_id: str, pack_names: list[str], as_json: bool) -> None:
    """``info`` for a dataset with no local inventory builder: only what an installed pack knows.

    ``pack_names`` are the healthy installed packs that publish ``dataset_id`` (as ``datasets
    list`` already shows for such an id); the caller has already checked this is non-empty.
    """
    schemes, problems = _pack_schemes(dataset_id)
    card = _pack_card(dataset_id) or {}
    note = "no local inventory builder is registered for it: the folder and layout can't be shown"

    if as_json:
        emit_json(
            {
                "id": dataset_id,
                "builder": None,
                "card": card,
                "layout": None,
                "location": None,
                "schemes": schemes,
                "problems": problems,
                "packs": pack_names,
                "note": note,
            }
        )
        return

    release = card.get("release")
    title = f"{card.get('name', dataset_id)} ({dataset_id})"
    click.echo(title + (f"  release {release}" if release else ""))
    for line in _card_lines(card):
        click.echo(line)
    click.echo("")
    click.echo(f"known from: {', '.join(pack_names)}")
    click.echo(f"note: {note}")
    click.echo("")
    if schemes:
        click.echo("schemes in installed protocol packs:")
        click.echo(_schemes_table(schemes))
    for problem in problems:
        click.echo(f"warning: {problem}", err=True)


@datasets.command("info")
@click.argument("dataset")
@json_option
def info(dataset: str, as_json: bool) -> None:
    """Show DATASET's details, its expected layout, its local folder and its schemes."""
    from dfwb.core.errors import UnknownKeyError
    from dfwb.core.registry import catalogue_requirement
    from dfwb.preprocess.inventory.runner import dataset_copies, folder_status, get_builder

    try:
        builder = get_builder(dataset)
    except UnknownKeyError:
        # Not registered locally: maybe it is only known from an installed protocol pack, the way
        # `datasets list` already shows it. A dataset in neither is genuinely unknown -- re-raise
        # the original error, with its did-you-mean over the registered builders.
        pack_names = _pack_names().get(dataset)
        if not pack_names:
            raise
        _info_pack_only(dataset, pack_names, as_json)
        return
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
        click.echo(_schemes_table(schemes))
    else:
        requirement = catalogue_requirement("protocol_packs", dataset_id)
        suggestion = f" (pip install {requirement})" if requirement else ""
        click.echo(f"no installed protocol pack publishes {dataset_id}{suggestion}")
    for problem in problems:
        click.echo(f"warning: {problem}", err=True)


@datasets.command("synth")
@click.argument("dataset", type=click.Choice(["toyfake"]))
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Where to write the dataset folder, e.g. a datasets root.",
)
@click.option(
    "--videos",
    type=click.IntRange(min=1),
    default=200,
    show_default=True,
    help="How many videos: about 40% reals and 30% of each fake method.",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Decides the ids, the pairs, the official split and every frame.",
)
@click.option(
    "--no-media",
    is_flag=True,
    help="Write the file tree with empty placeholder videos (no PyAV needed).",
)
@json_option
def synth(dataset: str, out: Path, videos: int, seed: int, no_media: bool, as_json: bool) -> None:
    """Generate the synthetic DATASET (toyfake) into --out/DATASET, deterministically.

    Real videos are smooth moving textures; fakes blend in a patch with a high-frequency
    artefact. Encoding the videos needs the preprocess extra (PyAV).
    """
    import shlex

    from dfwb.core.paths import absolute
    from dfwb.preprocess.toyfake import DEFAULT_SEED, DEFAULT_VIDEOS
    from dfwb.preprocess.toyfake import synth as run_synth

    result = run_synth(absolute(out), videos=videos, seed=seed, write_media=not no_media)
    commands = [
        f"dfwb inventory build {dataset} --root {shlex.quote(str(result.root))}",
        f"dfwb protocols verify {dataset}",
    ]
    note = (
        None
        if (videos, seed) == (DEFAULT_VIDEOS, DEFAULT_SEED)
        else f"the built-in {dataset} pack describes the tree made with --videos "
        f"{DEFAULT_VIDEOS} --seed {DEFAULT_SEED}, so verify reports this one as partial coverage"
    )
    if as_json:
        emit_json(
            {
                "dataset_id": dataset,
                "root": str(result.root),
                "n_videos": result.n_videos,
                "by_task": result.by_task,
                "seed": seed,
                "media": not no_media,
                "next": commands,
                "note": note,
            }
        )
        return

    counts = ", ".join(f"{task} {count}" for task, count in result.by_task.items())
    media = "no media: empty placeholder files" if no_media else "FFV1 video"
    click.echo(
        f"wrote {result.n_videos} {dataset} videos to {result.root} ({counts}; seed {seed}; "
        f"{media})"
    )
    click.echo("next:")
    for command in commands:
        click.echo(f"  {command}")
    if note is not None:
        click.echo(f"note: {note}")


@datasets.command("unpack")
@click.argument("dataset", type=click.Choice(["wilddeepfake"]))
@click.option(
    "--from",
    "from_dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Directory holding the release's real_train/real_test/fake_train/fake_test tar shards.",
)
@click.option(
    "--to",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Where to write the unpacked dataset folder (default: DATASET's expected folder, e.g. "
    "WildDeepfake, under the first configured datasets root -- see DFWB_DATASETS_ROOT and "
    "`dfwb doctor`).",
)
@json_option
def unpack(dataset: str, from_dir: Path, to: Path | None, as_json: bool) -> None:
    """Safely unpack DATASET's release tar shards into the layout its inventory builder reads.

    Every archive member is checked before anything is written: an absolute path, a '..'
    segment, a symlink, a hard link, or a device or FIFO file refuses the whole archive, naming
    the first offending member. A frame filed under an inner label that disagrees with its
    archive's own real_train/real_test/fake_train/fake_test category is not refused -- it is
    unpacked under the category's label regardless, and reported as a warning. Unpacking is
    resumable: a sequence already holding exactly its expected frames is left alone, and any
    other sequence folder is removed and rewritten from scratch, so re-running after an
    interrupted unpack is always safe and never duplicates frames. Discovery is then run on the
    result and its counts are reported.
    """
    from dfwb.core.paths import absolute, require_root, resolve_roots
    from dfwb.preprocess.inventory.runner import get_builder
    from dfwb.preprocess.wilddeepfake_unpack import unpack_wilddeepfake

    builder = get_builder(dataset)
    source = absolute(from_dir)
    if to is not None:
        target = absolute(to)
    else:
        target = require_root("datasets", resolve_roots()) / builder.expected_folder

    result = unpack_wilddeepfake(builder, source, target)
    shown = result.warnings[:_WARNINGS_SHOWN]

    if as_json:
        emit_json(
            {
                "dataset_id": result.dataset_id,
                "from": str(result.from_dir),
                "to": str(result.to),
                "categories": list(result.categories),
                "shards_found": result.shards_found,
                "sequences_total": result.sequences_total,
                "sequences_written": result.sequences_written,
                "sequences_skipped": result.sequences_skipped,
                "frames_written": result.frames_written,
                "by_task": result.by_task,
                "warnings": {"count": len(result.warnings), "first": list(shown)},
            }
        )
        return

    click.echo(
        f"unpacked {result.sequences_written} sequence(s) ({result.sequences_skipped} already "
        f"complete) from {result.shards_found} shard(s) in {', '.join(result.categories)} into "
        f"{result.to}"
    )
    counts = ", ".join(f"{task} {count}" for task, count in result.by_task.items())
    click.echo(f"discovery now finds {sum(result.by_task.values())} record(s) ({counts})")
    if result.warnings:
        more = len(result.warnings) - len(shown)
        suffix = f"; and {more} more" if more else ""
        click.echo(
            f"warning: {len(result.warnings)} frame(s) filed under a label that disagrees with "
            f"their category: {'; '.join(shown)}{suffix}"
        )
