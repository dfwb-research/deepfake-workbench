"""``dfwb protocols``: list and inspect protocol packs, schemes and their per-split counts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import click

from dfwb.cli._output import emit_json, json_option, table


@click.group()
def protocols() -> None:
    """List and inspect installed protocol packs and their schemes."""


def _info_row(info: Any) -> dict[str, Any]:
    return {
        "dataset_id": info.dataset_id,
        "scheme": info.scheme,
        "pack": info.pack,
        "version": info.version,
        "kind": info.kind,
        "default": info.default,
        "counts": info.counts,
        "broken": info.broken,
    }


@protocols.command("list")
@json_option
def list_(as_json: bool) -> None:
    """List every scheme of every dataset in every installed pack (``*`` marks the default)."""
    from dfwb.protocols.protocol import list_protocols

    rows = list_protocols()
    if as_json:
        emit_json([_info_row(r) for r in rows])
        return

    healthy = [r for r in rows if r.broken is None]
    broken = [r for r in rows if r.broken is not None]
    headers = ["PROTOCOL", "PACK", "VERSION", "KIND", "TRAIN", "VAL", "TEST"]
    table_rows = []
    for r in healthy:
        name = f"{r.dataset_id}/{r.scheme}" + ("*" if r.default else "")
        counts = r.counts or {}
        train, val, test = (counts.get(s, "-") for s in ("train", "val", "test"))
        table_rows.append([name, r.pack, r.version, r.kind, train, val, test])
    for r in broken:
        table_rows.append([f"({r.pack})", r.pack, "-", "BROKEN", "-", "-", "-"])
    if table_rows:
        click.echo(table(headers, table_rows))
    else:
        click.echo("no protocol packs installed")
    for r in broken:
        click.echo(f"warning: pack {r.pack!r} is broken: {r.broken}", err=True)


@protocols.command("info")
@click.argument("ref")
@json_option
def info(ref: str, as_json: bool) -> None:
    """Show a protocol's card summary, scheme card and counts per split x compression x label."""
    from dfwb.protocols.protocol import load

    protocol = load(ref)
    split_by_key = {(row.key, row.compression): row.split for row in protocol.split_rows()}
    tallies: Counter[tuple[str, str | None, str]] = Counter()
    for video in protocol.records():
        split = split_by_key[(video.key, video.compression)]
        tallies[(split, video.compression, video.label_key)] += 1
    counts = [
        {"split": split, "compression": compression, "label_key": label_key, "count": count}
        for (split, compression, label_key), count in sorted(
            tallies.items(), key=lambda item: (item[0][0], item[0][1] or "", item[0][2])
        )
    ]

    data = {
        "ref": protocol.ref,
        "pack": protocol.pack.name,
        "pack_version": protocol.pack_version,
        "sha256": protocol.sha256,
        "card": protocol.card.model_dump(mode="json"),
        "scheme_card": protocol.scheme_card.model_dump(mode="json"),
        "counts": counts,
    }
    if as_json:
        emit_json(data)
        return

    click.echo(f"{protocol.ref}  (pack {protocol.pack.name} {protocol.pack_version})")
    click.echo(f"sha256: {protocol.sha256}")
    click.echo(f"dataset: {protocol.card.name}  release {protocol.card.release}")
    click.echo(f"scheme: {protocol.scheme}  kind={protocol.scheme_card.kind}")
    click.echo("")
    rows = [[c["split"], c["compression"] or "-", c["label_key"], c["count"]] for c in counts]
    click.echo(table(["SPLIT", "COMPRESSION", "LABEL", "COUNT"], rows))


@protocols.command("verify")
@click.argument("ref")
@click.option(
    "--inventory",
    "inventory",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Inventory file to check (default: <work root>/<dataset>/inventory.jsonl).",
)
@click.option(
    "--split",
    "splits",
    multiple=True,
    help="A split to require full coverage of (repeatable); default: every split this scheme "
    "assigns, except exclude.",
)
@json_option
def verify(ref: str, inventory: Path | None, splits: tuple[str, ...], as_json: bool) -> int:
    """Join the local inventory to a protocol's pack videos and report coverage.

    Exits 0 on full coverage, 3 if any requested split is short a video, 4 if any video is
    relabelled. A missing inventory is a usage error (exit 2).
    """
    from dfwb.core.paths import require_root, resolve_roots
    from dfwb.protocols.verify import verify as run_verify
    from dfwb.protocols.verify import write_report

    work_root = require_root("work", resolve_roots())
    report = run_verify(ref, inventory=inventory, splits=splits or None, work_root=work_root)
    report_path = write_report(report, work_root)

    if as_json:
        emit_json({**report.to_json(), "report_path": str(report_path)})
        return report.exit_code

    click.echo(f"{report.dataset}/{report.scheme}  (pack {report.pack} {report.pack_version})")
    click.echo("requested splits: " + ", ".join(report.requested_splits))
    for bucket, count in report.counts.items():
        click.echo(f"{bucket}: {count}")
        for sample in report.samples.get(bucket, []):
            click.echo(f"  {sample}")
    for warning in report.warnings:
        click.echo(f"warning: {warning}")
    click.echo(f"report written to {report_path}")
    return report.exit_code
