"""``dfwb protocols``: list and inspect protocol packs, schemes and their per-split counts."""

from __future__ import annotations

from collections import Counter
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
