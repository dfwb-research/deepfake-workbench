"""``dfwb inventory``: build a dataset's local inventory from its raw files, and summarise it."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import click

from dfwb.cli._output import emit_json, json_option, table

# The columns `dfwb inventory show --by` counts by.
_BY = ("task", "method", "label", "compression")


@click.group()
def inventory() -> None:
    """Build and summarise local dataset inventories."""


def _split_list(value: str | None) -> list[str] | None:
    """``"c23, c40"`` -> ``["c23", "c40"]``; unset or empty means no filter (``None``)."""
    parts = [part.strip() for part in (value or "").split(",") if part.strip()]
    return parts or None


@inventory.command("build")
@click.argument("dataset")
@click.option(
    "--root",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="The dataset folder (default: found through the datasets roots).",
)
@click.option(
    "--compressions",
    default=None,
    metavar="A,B",
    help="Only these compressions, comma-separated (default: every known one).",
)
@click.option("--probe", is_flag=True, help="Probe each video's frames, fps and size.")
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Parallel probes.",
)
@json_option
def build(
    dataset: str,
    root: Path | None,
    compressions: str | None,
    probe: bool,
    jobs: int,
    as_json: bool,
) -> None:
    """Scan DATASET's raw files and write <work root>/DATASET/inventory.jsonl."""
    from dfwb.preprocess.inventory.runner import build_inventory

    result = build_inventory(
        dataset, root=root, compressions=_split_list(compressions), probe=probe, jobs=jobs
    )
    if as_json:
        emit_json(
            {
                "dataset_id": result.dataset_id,
                "path": str(result.path),
                "count": result.count,
                "by_task": result.by_task,
                "dataset_dir": str(result.dataset_dir),
                "location_source": result.location_source,
            }
        )
        return
    sources = ", ".join(f"{k}: {v}" for k, v in sorted(result.location_source.items()))
    click.echo(
        f"wrote {result.count} records to {result.path} "
        f"(dataset folder: {result.dataset_dir}, from {sources})"
    )


@inventory.command("show")
@click.argument("dataset")
@click.option(
    "--by",
    type=click.Choice(_BY),
    default="task",
    show_default=True,
    help="The column to count by.",
)
@json_option
def show(dataset: str, by: str, as_json: bool) -> None:
    """Count DATASET's inventory records by task, method, label or compression."""
    from dfwb.core.paths import require_root, resolve_roots
    from dfwb.preprocess.inventory.runner import read_inventory
    from dfwb.protocols.rules import task_of

    records = read_inventory(dataset, require_root("work", resolve_roots()))
    counts: Counter[str | None] = Counter()
    for record in records:
        if by == "task":
            counts[task_of(record.key)] += 1
        elif by == "method":
            counts[record.method] += 1
        elif by == "label":
            counts[record.label_key] += 1
        else:
            counts[record.compression] += 1
    rows = sorted(counts.items(), key=lambda item: (item[0] is not None, item[0] or ""))
    if as_json:
        emit_json(
            {
                "dataset_id": dataset,
                "by": by,
                "counts": [{"value": value, "count": count} for value, count in rows],
                "total": len(records),
            }
        )
        return
    body = [["-" if value is None else value, count] for value, count in rows]
    click.echo(table([by.upper(), "COUNT"], [*body, ["total", len(records)]]))
