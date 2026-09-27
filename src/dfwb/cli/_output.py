"""Output helpers shared by commands: ``--json`` and plain-text tables."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import click

__all__ = ["distribution_text", "emit_json", "hint_line", "json_option", "table"]


def json_option[F: Callable[..., Any]](function: F) -> F:
    """Add ``--json`` (machine-readable output) to a command."""
    return click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON.")(
        function
    )


def emit_json(data: Any) -> None:
    """Print ``data`` as indented JSON."""
    click.echo(json.dumps(data, indent=2, default=str))


def hint_line(hint: str) -> None:
    """Print a ``hint: <hint>`` line on stderr.

    For a command that reports its own non-zero exit code instead of raising a
    :class:`~dfwb.core.errors.DFWBError` (so its normal output, JSON included, still reaches
    stdout): the one place that prints this prefix, matching ``dfwb.cli.main._report``.
    """
    click.echo(f"hint: {hint}", err=True)


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Left-aligned columns separated by two spaces."""
    cells = [
        [str(h) for h in headers],
        *[["" if v is None else str(v) for v in row] for row in rows],
    ]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    return "\n".join(
        "  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)).rstrip() for row in cells
    )


def distribution_text(distribution: str, materialized: bool | None) -> str:
    """A dataset's distribution for a table: ``list``, ``undecided``, or ``recipe`` with whether
    it is materialised here when its pack ships no key lists (``materialized`` is not None)."""
    if materialized is None:
        return distribution or "-"
    return f"{distribution}: {'materialised' if materialized else 'not materialised'}"
