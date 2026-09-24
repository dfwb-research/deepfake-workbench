"""Output helpers shared by commands: ``--json`` and plain-text tables."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import click

__all__ = ["emit_json", "json_option", "table"]


def json_option[F: Callable[..., Any]](function: F) -> F:
    """Add ``--json`` (machine-readable output) to a command."""
    return click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON.")(
        function
    )


def emit_json(data: Any) -> None:
    """Print ``data`` as indented JSON."""
    click.echo(json.dumps(data, indent=2, default=str))


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
