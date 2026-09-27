"""``--where KEY=VALUE``: the filter option every command that restricts a protocol split by a
``VideoRecord`` field shares (``dfwb preprocess run``, ``dfwb score``)."""

from __future__ import annotations

from typing import Any

import click

from dfwb.core.errors import ConfigError

__all__ = ["parse_where", "where_option"]


def parse_where(pairs: tuple[str, ...]) -> dict[str, Any]:
    """``("compression=c23", "identity=000", "identity=002")`` -> ``{"compression": "c23",
    "identity": ["000", "002"]}``: repeating a key collects its values as a list, meaning any of
    them (the same ``where`` semantics :func:`dfwb.preprocess.face.runner.run` uses)."""
    where: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(
                f"--where {pair!r} is not key=value",
                hint="use --where key=value, e.g. "
                "--where compression=c23 (repeat --where to give a key more than one value)",
            )
        if key in where:
            existing = where[key]
            where[key] = [*existing, value] if isinstance(existing, list) else [existing, value]
        else:
            where[key] = value
    return where


where_option = click.option(
    "--where",
    "where_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Restrict to videos matching KEY=VALUE (repeatable; repeat a key for any-of).",
)
