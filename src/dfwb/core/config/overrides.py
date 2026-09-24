"""Command-line overrides: ``dotted.path=value``, with YAML parsing of the value."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import yaml

from dfwb.core.config._yaml import load_yaml
from dfwb.core.config.dotted import Segment, parse_path, set_path
from dfwb.core.errors import ConfigError

__all__ = ["apply_overrides", "parse_override"]


def parse_override(text: str) -> tuple[tuple[Segment, ...], Any]:
    """Split ``optim.lr=3e-4`` into a path and a YAML-parsed value.

    Values are parsed as YAML flow scalars/collections (``[a, b]``, ``{k: v}``, ``true``, ``3``,
    ``3e-4``), exactly as the same value in a config file would be.
    """
    path_text, sep, raw = text.partition("=")
    if not sep or not path_text.strip():
        raise ConfigError(
            f"invalid override {text!r}",
            hint="overrides look like dotted.path=value, e.g. optim.lr=3e-4",
        )
    path = parse_path(path_text.strip())
    try:
        value = load_yaml(raw) if raw.strip() else ""
    except yaml.YAMLError:
        value = raw
    return path, value


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Return a copy of ``data`` with each override applied in order."""
    result = copy.deepcopy(data)
    for text in overrides:
        path, value = parse_override(text)
        set_path(result, path, value)
    return result
