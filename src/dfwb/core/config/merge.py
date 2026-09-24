"""Deep merge for ``extends``: mappings merge, lists and scalars replace, ``+key`` appends."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from dfwb.core.errors import ConfigError, format_loc

__all__ = ["APPEND_PREFIX", "deep_merge"]

APPEND_PREFIX = "+"


def deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any], *, _loc: tuple[str | int, ...] = ()
) -> dict[str, Any]:
    """Merge ``override`` onto ``base`` and return a new dict; neither input is modified.

    * mapping onto mapping: merged recursively;
    * anything else: the override value replaces the base value;
    * ``+key: [..]``: the list is appended to ``base[key]`` (which must be a list or absent).
    """
    result: dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for raw_key, value in override.items():
        if isinstance(raw_key, str) and raw_key.startswith(APPEND_PREFIX):
            key = raw_key[len(APPEND_PREFIX) :]
            where = format_loc((*_loc, key)) or key
            if not key:
                raise ConfigError(
                    f"{format_loc(_loc) or '<root>'}: '+' needs a key name",
                    hint="write +key: [...]",
                )
            if key in override:
                raise ConfigError(
                    f"{where}: both '{key}' and '+{key}' are set", hint="keep only one of them"
                )
            if not isinstance(value, list):
                raise ConfigError(
                    f"{where}: '+{key}' must be a list", hint=f"write +{key}: [item, ...]"
                )
            existing = result.get(key, [])
            if not isinstance(existing, list):
                raise ConfigError(
                    f"{where}: cannot append to a {type(existing).__name__}",
                    hint=f"use '{key}:' to replace the value instead",
                )
            result[key] = [*existing, *copy.deepcopy(value)]
        elif isinstance(value, Mapping):
            existing = result.get(raw_key)
            start = existing if isinstance(existing, Mapping) else {}
            result[raw_key] = deep_merge(start, value, _loc=(*_loc, raw_key))
        else:
            result[raw_key] = copy.deepcopy(value)
    return result
