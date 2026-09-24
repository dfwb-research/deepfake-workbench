"""Dotted paths with list indices (``data.train[0].split``) for overrides and ``${ref:…}``."""

from __future__ import annotations

from typing import Any

from dfwb.core.errors import ConfigError, format_loc

__all__ = ["Segment", "get_path", "parse_path", "set_path"]

Segment = str | int

_HINT = "paths look like section.key or section.list[0].key"


def parse_path(text: str) -> tuple[Segment, ...]:
    """Parse ``a.b[0].c`` into ``("a", "b", 0, "c")``."""
    segments: list[Segment] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "[":
            end = text.find("]", i)
            number = text[i + 1 : end] if end != -1 else ""
            if not segments or not number.isdigit():
                raise ConfigError(f"invalid path {text!r}", hint=_HINT)
            segments.append(int(number))
            i = end + 1
        else:
            end = i
            while end < n and text[end] not in ".[]":
                end += 1
            name = text[i:end]
            if not name or (end < n and text[end] == "]"):
                raise ConfigError(f"invalid path {text!r}", hint=_HINT)
            segments.append(name)
            i = end
        if i < n and text[i] == ".":
            i += 1
            if i == n or text[i] in ".[":
                raise ConfigError(f"invalid path {text!r}", hint=_HINT)
        elif i < n and text[i] != "[":
            raise ConfigError(f"invalid path {text!r}", hint=_HINT)
    if not segments:
        raise ConfigError("empty path", hint=_HINT)
    return tuple(segments)


def get_path(data: Any, path: tuple[Segment, ...]) -> Any:
    """Return the value at ``path``; raises ``KeyError`` if any step is missing."""
    current = data
    for segment in path:
        if isinstance(segment, int):
            if not isinstance(current, list) or segment >= len(current):
                raise KeyError(format_loc(path))
            current = current[segment]
        else:
            if not isinstance(current, dict) or segment not in current:
                raise KeyError(format_loc(path))
            current = current[segment]
    return current


def set_path(data: dict[str, Any], path: tuple[Segment, ...], value: Any) -> None:
    """Set ``value`` at ``path`` in place, creating missing mappings along the way."""
    current: Any = data
    for depth, segment in enumerate(path):
        last = depth == len(path) - 1
        where = format_loc(path[: depth + 1])
        if isinstance(segment, int):
            if not isinstance(current, list):
                raise ConfigError(f"{where}: {format_loc(path[:depth])} is not a list", hint=_HINT)
            if segment >= len(current):
                raise ConfigError(
                    f"{where}: index {segment} is out of range (the list has {len(current)} items)",
                    hint="overrides can change existing list items, not add new ones",
                )
        elif not isinstance(current, dict):
            parent = format_loc(path[:depth]) or "<root>"
            raise ConfigError(
                f"{where}: cannot set a key inside {parent}, which is not a mapping", hint=_HINT
            )
        if last:
            current[segment] = value
            return
        if isinstance(segment, str) and segment not in current:
            current[segment] = {}
        current = current[segment]
