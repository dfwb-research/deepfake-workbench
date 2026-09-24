"""Interpolation: ``${env:VAR}``, ``${env:VAR,default}`` and ``${ref:dotted.path}``."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from dfwb.core.config.dotted import Segment, get_path, parse_path
from dfwb.core.errors import ConfigError, format_loc

__all__ = ["interpolate"]

_TOKEN = re.compile(r"\$\{([a-z]+):([^{}]*)\}")
_HINT = "use ${env:VAR}, ${env:VAR,default} or ${ref:dotted.path}"


def interpolate(data: dict[str, Any], *, env: Mapping[str, str]) -> dict[str, Any]:
    """Return a copy of ``data`` with every interpolation resolved.

    A string that is exactly one ``${ref:…}`` keeps the referenced value's type (a list stays a
    list); interpolations embedded in a longer string are converted to text.
    """
    resolved = _Resolver(data, env).value(data, ())
    assert isinstance(resolved, dict)
    return resolved


class _Resolver:
    def __init__(self, root: dict[str, Any], env: Mapping[str, str]) -> None:
        self.root = root
        self.env = env
        self.stack: list[tuple[Segment, ...]] = []

    def value(self, value: Any, loc: tuple[Segment, ...]) -> Any:
        if isinstance(value, str):
            return self.string(value, loc)
        if isinstance(value, dict):
            return {k: self.value(v, (*loc, k)) for k, v in value.items()}
        if isinstance(value, list):
            return [self.value(v, (*loc, i)) for i, v in enumerate(value)]
        return value

    def string(self, text: str, loc: tuple[Segment, ...]) -> Any:
        matches = list(_TOKEN.finditer(text))
        where = format_loc(loc) or "<root>"
        if not matches:
            if "${" in text:
                raise ConfigError(f"{where}: malformed interpolation in {text!r}", hint=_HINT)
            return text
        if len(matches) == 1 and matches[0].span() == (0, len(text)):
            return self.token(matches[0], loc)
        parts: list[str] = []
        last = 0
        for match in matches:
            parts.append(text[last : match.start()])
            value = self.token(match, loc)
            if isinstance(value, (dict, list)):
                raise ConfigError(
                    f"{where}: cannot embed a {type(value).__name__} inside a string",
                    hint="reference it as the whole value: key: ${ref:path}",
                )
            parts.append(_text(value))
            last = match.end()
        parts.append(text[last:])
        return "".join(parts)

    def token(self, match: re.Match[str], loc: tuple[Segment, ...]) -> Any:
        kind, arg = match.group(1), match.group(2)
        where = format_loc(loc) or "<root>"
        if kind == "env":
            name, has_default, default = arg.partition(",")
            name = name.strip()
            if name in self.env:
                return self.env[name]
            if has_default:
                return default.strip()
            raise ConfigError(
                f"{where}: environment variable {name} is not set",
                hint=f"export {name}=... or give a default: ${{env:{name},<default>}}",
            )
        if kind == "ref":
            target = parse_path(arg.strip())
            if target in self.stack or target == loc:
                chain = " -> ".join(format_loc(p) for p in [*self.stack, target])
                raise ConfigError(f"{where}: reference cycle: {chain}", hint="break the cycle")
            try:
                raw = get_path(self.root, target)
            except KeyError:
                raise ConfigError(
                    f"{where}: ${{ref:{arg.strip()}}} points to a missing key",
                    hint="check the dotted path",
                ) from None
            self.stack.append(target)
            try:
                return copy.deepcopy(self.value(raw, target))
            finally:
                self.stack.pop()
        raise ConfigError(f"{where}: unknown interpolation '{kind}'", hint=_HINT)


def _text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
