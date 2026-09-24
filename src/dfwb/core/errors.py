"""Framework errors.

Every error carries a one-line ``hint`` and maps to a CLI exit code, so the CLI can always print
``error: <message>`` followed by ``hint: <hint>`` and exit with a meaningful status.
"""

from __future__ import annotations

import copy
import difflib
import re
import types
import typing
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    from pydantic import ValidationError

Loc = tuple[str | int, ...]

__all__ = [
    "UNION_TAG_PREFIX",
    "AmbiguousKeyError",
    "ConfigError",
    "ContractError",
    "CoverageError",
    "DFWBError",
    "InstallationError",
    "PluginError",
    "UnknownKeyError",
    "did_you_mean",
    "format_loc",
    "model_fields_at",
    "validation_messages",
    "validation_problems",
]

# Tags of discriminated unions in our models start with this; they are dropped from error paths.
UNION_TAG_PREFIX = "@"


def _rebuild(cls: type[DFWBError], message: str, hint: str) -> DFWBError:
    return cls(message, hint=hint)


class DFWBError(Exception):
    """Base class for every error the framework raises on purpose.

    Args:
        message: The cause, shown as ``error: <message>``.
        hint: The remedy, shown as ``hint: <hint>`` (install command, did-you-mean, doc link).
    """

    exit_code: ClassVar[int] = 1

    def __init__(self, message: str, *, hint: str) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (type(self), self.message, self.hint), self.__dict__)

    def with_prefix(self, prefix: str) -> Self:
        """Return a copy of this error whose message starts with ``prefix``."""
        new = copy.copy(self)
        new.message = f"{prefix}{self.message}"
        new.args = (new.message,)
        return new


class ConfigError(DFWBError):
    """A config, override or parameter is invalid (exit code 2).

    ``problems`` optionally holds the individual ``(location, text)`` findings, so a caller can
    re-anchor them under its own path (e.g. a component's params under ``model.backbone``).
    """

    exit_code = 2

    def __init__(
        self, message: str, *, hint: str, problems: Sequence[tuple[Loc, str]] = ()
    ) -> None:
        super().__init__(message, hint=hint)
        self.problems = tuple(problems)


class UnknownKeyError(DFWBError):
    """A registry key is not registered (exit code 2)."""

    exit_code = 2


class AmbiguousKeyError(DFWBError):
    """A plain registry key is provided by more than one plugin (exit code 2)."""

    exit_code = 2


class CoverageError(DFWBError):
    """Results were written but coverage is below target (exit code 3)."""

    exit_code = 3


class ContractError(DFWBError):
    """A file or object breaks one of the C1-C5 contracts (exit code 4)."""

    exit_code = 4


class InstallationError(DFWBError):
    """A needed package or extra is not installed (exit code 5)."""

    exit_code = 5


class PluginError(DFWBError):
    """A plugin is broken: bad registration, missing target, failing import (exit code 1)."""

    exit_code = 1


def did_you_mean(word: str, candidates: Iterable[str], *, n: int = 3) -> str:
    """Return ``" (did you mean 'x'?)"`` for close matches of ``word``, or ``""``."""
    matches = difflib.get_close_matches(word, sorted(set(candidates)), n=n, cutoff=0.6)
    if not matches:
        return ""
    return " (did you mean " + " or ".join(repr(m) for m in matches) + "?)"


def format_loc(loc: Sequence[str | int]) -> str:
    """Format a location tuple as a dotted path with list indices: ``data.train[0].split``."""
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out


_QUOTED = re.compile(r"'([^']*)'")


def validation_problems(
    exc: ValidationError, *, fields_at: Callable[[Loc], Iterable[str]] | None = None
) -> list[tuple[Loc, str]]:
    """Turn a pydantic ``ValidationError`` into ``(location, problem)`` pairs with suggestions.

    Args:
        exc: The validation error.
        fields_at: Optional callable ``loc -> keys`` giving the valid keys of the section at
            ``loc``; used to suggest near-matches for unknown keys.
    """
    problems: list[tuple[Loc, str]] = []
    for err in exc.errors(include_url=False):
        loc: Loc = tuple(
            p for p in err["loc"] if not (isinstance(p, str) and p.startswith(UNION_TAG_PREFIX))
        )
        kind = err["type"]
        value = err.get("input")
        if kind == "extra_forbidden":
            known = list(fields_at(loc[:-1])) if fields_at else []
            problems.append((loc, f"unknown key{did_you_mean(str(loc[-1]), known)}"))
        elif kind == "missing":
            problems.append((loc, "required key is missing"))
        elif kind == "literal_error":
            expected = str(err.get("ctx", {}).get("expected", ""))
            choices = _QUOTED.findall(expected)
            shown = "[" + (", ".join(repr(c) for c in choices) if choices else expected) + "]"
            suggestion = did_you_mean(str(value), choices) if isinstance(value, str) else ""
            problems.append((loc, f"{value!r} is not one of {shown}{suggestion}"))
        else:
            problems.append((loc, str(err["msg"])))
    return problems


def validation_messages(
    exc: ValidationError,
    *,
    prefix: Loc = (),
    fields_at: Callable[[Loc], Iterable[str]] | None = None,
) -> list[str]:
    """:func:`validation_problems` rendered as ``path: problem`` lines under ``prefix``."""
    return [
        f"{format_loc((*prefix, *loc)) or '<root>'}: {text}"
        for loc, text in validation_problems(exc, fields_at=fields_at)
    ]


def _models_in(annotation: Any) -> list[type[Any]]:
    from pydantic import BaseModel

    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _models_in(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType, list, dict, tuple):
        return [m for arg in typing.get_args(annotation) for m in _models_in(arg)]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return []


def model_fields_at(model: type[Any], loc: Loc) -> list[str]:
    """Valid keys of the pydantic model section at ``loc`` (best effort, for did-you-mean)."""
    current = _models_in(model)
    for part in loc:
        if isinstance(part, int):
            continue
        following: list[type[Any]] = []
        for cls in current:
            info = cls.model_fields.get(str(part))
            if info is None:
                info = next((f for f in cls.model_fields.values() if f.alias == part), None)
            if info is not None:
                following.extend(_models_in(info.annotation))
        current = following
    return sorted({f.alias or n for cls in current for n, f in cls.model_fields.items()})
