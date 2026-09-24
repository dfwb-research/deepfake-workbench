"""Per-machine ``.env`` loading, applied explicitly by the CLI at startup, never at import.

The supported grammar is a small, well-defined subset of a shell environment file:

- Blank lines and ``#`` comment lines are skipped. An optional ``export `` prefix is allowed.
- A line is ``KEY=VALUE``, where ``KEY`` matches ``[A-Za-z_][A-Za-z0-9_]*``.
- An unquoted value is trimmed, and an inline `` #`` starts a comment.
- A ``'single'``-quoted value is literal.
- A ``"double"``-quoted value supports the escapes ``\\n``, ``\\"`` and ``\\\\``.
- ``${NAME}`` inside an unquoted or double-quoted value expands, first from keys earlier in the
  same file and then from the process environment. An unknown name is a :class:`ConfigError`.

Every other line is a :class:`ConfigError`. This module is stdlib-only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

from dfwb.core.errors import ConfigError

__all__ = [
    "ENV_FILE_VAR",
    "AppliedEnv",
    "apply_env_file",
    "find_env_file",
    "last_applied",
    "parse_env_file",
]

ENV_FILE_VAR = "DFWB_ENV_FILE"

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EXPAND_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_EXPORT_PREFIX = "export "


@dataclass(frozen=True)
class AppliedEnv:
    """What loading one ``.env`` file did: which keys were set and which were already present."""

    file: Path
    applied: tuple[str, ...]
    skipped: tuple[str, ...]


def _fail(path: Path, lineno: int, message: str, *, hint: str) -> ConfigError:
    return ConfigError(f"{path}:{lineno}: {message}", hint=hint)


def _unquote_single(path: Path, lineno: int, text: str) -> str:
    """Parse a ``'...'`` value; ``text[0]`` is the opening quote. No escapes are supported."""
    end = text.find("'", 1)
    if end == -1:
        raise _fail(path, lineno, "unterminated quote", hint="add the closing '")
    return text[1:end]


def _unquote_double(path: Path, lineno: int, text: str) -> str:
    """Parse a ``"..."`` value; ``text[0]`` is the opening quote. Escapes: \\n \\" \\\\."""
    chars: list[str] = []
    i = 1
    while i < len(text):
        char = text[i]
        if char == '"':
            return "".join(chars)
        if char == "\\" and i + 1 < len(text) and text[i + 1] in ("n", '"', "\\"):
            chars.append("\n" if text[i + 1] == "n" else text[i + 1])
            i += 2
            continue
        chars.append(char)
        i += 1
    raise _fail(path, lineno, "unterminated quote", hint='add the closing "')


def _expand(path: Path, lineno: int, text: str, so_far: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in so_far:
            return so_far[name]
        if name in os.environ:
            return os.environ[name]
        raise _fail(path, lineno, f"unknown variable {name!r}", hint=f"define {name} first")

    return _EXPAND_RE.sub(replace, text)


def _parse_line(
    path: Path, lineno: int, line: str, so_far: Mapping[str, str]
) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith(_EXPORT_PREFIX):
        stripped = stripped[len(_EXPORT_PREFIX) :].lstrip()

    key_text, sep, value_text = stripped.partition("=")
    if not sep:
        raise _fail(path, lineno, "expected KEY=VALUE", hint="use KEY=VALUE, one per line")
    key = key_text.rstrip()
    if not _KEY_RE.fullmatch(key):
        raise _fail(path, lineno, "expected KEY=VALUE", hint="use KEY=VALUE, one per line")

    lead = value_text.lstrip()
    if lead.startswith("'"):
        value = _unquote_single(path, lineno, lead)
    elif lead.startswith('"'):
        value = _expand(path, lineno, _unquote_double(path, lineno, lead), so_far)
    else:
        comment_at = lead.find(" #")
        raw = lead if comment_at == -1 else lead[:comment_at]
        value = _expand(path, lineno, raw.strip(), so_far)
    return key, value


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse ``path`` per the supported grammar (see module docstring)."""
    values: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        parsed = _parse_line(path, lineno, line, values)
        if parsed is not None:
            key, value = parsed
            values[key] = value
    return values


def apply_env_file(path: Path, environ: MutableMapping[str, str]) -> AppliedEnv:
    """Apply ``path``'s keys to ``environ``: a key already in ``environ`` is left untouched."""
    applied: list[str] = []
    skipped: list[str] = []
    for key, value in parse_env_file(path).items():
        if key in environ:
            skipped.append(key)
        else:
            environ[key] = value
            applied.append(key)
    return AppliedEnv(path, tuple(applied), tuple(skipped))


def find_env_file(cwd: Path, environ: Mapping[str, str]) -> Path | None:
    """The ``.env`` file the CLI should load: ``DFWB_ENV_FILE`` if set, else ``cwd/.env``.

    Raises:
        ConfigError: ``DFWB_ENV_FILE`` is set but does not name an existing file.
    """
    override = environ.get(ENV_FILE_VAR)
    if override:
        path = Path(override)
        if not path.is_file():
            raise ConfigError(
                f"{ENV_FILE_VAR}={override}: no such file",
                hint=f"fix or unset {ENV_FILE_VAR}",
            )
        return path
    candidate = cwd / ".env"
    return candidate if candidate.is_file() else None


_last: AppliedEnv | None = None


def last_applied() -> AppliedEnv | None:
    """The :class:`AppliedEnv` record of the env file the CLI most recently loaded, if any."""
    return _last


def _remember(value: AppliedEnv | None) -> None:
    """Record what the running CLI invocation loaded, for :func:`last_applied` (CLI-only)."""
    global _last
    _last = value
