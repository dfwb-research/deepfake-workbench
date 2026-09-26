"""Licence acknowledgements for gated model weights.

Some model weights ship under terms stricter than the code that reads them -- research-only or
non-commercial licences, for instance. Before dfwb uses such weights it requires an explicit,
one-time acknowledgement from whoever runs it, recorded here so later runs on the same machine
never ask again. The store is a small JSON file under ``platformdirs.user_state_dir("dfwb")``
(``DFWB_STATE_DIR`` overrides the directory, for tests and for machines with a non-default layout),
written atomically so a crash mid-write never corrupts it.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import platformdirs

from dfwb.core.errors import ContractError, InstallationError

__all__ = [
    "STATE_DIR_ENV",
    "Acceptance",
    "accept",
    "all_accepted",
    "is_accepted",
    "require_accepted",
    "state_file",
]

STATE_DIR_ENV = "DFWB_STATE_DIR"
_FILE_NAME = "licenses.json"


@dataclass(frozen=True)
class Acceptance:
    """One recorded acknowledgement."""

    license: str
    accepted_at: str  # ISO-8601, UTC


def _state_dir() -> Path:
    override = os.environ.get(STATE_DIR_ENV)
    return Path(override) if override else Path(platformdirs.user_state_dir("dfwb"))


def state_file() -> Path:
    """Where acknowledgements are stored on this machine."""
    return _state_dir() / _FILE_NAME


def _corrupt(path: Path, detail: str) -> ContractError:
    return ContractError(
        f"{path}: licence store is corrupt ({detail})",
        hint="fix or delete the file; accept the licence again with --accept-license",
    )


def _read() -> dict[str, Acceptance]:
    path = state_file()
    if not path.is_file():
        return {}
    try:
        data: Any = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise _corrupt(path, f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise _corrupt(path, f"expected a JSON object, got {type(data).__name__}")
    entries: dict[str, Acceptance] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            raise _corrupt(path, f"{name!r} is not an object")
        license_value = entry.get("license")
        if not isinstance(license_value, str):
            raise _corrupt(path, f"{name!r} is missing a 'license' string")
        entries[name] = Acceptance(
            license=license_value, accepted_at=str(entry.get("accepted_at", ""))
        )
    return entries


def _write(entries: dict[str, Acceptance]) -> None:
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {"license": entry.license, "accepted_at": entry.accepted_at}
        for name, entry in sorted(entries.items())
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def is_accepted(name: str) -> bool:
    """Whether ``name`` has already been acknowledged on this machine."""
    return name in _read()


def accept(name: str, *, license: str) -> None:
    """Record that ``name`` has been acknowledged, under the terms of ``license``.

    This is a read-modify-write over the whole store, not locked against other processes: two
    concurrent calls can race and the loser's update is silently lost (last writer wins). The
    file itself is never left half-written either way -- the write underneath is always atomic.
    """
    entries = _read()
    now = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    entries[name] = Acceptance(license=license, accepted_at=now)
    _write(entries)


def all_accepted() -> dict[str, Acceptance]:
    """Every acknowledgement recorded on this machine, keyed by name."""
    return _read()


def require_accepted(name: str, *, terms: str) -> None:
    """Refuse to go on until the licence named ``name`` has been acknowledged on this machine.

    The single implementation of the licence gate: anything that must not touch a gated file
    (model weights whose terms are stricter than dfwb's own, a zoo adapter or its weights that
    declare ``requires_ack``) calls this first. The acknowledgement is recorded once
    (``--accept-license``) and never asked again on the same machine.

    Args:
        name: The name the acknowledgement is recorded under, e.g. ``"insightface-buffalo_l"``.
        terms: The licence terms in a few words, shown to the user.

    Raises:
        InstallationError: The licence has not been acknowledged (exit code 5).
    """
    if not is_accepted(name):
        raise InstallationError(
            f"{name}: these model weights need a one-time licence acknowledgement before first use",
            hint=f"{terms}; if your use fits those terms, re-run with --accept-license "
            "(this machine will not ask again)",
        )
