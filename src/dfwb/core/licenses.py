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

__all__ = ["STATE_DIR_ENV", "Acceptance", "accept", "all_accepted", "is_accepted", "state_file"]

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


def _read() -> dict[str, Acceptance]:
    path = state_file()
    if not path.is_file():
        return {}
    data: Any = json.loads(path.read_text("utf-8"))
    entries: dict[str, Acceptance] = {}
    for name, entry in data.items():
        entries[name] = Acceptance(
            license=entry["license"], accepted_at=str(entry.get("accepted_at", ""))
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
    """Record that ``name`` has been acknowledged, under the terms of ``license``."""
    entries = _read()
    now = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    entries[name] = Acceptance(license=license, accepted_at=now)
    _write(entries)


def all_accepted() -> dict[str, Acceptance]:
    """Every acknowledgement recorded on this machine, keyed by name."""
    return _read()
