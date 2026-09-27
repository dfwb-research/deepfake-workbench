"""Shared pieces of the record contracts: strict base model, name patterns, absolute-path guard."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints

from dfwb.core.errors import ContractError

__all__ = [
    "ABSOLUTE_PATH",
    "DatasetId",
    "RecordModel",
    "SchemeName",
    "Sha256",
    "assert_no_absolute_paths",
]

DatasetId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
SchemeName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:[-+][a-z0-9]+)*$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RecordModel(BaseModel):
    """Base for document-style records (cards, metadata): unknown keys are contract errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


#: What the absolute-path guard flags: a path that starts a value or follows a separator ("/x",
#: "~/x", "run:/x", "--out=/x"), or a file URL ("file:///x"). Not paths: "//" as in "https://",
#: and a slash before a space ("real / fake"). :func:`dfwb.core.runmeta.sanitize_command` uses
#: this same pattern to find what it must still clean, so the two always agree.
ABSOLUTE_PATH = re.compile(r"(?:^|[\s=:,;\"'(])(?:~/|/(?![/\s]))|file:///")


def _walk(value: Any, where: str) -> None:
    if isinstance(value, str):
        if ABSOLUTE_PATH.search(value):
            raise ContractError(
                f"{where}: {value!r} looks like an absolute path",
                hint="records store paths relative to a DFWB root (see dfwb.core.paths.RelPath)",
            )
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _walk(getattr(value, name), f"{where}.{name}")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            _walk(getattr(value, f.name), f"{where}.{f.name}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _walk(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, f"{where}[{index}]")


def assert_no_absolute_paths(record: Any, *, where: str = "record") -> None:
    """Raise :class:`ContractError` if any string inside ``record`` looks like an absolute path."""
    _walk(record, where)
