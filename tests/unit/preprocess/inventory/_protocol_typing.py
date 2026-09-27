"""Type-checked by ``test_protocol_typing.py``: three builder styles satisfy ``InventoryBuilder``.

A base-class subclass (class variables), a builder that sets its attributes in ``__init__``, and a
plain class with class attributes must all type-check as the contract. A class missing a member
must not: its assignment carries a ``type: ignore`` that strict mypy reports as unused if the
assignment were accepted.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from dfwb.core.records import InventoryRecord
from dfwb.preprocess.inventory.base import (
    BaseBuilder,
    InventoryBuilder,
    LabelSpec,
    SchemeSpec,
    TaskSpec,
)


class Subclassed(BaseBuilder):
    dataset_id = "subclassed"
    expected_folder = "Subclassed"
    label_prefix = "SUB"
    tasks = (TaskSpec("REAL", "Real", "real", "real", "original"),)
    labels = {"REAL": LabelSpec(0, 0, 0, "real")}
    schemes = {"all-test": SchemeSpec("all-test", "subset")}
    default_scheme = "all-test"
    card_info = {"name": "Subclassed"}


class InitAttributes:
    def __init__(self) -> None:
        self.dataset_id = "init"
        self.version = "1"
        self.expected_folder = "Init"

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        return iter(())

    def describe_layout(self) -> str:
        return "one folder"


class ClassAttributes:
    dataset_id = "plain"
    version = "1"
    expected_folder = "Plain"

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        return iter(())

    def describe_layout(self) -> str:
        return "one folder"


class MissingFolder:
    dataset_id = "missing"
    version = "1"

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        return iter(())

    def describe_layout(self) -> str:
        return "one folder"


def builders() -> list[InventoryBuilder]:
    return [Subclassed(), InitAttributes(), ClassAttributes()]


def not_a_builder() -> InventoryBuilder:
    return MissingFolder()  # type: ignore[return-value]
