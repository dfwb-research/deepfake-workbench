"""Breakdowns by method, compression, label_key and (via the pack's labels.yaml) family."""

from __future__ import annotations

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import ScoreRow
from dfwb.eval.breakdown import group_rows

from .conftest import make_meta

ROWS = [
    ScoreRow(
        "toyone",
        "REAL/r1",
        "c23",
        0,
        0.1,
        "ok",
        label_key="TOYONE-REAL",
        method="original",
    ),
    ScoreRow("toyone", "FAKE_A/a1", None, 1, 0.9, "ok", label_key="TOYONE-FAKE_A", method="FakeA"),
    ScoreRow("toyone", "FAKE_B/b1", None, 1, 0.7, "ok", label_key="TOYONE-FAKE_B", method="FakeB"),
    ScoreRow("toyone", "REAL/r2", "c40", 0, 0.2, "ok", label_key="TOYONE-REAL", method="original"),
]


def test_group_rows_by_method():
    groups = group_rows(ROWS, "method")
    assert set(groups) == {"original", "FakeA", "FakeB"}
    assert len(groups["original"]) == 2


def test_group_rows_by_compression():
    groups = group_rows(ROWS, "compression")
    assert set(groups) == {"c23", "c40"}


def test_group_rows_rejects_unknown_dimension():
    with pytest.raises(ConfigError, match="unknown breakdown"):
        group_rows(ROWS, "nope")


def test_group_rows_by_family_needs_meta():
    with pytest.raises(ContractError, match="needs the score file's"):
        group_rows(ROWS, "family")


def test_group_rows_by_family_uses_the_pack_vocab(family_pack):
    groups = group_rows(ROWS, "family", meta=make_meta())
    assert set(groups) == {"real", "face-swap", "lip-sync"}
    assert [r.key for r in groups["real"]] == ["REAL/r1", "REAL/r2"]
    assert [r.key for r in groups["face-swap"]] == ["FAKE_A/a1"]
    assert [r.key for r in groups["lip-sync"]] == ["FAKE_B/b1"]
