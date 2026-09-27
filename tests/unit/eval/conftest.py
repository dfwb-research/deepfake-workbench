"""Fixtures for ``dfwb.eval`` tests: C5 score files, and a small installed protocol pack.

Building a real, hash-checked protocol pack is exactly what ``tests/unit/protocols/conftest.py``
already does for ``dfwb.protocols`` tests; :func:`family_pack` below reuses its ``toyone`` builder
rather than writing a second copy of that fixture, since breaking down by ``family`` genuinely
needs a real, loadable pack (``dfwb.protocols.load`` hash-checks the split file it reads).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.unit.protocols.conftest import make_pack, write_toyone_dataset

from dfwb.core import plugins
from dfwb.core.records import ScoreMeta, ScoreRow

__all__ = ["family_pack", "make_meta", "make_rows"]


class _FakeEntryPoint:
    """Duck-typed stand-in for ``importlib.metadata.EntryPoint`` (mirrors the protocols tests)."""

    def __init__(self, name: str, register: Callable[[Any], None], *, dist: str) -> None:
        self.name = name
        self.value = f"fake_module_{name}:register"
        self.dist = SimpleNamespace(name=dist, version="0.0.0")
        self._register = register

    def load(self) -> Callable[[Any], None]:
        return self._register


@pytest.fixture
def family_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install the ``toyone`` fixture dataset, so ``dfwb.protocols.load("toyone/official")``
    resolves for real (its ``family`` mapping: ``TOYONE-REAL`` -> real, ``TOYONE-FAKE_A`` ->
    face-swap, ``TOYONE-FAKE_B`` -> lip-sync).

    Unlike ``tests/unit/protocols/conftest.py``'s ``register_packs`` (which replaces *every*
    entry-point group with the fixture's own, hiding the real ``dfwb.builtins`` registration and
    so the ``metrics`` registry too), this only intercepts the ``dfwb.plugins`` group, so a test
    using both a fixture pack and a real metric (``dfwb eval``'s usual combination) gets both.
    """
    root = make_pack(
        tmp_path, "toyone-pack", {"toyone": {}}, builders={"toyone": write_toyone_dataset}
    )
    monkeypatch.syspath_prepend(str(root.parent.parent))

    def _register(api: Any) -> None:
        api.protocol_packs.add(
            "toyone-pack",
            target=f"{root.parent.name}:{root.name}",
            summary="fixture pack 'toyone-pack'",
        )

    real_entry_points = plugins._entry_points

    def _entry_points(group: str) -> list[Any]:
        if group == plugins.ENTRY_POINT_GROUP:
            fake = _FakeEntryPoint("toyone-pack", _register, dist="fixture-packs")
            return [*real_entry_points(group), fake]
        return real_entry_points(group)

    monkeypatch.setattr(plugins, "_entry_points", _entry_points)
    return root / "toyone"


def make_meta(**change: object) -> ScoreMeta:
    """A minimal, valid :class:`ScoreMeta`, overridable field by field."""
    data: dict[str, object] = {
        "schema": "dfwb.scores/1",
        "detector": {
            "name": "tiny",
            "version": "0.1.0",
            "source": "run:abc",
            "checkpoint_sha256": "d" * 64,
            "contract_version": [1, 0],
        },
        "protocol": {
            "id": "toyone/official",
            "split": "test",
            "where": {},
            "pack": "toyone-pack",
            "pack_version": "1.0.0",
            "scheme_sha256": "e" * 64,
        },
        "labels": "binary",
        "processing_profile": None,
        "input_adaptation": {"derived_crop": False, "mismatch_override": False},
        "aggregation": {"clip_to_video": "mean-prob", "clips_per_video": 1},
        "coverage": {"expected": 1, "ok": 1, "missing": 0, "error": 0},
        "seed": 0,
        "env": {"dfwb": "0.1.0a1", "python": "3.12.14"},
        "git": None,
        "command": "dfwb score --detector run:abc --split test",
        "created": "2026-09-24T00:00:00Z",
    }
    data.update(change)
    return ScoreMeta.model_validate(data)


def make_rows(
    n_real: int, n_fake: int, *, real_score: float = 0.2, fake_score: float = 0.8
) -> list[ScoreRow]:
    """``n_real`` real rows scored near 0, ``n_fake`` fake rows scored near 1, all ``status=ok``."""
    rows = [
        ScoreRow("d", f"real/{i:04d}", None, 0, real_score, "ok", label_key="TOYONE-REAL")
        for i in range(n_real)
    ]
    rows += [
        ScoreRow("d", f"fake/{i:04d}", None, 1, fake_score, "ok", label_key="TOYONE-FAKE_A")
        for i in range(n_fake)
    ]
    return rows
