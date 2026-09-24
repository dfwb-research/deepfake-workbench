"""Fixtures for ``dfwb.protocols`` tests: build and register throwaway protocol packs (C3a).

Follows the fake-entry-point pattern of ``tests/unit/core/test_plugins.py``: a registration goes
through the real :class:`~dfwb.core.registry.Registry`, so ``.load()`` locates the pack directory
without importing any code from it, exactly as it would for a real installed pack.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from dfwb.core import plugins
from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PackCard,
    SchemeCard,
    SplitRow,
    VideoRecord,
    write_jsonl,
    write_split_tsv,
)
from dfwb.core.records.protocol import LicenseInfo

__all__ = ["fixture_packs", "make_pack", "register_packs"]

DatasetSpec = Mapping[str, Mapping[str, Any]]
PackSpec = Mapping[str, DatasetSpec]


def _dump(model: BaseModel) -> str:
    return yaml.safe_dump(model.model_dump(mode="json", by_alias=True), sort_keys=False)


def _package_name(pack_name: str) -> str:
    """A valid Python identifier for the throwaway package that wraps one fixture pack."""
    return "dfwb_fixture_pack_" + re.sub(r"[^0-9a-zA-Z_]", "_", pack_name)


def _write_dataset(dataset_dir: Path, dataset_id: str) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits").mkdir()
    label_key = f"{dataset_id.upper()}-REAL"
    video = VideoRecord(f"{dataset_id}/0000", None, label_key, "original", identity="0000")
    write_jsonl(dataset_dir / "videos.jsonl.gz", [video])
    sha256 = write_split_tsv(
        dataset_dir / "splits" / "official.tsv.gz", [SplitRow(video.key, None, "train")]
    )
    card = DatasetCard(
        id=dataset_id,
        name=dataset_id,
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only",
        modalities=["video"],
        key_rule="fixture",
        schemes={"official": SchemeCard(kind="official", source="fixture", sha256=sha256)},
        default_scheme="official",
    )
    (dataset_dir / "dataset.yaml").write_text(_dump(card))
    labels = LabelVocab(vocab={label_key: {"binary": 0}}, mappings={"binary": {"from": "binary"}})
    (dataset_dir / "labels.yaml").write_text(_dump(labels))


def make_pack(tmp_path: Path, name: str, datasets: DatasetSpec) -> Path:
    """Write a minimal, valid protocol pack under a throwaway importable package in ``tmp_path``.

    Returns the pack root (the directory holding ``pack.yaml``) -- exactly what the registry's
    ``.load()`` resolves to once :func:`register_packs` installs it, so mutating the returned
    path (e.g. corrupting ``pack.yaml``) is visible to code under test.
    """
    package_dir = tmp_path / _package_name(name)
    root = package_dir / "pack"
    root.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    for dataset_id in datasets:
        _write_dataset(root / dataset_id, dataset_id)
    card = PackCard(schema_version=1, name=name, version="1.0.0", datasets=sorted(datasets))
    (root / "pack.yaml").write_text(_dump(card))
    return root


class _FakePackEntryPoint:
    """Duck-typed stand-in for ``importlib.metadata.EntryPoint`` (mirrors test_plugins.py)."""

    def __init__(self, name: str, register: Callable[[Any], None]) -> None:
        self.name = name
        self.value = f"fake_module_{name}:register"
        self.dist = SimpleNamespace(name="fixture-packs", version="0.0.0")
        self._register = register

    def load(self) -> Callable[[Any], None]:
        return self._register


def register_packs(monkeypatch: pytest.MonkeyPatch, packs: Mapping[str, Path]) -> None:
    """Install fake ``protocol_packs`` entries so each ``name`` in ``packs`` resolves to its root.

    A single fake ``dfwb.plugins`` entry point registers every pack in one ``register(api)`` call
    (``api.protocol_packs.add(name, target=...)``); the target names the importable package
    :func:`make_pack` wrote, so lookup goes through the real, data-kind registry path.
    """
    seen: set[Path] = set()
    for path in packs.values():
        parent = path.parent.parent
        if parent not in seen:
            monkeypatch.syspath_prepend(str(parent))
            seen.add(parent)

    def _register(api: Any) -> None:
        for name, path in packs.items():
            api.protocol_packs.add(
                name, target=f"{path.parent.name}:{path.name}", summary=f"fixture pack {name!r}"
            )

    entry_point = _FakePackEntryPoint("fixture-packs", _register)
    monkeypatch.setattr(
        plugins,
        "_entry_points",
        lambda group: [entry_point] if group == plugins.ENTRY_POINT_GROUP else [],
    )


@pytest.fixture
def fixture_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[PackSpec], dict[str, Path]]:
    """Build and register one or more fixture protocol packs; returns ``{pack_name: pack_root}``."""

    def _install(packs: PackSpec) -> dict[str, Path]:
        roots = {name: make_pack(tmp_path, name, datasets) for name, datasets in packs.items()}
        register_packs(monkeypatch, roots)
        return roots

    return _install
