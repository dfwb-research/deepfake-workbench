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
    PairRecord,
    SchemeCard,
    SplitRow,
    VideoRecord,
    write_jsonl,
    write_split_tsv,
)
from dfwb.core.records.protocol import LabelMappingSpec, LicenseInfo

__all__ = [
    "fixture_packs",
    "make_pack",
    "register_packs",
    "register_provider_packs",
    "release_pack",
    "toyone_pack",
    "write_release_mismatch_dataset",
    "write_toyone_dataset",
]

DatasetSpec = Mapping[str, Mapping[str, Any]]
PackSpec = Mapping[str, DatasetSpec]
Builder = Callable[[Path, str], None]


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


# Real identities, by index (shared by REAL/FAKE_A/FAKE_B so "the same identity" carries a
# consistent attrs.lang across tasks): 1,3 -> en; 2,4 -> fr.
_TOYONE_LANG = {"1": "en", "2": "fr", "3": "en", "4": "fr"}


def write_toyone_dataset(dataset_dir: Path, dataset_id: str) -> None:
    """Write the richer ``toyone`` dataset: attrs, two schemes, three labels, pairs.

    12 videos: ``REAL/r1..r4`` (compressions ``c23`` and ``c40``, so 8 ``VideoRecord`` rows),
    ``FAKE_A/a1..a4`` and ``FAKE_B/b1..b4`` (no compression variants, 4 rows each) -- 16 rows in
    total, each with ``attrs: {lang: en|fr}``. Two schemes: ``official`` (train/val/test, with
    ``r4``/``a4``-paired-by-index left **unassigned** by index 4's real, to exercise that an
    unassigned record is absent rather than excluded, and that pairing falls back to the fake's
    split) and ``all-test`` (every row -> test). A ``labels.yaml``
    with ``binary``, ``audiovisual-binary`` and ``family`` mappings (``binary`` overrides
    ``TOYONE-FAKE_B`` to ``"exclude"``), and four fake/real pairs in ``pairs.jsonl.gz``.
    """
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits").mkdir()

    videos: list[VideoRecord] = []
    for i in range(1, 5):
        idx = str(i)
        for compression in ("c23", "c40"):
            videos.append(
                VideoRecord(
                    f"REAL/r{i}",
                    compression,
                    "TOYONE-REAL",
                    "original",
                    identity=f"r{i}",
                    attrs={"lang": _TOYONE_LANG[idx]},
                )
            )
    fake_tasks = (("FAKE_A", "TOYONE-FAKE_A", "FakeA"), ("FAKE_B", "TOYONE-FAKE_B", "FakeB"))
    for task, label_key, method in fake_tasks:
        prefix = "a" if task == "FAKE_A" else "b"
        for i in range(1, 5):
            idx = str(i)
            videos.append(
                VideoRecord(
                    f"{task}/{prefix}{i}",
                    None,
                    label_key,
                    method,
                    identity=f"{prefix}{i}",
                    attrs={"lang": _TOYONE_LANG[idx]},
                )
            )
    write_jsonl(dataset_dir / "videos.jsonl.gz", videos)

    # "official": index 1 -> train, 2 -> val, 3 -> test, 4 -> train, except REAL/r4 is left
    # unassigned entirely (both compressions), so it never appears in official records() results,
    # and its pair (FAKE_B/b4, REAL/r4) can only match a split through the fake's split.
    official_rows = [
        SplitRow("REAL/r1", "c23", "train"),
        SplitRow("REAL/r1", "c40", "train"),
        SplitRow("REAL/r2", "c23", "val"),
        SplitRow("REAL/r2", "c40", "val"),
        SplitRow("REAL/r3", "c23", "test"),
        SplitRow("REAL/r3", "c40", "test"),
        SplitRow("FAKE_A/a1", None, "train"),
        SplitRow("FAKE_A/a2", None, "val"),
        SplitRow("FAKE_A/a3", None, "test"),
        SplitRow("FAKE_A/a4", None, "train"),
        SplitRow("FAKE_B/b1", None, "train"),
        SplitRow("FAKE_B/b2", None, "val"),
        SplitRow("FAKE_B/b3", None, "test"),
        SplitRow("FAKE_B/b4", None, "train"),
    ]
    official_sha = write_split_tsv(dataset_dir / "splits" / "official.tsv.gz", official_rows)

    all_test_rows = [SplitRow(v.key, v.compression, "test") for v in videos]
    all_test_sha = write_split_tsv(dataset_dir / "splits" / "all-test.tsv.gz", all_test_rows)

    def _counts(rows: list[SplitRow]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.split] = counts.get(row.split, 0) + 1
        return counts

    card = DatasetCard(
        id=dataset_id,
        name="Toy One",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only",
        modalities=["video"],
        key_rule="fixture",
        schemes={
            "official": SchemeCard(
                kind="official",
                source="fixture",
                sha256=official_sha,
                counts=_counts(official_rows),
            ),
            "all-test": SchemeCard(
                kind="subset",
                source="fixture",
                sha256=all_test_sha,
                counts=_counts(all_test_rows),
            ),
        },
        default_scheme="official",
    )
    (dataset_dir / "dataset.yaml").write_text(_dump(card))

    vocab = {
        "TOYONE-REAL": {
            "binary": 0,
            "binary_av": 0,
            "multiclass": "real",
            "family": "real",
            "method": "original",
            "task": "REAL",
        },
        "TOYONE-FAKE_A": {
            "binary": 1,
            "binary_av": 1,
            "multiclass": "fake_a",
            "family": "face-swap",
            "method": "FakeA",
            "task": "FAKE_A",
        },
        "TOYONE-FAKE_B": {
            "binary": 1,
            "binary_av": 0,
            "multiclass": "fake_b",
            "family": "lip-sync",
            "method": "FakeB",
            "task": "FAKE_B",
        },
    }
    mappings = {
        "binary": LabelMappingSpec(from_="binary", override={"TOYONE-FAKE_B": "exclude"}),
        "audiovisual-binary": LabelMappingSpec(from_="binary_av"),
        "family": LabelMappingSpec(from_="family"),
    }
    labels = LabelVocab(vocab=vocab, mappings=mappings)
    (dataset_dir / "labels.yaml").write_text(_dump(labels))

    pairs = [
        PairRecord("FAKE_A/a1", "REAL/r1", "target-id"),
        PairRecord("FAKE_A/a2", "REAL/r2", "target-id"),
        PairRecord("FAKE_B/b3", "REAL/r3", "target-id"),
        PairRecord("FAKE_B/b4", "REAL/r4", "target-id"),
    ]
    write_jsonl(dataset_dir / "pairs.jsonl.gz", pairs)


def write_release_mismatch_dataset(dataset_dir: Path, dataset_id: str) -> None:
    """30 ``FAKE_A`` videos, all assigned to ``test`` (the release-mismatch fixture).

    Big enough that a test inventory can drop 10 and add 10 differently-keyed ones under the same
    task, to cross ``verify``'s release-mismatch heuristic (>=10 missing and >=10 extra sharing a
    task prefix).
    """
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits").mkdir()
    label_key = f"{dataset_id.upper()}-FAKE_A"
    videos = [
        VideoRecord(f"FAKE_A/v{i:02d}", None, label_key, "FakeA", identity=f"v{i:02d}")
        for i in range(1, 31)
    ]
    write_jsonl(dataset_dir / "videos.jsonl.gz", videos)
    rows = [SplitRow(v.key, v.compression, "test") for v in videos]
    sha256 = write_split_tsv(dataset_dir / "splits" / "official.tsv.gz", rows)
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
    labels = LabelVocab(vocab={label_key: {"binary": 1}}, mappings={"binary": {"from": "binary"}})
    (dataset_dir / "labels.yaml").write_text(_dump(labels))


def make_pack(
    tmp_path: Path,
    name: str,
    datasets: DatasetSpec,
    *,
    builders: Mapping[str, Builder] | None = None,
) -> Path:
    """Write a minimal, valid protocol pack under a throwaway importable package in ``tmp_path``.

    Each dataset is written by :func:`_write_dataset` (one video, one ``official`` scheme) unless
    ``builders`` maps its id to a different ``builder(dataset_dir, dataset_id)`` -- e.g.
    :func:`write_toyone_dataset` for the richer ``toyone`` fixture -- so tests needing a richer
    dataset reuse this function's package/registration scaffolding instead of duplicating it.

    Returns the pack root (the directory holding ``pack.yaml``) -- exactly what the registry's
    ``.load()`` resolves to once :func:`register_packs` installs it, so mutating the returned
    path (e.g. corrupting ``pack.yaml``) is visible to code under test.
    """
    builders = builders or {}
    package_dir = tmp_path / _package_name(name)
    root = package_dir / "pack"
    root.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    for dataset_id in datasets:
        builder = builders.get(dataset_id, _write_dataset)
        builder(root / dataset_id, dataset_id)
    card = PackCard(schema_version=1, name=name, version="1.0.0", datasets=sorted(datasets))
    (root / "pack.yaml").write_text(_dump(card))
    return root


class _FakePackEntryPoint:
    """Duck-typed stand-in for ``importlib.metadata.EntryPoint`` (mirrors test_plugins.py)."""

    def __init__(
        self, name: str, register: Callable[[Any], None], *, dist: str = "fixture-packs"
    ) -> None:
        self.name = name
        self.value = f"fake_module_{name}:register"
        self.dist = SimpleNamespace(name=dist, version="0.0.0")
        self._register = register

    def load(self) -> Callable[[Any], None]:
        return self._register


def _pack_register(packs: Mapping[str, Path]) -> Callable[[Any], None]:
    def _register(api: Any) -> None:
        for name, path in packs.items():
            api.protocol_packs.add(
                name, target=f"{path.parent.name}:{path.name}", summary=f"fixture pack {name!r}"
            )

    return _register


def register_provider_packs(
    monkeypatch: pytest.MonkeyPatch, providers: Mapping[str, Mapping[str, Path]]
) -> None:
    """Install one fake entry point per provider: ``{dist_name: {pack_name: pack_root}}``.

    Unlike :func:`register_packs` (every pack from one distribution), this lets a test register
    the *same* pack name from more than one distribution, to exercise the registry's own collision
    handling: two providers registering one key is ambiguous, never resolved silently (C1).
    """
    seen: set[Path] = set()
    for packs in providers.values():
        for path in packs.values():
            parent = path.parent.parent
            if parent not in seen:
                monkeypatch.syspath_prepend(str(parent))
                seen.add(parent)

    entry_points = [
        _FakePackEntryPoint(dist, _pack_register(packs), dist=dist)
        for dist, packs in providers.items()
    ]
    monkeypatch.setattr(
        plugins,
        "_entry_points",
        lambda group: entry_points if group == plugins.ENTRY_POINT_GROUP else [],
    )


def register_packs(monkeypatch: pytest.MonkeyPatch, packs: Mapping[str, Path]) -> None:
    """Install fake ``protocol_packs`` entries so each ``name`` in ``packs`` resolves to its root.

    A single fake ``dfwb.plugins`` entry point registers every pack in one ``register(api)`` call
    (``api.protocol_packs.add(name, target=...)``); the target names the importable package
    :func:`make_pack` wrote, so lookup goes through the real, data-kind registry path. All packs
    share one fake distribution; use :func:`register_provider_packs` for more than one provider.
    """
    register_provider_packs(monkeypatch, {"fixture-packs": packs})


@pytest.fixture
def fixture_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., dict[str, Path]]:
    """Build and register one or more fixture protocol packs; returns ``{pack_name: pack_root}``.

    ``builders`` optionally overrides how one dataset's files are written, keyed
    ``{pack_name: {dataset_id: builder}}`` (e.g. :func:`write_toyone_dataset`) instead of the
    default single-video dataset; see :func:`make_pack`.
    """

    def _install(
        packs: PackSpec, *, builders: Mapping[str, Mapping[str, Builder]] | None = None
    ) -> dict[str, Path]:
        builders = builders or {}
        roots = {
            name: make_pack(tmp_path, name, datasets, builders=builders.get(name))
            for name, datasets in packs.items()
        }
        register_packs(monkeypatch, roots)
        return roots

    return _install


@pytest.fixture
def toyone_pack(fixture_packs: Callable[..., dict[str, Path]]) -> Path:
    """Install the richer ``toyone`` dataset in its own pack; returns the dataset dir."""
    roots = fixture_packs(
        {"toyone-pack": {"toyone": {}}}, builders={"toyone-pack": {"toyone": write_toyone_dataset}}
    )
    return roots["toyone-pack"] / "toyone"


@pytest.fixture
def release_pack(fixture_packs: Callable[..., dict[str, Path]]) -> Path:
    """Install the 30-``FAKE_A``-video dataset in its own pack; returns the dataset dir."""
    roots = fixture_packs(
        {"release-pack": {"release": {}}},
        builders={"release-pack": {"release": write_release_mismatch_dataset}},
    )
    return roots["release-pack"] / "release"
