"""A tiny fixture pack, processed store(s) and fake detector for ``dfwb.score`` tests.

``scoretoy``: 4 real (``REAL/r00..r03``) and 4 fake (``FAKE/f00..f03``) videos, all in the
``test`` split of the ``official`` scheme. The fake ``fake:`` detector source
(:func:`load_fake`) never touches torch weights: it scores a clip by the mean of its (already
adapted) pixel values, which are always in ``[0, 1]``, and can be told to raise for a chosen set
of keys, so a batch's failure is deterministic and reproducible.

Only the ``dfwb.plugins`` entry-point group is intercepted (to add the pack and the ``fake``
detector source), so the real ``dfwb.builtins`` registrations (the ``run`` source, metrics, ...)
stay available.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.unit.data.conftest import processed_record, write_store_frames, write_store_index
from tests.unit.protocols.conftest import make_pack

from dfwb.core import plugins
from dfwb.core.detector import DETECTOR_CONTRACT_VERSION, DetectorMeta, DetectorOutput, InputSpec
from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    SchemeCard,
    SplitRow,
    VideoRecord,
    write_jsonl,
    write_split_tsv,
)
from dfwb.core.records.local import (
    BackendSpec,
    CropSpec,
    DecodeSpec,
    ExtrasSpec,
    ProcessingProfile,
    SamplingSpec,
    TrackSpec,
)
from dfwb.core.records.protocol import LabelMappingSpec, LicenseInfo

__all__ = [
    "DATASET",
    "PACK",
    "PROTOCOL",
    "FakeDetector",
    "install_scoretoy_pack",
    "load_fake",
    "toy_profile",
    "write_toy_store",
]

DATASET = "scoretoy"
PACK = "scoretoy-pack"
PROTOCOL = f"{PACK}:{DATASET}/official"
N_VIDEOS = 4  # per class


def _dump(model: Any) -> str:
    import yaml

    return yaml.safe_dump(model.model_dump(mode="json", by_alias=True), sort_keys=False)


def _keys() -> list[tuple[str, int]]:
    return [(f"REAL/r{i:02d}", 0) for i in range(N_VIDEOS)] + [
        (f"FAKE/f{i:02d}", 1) for i in range(N_VIDEOS)
    ]


def write_scoretoy_dataset(dataset_dir: Path, dataset_id: str) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits").mkdir()
    videos: list[VideoRecord] = []
    rows: list[SplitRow] = []
    for key, label in _keys():
        videos.append(
            VideoRecord(
                key,
                None,
                "SCORETOY-FAKE" if label else "SCORETOY-REAL",
                "swap" if label else "original",
                identity=key.rsplit("/", 1)[1],
            )
        )
        rows.append(SplitRow(key, None, "test"))
    write_jsonl(dataset_dir / "videos.jsonl.gz", videos)
    sha256 = write_split_tsv(dataset_dir / "splits" / "official.tsv.gz", rows)
    card = DatasetCard(
        id=dataset_id,
        name="Score Toy",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only",
        modalities=["video"],
        key_rule="fixture",
        schemes={"official": SchemeCard(kind="official", source="fixture", sha256=sha256)},
        default_scheme="official",
    )
    (dataset_dir / "dataset.yaml").write_text(_dump(card))
    vocab = {"SCORETOY-REAL": {"binary": 0}, "SCORETOY-FAKE": {"binary": 1}}
    mappings = {
        "binary": LabelMappingSpec(from_="binary"),
        "binary-exclude-fake": LabelMappingSpec(
            from_="binary", override={"SCORETOY-FAKE": "exclude"}
        ),
    }
    labels = LabelVocab(vocab=vocab, mappings=mappings)
    (dataset_dir / "labels.yaml").write_text(_dump(labels))


class _FakeEntryPoint:
    def __init__(self, name: str, register: Callable[[Any], None]) -> None:
        self.name = name
        self.value = f"fake_module_{name}:register"
        self.dist = SimpleNamespace(name="fixture-packs", version="0.0.0")
        self._register = register

    def load(self) -> Callable[[Any], None]:
        return self._register


def install_scoretoy_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install ``scoretoy`` and the ``fake`` detector source; returns its dataset directory."""
    root = make_pack(tmp_path, PACK, {DATASET: {}}, builders={DATASET: write_scoretoy_dataset})
    monkeypatch.syspath_prepend(str(root.parent.parent))

    def _register(api: Any) -> None:
        api.protocol_packs.add(
            PACK, target=f"{root.parent.name}:{root.name}", summary="fixture pack"
        )
        api.detector_sources.add(
            "fake", target="tests.unit.score._toy:load_fake", summary="fake in-memory detector"
        )

    real_entry_points = plugins._entry_points

    def _entry_points(group: str) -> list[Any]:
        if group == plugins.ENTRY_POINT_GROUP:
            return [*real_entry_points(group), _FakeEntryPoint(PACK, _register)]
        return real_entry_points(group)

    monkeypatch.setattr(plugins, "_entry_points", _entry_points)
    return root / DATASET


def toy_profile(
    profile_id: str, *, backend: str = "insightface", scale: float = 1.3, size: int = 32
) -> ProcessingProfile:
    return ProcessingProfile(
        id=profile_id,
        backend=BackendSpec(name=backend),
        track=TrackSpec(iou=0.5, strategy="greedy"),
        crop=CropSpec(scale=scale, size=size, square=True, align="none"),
        sampling=SamplingSpec(mode="uniform", frames=4),
        decode=DecodeSpec(library="opencv", color="rgb"),
        extras=ExtrasSpec(landmarks=False, mesh=False, masks=False),
    )


def write_toy_store(
    work_root: Path, profile: ProcessingProfile, *, skip: Iterable[str] = (), n_frames: int = 4
) -> Path:
    """Write a processed store for ``profile`` under ``work_root``: every ``scoretoy`` video
    except ``skip`` (left unprocessed, so it becomes a ``missing`` row). Returns the store dir."""
    store_dir = work_root / DATASET / "processed" / profile.profile_id()
    store_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile.model_dump(mode="json"),
        "sha256": profile.sha256(),
        "profile_id": profile.profile_id(),
        "backend": {"name": profile.backend.name, "version": None, "license": None, "meta": {}},
    }
    (store_dir / "profile.json").write_text(json.dumps(payload), encoding="utf-8")
    skipped = set(skip)
    records = []
    for key, _ in _keys():
        if key in skipped:
            continue
        record = processed_record(key, n_frames=n_frames)
        records.append(record)
        write_store_frames(store_dir, record)
    write_store_index(store_dir, records)
    return store_dir


class FakeDetector:
    """An in-memory C4 detector: scores a clip by its mean pixel value (already in ``[0, 1]``
    once adapted), and can be told to raise for a chosen set of video keys."""

    def __init__(self, spec: InputSpec, *, raise_for: frozenset[str] = frozenset()) -> None:
        self.meta = DetectorMeta(
            name="fake-detector",
            version="0",
            contract_version=DETECTOR_CONTRACT_VERSION,
            input=spec,
            license="MIT",
            weights_license=None,
            citation=None,
            source="fake:test",
        )
        self._raise_for = raise_for

    def to(self, device: Any) -> FakeDetector:
        return self

    def predict(self, batch: Any) -> DetectorOutput:
        if self._raise_for.intersection(batch.keys):
            raise RuntimeError("fake detector: configured to fail on this batch")
        score = batch.clips.mean(dim=tuple(range(1, batch.clips.ndim))).clamp(0.0, 1.0)
        return DetectorOutput(score=score)


def load_fake(ref: str) -> FakeDetector:
    """The ``fake:`` detector source: ``fake:key=value&key=value...`` configures the returned
    :class:`FakeDetector` -- ``crop`` (default ``face``), ``scale`` (default ``1.3``), ``size``
    (default ``32``), ``preferred`` (``InputSpec.preferred_profile``), and ``raise`` (a
    comma-separated list of video keys :meth:`FakeDetector.predict` raises for)."""
    options: dict[str, str] = {}
    for part in ref.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        options[key] = value
    crop = options.get("crop", "face")
    scale = float(options.get("scale", "1.3"))
    size = int(options.get("size", "32"))
    preferred = options.get("preferred")
    raise_for = frozenset(options["raise"].split(",")) if options.get("raise") else frozenset()
    spec = InputSpec(
        crop=crop,  # type: ignore[arg-type]  # test-only: trusted fixture input
        crop_scale=scale,
        size=(size, size),
        frames=1,
        preferred_profile=preferred,
    )
    return FakeDetector(spec, raise_for=raise_for)
