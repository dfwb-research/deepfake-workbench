"""A tiny, separable training setup for the Lightning layer's tests.

- ``toytrain``: a hand-built protocol pack with 12 real (``REAL/rNN``) and 12 fake (``FAKE/fNN``)
  videos. Indices 0-7 are ``train`` and 8-11 are ``val`` in the ``official`` scheme; every fake
  is paired with the real of the same index; ``attrs.group`` alternates ``a``/``b`` by index.
- A hand-built processed store for a face profile (crop scale 1.3, 32 px), written with the store
  helpers the data tests use. Real frames are dark noise and fake frames bright noise, so
  ``tiny-cnn`` separates them within a few steps.
- :func:`toy_config`: a ``dfwb.train/1`` config over both, and :func:`fit_toy`, which trains it
  with the same pieces a run uses: data module, module, callbacks and loggers.

Only the ``dfwb.plugins`` entry-point group is intercepted (to add the pack), so the real
``dfwb.builtins`` registrations (``tiny-cnn``, the losses, the metrics) stay available.
"""

from __future__ import annotations

import csv
import json
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("lightning")

import cv2
import lightning.pytorch as L
import numpy as np
import yaml
from lightning.pytorch.callbacks import Callback
from pydantic import BaseModel
from tests.unit.data.conftest import processed_record, write_store_index
from tests.unit.protocols.conftest import make_pack

from dfwb.core import plugins
from dfwb.core.config.schema import TrainConfig
from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PairRecord,
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
from dfwb.core.seed import seed_everything
from dfwb.models.detector import AssembledDetector, build_detector
from dfwb.train.callbacks import build_callbacks
from dfwb.train.datamodule import ProtocolDataModule
from dfwb.train.loggers import build_loggers
from dfwb.train.module import DetectorModule

__all__ = [
    "DATASET",
    "PROFILE",
    "PROTOCOL",
    "ToyRun",
    "column",
    "fit_toy",
    "install_toytrain_pack",
    "make_parts",
    "read_metrics_csv",
    "toy_config",
    "toy_profile",
    "toy_source",
    "write_toy_config",
    "write_toy_store",
]

DATASET = "toytrain"
PACK = "toytrain-pack"
PROTOCOL = f"{PACK}:{DATASET}/official"
PROFILE = "toy-face-32"
N_VIDEOS = 12  # per class
N_TRAIN = 8  # indices below this are train, the rest val


def _dump(model: BaseModel) -> str:
    return yaml.safe_dump(model.model_dump(mode="json", by_alias=True), sort_keys=False)


def _keys() -> list[tuple[str, int]]:
    """``(key, label)`` for every video, reals first."""
    return [(f"REAL/r{i:02d}", 0) for i in range(N_VIDEOS)] + [
        (f"FAKE/f{i:02d}", 1) for i in range(N_VIDEOS)
    ]


def write_toytrain_dataset(dataset_dir: Path, dataset_id: str) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits").mkdir()
    videos: list[VideoRecord] = []
    rows: list[SplitRow] = []
    for key, label in _keys():
        index = int(key[-2:])
        videos.append(
            VideoRecord(
                key,
                None,
                "TOYTRAIN-FAKE" if label else "TOYTRAIN-REAL",
                "swap" if label else "original",
                identity=key.rsplit("/", 1)[1],
                attrs={"group": "ab"[index % 2]},
            )
        )
        rows.append(SplitRow(key, None, "train" if index < N_TRAIN else "val"))
    write_jsonl(dataset_dir / "videos.jsonl.gz", videos)
    sha256 = write_split_tsv(dataset_dir / "splits" / "official.tsv.gz", rows)
    card = DatasetCard(
        id=dataset_id,
        name="Toy Train",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only",
        modalities=["video"],
        key_rule="fixture",
        schemes={"official": SchemeCard(kind="official", source="fixture", sha256=sha256)},
        default_scheme="official",
    )
    (dataset_dir / "dataset.yaml").write_text(_dump(card))
    vocab = {"TOYTRAIN-REAL": {"binary": 0}, "TOYTRAIN-FAKE": {"binary": 1}}
    labels = LabelVocab(vocab=vocab, mappings={"binary": LabelMappingSpec(from_="binary")})
    (dataset_dir / "labels.yaml").write_text(_dump(labels))
    pairs = [PairRecord(f"FAKE/f{i:02d}", f"REAL/r{i:02d}", "fixture") for i in range(N_VIDEOS)]
    write_jsonl(dataset_dir / "pairs.jsonl.gz", pairs)


class _FakeEntryPoint:
    def __init__(self, register: Callable[[Any], None]) -> None:
        self.name = PACK
        self.value = f"fake_module_{PACK}:register"
        self.dist = SimpleNamespace(name="fixture-packs", version="0.0.0")
        self._register = register

    def load(self) -> Callable[[Any], None]:
        return self._register


def install_toytrain_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install ``toytrain`` as pack ``toytrain-pack``; returns its dataset directory."""
    root = make_pack(tmp_path, PACK, {DATASET: {}}, builders={DATASET: write_toytrain_dataset})
    monkeypatch.syspath_prepend(str(root.parent.parent))

    def _register(api: Any) -> None:
        api.protocol_packs.add(
            PACK, target=f"{root.parent.name}:{root.name}", summary="fixture pack"
        )

    real_entry_points = plugins._entry_points

    def _entry_points(group: str) -> list[Any]:
        if group == plugins.ENTRY_POINT_GROUP:
            return [*real_entry_points(group), _FakeEntryPoint(_register)]
        return real_entry_points(group)

    monkeypatch.setattr(plugins, "_entry_points", _entry_points)
    return root / DATASET


def toy_profile(
    *, backend: str = "insightface", scale: float = 1.3, size: int = 32
) -> ProcessingProfile:
    return ProcessingProfile(
        id=PROFILE,
        backend=BackendSpec(name=backend),
        track=TrackSpec(iou=0.5, strategy="greedy"),
        crop=CropSpec(scale=scale, size=size, square=True, align="none"),
        sampling=SamplingSpec(mode="uniform", frames=4),
        decode=DecodeSpec(library="opencv", color="rgb"),
        extras=ExtrasSpec(landmarks=False, mesh=False, masks=False),
    )


def _write_frames(frame_dir: Path, key: str, label: int, n: int, size: int) -> None:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    low, high = (190, 250) if label else (0, 60)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        image = rng.integers(low, high, size=(size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_dir / f"frame_{i:06d}.png"), image)


def write_toy_store(
    work_root: Path,
    *,
    profile: ProcessingProfile | None = None,
    skip: Iterable[str] = (),
    n_frames: int = 4,
) -> Path:
    """Write the processed store for ``profile`` (default :func:`toy_profile`) under
    ``work_root``: every video except ``skip`` (left unprocessed). Returns the store directory."""
    profile = profile or toy_profile()
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
    for key, label in _keys():
        if key in skipped:
            continue
        record = processed_record(key, n_frames=n_frames)
        records.append(record)
        _write_frames(store_dir / record.relpath, key, label, n_frames, profile.crop.size)
    write_store_index(store_dir, records)
    return store_dir


def toy_source(split: str, **where: Any) -> dict[str, Any]:
    """One ``data.train``/``data.val`` entry over the toy protocol."""
    return {"protocol": PROTOCOL, "split": split, "where": dict(where)}


def toy_config(**changes: Any) -> TrainConfig:
    """The toy ``dfwb.train/1`` config; each keyword replaces (or, for a dict, updates) one
    top-level section, e.g. ``toy_config(data={"val": [...]})``."""
    config: dict[str, Any] = {
        "schema": "dfwb.train/1",
        "run": {"name": "toy", "seeds": [0]},
        "data": {
            "processing": PROFILE,
            "clip": {
                "frames": 1,
                "sampling": "uniform",
                "clips_per_video": {"train": 2, "eval": 2},
            },
            "labels": "binary",
            "train": [toy_source("train")],
            "val": [toy_source("val")],
            "loader": {"batch_size": 8, "num_workers": 0},
        },
        "model": {
            "backbone": {"name": "tiny-cnn"},
            "temporal_pool": {"name": "mean"},
            "head": {"name": "linear"},
        },
        "loss": {"name": "bce"},
        "optim": {"name": "adamw", "lr": 0.01, "weight_decay": 0.0},
        "schedule": {"name": "constant"},
        "train": {
            "max_epochs": 1,
            "precision": "32-true",
            "devices": 1,
            "monitor": "val/video_auc",
            "mode": "max",
        },
        "eval": {"metrics": ["auc", "eer", "brier", "ece"], "aggregate": "mean-prob"},
    }
    for section, value in changes.items():
        if isinstance(value, Mapping) and isinstance(config.get(section), dict):
            config[section] = {**config[section], **value}
        else:
            config[section] = value
    return TrainConfig.model_validate(config)


def write_toy_config(directory: Path, config: TrainConfig, name: str = "exp.yaml") -> Path:
    """Write ``config`` as a YAML config file (as a user would, before resolution)."""
    path = directory / name
    path.write_text(yaml.safe_dump(config.model_dump(mode="json", by_alias=True), sort_keys=False))
    return path


@dataclass
class ToyRun:
    run_dir: Path
    trainer: L.Trainer
    module: DetectorModule
    datamodule: ProtocolDataModule
    config: TrainConfig


def make_parts(
    config: TrainConfig,
    *,
    work_root: Path,
    seed: int = 0,
    pairs: bool = False,
) -> tuple[AssembledDetector, ProtocolDataModule, DetectorModule]:
    seed_everything(seed)
    detector = build_detector(config.model)
    datamodule = ProtocolDataModule(
        config.data, input_spec=detector.meta.input, work_root=work_root, seed=seed, pairs=pairs
    )
    module = DetectorModule(
        detector,
        loss=config.loss,
        optim=config.optim,
        schedule=config.schedule,
        train=config.train,
        eval=config.eval,
    )
    return detector, datamodule, module


def fit_toy(
    run_dir: Path,
    config: TrainConfig,
    *,
    work_root: Path,
    seed: int = 0,
    pairs: bool = False,
    callbacks: Sequence[Callback] = (),
    callback_options: Mapping[str, Any] | None = None,
    module_hook: Callable[[DetectorModule], None] | None = None,
    **trainer_options: Any,
) -> ToyRun:
    """Train ``config`` on CPU, deterministically, the way a run does: the framework's own
    callbacks and loggers, and Lightning's own checkpointing off."""
    _, datamodule, module = make_parts(config, work_root=work_root, seed=seed, pairs=pairs)
    if module_hook is not None:
        module_hook(module)
    options: dict[str, Any] = {
        "accelerator": "cpu",
        "devices": 1,
        "max_epochs": config.train.max_epochs,
        "deterministic": True,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 1,
        "default_root_dir": run_dir,
        "logger": build_loggers(run_dir),
        "callbacks": [
            *build_callbacks(run_dir, config.model, **(callback_options or {})),
            *callbacks,
        ],
    }
    options.update(trainer_options)
    trainer = L.Trainer(**options)
    trainer.fit(module, datamodule=datamodule)
    return ToyRun(run_dir, trainer, module, datamodule, config)


def read_metrics_csv(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "logs" / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def column(rows: Sequence[Mapping[str, str]], name: str) -> list[float]:
    """Every non-empty value of one logged column, in order."""
    return [float(row[name]) for row in rows if row.get(name)]
