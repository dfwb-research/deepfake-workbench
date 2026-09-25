"""Checkpoints: ``model.safetensors`` plus ``detector.json``, never pickle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dfwb.core.config.schema import ComponentSpec, ModelSection
from dfwb.core.errors import InstallationError
from dfwb.core.plugins import api
from dfwb.models import checkpoint
from dfwb.models.detector import build_detector

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NEW_MODULES = (
    "src/dfwb/models/pools.py",
    "src/dfwb/models/heads.py",
    "src/dfwb/models/detector.py",
    "src/dfwb/models/checkpoint.py",
    "src/dfwb/models/source.py",
)


def _model_cfg(stem: str | None = None) -> ModelSection:
    return ModelSection(
        backbone=ComponentSpec(name="tiny-cnn"),
        temporal_pool=ComponentSpec(name="mean"),
        head=ComponentSpec(name="linear"),
        stem=ComponentSpec(name=stem) if stem else None,
    )


def _batch(b: int, t: int, size: int = 64) -> object:
    from dfwb.core.detector import ClipBatch

    return ClipBatch(
        clips=torch.rand(b, t, 3, size, size),
        keys=[f"k{i}" for i in range(b)],
        dataset_ids=["toy"] * b,
        compressions=[None] * b,
        clip_index=torch.zeros(b, dtype=torch.long),
        frame_indices=torch.zeros(b, t, dtype=torch.long),
    )


def test_no_pickle_or_torch_load_in_the_new_modules():
    for rel in _NEW_MODULES:
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "torch.load" not in text, rel
        assert "pickle" not in text, rel


def test_checkpoint_directory_holds_only_safetensors_and_json(tmp_path):
    cfg = _model_cfg()
    detector = build_detector(cfg)
    checkpoint.save(tmp_path, detector, cfg)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["detector.json", "model.safetensors"]


def test_checkpoint_round_trip_gives_identical_outputs(tmp_path):
    torch.manual_seed(0)
    cfg = _model_cfg()
    detector = build_detector(cfg, source="run:abc123")
    detector.eval()
    batch = _batch(2, 3)
    before = detector.predict(batch)

    checkpoint.save(tmp_path, detector, cfg)
    restored = checkpoint.load(tmp_path)
    restored.eval()
    after = restored.predict(batch)

    assert torch.allclose(before.score, after.score)
    assert torch.allclose(before.logit, after.logit)
    assert restored.meta.source == "run:abc123"
    assert restored.meta.name == detector.meta.name


def test_checkpoint_round_trip_with_a_stem(tmp_path):
    api.layers.add(
        "fake-stem-ckpt", target="tests.unit.models._fake_layers:FakeHP", summary="fake stem"
    )
    torch.manual_seed(0)
    cfg = _model_cfg(stem="fake-stem-ckpt")
    detector = build_detector(cfg)
    detector.eval()
    batch = _batch(2, 2)
    before = detector.predict(batch)

    checkpoint.save(tmp_path, detector, cfg)
    restored = checkpoint.load(tmp_path)
    restored.eval()
    after = restored.predict(batch)
    assert torch.allclose(before.score, after.score)


def test_checkpoint_load_returns_the_detector_in_eval_mode(tmp_path):
    cfg = _model_cfg()
    detector = build_detector(cfg)
    detector.train()  # deliberately saved mid-training, in train mode
    checkpoint.save(tmp_path, detector, cfg)
    restored = checkpoint.load(tmp_path)
    assert restored.training is False


def test_detector_json_records_meta_as_json_safe_types(tmp_path):
    cfg = _model_cfg()
    detector = build_detector(cfg, source="run:xyz")
    checkpoint.save(tmp_path, detector, cfg)
    payload = json.loads((tmp_path / "detector.json").read_text())
    assert payload["meta"]["input"]["size"] == [64, 64]
    assert payload["meta"]["contract_version"] == [1, 0]
    assert payload["meta"]["source"] == "run:xyz"
    assert payload["model"]["backbone"]["name"] == "tiny-cnn"


def test_detector_json_lists_every_component_used(tmp_path):
    api.layers.add(
        "fake-stem-list", target="tests.unit.models._fake_layers:FakeHP", summary="fake stem"
    )
    cfg = _model_cfg(stem="fake-stem-list")
    detector = build_detector(cfg)
    checkpoint.save(tmp_path, detector, cfg)
    payload = json.loads((tmp_path / "detector.json").read_text())
    components = payload["components"]
    registries = {c["registry"] for c in components}
    assert registries == {"backbones", "temporal_pools", "heads", "layers"}
    for component in components:
        assert component["key"]
        assert component["provider"]


def test_missing_provider_raises_installation_error(tmp_path):
    cfg = _model_cfg()
    detector = build_detector(cfg)
    checkpoint.save(tmp_path, detector, cfg)
    meta_path = tmp_path / "detector.json"
    payload = json.loads(meta_path.read_text())
    payload["components"][0]["provider"] = "dfwb-totally-not-a-real-provider"
    meta_path.write_text(json.dumps(payload))

    with pytest.raises(InstallationError) as info:
        checkpoint.load(tmp_path)
    assert "dfwb-totally-not-a-real-provider" in info.value.message
    assert info.value.hint


def test_incompatible_major_version_raises_installation_error(tmp_path):
    cfg = _model_cfg()
    detector = build_detector(cfg)
    checkpoint.save(tmp_path, detector, cfg)
    meta_path = tmp_path / "detector.json"
    payload = json.loads(meta_path.read_text())
    for component in payload["components"]:
        if component["provider"] == "dfwb":
            component["version"] = "999.0.0"
    meta_path.write_text(json.dumps(payload))

    with pytest.raises(InstallationError, match="999"):
        checkpoint.load(tmp_path)


def test_lora_checkpoint_round_trip(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tests.unit.models.test_hf_vision import _save_clip

    model_dir = _save_clip(tmp_path)
    cfg = ModelSection(
        backbone=ComponentSpec(
            name="hf-vision",
            model=str(model_dir),
            pretrained=False,
            freeze={"mode": "lora", "targets": ["q_proj", "v_proj"]},
        ),
        temporal_pool=ComponentSpec(name="mean"),
        head=ComponentSpec(name="linear"),
    )
    detector = build_detector(cfg)
    detector.eval()
    height, width = detector.meta.input.size
    assert height == width
    batch = _batch(2, 1, size=height)
    before = detector.predict(batch)

    out_dir = tmp_path / "ckpt"
    checkpoint.save(out_dir, detector, cfg)
    restored = checkpoint.load(out_dir)
    restored.eval()
    after = restored.predict(batch)

    assert torch.allclose(before.score, after.score, atol=1e-5)
    lora_keys = [k for k in restored.state_dict() if "lora" in k]
    assert lora_keys
