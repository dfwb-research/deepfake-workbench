"""Checkpoints: ``model.safetensors`` plus ``detector.json``, never pickle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

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


# --------------------------------------------------------------------- what a checkpoint keeps


def test_load_restores_the_saved_meta_rather_than_regenerating_it(tmp_path):
    import dataclasses

    cfg = _model_cfg()
    detector = build_detector(cfg, source="run:abc")
    detector.meta = dataclasses.replace(
        detector.meta,
        name="my-detector",
        version="0.0.7",
        license="Apache-2.0",
        weights_license="CC-BY-4.0",
        training_data=("toyfake-pack:toyfake/official@0.1.0",),
    )
    checkpoint.save(tmp_path, detector, cfg)
    restored = checkpoint.load(tmp_path)
    assert restored.meta == detector.meta


def test_a_provider_below_one_is_checked_on_major_and_minor(tmp_path):
    # before 1.0 a minor release may break things, so 0.x checkpoints are matched on 0.x
    from dfwb import __version__

    cfg = _model_cfg()
    checkpoint.save(tmp_path, build_detector(cfg), cfg)
    meta_path = tmp_path / "detector.json"
    payload = json.loads(meta_path.read_text())
    major, minor = __version__.split(".")[:2]
    assert major == "0"

    def _record(version: str) -> None:
        for component in payload["components"]:
            if component["provider"] == "dfwb":
                component["version"] = version
        meta_path.write_text(json.dumps(payload))

    _record(f"0.{int(minor) + 1}.0")
    with pytest.raises(InstallationError, match=rf"0\.{int(minor) + 1}") as caught:
        checkpoint.load(tmp_path)
    assert f'pip install "deepfake-workbench==0.{int(minor) + 1}.*"' in caught.value.hint

    _record(f"0.{minor}.99")
    checkpoint.load(tmp_path)  # the same 0.x: loads


# ------------------------------------------------------------------ loading never downloads


def _refuse_pretrained(monkeypatch: pytest.MonkeyPatch, module: object, name: str) -> None:
    original = getattr(module, name)

    def _guard(*args: object, **kwargs: object) -> object:
        if kwargs.get("pretrained"):
            raise AssertionError(f"{name}(pretrained=True) while loading a checkpoint")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, _guard)


def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DFWB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def test_loading_a_timm_checkpoint_never_fetches_pretrained_weights(tmp_path, monkeypatch):
    timm = pytest.importorskip("timm")
    built = ModelSection(
        backbone=ComponentSpec(name="timm", model="test_vit", pretrained=False),
        temporal_pool=ComponentSpec(name="mean"),
        head=ComponentSpec(name="linear"),
    )
    # what the run's config said: the weights it started from were downloaded
    saved = built.model_copy(
        update={"backbone": ComponentSpec(name="timm", model="test_vit", pretrained=True)}
    )
    detector = build_detector(built)
    detector.eval()
    batch = _batch(2, 1, size=detector.meta.input.size[0])
    before = detector.predict(batch)
    checkpoint.save(tmp_path, detector, saved)

    _refuse_pretrained(monkeypatch, timm, "create_model")
    _offline(monkeypatch)
    restored = checkpoint.load(tmp_path)
    torch.testing.assert_close(restored.predict(batch).score, before.score)


@pytest.mark.parametrize("kind", ["vit", "clip"])
def test_loading_an_hf_checkpoint_never_touches_the_hub(tmp_path, monkeypatch, kind):
    pytest.importorskip("transformers")
    import shutil

    import transformers
    from tests.unit.models.test_hf_vision import _save_clip, _save_vit

    model_dir = (_save_vit if kind == "vit" else _save_clip)(tmp_path)
    built = ModelSection(
        backbone=ComponentSpec(name="hf-vision", model=str(model_dir), pretrained=False),
        temporal_pool=ComponentSpec(name="mean"),
        head=ComponentSpec(name="linear"),
    )
    saved = built.model_copy(
        update={
            "backbone": ComponentSpec(name="hf-vision", model="some-org/not-here", pretrained=True)
        }
    )
    detector = build_detector(built)
    detector.eval()
    batch = _batch(2, 1, size=detector.meta.input.size[0])
    before = detector.predict(batch)
    out = tmp_path / "ckpt"
    checkpoint.save(out, detector, saved)
    shutil.rmtree(model_dir)  # nothing left to read the model from but the checkpoint

    def _no_hub(*args: object, **kwargs: object) -> object:
        raise AssertionError("the hub (or a model directory) was read while loading a checkpoint")

    for name in ("AutoModel", "AutoConfig", "AutoImageProcessor"):
        monkeypatch.setattr(getattr(transformers, name), "from_pretrained", _no_hub)
    _offline(monkeypatch)
    restored = checkpoint.load(out)
    torch.testing.assert_close(restored.predict(batch).score, before.score)
    assert restored.meta == detector.meta
