"""``adapt()``: the deterministic chain from a store's ``ProcessingProfile`` to a detector's
``InputSpec``, and ``available_profiles()``, which reads a work root's stores for candidates.

Every expected tensor below is computed independently of ``adapt.py``'s own code -- by hand-slicing
tensors, or by calling ``torchvision.transforms.v2`` primitives directly on the same input -- never
by trusting the chain's own output back against itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ContractError
from dfwb.core.records.local import (
    BackendSpec,
    CropSpec,
    DecodeSpec,
    ExtrasSpec,
    ProcessingProfile,
    SamplingSpec,
    TrackSpec,
)
from dfwb.data.adapt import adapt, available_profiles


def _profile(
    *,
    id: str = "toy-face",
    backend: str = "insightface",
    scale: float = 1.3,
    size: int = 100,
) -> ProcessingProfile:
    return ProcessingProfile(
        id=id,
        backend=BackendSpec(name=backend),
        track=TrackSpec(iou=0.5, strategy="greedy"),
        crop=CropSpec(scale=scale, size=size, square=True, align="none"),
        sampling=SamplingSpec(mode="uniform", frames=1),
        decode=DecodeSpec(library="opencv", color="rgb"),
        extras=ExtrasSpec(landmarks=False, mesh=False, masks=False),
    )


def _spec(**overrides: object) -> InputSpec:
    params: dict[str, object] = {
        "modality": "frames",
        "crop": "face",
        "crop_scale": 1.3,
        "size": (100, 100),
        "frames": 1,
        "sampling": "any",
        "color": "rgb",
        "value_range": (0.0, 1.0),
        "mean": None,
        "std": None,
    }
    params.update(overrides)
    return InputSpec(**params)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------- derived crop


def test_smaller_derived_scale_is_allowed_and_crops_to_the_exact_hand_computed_region():
    profile = _profile(scale=1.3, size=100)
    spec = _spec(crop_scale=1.2, size=(92, 92))  # round(100 * 1.2 / 1.3) == 92

    result = adapt(spec, profile)

    assert result.derived_crop is True
    assert result.mismatch is False
    assert result.reason is None

    clip = torch.rand(2, 3, 100, 100)
    out = result.chain(clip)

    # a centre crop to 92 from 100: top = left = (100 - 92) // 2 = 4, computed by hand.
    expected = clip[:, :, 4 : 4 + 92, 4 : 4 + 92]
    assert out.shape == (2, 3, 92, 92)
    torch.testing.assert_close(out, expected)


def test_derived_crop_is_skipped_when_the_spec_scale_already_matches_the_store():
    profile = _profile(scale=1.3, size=100)
    spec = _spec(crop_scale=1.3, size=(100, 100))

    result = adapt(spec, profile)

    assert result.derived_crop is False
    clip = torch.rand(1, 3, 100, 100)
    torch.testing.assert_close(result.chain(clip), clip)


def test_derived_crop_is_skipped_when_the_spec_crop_scale_is_none():
    profile = _profile(scale=1.3, size=100)
    spec = _spec(crop_scale=None, size=(100, 100))

    result = adapt(spec, profile)

    assert result.derived_crop is False
    clip = torch.rand(1, 3, 100, 100)
    torch.testing.assert_close(result.chain(clip), clip)


# ------------------------------------------------------------------------------------- refusals


def test_adapt_refuses_larger_scale_and_names_profile():
    profile = _profile(id="store-face", scale=1.3, size=100)
    compatible = _profile(id="wide-face", scale=2.0, size=100)
    spec = _spec(crop_scale=2.0)

    with pytest.raises(ContractError) as excinfo:
        adapt(spec, profile, candidates=[compatible])

    message = str(excinfo.value)
    assert compatible.profile_id() in message
    assert repr(compatible.id) in message
    assert "dfwb preprocess profiles" not in message  # only the no-candidate path says that


def test_adapt_refuses_larger_scale_with_no_compatible_candidate():
    profile = _profile(id="store-face", scale=1.3, size=100)
    spec = _spec(crop_scale=2.0)

    with pytest.raises(ContractError) as excinfo:
        adapt(spec, profile)

    exc = excinfo.value
    assert "face" in exc.message
    assert "2.0" in exc.message
    assert "dfwb preprocess profiles" in exc.hint


def test_adapt_refuses_a_face_spec_against_a_full_frame_profile():
    profile = _profile(id="store-full", backend="center", scale=1.0, size=100)
    spec = _spec(crop="face", crop_scale=None)

    with pytest.raises(ContractError, match="full-frame"):
        adapt(spec, profile)


def test_adapt_refuses_a_full_frame_spec_against_a_face_profile():
    profile = _profile(id="store-face", backend="insightface", scale=1.3, size=100)
    spec = _spec(crop="full-frame", crop_scale=None)

    with pytest.raises(ContractError, match="face"):
        adapt(spec, profile)


def test_refusal_names_the_best_of_several_compatible_candidates():
    profile = _profile(id="store-face", scale=1.3, size=100)
    too_small = _profile(id="too-small", scale=1.5, size=100)
    just_right = _profile(id="just-right", scale=2.0, size=100)
    bigger_than_needed = _profile(id="bigger", scale=3.0, size=100)
    spec = _spec(crop_scale=2.0)

    with pytest.raises(ContractError) as excinfo:
        adapt(spec, profile, candidates=[too_small, bigger_than_needed, just_right])

    assert just_right.profile_id() in str(excinfo.value)


# --------------------------------------------------------------------------------- allow_mismatch


def test_allow_mismatch_on_crop_kind_skips_the_crop_and_resizes_the_full_stored_crop():
    profile = _profile(id="store-full", backend="center", scale=1.0, size=100)
    spec = _spec(crop="face", crop_scale=None, size=(50, 50))

    result = adapt(spec, profile, allow_mismatch=True)

    assert result.mismatch is True
    assert result.derived_crop is False
    assert result.reason is not None
    assert "full-frame" in result.reason

    clip = torch.rand(1, 3, 100, 100)
    out = result.chain(clip)
    assert out.shape == (1, 3, 50, 50)


def test_allow_mismatch_on_too_large_scale_skips_the_crop_and_resizes_the_full_stored_crop():
    profile = _profile(id="store-face", scale=1.3, size=100)
    spec = _spec(crop_scale=2.0, size=(64, 64))

    result = adapt(spec, profile, allow_mismatch=True)

    assert result.mismatch is True
    assert result.derived_crop is False
    assert result.reason is not None
    assert "2.0" in result.reason

    clip = torch.rand(1, 3, 100, 100)
    out = result.chain(clip)
    assert out.shape == (1, 3, 64, 64)


def test_allow_mismatch_false_by_default():
    profile = _profile(scale=1.3, size=100)
    spec = _spec(crop_scale=2.0)
    with pytest.raises(ContractError):
        adapt(spec, profile)


# --------------------------------------------------------------------------------------- resize


def test_resize_is_skipped_when_the_size_already_matches():
    profile = _profile(scale=1.3, size=100)
    spec = _spec(crop_scale=None, size=(100, 100))
    result = adapt(spec, profile)
    clip = torch.rand(1, 3, 100, 100)
    torch.testing.assert_close(result.chain(clip), clip)


def test_resize_changes_spatial_shape_when_the_size_differs():
    profile = _profile(scale=1.3, size=100)
    spec = _spec(crop_scale=None, size=(50, 40))
    result = adapt(spec, profile)
    clip = torch.rand(3, 3, 100, 100)
    out = result.chain(clip)
    assert out.shape == (3, 3, 50, 40)


# --------------------------------------------------------------------------------- colour order


def test_bgr_flips_the_channel_axis():
    profile = _profile(scale=1.3, size=4)
    spec = _spec(crop_scale=None, size=(4, 4), color="bgr")
    result = adapt(spec, profile)

    clip = torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, 4, 4).clone()
    out = result.chain(clip)

    expected = clip.flip(dims=(1,))
    torch.testing.assert_close(out, expected)
    assert out[0, 0, 0, 0].item() == 2.0  # channel 0 (R=2 originally) is now first: BGR
    assert out[0, 2, 0, 0].item() == 0.0


def test_rgb_leaves_the_channel_axis_untouched():
    profile = _profile(scale=1.3, size=4)
    spec = _spec(crop_scale=None, size=(4, 4), color="rgb")
    result = adapt(spec, profile)
    clip = torch.rand(1, 3, 4, 4)
    torch.testing.assert_close(result.chain(clip), clip)


# ----------------------------------------------------------------------------------- value range


def test_value_range_maps_0_1_to_the_given_range():
    profile = _profile(scale=1.3, size=4)
    spec = _spec(crop_scale=None, size=(4, 4), value_range=(-1.0, 1.0))
    result = adapt(spec, profile)

    clip = torch.rand(2, 3, 4, 4)
    out = result.chain(clip)

    expected = clip * 2.0 - 1.0  # x*(hi-lo)+lo, hand-computed for lo=-1, hi=1
    torch.testing.assert_close(out, expected)


def test_default_0_1_value_range_is_a_no_op():
    profile = _profile(scale=1.3, size=4)
    spec = _spec(crop_scale=None, size=(4, 4), value_range=(0.0, 1.0))
    result = adapt(spec, profile)
    clip = torch.rand(1, 3, 4, 4)
    torch.testing.assert_close(result.chain(clip), clip)


# --------------------------------------------------------------------------------- normalisation


def test_full_chain_matches_a_hand_computed_tensor_normalisation_applied_once():
    profile = _profile(scale=1.3, size=100)
    spec = _spec(
        crop_scale=1.2,
        size=(92, 92),
        color="bgr",
        value_range=(-1.0, 1.0),
        mean=(0.1, 0.2, 0.3),
        std=(0.5, 0.6, 0.7),
    )
    result = adapt(spec, profile)

    torch.manual_seed(0)
    clip = torch.rand(2, 3, 100, 100)
    out = result.chain(clip)

    # hand-computed, step by step, independently of adapt.py: crop -> flip -> range -> normalise.
    cropped = clip[:, :, 4:96, 4:96]
    flipped = cropped.flip(dims=(1,))
    ranged = flipped * 2.0 - 1.0
    mean = torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.6, 0.7]).view(1, 3, 1, 1)
    expected = (ranged - mean) / std

    assert out.shape == (2, 3, 92, 92)
    torch.testing.assert_close(out, expected)

    # applied exactly once: subtracting mean/std a second time would not round-trip back.
    twice = (out - mean) / std
    assert not torch.allclose(out, twice)


def test_no_normalisation_when_mean_and_std_are_unset():
    profile = _profile(scale=1.3, size=4)
    spec = _spec(crop_scale=None, size=(4, 4), mean=None, std=None)
    result = adapt(spec, profile)
    clip = torch.rand(1, 3, 4, 4)
    torch.testing.assert_close(result.chain(clip), clip)


# --------------------------------------------------------------------------------- determinism


def test_chain_is_deterministic_and_has_no_randomness():
    profile = _profile(scale=1.3, size=100)
    spec = _spec(crop_scale=1.2, size=(92, 92), mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
    result = adapt(spec, profile)
    clip = torch.rand(1, 3, 100, 100)
    torch.testing.assert_close(result.chain(clip), result.chain(clip))


# ------------------------------------------------------------------------------ available_profiles


def _write_profile_json(store_dir: Path, profile: ProcessingProfile) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile.model_dump(mode="json"),
        "sha256": profile.sha256(),
        "profile_id": profile.profile_id(),
        "backend": {"name": profile.backend.name, "version": "0", "license": "MIT", "meta": {}},
    }
    (store_dir / "profile.json").write_text(json.dumps(payload), encoding="utf-8")


def test_available_profiles_reads_and_validates_every_store(tmp_path: Path):
    work_root = tmp_path / "work"
    a = _profile(id="face-a", scale=1.3, size=100)
    b = _profile(id="face-b", backend="center", scale=1.0, size=64)
    _write_profile_json(work_root / "toy" / "processed" / a.profile_id(), a)
    _write_profile_json(work_root / "toy" / "processed" / b.profile_id(), b)

    found = available_profiles(work_root, "toy")

    assert {p.id for p in found} == {"face-a", "face-b"}
    assert {p.profile_id() for p in found} == {a.profile_id(), b.profile_id()}


def test_available_profiles_returns_empty_list_when_nothing_is_processed(tmp_path: Path):
    assert available_profiles(tmp_path / "work", "toy") == []


def test_available_profiles_skips_a_store_dir_with_no_profile_json(tmp_path: Path):
    work_root = tmp_path / "work"
    # a store directory can exist (e.g. its index.jsonl was started) before profile.json is
    # ever written -- that must be skipped quietly, not treated as an error.
    (work_root / "toy" / "processed" / "not-started-yet").mkdir(parents=True)
    assert available_profiles(work_root, "toy") == []


def test_available_profiles_skips_an_unreadable_file_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    work_root = tmp_path / "work"
    good = _profile(id="face-good", scale=1.3, size=100)
    _write_profile_json(work_root / "toy" / "processed" / good.profile_id(), good)

    broken_dir = work_root / "toy" / "processed" / "broken-store"
    broken_dir.mkdir(parents=True)
    (broken_dir / "profile.json").write_text("not json at all", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        found = available_profiles(work_root, "toy")

    assert [p.id for p in found] == ["face-good"]
    assert any("broken-store" in message for message in caplog.messages)


def test_available_profiles_skips_a_file_missing_the_profile_key(tmp_path: Path):
    work_root = tmp_path / "work"
    store_dir = work_root / "toy" / "processed" / "half-written"
    store_dir.mkdir(parents=True)
    (store_dir / "profile.json").write_text(json.dumps({"sha256": "x"}), encoding="utf-8")

    assert available_profiles(work_root, "toy") == []
