"""Clip-consistent transforms: clip consistency, generator determinism, jpeg's one-quality-per-
clip, unknown-name/param refusals and the normalize refusal -- exercised exactly as a config's
``transforms.train:`` list would use them, through :func:`build_transforms` and the ``transforms``
registry.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from dfwb.core.config.schema import ComponentSpec
from dfwb.core.errors import ConfigError
from dfwb.core.plugins import get_registry
from dfwb.data import transforms as transforms_module
from dfwb.data.transforms import build_transforms

# Every built-in that draws a random parameter: (registry key, params) pairs a config could pass.
_RANDOM_TRANSFORMS: list[tuple[str, dict[str, Any]]] = [
    ("random-resized-crop", {"size": 6, "scale": (0.5, 1.0), "ratio": (0.9, 1.1)}),
    ("hflip", {"p": 0.5}),
    ("color-jitter", {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.2}),
    ("grayscale", {"p": 0.5}),
    ("gaussian-blur", {"kernel_size": 3, "sigma": (0.5, 2.0)}),
    ("gaussian-noise", {"std": 0.2}),
    ("jpeg", {"quality": (10, 90)}),
]


def _frame(seed: int, *, c: int = 3, h: int = 8, w: int = 8) -> torch.Tensor:
    """A reproducible, non-constant frame: geometric and colour transforms actually move pixels."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((c, h, w), generator=generator)


def _clip(frame: torch.Tensor, t: int) -> torch.Tensor:
    return frame.unsqueeze(0).repeat(t, 1, 1, 1)


def _build(name: str, **params: Any) -> Any:
    return build_transforms([ComponentSpec(name=name, **params)])


# --------------------------------------------------------------------------------- clip consistency


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    base_seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=20, deadline=None)
def test_every_random_transform_is_clip_consistent(seed: int, base_seed: int) -> None:
    frame = _frame(base_seed)
    clip = _clip(frame, t=5)
    for name, params in _RANDOM_TRANSFORMS:
        transform = _build(name, **params)
        generator = torch.Generator().manual_seed(seed)

        out = transform(clip, generator=generator)

        first = out[0]
        for i in range(1, out.shape[0]):
            assert torch.equal(out[i], first), f"{name}: frame {i} differs from frame 0"


def test_a_deterministic_transform_leaves_identical_frames_identical() -> None:
    frame = _frame(1)
    clip = _clip(frame, t=4)
    transform = _build("resize", size=(5, 5))

    out = transform(clip, generator=None)

    first = out[0]
    for i in range(1, out.shape[0]):
        assert torch.equal(out[i], first)


# ------------------------------------------------------------------------------------- determinism


def test_same_generator_seed_gives_the_same_output_for_every_random_transform() -> None:
    frame = _frame(7)
    clip = _clip(frame, t=3)
    for name, params in _RANDOM_TRANSFORMS:
        transform = _build(name, **params)

        first = transform(clip, generator=torch.Generator().manual_seed(11))
        second = transform(clip, generator=torch.Generator().manual_seed(11))

        assert torch.equal(first, second), f"{name}: same seed produced different output"


def test_different_generator_seeds_give_different_output_for_every_random_transform() -> None:
    frame = _frame(7)
    clip = _clip(frame, t=3)
    for name, params in _RANDOM_TRANSFORMS:
        transform = _build(name, **params)

        outputs = [
            transform(clip, generator=torch.Generator().manual_seed(seed)) for seed in range(20)
        ]
        distinct = {out.contiguous().numpy().tobytes() for out in outputs}

        assert len(distinct) > 1, f"{name}: 20 different seeds all gave the same output"


def test_global_rng_state_is_unchanged_after_a_v2_backed_transform() -> None:
    transform = _build("color-jitter", brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)
    clip = _clip(_frame(3), t=2)
    before = torch.get_rng_state()

    transform(clip, generator=torch.Generator().manual_seed(9))

    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_global_rng_state_is_unchanged_even_when_torch_is_used_elsewhere_first() -> None:
    # A more adversarial check: the global RNG is in an arbitrary, already-advanced state before
    # the transform runs (as it would be mid-training), not freshly seeded.
    torch.manual_seed(1234)
    torch.rand(100)  # advance the global RNG away from a fresh seed
    before = torch.get_rng_state()

    transform = _build("hflip", p=0.5)
    transform(_clip(_frame(4), t=2), generator=torch.Generator().manual_seed(5))

    after = torch.get_rng_state()
    assert torch.equal(before, after)


# ----------------------------------------------------------------------------------------- jpeg


def test_jpeg_draws_one_quality_per_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    original = transforms_module.encode_jpeg

    def spy(frames: list[torch.Tensor], quality: int) -> list[torch.Tensor]:
        calls.append(quality)
        result: list[torch.Tensor] = original(frames, quality=quality)
        return result

    monkeypatch.setattr(transforms_module, "encode_jpeg", spy)

    transform = _build("jpeg", quality=(20, 80))
    clip = _clip(_frame(0), t=6)
    transform(clip, generator=torch.Generator().manual_seed(5))

    assert len(calls) == 1  # every frame of the clip goes through one encode_jpeg call
    assert 20 <= calls[0] <= 80


def test_jpeg_with_a_fixed_quality_uses_it_every_time(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    original = transforms_module.encode_jpeg

    def spy(frames: list[torch.Tensor], quality: int) -> list[torch.Tensor]:
        calls.append(quality)
        result: list[torch.Tensor] = original(frames, quality=quality)
        return result

    monkeypatch.setattr(transforms_module, "encode_jpeg", spy)

    transform = _build("jpeg", quality=42)
    transform(_clip(_frame(0), t=3), generator=None)

    assert calls == [42]


def test_jpeg_with_a_range_needs_a_generator() -> None:
    jpeg = get_registry("transforms").build("jpeg", quality=(10, 90))
    with pytest.raises(ValueError, match="generator"):
        jpeg(_clip(_frame(0), t=2))


# --------------------------------------------------------------------------------------- shapes


def test_resize_changes_h_and_w() -> None:
    clip = _clip(_frame(0, h=10, w=10), t=2)
    out = _build("resize", size=(4, 6))(clip, generator=None)
    assert out.shape == (2, 3, 4, 6)


def test_center_crop_changes_h_and_w() -> None:
    clip = _clip(_frame(0, h=10, w=10), t=2)
    out = _build("center-crop", size=(4, 4))(clip, generator=None)
    assert out.shape == (2, 3, 4, 4)


def test_random_resized_crop_changes_h_and_w() -> None:
    clip = _clip(_frame(0, h=10, w=10), t=2)
    out = _build("random-resized-crop", size=(5, 5))(
        clip, generator=torch.Generator().manual_seed(0)
    )
    assert out.shape == (2, 3, 5, 5)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("hflip", {"p": 0.5}),
        ("color-jitter", {"brightness": 0.3}),
        ("grayscale", {"p": 0.3}),
        ("gaussian-blur", {"kernel_size": 3}),
        ("gaussian-noise", {"std": 0.05}),
        ("jpeg", {"quality": 50}),
    ],
)
def test_other_built_ins_preserve_shape(name: str, params: dict[str, Any]) -> None:
    clip = _clip(_frame(0), t=3)
    out = _build(name, **params)(clip, generator=torch.Generator().manual_seed(0))
    assert out.shape == clip.shape


def test_output_stays_float32_in_0_1_after_gaussian_noise() -> None:
    clip = _clip(_frame(0), t=2)
    out = _build("gaussian-noise", std=0.5)(clip, generator=torch.Generator().manual_seed(0))
    assert out.dtype == torch.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_output_stays_float32_in_0_1_after_jpeg() -> None:
    clip = _clip(_frame(0), t=2)
    out = _build("jpeg", quality=30)(clip, generator=torch.Generator().manual_seed(0))
    assert out.dtype == torch.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# --------------------------------------------------------------------------------- composition


def test_build_transforms_with_no_specs_is_the_identity() -> None:
    clip = _clip(_frame(0), t=2)
    out = build_transforms([])(clip, generator=torch.Generator().manual_seed(0))
    assert torch.equal(out, clip)


def test_build_transforms_composes_in_order() -> None:
    specs = [
        ComponentSpec(name="resize", size=(6, 6)),
        ComponentSpec(name="center-crop", size=(4, 4)),
    ]
    clip = _clip(_frame(0, h=10, w=10), t=2)
    out = build_transforms(specs)(clip, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 3, 4, 4)


# -------------------------------------------------------------------------------------- refusals


def test_unknown_param_gives_config_error_with_the_dotted_path() -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_transforms([ComponentSpec(name="resize", size=4, bogus=1)])
    assert "transforms.train[0].bogus" in str(excinfo.value)


def test_unknown_param_at_a_later_index_uses_its_own_index() -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_transforms(
            [
                ComponentSpec(name="resize", size=4),
                ComponentSpec(name="hflip", p=0.5, oops=True),
            ]
        )
    assert "transforms.train[1].oops" in str(excinfo.value)


def test_unknown_transform_name_gives_config_error_with_the_dotted_path() -> None:
    with pytest.raises(ConfigError) as excinfo:
        build_transforms([ComponentSpec(name="not-a-real-transform")])
    assert "transforms.train[0]" in str(excinfo.value)


def test_normalize_is_refused_inside_transforms() -> None:
    with pytest.raises(ConfigError, match="normalize") as excinfo:
        build_transforms([ComponentSpec(name="normalize", mean=[0.5] * 3, std=[0.5] * 3)])
    assert "transforms.train[0]" in str(excinfo.value)
    assert "input spec" in excinfo.value.hint or "InputSpec" in excinfo.value.hint


def test_normalize_is_refused_even_when_provider_qualified() -> None:
    with pytest.raises(ConfigError, match="normalize"):
        build_transforms([ComponentSpec(name="dfwb:normalize", mean=[0.0] * 3, std=[1.0] * 3)])


def test_normalize_stays_registered_for_adapt_to_use_later() -> None:
    registry = get_registry("transforms")
    assert "normalize" in registry
    entry = registry.entry("normalize")
    assert entry.key == "normalize"


def test_normalize_component_normalises_per_channel() -> None:
    normalize = get_registry("transforms").build(
        "normalize", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )
    clip = torch.full((2, 3, 4, 4), 0.75)
    out = normalize(clip, generator=None)
    assert torch.allclose(out, torch.full_like(out, 0.5))


# --------------------------------------------------------------------------- generator required


def test_gaussian_noise_needs_a_generator() -> None:
    noise = get_registry("transforms").build("gaussian-noise", std=0.1)
    with pytest.raises(ValueError, match="generator"):
        noise(_clip(_frame(0), t=2))


def test_a_random_v2_backed_transform_needs_a_generator() -> None:
    hflip = get_registry("transforms").build("hflip", p=0.5)
    with pytest.raises(ValueError, match="generator"):
        hflip(_clip(_frame(0), t=2))
