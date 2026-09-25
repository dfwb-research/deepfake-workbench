"""Clip-consistent transforms, registered in the ``transforms`` registry.

A clip is ``[T, C, H, W]``, float32 in ``[0, 1]``. Every random transform here draws its
parameters exactly once per call and applies them identically to all ``T`` frames, so a clip
never shows one frame cropped or coloured differently from its neighbours. Two techniques do
this, depending on what the parameters are:

* A torchvision v2 transform is called directly on the whole ``[T, C, H, W]`` tensor. Those
  transforms sample their random parameters once per call and apply them through ordinary tensor
  indexing over the last two dimensions, so a leading ``T`` dimension rides along for free -- the
  crop box, the flip decision, the colour factors are the same for every frame. Because the v2
  transforms draw from torch's *global* RNG rather than an explicit generator, the call happens
  inside :func:`torch.random.fork_rng`, with the global RNG reseeded from a value drawn from the
  caller's own generator first: deterministic in the generator's seed, and the global RNG is
  exactly as the caller left it once the call returns.
* Where the parameter is drawn by hand (a JPEG quality, a noise field), it is drawn once, directly
  from the given generator, and reused for every frame.

Normalisation is deliberately not one of the transforms a training pipeline can select here:
:func:`build_transforms` refuses a component named ``normalize`` by name, because mean/std belong
to input adaptation (built from a detector's own input spec), never to a fixed transform list --
that is what keeps a value baked in at the wrong layer from ever becoming a silent, unfixable
default again.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import Tensor
from torchvision.transforms import v2

from dfwb.core.config.schema import ComponentSpec
from dfwb.core.errors import ConfigError, DFWBError, Loc, format_loc
from dfwb.core.plugins import get_registry
from dfwb.data._images import jpeg_round_trip
from dfwb.data.dataset import ClipTransform

__all__ = [
    "CenterCrop",
    "ColorJitter",
    "GaussianBlur",
    "GaussianNoise",
    "Grayscale",
    "HorizontalFlip",
    "Jpeg",
    "Normalize",
    "RandomResizedCrop",
    "Resize",
    "build_transforms",
]

# ------------------------------------------------------------------------------------- seeding


def _draw_seed(generator: torch.Generator | None) -> int:
    """A fresh 63-bit seed, drawn from ``generator`` so it depends only on the generator's state."""
    if generator is None:
        raise ValueError(
            "a clip-consistent random transform needs a generator (ClipDataset always passes one)"
        )
    high = 2**63 - 1
    return int(torch.randint(0, high, (1,), generator=generator, dtype=torch.int64).item())


@contextmanager
def _seeded_fork(generator: torch.Generator | None) -> Iterator[None]:
    """Runs the block with torch's *global* RNG reseeded from ``generator``, inside a fork that
    restores the global RNG to exactly what it was, whatever the block does to it. ``devices=[]``
    keeps this from ever touching CUDA state on a CPU-only run (CPU state is always forked)."""
    seed = _draw_seed(generator)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


# --------------------------------------------------------------------------- v2-backed transforms


class _V2Transform:
    """Wraps one torchvision v2 transform that has no randomness of its own: no generator, no
    fork, just the transform applied to the whole ``[T, C, H, W]`` tensor."""

    def __init__(self, inner: torch.nn.Module) -> None:
        self._inner = inner

    def __call__(self, clip: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        result: Tensor = self._inner(clip)
        return result


class _ForkedV2Transform:
    """Wraps one torchvision v2 transform so every call is clip-consistent: the wrapped
    transform's own random draw happens once, inside :func:`_seeded_fork`, seeded from the caller's
    generator, and is applied to the whole ``[T, C, H, W]`` tensor at once."""

    def __init__(self, inner: torch.nn.Module) -> None:
        self._inner = inner

    def __call__(self, clip: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        with _seeded_fork(generator):
            result: Tensor = self._inner(clip)
        return result


class Resize(_V2Transform):
    """Deterministic resize to ``size``."""

    def __init__(self, size: int | tuple[int, int]) -> None:
        super().__init__(v2.Resize(size=size, antialias=True))


class CenterCrop(_V2Transform):
    """Deterministic centre crop to ``size``."""

    def __init__(self, size: int | tuple[int, int]) -> None:
        super().__init__(v2.CenterCrop(size=size))


class RandomResizedCrop(_ForkedV2Transform):
    """A random crop (area ``scale`` of the original, aspect ``ratio``), resized to ``size`` --
    one crop box drawn per clip, applied to every frame."""

    def __init__(
        self,
        size: int | tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ) -> None:
        super().__init__(
            v2.RandomResizedCrop(size=size, scale=tuple(scale), ratio=tuple(ratio), antialias=True)
        )


class HorizontalFlip(_ForkedV2Transform):
    """Flips the whole clip left-right with probability ``p`` -- one coin flip per clip."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(v2.RandomHorizontalFlip(p=p))


class ColorJitter(_ForkedV2Transform):
    """Brightness/contrast/saturation/hue jitter, one factor per clip, applied to every frame."""

    def __init__(
        self,
        brightness: float | tuple[float, float] | None = None,
        contrast: float | tuple[float, float] | None = None,
        saturation: float | tuple[float, float] | None = None,
        hue: float | tuple[float, float] | None = None,
    ) -> None:
        super().__init__(
            v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        )


class Grayscale(_ForkedV2Transform):
    """Converts the whole clip to grayscale (replicated back to the original channel count) with
    probability ``p`` -- one decision per clip, not one per frame."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__(v2.RandomGrayscale(p=p))


class GaussianBlur(_ForkedV2Transform):
    """Gaussian blur with a sigma drawn once per clip from ``sigma``."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        sigma: float | tuple[float, float] = (0.1, 2.0),
    ) -> None:
        super().__init__(v2.GaussianBlur(kernel_size=kernel_size, sigma=sigma))


# --------------------------------------------------------------------------- hand-drawn transforms


class GaussianNoise:
    """Adds one noise field, drawn once per clip with standard deviation ``std``, to every frame,
    then clamps back to ``[0, 1]``. Drawing one ``[C, H, W]`` field and broadcasting it over ``T``
    (rather than an independent field per frame) is what makes this clip-consistent: the same
    frame content in, the same noisy frame out, for every position of the clip."""

    def __init__(self, std: float) -> None:
        self.std = float(std)

    def __call__(self, clip: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        if generator is None:
            raise ValueError("gaussian-noise needs a generator (ClipDataset always passes one)")
        noise = torch.randn(clip.shape[1:], generator=generator) * self.std
        return (clip + noise.unsqueeze(0)).clamp(0.0, 1.0)


class Jpeg:
    """Round-trips the clip through JPEG (encoded and decoded with Pillow), at one quality per
    clip: a fixed ``quality``, or one drawn uniformly from the closed range ``quality`` gives."""

    def __init__(self, quality: int | tuple[int, int]) -> None:
        self.quality = quality

    def _quality(self, generator: torch.Generator | None) -> int:
        if isinstance(self.quality, int):
            return self.quality
        low, high = self.quality
        if generator is None:
            raise ValueError(
                "jpeg needs a generator to draw a quality (ClipDataset always passes one)"
            )
        return int(torch.randint(low, high + 1, (1,), generator=generator).item())

    def __call__(self, clip: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        quality = self._quality(generator)
        frames = [(frame * 255.0).round().clamp(0, 255).to(torch.uint8) for frame in clip]
        decoded = jpeg_round_trip(frames, quality)
        return torch.stack(decoded).to(torch.float32) / 255.0


class Normalize(_V2Transform):
    """Per-channel ``(x - mean) / std``, applied identically to every frame. Not a training-time
    augmentation: this exists so input adaptation has a registered target to build, not so a
    transform list can normalise itself (:func:`build_transforms` refuses that -- see the module
    docstring)."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__(v2.Normalize(mean=list(mean), std=list(std)))


# --------------------------------------------------------------------------------- build_transforms


class _Sequential:
    """Composes several clip-consistent transforms, applied in order to the same clip."""

    def __init__(self, transforms: Sequence[ClipTransform]) -> None:
        self._transforms = tuple(transforms)

    def __call__(self, clip: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        for transform in self._transforms:
            clip = transform(clip, generator=generator)
        return clip


def _bare_key(name: str) -> str:
    """``name`` with any ``provider:`` qualification stripped, lower-cased."""
    return name.strip().lower().rpartition(":")[2]


def build_transforms(specs: Sequence[ComponentSpec]) -> ClipTransform:
    """Builds ``transforms.train`` in order, through the ``transforms`` registry.

    Raises:
        ConfigError: A component names ``normalize`` (normalisation is applied by input
            adaptation, from the detector's input spec, never by a fixed transform list), or an
            unknown transform name or parameter -- either way, at ``transforms.train[<i>]``.
    """
    registry = get_registry("transforms")
    built: list[ClipTransform] = []
    for index, spec in enumerate(specs):
        loc: Loc = ("transforms", "train", index)
        if _bare_key(spec.name) == "normalize":
            raise ConfigError(
                f"{format_loc(loc)}: normalize does not belong in transforms.train",
                hint="normalisation is applied by adapt(), from the detector's input spec",
            )
        try:
            component = registry.build(spec.name, **spec.params)
        except ConfigError as exc:
            pairs = exc.problems or (((), exc.message),)
            lines = [f"{format_loc((*loc, *where))}: {text}" for where, text in pairs]
            raise ConfigError("\n".join(lines), hint=exc.hint) from None
        except DFWBError as exc:
            raise ConfigError(f"{format_loc(loc)}: {exc.message}", hint=exc.hint) from None
        built.append(component)
    return _Sequential(built)
