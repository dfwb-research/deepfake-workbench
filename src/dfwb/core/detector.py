"""The detector and adapter contract (C4): the one interface score, eval and the zoo agree on.

Torch types appear only in annotations, so this module imports without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from dfwb.core.errors import ContractError

if TYPE_CHECKING:
    import torch
    from torch import Tensor

__all__ = [
    "DETECTOR_CONTRACT_VERSION",
    "ClipBatch",
    "Detector",
    "DetectorMeta",
    "DetectorOutput",
    "InputSpec",
]

DETECTOR_CONTRACT_VERSION: tuple[int, int] = (1, 0)


def _bad(field_name: str, problem: str) -> ContractError:
    return ContractError(f"InputSpec.{field_name}: {problem}", hint="see `dfwb schema export c4`")


@dataclass(frozen=True)
class InputSpec:
    """What a detector consumes. The harness adapts stored clips to it (or refuses)."""

    modality: Literal["frames", "audio", "audiovisual"] = "frames"
    crop: Literal["face", "full-frame"] = "face"
    crop_scale: float | None = 1.3
    size: tuple[int, int] = (224, 224)
    frames: int = 1
    sampling: Literal["uniform", "consecutive", "any"] = "any"
    color: Literal["rgb", "bgr"] = "rgb"
    value_range: tuple[float, float] = (0.0, 1.0)
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    preferred_profile: str | None = None

    def __post_init__(self) -> None:
        if self.crop_scale is not None and self.crop_scale <= 0:
            raise _bad("crop_scale", f"must be positive, got {self.crop_scale}")
        if len(self.size) != 2 or min(self.size) < 1:
            raise _bad("size", f"must be two positive integers, got {self.size}")
        if self.frames < 1:
            raise _bad("frames", f"must be at least 1, got {self.frames}")
        low, high = self.value_range
        if not low < high:
            raise _bad("value_range", f"low must be below high, got {self.value_range}")
        if (self.mean is None) != (self.std is None):
            raise _bad("mean", "mean and std must be given together")
        if self.mean is not None and self.std is not None:
            if len(self.mean) != len(self.std):
                raise _bad("std", "mean and std must have the same length")
            if min(self.std) <= 0:
                raise _bad("std", "every std must be positive")


@dataclass(frozen=True)
class DetectorMeta:
    """Identity, licensing and provenance of a detector."""

    name: str
    version: str
    contract_version: tuple[int, int]
    input: InputSpec
    license: str  # SPDX identifier for the code
    weights_license: str | None  # SPDX or LicenseRef-*
    citation: str | None  # BibTeX
    source: str | None  # "https://github.com/x/y@<commit>" or "run:<fingerprint>"
    training_data: tuple[str, ...] = ()  # protocol refs the weights were trained on


@dataclass
class ClipBatch:
    """A batch of clips, already matched to the detector's :class:`InputSpec`."""

    clips: Tensor  # [B, T, C, H, W], float
    keys: list[str]
    dataset_ids: list[str]
    compressions: list[str | None]
    clip_index: Tensor  # [B] which clip of the video this is
    frame_indices: Tensor  # [B, T] source frame numbers
    audio: Tensor | None = None
    labels: Tensor | None = None  # [B], training and validation batches only
    extras: dict[str, Any] = field(default_factory=dict)  # plugin-owned, keys "<provider>/<name>"


@dataclass
class DetectorOutput:
    """Detector output. ``score`` is P(fake) in [0, 1]: higher means more fake, always."""

    score: Tensor  # [B]
    logit: Tensor | None = None  # [B]
    frame_scores: Tensor | None = None  # [B, T]
    features: Tensor | None = None  # [B, D]


@runtime_checkable
class Detector(Protocol):
    """Anything that can score clips: trained runs, zoo adapters, users' own code."""

    meta: DetectorMeta

    def to(self, device: torch.device) -> Detector: ...

    def predict(self, batch: ClipBatch) -> DetectorOutput:
        """Score a batch; called under ``torch.inference_mode()``."""
        ...
