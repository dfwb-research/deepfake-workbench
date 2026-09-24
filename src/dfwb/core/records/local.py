"""Local records (contracts C3b, C3c): inventories and processed stores. Never published."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ConfigDict, Field, with_config

from dfwb.core.errors import ContractError
from dfwb.core.hashing import fingerprint
from dfwb.core.paths import is_relative_posix_path
from dfwb.core.records._base import RecordModel
from dfwb.core.records.protocol import VideoRecord

__all__ = [
    "BuilderRef",
    "InventoryRecord",
    "Probe",
    "ProcessedRecord",
    "ProcessingProfile",
    "TrackStats",
    "to_video_record",
]


@with_config(ConfigDict(extra="forbid"))
@dataclass(slots=True, frozen=True)
class Probe:
    """Optional media probe (``--probe``)."""

    frames: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    has_audio: bool | None = None
    codec: str | None = None


@with_config(ConfigDict(extra="forbid"))
@dataclass(slots=True, frozen=True)
class BuilderRef:
    id: str
    version: str


@with_config(ConfigDict(extra="forbid"))
@dataclass(slots=True, frozen=True)
class InventoryRecord:
    """One line of ``$DFWB_WORK_ROOT/<dataset_id>/inventory.jsonl``."""

    key: str
    compression: str | None
    label_key: str
    method: str
    relpath: str  # relative to $DFWB_DATASETS_ROOT/<dataset folder>
    builder: BuilderRef
    identity: str | None = None
    source_id: str | None = None
    target_id: str | None = None
    pair_key: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    probe: Probe | None = None
    folder: str | None = None  # set when this record lives in a different dataset folder (J10)

    def __post_init__(self) -> None:
        if not is_relative_posix_path(self.relpath):
            raise ContractError(
                f"{self.key}: relpath {self.relpath!r} must be relative, POSIX, without '..'",
                hint="builders record paths relative to the dataset folder",
            )


def to_video_record(record: InventoryRecord) -> VideoRecord:
    """The portable columns of ``record``: drops ``relpath``, ``builder``, ``probe``, ``folder``."""
    return VideoRecord(
        key=record.key,
        compression=record.compression,
        label_key=record.label_key,
        method=record.method,
        identity=record.identity,
        source_id=record.source_id,
        target_id=record.target_id,
        pair_key=record.pair_key,
        attrs=record.attrs,
    )


@with_config(ConfigDict(extra="forbid"))
@dataclass(slots=True, frozen=True)
class TrackStats:
    mean_confidence: float | None = None
    identity_switch: bool | None = None


@with_config(ConfigDict(extra="forbid"))
@dataclass(slots=True, frozen=True)
class ProcessedRecord:
    """One line of ``processed/<profile_id>/index.jsonl``."""

    key: str
    compression: str | None
    status: Literal["ok", "no_face", "decode_error", "too_short", "skipped"]
    n_frames: int
    frame_indices: list[int]
    relpath: str
    track: TrackStats | None = None
    reason: str | None = None


class BackendSpec(RecordModel):
    """The face backend: a ``face_backends`` registry key plus its parameters."""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str


class TrackSpec(RecordModel):
    iou: float = Field(gt=0, le=1)
    strategy: str


class CropSpec(RecordModel):
    scale: float = Field(gt=0)
    size: int = Field(ge=1)
    square: bool
    align: str


class SamplingSpec(RecordModel):
    mode: Literal["uniform", "stride", "first-consecutive", "all"]
    frames: int | None = Field(default=None, ge=1)


class DecodeSpec(RecordModel):
    library: str
    color: Literal["rgb", "bgr"]


class ExtrasSpec(RecordModel):
    landmarks: bool
    mesh: bool
    masks: bool


class ProcessingProfile(RecordModel):
    """A named, hashed recipe for turning videos into face clips (C3c)."""

    id: str = Field(pattern=r"^[a-z0-9]+(?:[-.][a-z0-9]+)*$")
    backend: BackendSpec
    track: TrackSpec
    crop: CropSpec
    sampling: SamplingSpec
    decode: DecodeSpec
    extras: ExtrasSpec

    def sha256(self) -> str:
        """sha256 of the canonical JSON of the whole profile."""
        return fingerprint(self.model_dump(mode="json"))

    def profile_id(self) -> str:
        """Readable slug plus the first 8 hex characters of :meth:`sha256`."""
        return f"{self.id}-{self.sha256()[:8]}"
