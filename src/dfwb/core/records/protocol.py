"""Portable protocol records (contract C3a): what protocol packs publish."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator, with_config

from dfwb.core.records._base import DatasetId, RecordModel, SchemeName, Sha256

__all__ = [
    "PACK_SCHEMA_VERSION",
    "DatasetCard",
    "LabelMappingSpec",
    "LabelVocab",
    "LicenseInfo",
    "PackCard",
    "PackProvenance",
    "PairRecord",
    "PaperRef",
    "SchemeCard",
    "SplitRow",
    "TermsInfo",
    "VideoRecord",
]

PACK_SCHEMA_VERSION = 1

Split = Literal["train", "val", "test", "exclude"]


class PackCard(RecordModel):
    """``pack.yaml``."""

    schema_version: Literal[1]
    name: str
    version: str
    datasets: list[DatasetId]
    withheld: list[DatasetId] = Field(default_factory=list)


class PaperRef(RecordModel):
    title: str
    venue: str | None = None
    year: int | None = None
    doi: str | None = None


class LicenseInfo(RecordModel):
    spdx: str | None = None
    summary: str
    url: str | None = None


class TermsInfo(RecordModel):
    """Outcome of the per-dataset terms review (decided per dataset, later)."""

    source: str | None = None
    reviewed: datetime.date | None = None
    notes: str | None = None


class SchemeCard(RecordModel):
    """One split scheme of a dataset."""

    kind: Literal["official", "derived", "subset"]
    source: str | None = None
    rule: str | None = None
    ratios: list[int] | None = None
    splits: list[Literal["train", "val", "test"]] | None = None
    sha256: Sha256
    rationale: str | None = None
    counts: dict[Literal["train", "val", "test", "exclude"], int] | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class DatasetCard(RecordModel):
    """``<dataset_id>/dataset.yaml``."""

    id: DatasetId
    name: str
    aliases: list[str] = Field(default_factory=list)
    release: str
    homepage: str | None = None
    paper: PaperRef | None = None
    license: LicenseInfo
    access: str
    distribution: Literal["undecided", "list", "recipe"] = "undecided"
    terms: TermsInfo = Field(default_factory=TermsInfo)
    modalities: list[str] = Field(min_length=1)
    compressions: list[str] | None = None
    key_rule: str
    schemes: dict[SchemeName, SchemeCard] = Field(min_length=1)
    default_scheme: SchemeName

    @model_validator(mode="after")
    def _default_scheme_exists(self) -> DatasetCard:
        if self.default_scheme not in self.schemes:
            raise ValueError(f"default_scheme {self.default_scheme!r} is not one of the schemes")
        return self


class LabelMappingSpec(RecordModel):
    """A named label mapping: take attribute ``from`` of each vocab entry, with overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    from_: str = Field(alias="from")
    override: dict[str, int | Literal["exclude"]] = Field(default_factory=dict)


class LabelVocab(RecordModel):
    """``<dataset_id>/labels.yaml``."""

    vocab: dict[str, dict[str, str | int | float | bool | None]]
    mappings: dict[str, LabelMappingSpec]


@with_config(ConfigDict(extra="forbid"))
@dataclass(slots=True, frozen=True)
class VideoRecord:
    """One line of ``videos.jsonl.gz``: a video, identified by ``(key, compression)``."""

    key: str
    compression: str | None
    label_key: str
    method: str
    identity: str | None = None
    source_id: str | None = None
    target_id: str | None = None
    pair_key: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@with_config(ConfigDict(extra="forbid"))
@dataclass(slots=True, frozen=True)
class SplitRow:
    """One line of ``splits/<scheme>.tsv.gz``."""

    key: str
    compression: str | None
    split: Split


@with_config(ConfigDict(extra="forbid"))
@dataclass(slots=True, frozen=True)
class PairRecord:
    """One line of ``pairs.jsonl.gz``: a fake/real pairing produced by a pairing rule."""

    fake_key: str
    real_key: str
    rule: str


class PackProvenance(RecordModel):
    """``PROVENANCE.json``: how a pack was built, for byte-identical rebuilds."""

    builder: dict[str, str]
    dfwb: str
    source_listing_sha256: Sha256
    rules: dict[str, dict[str, Any]]
