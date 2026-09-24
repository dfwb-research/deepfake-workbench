"""Import targets used by the registry tests (the tests register these by import path)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import typing_extensions
from pydantic import BaseModel, ConfigDict, Field

IMPORT_COUNT = 0


class Stem:
    def __init__(self, channels: int = 3, mode: Literal["fixed", "learned"] = "fixed") -> None:
        self.channels = channels
        self.mode = mode


def make_head(dim: int, *, dropout: float = 0.0) -> dict[str, Any]:
    return {"dim": dim, "dropout": dropout}


def open_kwargs(**kwargs: Any) -> dict[str, Any]:
    return kwargs


class FreezeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["none", "full", "partial", "lora"] = "none"
    trainable_blocks: int = 0


class Freeze(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["none", "full", "partial", "lora"] = "none"
    trainable_blocks: int = 0


class TimmParams(BaseModel):
    """A nested params model, as a timm backbone would declare (C2's error example)."""

    model_config = ConfigDict(extra="forbid")
    model: str
    pretrained: bool = False
    freeze: Freeze = Freeze()


@dataclass
class PoolParams:
    kind: Literal["mean", "max"]
    temperature: float = 1.0


class LossParams(TypedDict):
    weight: float


class AliasedParams(BaseModel):
    learning_rate: float = Field(0.1, validation_alias="lr")  # input name differs: refused


class KeywordParams(BaseModel):
    lambda_: float = Field(1.0, alias="lambda")  # a Python keyword as a parameter name: fine


class ExtensionsLossParams(typing_extensions.TypedDict):
    weight: float


def backbone(mode: str = "none", trainable_blocks: int = 0) -> tuple[str, int]:
    return (mode, trainable_blocks)


def pool(kind: str, temperature: float = 1.0) -> tuple[str, float]:
    return (kind, temperature)


def loss(weight: float) -> float:
    return weight


NOT_CALLABLE = 42
