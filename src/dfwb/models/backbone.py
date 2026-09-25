"""The ``Backbone`` interface: feature extractors that plug into pool -> head assembly.

``FreezeSpec`` names how much of a backbone's parameters update during training.
``Backbone.apply_freeze`` implements every mode except ``lora``, which needs a backbone-specific
adapter that arrives with the first backbone that supports it. ``partial`` freezing goes only by a
backbone's own :meth:`Backbone.param_groups` ordering, never a name pattern over individual
parameters, so a backbone that reports its blocks honestly always freezes exactly the blocks it
says it has.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError

__all__ = ["Backbone", "BackboneOutput", "FreezeSpec"]

# Selected by module type, never by parameter-name substrings: BatchNorm and InstanceNorm share
# this private base, so one entry covers every rank of both.
_NORM_TYPES: tuple[type[nn.Module], ...] = (
    nn.modules.batchnorm._NormBase,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.RMSNorm,
)


class FreezeSpec(BaseModel):
    """How much of a backbone trains: ``none | full | partial | norm-only | lora``.

    ``trainable_blocks`` is required (and must be at least 1) when ``mode`` is ``"partial"``, and
    rejected otherwise. ``r``, ``alpha``, ``dropout`` and ``targets`` only take effect in ``lora``
    mode; they keep their defaults in every other mode.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "full", "partial", "norm-only", "lora"] = "none"
    trainable_blocks: int | None = None
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

    @model_validator(mode="after")
    def _check_trainable_blocks(self) -> FreezeSpec:
        if self.mode == "partial":
            if self.trainable_blocks is None or self.trainable_blocks < 1:
                raise ValueError("trainable_blocks must be an integer >= 1 when mode is 'partial'")
        elif self.trainable_blocks is not None:
            raise ValueError("trainable_blocks is only meaningful when mode is 'partial'")
        return self


@dataclass(frozen=True)
class BackboneOutput:
    """What every backbone's ``forward`` returns."""

    pooled: Tensor
    tokens: Tensor | None = None


class Backbone(nn.Module):  # type: ignore[misc, unused-ignore]  # Any without torch
    """Feature extractor: ``kind="image"`` sees ``[B*T,C,H,W]``; ``kind="video"`` sees
    ``[B,T,C,H,W]``."""

    kind: Literal["image", "video"]
    out_dim: int
    native_input: InputSpec

    def forward(self, x: Tensor) -> BackboneOutput:
        raise NotImplementedError

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named groups of this backbone's own parameters, e.g. ``{"blocks.0": [...], ...}``."""
        raise NotImplementedError

    def lora_module(self) -> nn.Module | None:
        """The submodule LoRA adapters are injected into, or ``None`` when this backbone has no
        LoRA support."""
        return None

    def checkpoint_state(self) -> dict[str, Any]:
        """JSON-safe data a checkpoint keeps so :meth:`from_checkpoint` can rebuild this
        backbone's exact architecture without fetching anything. Most backbones rebuild from
        their config parameters alone and keep nothing here."""
        return {}

    @classmethod
    def from_checkpoint(cls, params: Mapping[str, Any], state: Mapping[str, Any]) -> Self:
        """Rebuild the architecture a checkpoint was saved from, before its weights are loaded
        into it: ``params`` are the saved config parameters (``pretrained`` already switched off,
        since the weights come from the checkpoint), ``state`` what :meth:`checkpoint_state`
        returned at save time. Never downloads anything; override it when rebuilding from
        ``params`` alone would (as a hub model's config would be)."""
        return cls(**params)

    def apply_freeze(self, freeze: FreezeSpec) -> None:
        """Set ``requires_grad`` on this backbone's own parameters; never touches other modules."""
        if freeze.mode == "none":
            for p in self.parameters():
                p.requires_grad = True
        elif freeze.mode == "full":
            for p in self.parameters():
                p.requires_grad = False
        elif freeze.mode == "norm-only":
            self._freeze_all_but_norms()
        elif freeze.mode == "partial":
            self._freeze_all_but_last_blocks(freeze.trainable_blocks)
        elif freeze.mode == "lora":
            module = self.lora_module()
            if module is None:
                raise ConfigError(
                    "freeze.mode: 'lora' needs a backbone with LoRA support",
                    hint=(
                        "LoRA freezing needs a peft-enabled backbone "
                        "(the timm/HF backbones with the peft extra installed)"
                    ),
                )
            from dfwb.models.lora import apply_lora

            apply_lora(module, freeze)

    def _freeze_all_but_norms(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        for module in self.modules():
            if isinstance(module, _NORM_TYPES):
                for p in module.parameters(recurse=False):
                    p.requires_grad = True

    def _freeze_all_but_last_blocks(self, trainable_blocks: int | None) -> None:
        assert trainable_blocks is not None  # FreezeSpec guarantees this for mode="partial"
        groups = self.param_groups()
        block_names = [name for name in groups if name.startswith("blocks.")]
        if trainable_blocks > len(block_names):
            raise ConfigError(
                f"freeze.trainable_blocks: {trainable_blocks} exceeds the "
                f"{len(block_names)} block group(s) this backbone declares",
                hint="lower trainable_blocks, or check the backbone's param_groups()",
            )
        trainable_names = set(block_names[len(block_names) - trainable_blocks :])
        trainable_ids = {id(p) for name in trainable_names for p in groups[name]}
        for p in self.parameters():
            p.requires_grad = id(p) in trainable_ids
