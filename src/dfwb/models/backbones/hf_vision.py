"""``hf-vision``: any Hugging Face vision transformer, via ``AutoModel``."""

from __future__ import annotations

from typing import Any, Literal, cast

from torch import Tensor, nn

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError
from dfwb.models.backbone import Backbone, BackboneOutput, FreezeSpec
from dfwb.models.backbones._blocks import block_container, container_prefixes, partition_by_blocks

__all__ = ["HFVisionBackbone"]

# Older BERT-style encoders name their block list ``encoder.layer`` (singular); CLIP-style ones
# name it ``encoder.layers``; some current vision transformers flatten it to a top-level
# ``layers`` with no ``encoder`` wrapper at all. Tried in this order, first match wins.
_BLOCK_ATTRS: tuple[str, ...] = ("encoder.layer", "encoder.layers", "layers")

_DEFAULT_FREEZE = FreezeSpec()


class HFVisionBackbone(Backbone):
    """Any Hugging Face vision transformer (CLIP, DINOv2, SigLIP 2, plain ViT, ...) loaded
    through ``AutoModel``. ``model`` is a hub id or a local directory. ``.vision_model`` is used
    when the loaded model wraps one, as full multimodal checkpoints (CLIP) do.
    """

    kind = "image"

    def __init__(
        self,
        model: str,
        pretrained: bool = True,
        pool: Literal["cls", "mean"] = "cls",
        freeze: FreezeSpec = _DEFAULT_FREEZE,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        raw: Any
        if pretrained:
            raw = AutoModel.from_pretrained(model)
        else:
            raw = AutoModel.from_config(  # type: ignore[no-untyped-call]
                AutoConfig.from_pretrained(model)
            )
        vision_config = getattr(raw.config, "vision_config", raw.config)
        self.out_dim = vision_config.hidden_size
        self.pool: Literal["cls", "mean"] = pool
        self.model = getattr(raw, "vision_model", raw)
        self.native_input = _resolve_native_input(model)
        self.apply_freeze(freeze)

    def forward(self, x: Tensor) -> BackboneOutput:
        out = self.model(pixel_values=x)
        last_hidden = out.last_hidden_state
        if self.pool == "cls":
            pooler_output = getattr(out, "pooler_output", None)
            pooled = pooler_output if pooler_output is not None else last_hidden[:, 0]
        else:
            pooled = last_hidden.mean(dim=1)
        return BackboneOutput(pooled=pooled, tokens=last_hidden)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        container = block_container(self.model, _BLOCK_ATTRS)
        return partition_by_blocks(self.model, container_prefixes(container))

    def lora_module(self) -> nn.Module | None:
        return cast(nn.Module, self.model)


def _resolve_native_input(model: str) -> InputSpec:
    from transformers import AutoImageProcessor

    try:
        processor: Any = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            model
        )
    except OSError as exc:
        raise ConfigError(
            f"model: no image processor found for {model!r}",
            hint=(
                "hf-vision needs an AutoImageProcessor next to the model "
                "(a preprocessor_config.json); save one alongside the model, or pick a hub id "
                "that ships one"
            ),
        ) from exc
    size = processor.size
    height, width = size.get("height"), size.get("width")
    if height is None or width is None:
        edge = size.get("shortest_edge") or size.get("longest_edge")
        height = width = edge
    return InputSpec(
        size=(height, width),
        mean=tuple(processor.image_mean),
        std=tuple(processor.image_std),
        value_range=(0.0, 1.0),
    )
