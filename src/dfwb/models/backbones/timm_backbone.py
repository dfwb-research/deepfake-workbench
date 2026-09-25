"""``timm``: any timm image classification backbone, without its classifier head."""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor, nn

from dfwb.core.detector import InputSpec
from dfwb.models.backbone import Backbone, BackboneOutput, FreezeSpec
from dfwb.models.backbones._blocks import block_container, partition_by_blocks

__all__ = ["TimmBackbone"]

_BLOCK_ATTRS: tuple[str, ...] = ("blocks", "stages", "layers")

_DEFAULT_FREEZE = FreezeSpec()


class TimmBackbone(Backbone):
    """Any ``timm.create_model`` image backbone, pooled the model's own way.

    ``out_dim`` and ``native_input`` come straight from the model timm builds
    (``model.num_features`` and ``timm.data.resolve_data_config``), so any timm image model name
    works, not only ones this framework knows about by name. ``drop_path`` is the model's own
    stochastic depth rate.
    """

    kind = "image"

    def __init__(
        self,
        model: str,
        pretrained: bool = True,
        freeze: FreezeSpec = _DEFAULT_FREEZE,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        import timm
        from timm.data.config import resolve_data_config

        backbone: Any = timm.create_model(
            model, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path
        )
        self.model = backbone
        self.out_dim = backbone.num_features
        data_config: dict[str, Any] = resolve_data_config(  # type: ignore[no-untyped-call]
            {}, model=backbone
        )
        _, height, width = data_config["input_size"]
        self.native_input = InputSpec(
            size=(height, width),
            mean=tuple(data_config["mean"]),
            std=tuple(data_config["std"]),
            value_range=(0.0, 1.0),
        )
        self.apply_freeze(freeze)

    def forward(self, x: Tensor) -> BackboneOutput:
        features = self.model.forward_features(x)
        pooled = self.model.forward_head(features, pre_logits=True)
        tokens = features if features.ndim == 3 else None
        return BackboneOutput(pooled=pooled, tokens=tokens)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return partition_by_blocks(self.model, block_container(self.model, _BLOCK_ATTRS))

    def lora_module(self) -> nn.Module | None:
        return cast(nn.Module, self.model)
