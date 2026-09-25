"""The optional stem hook: puts a registered ``layers`` component in front of a backbone.

This is the generic mechanism that makes forensic front-ends (high-pass filters, learned residual
extractors, ...) usable ahead of any backbone without either depending on the utility package that
provides them or writing research code into this framework.
"""

from __future__ import annotations

from torch import nn

from dfwb.core.config.schema import ComponentSpec
from dfwb.core.errors import ContractError
from dfwb.core.plugins import api

__all__ = ["build_stem"]


def build_stem(spec: ComponentSpec, in_channels: int = 3) -> nn.Module:
    """Build ``layers/<spec.name>`` and adapt its output back to 3 channels if needed.

    ``in_channels`` is passed to the layer unless ``spec`` already sets it. The built layer must
    expose an ``out_channels: int`` attribute; when that differs from 3, a 1x1 convolution is
    appended so the result is always a standard 3-channel image, ready for any backbone.

    Raises:
        ContractError: The built layer has no ``out_channels`` attribute.
    """
    params: dict[str, object] = {"in_channels": in_channels, **spec.params}
    layer = api.layers.build(spec.name, **params)
    if not isinstance(layer, nn.Module):
        raise ContractError(
            f"layers/{spec.name}: must build an nn.Module",
            hint="a stem layer's registry target must return an nn.Module",
        )
    out_channels = getattr(layer, "out_channels", None)
    if not isinstance(out_channels, int):
        raise ContractError(
            f"layers/{spec.name}: must expose an int out_channels attribute",
            hint="a stem layer sets self.out_channels in its own __init__",
        )
    if out_channels != 3:
        return nn.Sequential(layer, nn.Conv2d(out_channels, 3, kernel_size=1))
    return layer
