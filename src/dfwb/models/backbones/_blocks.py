"""Shared block-group discovery and partitioning for the ``timm`` and ``hf-vision`` backbones.

``partial`` freezing always goes by a backbone's own block order (from its own module tree), never
a name pattern over individual parameters. Blocks are found one of two ways: most backbones
collect them into a single named container (``blocks``, ``encoder.layers``, ...), where each child
of the container is one block; the ResNet family instead exposes them as separate, numbered
top-level attributes (``layer1``, ``layer2``, ...), where each attribute is one whole block in its
own right.
"""

from __future__ import annotations

from torch import nn

__all__ = ["block_container", "container_prefixes", "numbered_blocks", "partition_by_blocks"]


def block_container(module: nn.Module, attr_paths: tuple[str, ...]) -> tuple[str, nn.Module] | None:
    """The first of ``attr_paths`` (dotted, e.g. ``"encoder.layers"``) that resolves to a
    non-empty submodule of ``module``, paired with that dotted path; ``None`` if none do."""
    for path in attr_paths:
        target: nn.Module | None = module
        for part in path.split("."):
            if target is None:
                break
            candidate = getattr(target, part, None)
            target = candidate if isinstance(candidate, nn.Module) else None
        if target is not None and len(list(target.children())) > 0:
            return path, target
    return None


def container_prefixes(container: tuple[str, nn.Module] | None) -> list[str]:
    """``"<path>.0"``, ``"<path>.1"``, ... for each child of ``container``; ``[]`` when
    ``container`` is ``None``."""
    if container is None:
        return []
    path, blocks = container
    return [f"{path}.{i}" for i in range(len(list(blocks.children())))]


def numbered_blocks(module: nn.Module, stem: str = "layer") -> list[str]:
    """Numbered top-level attributes ``f"{stem}{i}"`` for consecutive ``i`` starting at 1 (the
    ResNet family's ``layer1..layerN``), in that order, each one a whole block in its own right
    rather than a container of further blocks."""
    prefixes: list[str] = []
    index = 1
    while isinstance(getattr(module, f"{stem}{index}", None), nn.Module):
        prefixes.append(f"{stem}{index}")
        index += 1
    return prefixes


def partition_by_blocks(module: nn.Module, prefixes: list[str]) -> dict[str, list[nn.Parameter]]:
    """Partition every parameter of ``module`` into ``embed``, ``blocks.<i>`` and ``norm``, in
    that key order.

    ``blocks.<i>`` holds the parameters under ``prefixes[i]`` (in the backbone's own order).
    ``embed`` holds the non-block parameters that come before the first block in
    ``named_parameters()`` order; ``norm`` holds the ones that come after the last block. Every
    parameter of ``module`` ends up in exactly one group.
    """
    groups: dict[str, list[nn.Parameter]] = {f"blocks.{i}": [] for i in range(len(prefixes))}
    embed: list[nn.Parameter] = []
    norm: list[nn.Parameter] = []
    seen_block = False
    for name, param in module.named_parameters():
        index = _block_index(name, prefixes)
        if index is not None:
            groups[f"blocks.{index}"].append(param)
            seen_block = True
        elif seen_block:
            norm.append(param)
        else:
            embed.append(param)
    result: dict[str, list[nn.Parameter]] = {}
    if embed:
        result["embed"] = embed
    result.update(groups)
    if norm:
        result["norm"] = norm
    return result


def _block_index(name: str, prefixes: list[str]) -> int | None:
    for index, prefix in enumerate(prefixes):
        if name == prefix or name.startswith(f"{prefix}."):
            return index
    return None
