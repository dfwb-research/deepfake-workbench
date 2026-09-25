"""Shared block-group partitioning for backbones whose blocks live in a named ``nn.ModuleList``.

Both the ``timm`` and ``hf-vision`` backbones use this: ``partial`` freezing always goes by a
backbone's own block order (from its own module tree), never a name pattern over individual
parameters.
"""

from __future__ import annotations

from torch import nn

__all__ = ["block_container", "partition_by_blocks"]


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


def partition_by_blocks(
    module: nn.Module, container: tuple[str, nn.Module] | None
) -> dict[str, list[nn.Parameter]]:
    """Partition every parameter of ``module`` into ``blocks.<i>``, ``embed`` and ``norm``.

    ``blocks.<i>`` holds the parameters of the i-th child of ``container`` (in the backbone's own
    order). ``embed`` holds the non-block parameters that come before the first block in
    ``named_parameters()`` order; ``norm`` holds the ones that come after the last block. Every
    parameter of ``module`` ends up in exactly one group.
    """
    prefixes: list[str] = []
    if container is not None:
        path, blocks = container
        prefixes = [f"{path}.{i}" for i in range(len(list(blocks.children())))]
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
    result: dict[str, list[nn.Parameter]] = dict(groups)
    if embed:
        result["embed"] = embed
    if norm:
        result["norm"] = norm
    return result


def _block_index(name: str, prefixes: list[str]) -> int | None:
    for index, prefix in enumerate(prefixes):
        if name == prefix or name.startswith(f"{prefix}."):
            return index
    return None
