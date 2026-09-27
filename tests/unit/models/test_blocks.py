"""``_blocks``: block discovery and parameter partitioning, in isolation.

Exercised directly (rather than only through the timm/hf-vision backbones) so the fallthrough and
empty-group branches -- no attr path matches, an empty container, no embed or norm parameters,
the numbered ``layer1..layerN`` fallback -- are covered without needing a real timm or
transformers model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from torch import nn

from dfwb.models.backbones._blocks import (
    block_container,
    container_prefixes,
    numbered_blocks,
    partition_by_blocks,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)


class _WithEmbedAndNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(2, 2)
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.norm = nn.LayerNorm(2)


class _BlocksOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])


class _EmptyThenNonEmpty(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()  # present but empty: block_container must skip it
        self.layers = nn.ModuleList([_Block()])


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_Block(), _Block()])


class _Wrapped(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()


class _ResnetLike(nn.Module):
    """Numbered top-level ``layer1..layer3`` attributes, ResNet-style: no single container
    attribute collects them, and each ``layerN`` is a whole block, not a container to split
    further."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.layer1 = _Block()
        self.layer2 = _Block()
        self.layer3 = _Block()


# --------------------------------------------------------------------------- block_container


def test_block_container_returns_none_when_no_path_matches():
    assert block_container(nn.Linear(2, 2), ("blocks", "layers")) is None


def test_block_container_skips_an_empty_container_for_the_next_path():
    module = _EmptyThenNonEmpty()
    result = block_container(module, ("blocks", "layers"))
    assert result is not None
    path, target = result
    assert path == "layers"
    assert target is module.layers


def test_block_container_resolves_a_dotted_path():
    module = _Wrapped()
    assert block_container(module, ("encoder.layer",)) == ("encoder.layer", module.encoder.layer)


# --------------------------------------------------------------------------- container_prefixes


def test_container_prefixes_of_none_is_empty():
    assert container_prefixes(None) == []


def test_container_prefixes_names_each_child_by_index():
    module = _WithEmbedAndNorm()
    container = block_container(module, ("blocks",))
    assert container_prefixes(container) == ["blocks.0", "blocks.1"]


# --------------------------------------------------------------------------- numbered_blocks


def test_numbered_blocks_finds_consecutive_numbered_attributes_in_order():
    module = _ResnetLike()
    assert numbered_blocks(module) == ["layer1", "layer2", "layer3"]


def test_numbered_blocks_is_empty_when_there_is_no_layer1():
    assert numbered_blocks(_BlocksOnly()) == []


def test_numbered_blocks_honours_a_different_stem():
    class _Staged(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage1 = _Block()
            self.stage2 = _Block()

    assert numbered_blocks(_Staged(), stem="stage") == ["stage1", "stage2"]


# --------------------------------------------------------------------------- partition_by_blocks


def test_partition_with_no_prefixes_puts_everything_in_embed():
    module = nn.Linear(2, 2)
    groups = partition_by_blocks(module, [])
    assert set(groups) == {"embed"}
    assert {id(p) for p in groups["embed"]} == {id(p) for p in module.parameters()}


def test_partition_separates_embed_blocks_and_norm():
    module = _WithEmbedAndNorm()
    container = block_container(module, ("blocks",))
    groups = partition_by_blocks(module, container_prefixes(container))
    assert set(groups) == {"embed", "blocks.0", "blocks.1", "norm"}
    assert {id(p) for p in groups["embed"]} == {id(p) for p in module.embed.parameters()}
    assert {id(p) for p in groups["blocks.0"]} == {id(p) for p in module.blocks[0].parameters()}
    assert {id(p) for p in groups["blocks.1"]} == {id(p) for p in module.blocks[1].parameters()}
    assert {id(p) for p in groups["norm"]} == {id(p) for p in module.norm.parameters()}
    all_ids = [id(p) for params in groups.values() for p in params]
    assert len(all_ids) == len(set(all_ids))
    assert sorted(all_ids) == sorted(id(p) for p in module.parameters())


def test_partition_omits_embed_and_norm_keys_when_there_are_none():
    module = _BlocksOnly()
    container = block_container(module, ("blocks",))
    groups = partition_by_blocks(module, container_prefixes(container))
    assert set(groups) == {"blocks.0", "blocks.1"}


def test_partition_keys_are_ordered_embed_then_blocks_then_norm():
    module = _WithEmbedAndNorm()
    container = block_container(module, ("blocks",))
    groups = partition_by_blocks(module, container_prefixes(container))
    assert list(groups) == ["embed", "blocks.0", "blocks.1", "norm"]


def test_partition_with_numbered_blocks_treats_each_layer_as_one_whole_block():
    module = _ResnetLike()
    groups = partition_by_blocks(module, numbered_blocks(module))
    assert list(groups) == ["embed", "blocks.0", "blocks.1", "blocks.2"]
    assert {id(p) for p in groups["embed"]} == {id(p) for p in module.stem.parameters()}
    assert {id(p) for p in groups["blocks.0"]} == {id(p) for p in module.layer1.parameters()}
    assert {id(p) for p in groups["blocks.1"]} == {id(p) for p in module.layer2.parameters()}
    assert {id(p) for p in groups["blocks.2"]} == {id(p) for p in module.layer3.parameters()}
    all_ids = [id(p) for params in groups.values() for p in params]
    assert len(all_ids) == len(set(all_ids))
    assert sorted(all_ids) == sorted(id(p) for p in module.parameters())
