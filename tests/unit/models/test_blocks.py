"""``_blocks``: block-container resolution and parameter partitioning, in isolation.

Exercised directly (rather than only through the timm/hf-vision backbones) so the fallthrough and
empty-group branches -- no attr path matches, an empty container, no embed or norm parameters --
are covered without needing a real timm or transformers model.
"""

from __future__ import annotations

from torch import nn

from dfwb.models.backbones._blocks import block_container, partition_by_blocks


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


# --------------------------------------------------------------------------- partition_by_blocks


def test_partition_with_no_container_puts_everything_in_embed():
    module = nn.Linear(2, 2)
    groups = partition_by_blocks(module, None)
    assert set(groups) == {"embed"}
    assert {id(p) for p in groups["embed"]} == {id(p) for p in module.parameters()}


def test_partition_separates_embed_blocks_and_norm():
    module = _WithEmbedAndNorm()
    container = block_container(module, ("blocks",))
    groups = partition_by_blocks(module, container)
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
    groups = partition_by_blocks(module, container)
    assert set(groups) == {"blocks.0", "blocks.1"}
