"""Temporal pools: ``mean``, ``max`` and the learned attention pool, all ``[B,T,D] -> [B,D]``."""

from __future__ import annotations

import pytest
import torch

from dfwb.core.plugins import api, get_registry
from dfwb.models.pools import AttentionPool, MaxPool, MeanPool, TemporalPool

_POOL_CLASSES = [MeanPool, MaxPool, AttentionPool]


@pytest.mark.parametrize("pool_cls", _POOL_CLASSES)
def test_pool_reduces_the_time_dimension(pool_cls):
    pool = pool_cls(dim=8)
    assert isinstance(pool, TemporalPool)
    out = pool(torch.rand(3, 5, 8))
    assert out.shape == (3, 8)


@pytest.mark.parametrize("pool_cls", _POOL_CLASSES)
@pytest.mark.parametrize("t", [1, 8])
def test_pool_handles_a_single_frame_and_many_frames(pool_cls, t):
    pool = pool_cls(dim=4)
    out = pool(torch.rand(2, t, 4))
    assert out.shape == (2, 4)


def test_mean_pool_matches_torch_mean():
    pool = MeanPool(dim=4)
    x = torch.rand(2, 6, 4)
    assert torch.allclose(pool(x), x.mean(dim=1))


def test_max_pool_matches_torch_max():
    pool = MaxPool(dim=4)
    x = torch.rand(2, 6, 4)
    assert torch.allclose(pool(x), x.max(dim=1).values)


def test_attention_pool_weights_sum_to_one_over_time():
    pool = AttentionPool(dim=4)
    x = torch.rand(2, 5, 4)
    weights = torch.softmax(pool.score(x), dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1), atol=1e-6)


def test_attention_pool_is_a_convex_combination_of_the_frames():
    pool = AttentionPool(dim=3)
    x = torch.rand(2, 4, 3)
    out = pool(x)
    assert bool((out.min() >= x.min() - 1e-5) and (out.max() <= x.max() + 1e-5))


def test_attention_pool_takes_no_user_params_besides_dim():
    with pytest.raises(TypeError):
        AttentionPool(dim=4, extra=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("key", "cls"), [("mean", MeanPool), ("max", MaxPool), ("attention", AttentionPool)]
)
def test_registered_as_a_builtin_temporal_pool(key, cls):
    entry = get_registry("temporal_pools").entry(key)
    assert entry.provider == "dfwb"
    assert entry.requires == ("torch",)
    pool = api.temporal_pools.build(key, dim=4)
    assert isinstance(pool, cls)
