"""``collate_clips``: turns a batch of :class:`~dfwb.data.dataset.ClipSample` into a
:class:`~dfwb.core.detector.ClipBatch` (C4).

A ``DataLoader``'s default collate cannot be used here: ``labels`` must come out as ``None``
(not a batch of ``None``s) unless every sample actually carries one, and ``extras`` is
plugin-owned, so it needs its own, per-key rule rather than one fixed schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from dfwb.core.detector import ClipBatch
from dfwb.data.dataset import ClipSample

__all__ = ["collate_clips"]


def _collate_value(values: Sequence[Any]) -> Any:
    """One extras key's values, collated across a batch: stacked if every value is a tensor,
    a tensor of the right dtype if every value is a bool or an int (checked in that order, since
    ``bool`` is a subclass of ``int``), or, when the values are not uniform in one of those ways,
    just the plain list -- exactly what a plugin-owned value with no fixed shape needs."""
    if all(isinstance(value, Tensor) for value in values):
        return torch.stack(list(values))
    if all(isinstance(value, bool) for value in values):
        return torch.tensor(list(values), dtype=torch.bool)
    if all(isinstance(value, int) for value in values):
        return torch.tensor(list(values), dtype=torch.long)
    return list(values)


def _collate_extras(extras: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys: set[str] = set()
    for sample_extras in extras:
        keys.update(sample_extras)
    return {
        key: _collate_value([sample_extras.get(key) for sample_extras in extras])
        for key in sorted(keys)
    }


def collate_clips(samples: Sequence[ClipSample]) -> ClipBatch:
    """Stacks ``samples`` (one video's clip each) into one :class:`ClipBatch`.

    ``clips`` stacks to ``[B, T, C, H, W]`` and ``frame_indices`` to ``[B, T]``; every other
    per-sample field becomes a plain ``[B]``-length list or tensor, in ``samples`` order.
    ``labels`` is a ``[B]`` tensor only when every sample's ``label`` is not ``None`` -- a batch
    with even one unlabelled sample (e.g. scoring, where the true label may be unknown) gets
    ``labels=None`` rather than a tensor with a hole in it. ``extras`` is collated per key by
    :func:`_collate_value`.

    Raises:
        ValueError: ``samples`` is empty -- there is no batch shape to infer ``clips`` from.
    """
    if not samples:
        raise ValueError("collate_clips: samples must not be empty")

    clips = torch.stack([sample.clip for sample in samples])
    keys = [sample.key for sample in samples]
    dataset_ids = [sample.dataset for sample in samples]
    compressions = [sample.compression for sample in samples]
    clip_index = torch.tensor([sample.clip_index for sample in samples], dtype=torch.long)
    frame_indices = torch.tensor([sample.frame_indices for sample in samples], dtype=torch.long)

    labels: Tensor | None = None
    if all(sample.label is not None for sample in samples):
        labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)

    extras = _collate_extras([sample.extras for sample in samples])

    return ClipBatch(
        clips=clips,
        keys=keys,
        dataset_ids=dataset_ids,
        compressions=compressions,
        clip_index=clip_index,
        frame_indices=frame_indices,
        labels=labels,
        extras=extras,
    )
