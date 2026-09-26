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


def collate_clips(samples: Sequence[ClipSample | None]) -> ClipBatch:
    """Stacks ``samples`` (one video's clip each) into one :class:`ClipBatch`, dropping any
    ``None`` first -- :class:`~dfwb.data.dataset.ClipDataset` returns one for a sample whose every
    stored frame is corrupt and cannot be repeated from a neighbour (only when it tolerates a
    corrupt frame at all: a validation source, never training or scoring).

    ``clips`` stacks to ``[B, T, C, H, W]`` and ``frame_indices`` to ``[B, T]``; every other
    per-sample field becomes a plain ``[B]``-length list or tensor, in ``samples`` order.
    ``labels`` is a ``[B]`` tensor only when every sample's ``label`` is not ``None`` -- a batch
    with even one unlabelled sample (e.g. scoring, where the true label may be unknown) gets
    ``labels=None`` rather than a tensor with a hole in it. ``extras`` is collated per key by
    :func:`_collate_value`; a batch that dropped one or more ``None`` samples also gets
    ``extras["dfwb/videos_skipped"]``, the count dropped (never present, rather than ``0``, when
    nothing was).

    Raises:
        ValueError: ``samples`` holds no usable sample -- either it was empty, or every one of
            them was ``None`` -- so there is no batch shape to infer ``clips`` from.
    """
    kept = [sample for sample in samples if sample is not None]
    if not kept:
        raise ValueError("collate_clips: samples must not be empty")
    skipped = len(samples) - len(kept)

    clips = torch.stack([sample.clip for sample in kept])
    keys = [sample.key for sample in kept]
    dataset_ids = [sample.dataset for sample in kept]
    compressions = [sample.compression for sample in kept]
    clip_index = torch.tensor([sample.clip_index for sample in kept], dtype=torch.long)
    frame_indices = torch.tensor([sample.frame_indices for sample in kept], dtype=torch.long)

    labels: Tensor | None = None
    if all(sample.label is not None for sample in kept):
        labels = torch.tensor([sample.label for sample in kept], dtype=torch.long)

    extras = _collate_extras([sample.extras for sample in kept])
    if skipped:
        extras["dfwb/videos_skipped"] = skipped

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
