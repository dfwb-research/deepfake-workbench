"""Three samplers over :mod:`dfwb.data` datasets: label-balanced, source-balanced, and grouped by
video for eval.

Every sampler here draws a fresh, independent sequence per call to :meth:`set_epoch`: the epoch is
folded into the seed the same way :class:`~dfwb.data.dataset.ClipDataset` folds it into a train
clip's window, so re-running an epoch (e.g. after resuming) draws exactly the same indices again.
:class:`VideoGrouped` is the exception -- it needs no randomness at all, since eval always wants
the same, fixed grouping.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from dfwb.core.errors import ConfigError
from dfwb.data.dataset import ClipDataset, MultiSource
from dfwb.data.index import VideoIndex

__all__ = ["SourceBalanced", "VideoGrouped", "VideoLabelBalanced"]


def _seed_for(seed: int, epoch: int) -> int:
    """A 64-bit int, deterministic in ``(seed, epoch)``, for seeding one epoch's draws.

    A plain ``hash()`` is not usable here: Python randomises string hashing per process unless
    ``PYTHONHASHSEED`` is pinned, so the same ``(seed, epoch)`` would draw differently run to run.
    """
    digest = hashlib.sha256(f"{seed}:{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _clips_per_item(dataset: ClipDataset) -> int:
    """How many clips one video contributes to ``dataset``, inferred from its own length rather
    than reaching into its private state: ``len(dataset)`` is always an exact multiple of the
    number of videos it was built from."""
    n_items = len(dataset.index.items)
    return len(dataset) // n_items if n_items else 0


def _labels_of(source: VideoIndex | ClipDataset) -> list[int]:
    """One label per *sample* position of ``source``: a bare :class:`VideoIndex` is one sample
    per item; a :class:`ClipDataset` repeats each video's label once per clip it contributes, in
    the same order :meth:`ClipDataset.__getitem__` lays its samples out (video-major,
    clip-minor)."""
    if isinstance(source, VideoIndex):
        return [item.label for item in source.items]
    clips_per_item = _clips_per_item(source)
    return [item.label for item in source.index.items for _ in range(clips_per_item)]


class VideoLabelBalanced(Sampler[int]):
    """Draws sample indices of ``source`` so both labels are equally likely, regardless of how
    imbalanced ``source`` actually is.

    ``source`` is either a bare :class:`~dfwb.data.index.VideoIndex` (one sample per item) or a
    :class:`~dfwb.data.dataset.ClipDataset` (one sample per clip). Each label's total sampling
    weight is split evenly across its members (``1 / count[label]`` each), so
    ``torch.multinomial`` with those weights draws either label with probability 0.5 per draw --
    the standard inverse-frequency balancing a two-label training set needs. One epoch draws
    ``len(source)`` indices, with replacement (the only way to actually balance an imbalanced set).
    """

    def __init__(self, source: VideoIndex | ClipDataset, *, seed: int) -> None:
        self._labels = _labels_of(source)
        counts = Counter(self._labels)
        self._weights = torch.tensor(
            [1.0 / counts[label] for label in self._labels], dtype=torch.double
        )
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Every draw made after this call comes from ``epoch``'s own seeded sequence."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(_seed_for(self.seed, self._epoch))
        draws = torch.multinomial(
            self._weights, len(self._labels), replacement=True, generator=generator
        )
        return iter(draws.tolist())


class SourceBalanced(Sampler[int]):
    """Draws sample indices of a :class:`~dfwb.data.dataset.MultiSource` in proportion to
    ``multi.weights`` -- not to how many samples each source actually has.

    Each draw first picks a source (weighted by ``multi.weights``, already normalised to sum to
    1), then a uniformly random local index within that source, mapped to the source's slice of
    the concatenation. One epoch draws ``len(multi)`` indices, with replacement.
    """

    def __init__(self, multi: MultiSource, *, seed: int) -> None:
        self.multi = multi
        self.seed = seed
        self._epoch = 0
        self._lengths = [len(dataset) for dataset in multi.datasets]
        self._offsets = list(itertools.accumulate(self._lengths, initial=0))[:-1]

    def set_epoch(self, epoch: int) -> None:
        """Every draw made after this call comes from ``epoch``'s own seeded sequence."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.multi)

    def __iter__(self) -> Iterator[int]:
        n = len(self.multi)
        generator = torch.Generator().manual_seed(_seed_for(self.seed, self._epoch))
        weights = torch.tensor(self.multi.weights, dtype=torch.double)
        source_draws = torch.multinomial(weights, n, replacement=True, generator=generator)
        lengths = torch.tensor(self._lengths)
        offsets = torch.tensor(self._offsets)
        local = (torch.rand(n, generator=generator) * lengths[source_draws]).long()
        indices = offsets[source_draws] + local
        return iter(indices.tolist())


class VideoGrouped(Sampler[list[int]]):
    """A batch sampler for eval: every clip of one video lands in the same batch, so a video's
    clips are never split across two calls of a scoring loop.

    Deterministic and unseeded -- eval always wants the same, fixed grouping, never a shuffled
    one. Videos are packed into batches in ``dataset`` order, greedily: a video's whole run of
    clips is added to the current batch unless that would push it past ``batch_size``, in which
    case the current batch closes first. A single video with more clips than ``batch_size`` still
    gets one batch of its own, larger than ``batch_size`` -- its clips are never split to fit.

    Raises:
        ConfigError: ``batch_size`` is not a positive int.
    """

    def __init__(self, dataset: ClipDataset, batch_size: int) -> None:
        if batch_size < 1:
            raise ConfigError(
                f"VideoGrouped: batch_size must be positive, got {batch_size}",
                hint="pass how many clips one eval batch may hold, at least 1",
            )
        self.dataset = dataset
        self.batch_size = batch_size
        clips_per_item = _clips_per_item(dataset)
        self._batches = self._build_batches(len(dataset.index.items), clips_per_item, batch_size)

    @staticmethod
    def _build_batches(n_items: int, clips_per_item: int, batch_size: int) -> list[list[int]]:
        batches: list[list[int]] = []
        batch: list[int] = []
        for item_index in range(n_items):
            start = item_index * clips_per_item
            group = list(range(start, start + clips_per_item))
            if batch and len(batch) + len(group) > batch_size:
                batches.append(batch)
                batch = []
            batch.extend(group)
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)
