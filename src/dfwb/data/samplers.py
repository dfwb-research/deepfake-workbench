"""Samplers over :mod:`dfwb.data` datasets: label-balanced, source-balanced, grouped by video for
eval, and grouped by real/fake pair for paired training.

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
from dfwb.data.paired import PairedClipDataset

__all__ = ["PairGrouped", "SourceBalanced", "VideoGrouped", "VideoLabelBalanced"]


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


def _labels_of(source: VideoIndex | ClipDataset | MultiSource) -> list[int]:
    """One label per *sample* position of ``source``: a bare :class:`VideoIndex` is one sample
    per item; a :class:`ClipDataset` repeats each video's label once per clip it contributes, in
    the same order :meth:`ClipDataset.__getitem__` lays its samples out (video-major,
    clip-minor); a :class:`MultiSource` is its datasets' labels, concatenated in order.

    Raises:
        ConfigError: a :class:`MultiSource` holds paired data, which is already balanced one real
            and one fake per pair, and is batched pair by pair instead (:class:`PairGrouped`).
    """
    if isinstance(source, VideoIndex):
        return [item.label for item in source.items]
    if isinstance(source, MultiSource):
        labels: list[int] = []
        for dataset in source.datasets:
            if not isinstance(dataset, ClipDataset):
                raise ConfigError(
                    "label balancing cannot be applied to paired data",
                    hint="pairs are already one real and one fake each; drop the balance setting",
                )
            labels.extend(_labels_of(dataset))
        return labels
    clips_per_item = _clips_per_item(source)
    return [item.label for item in source.index.items for _ in range(clips_per_item)]


def _check_batch_size(name: str, batch_size: int) -> None:
    if batch_size < 1:
        raise ConfigError(
            f"{name}: batch_size must be positive, got {batch_size}",
            hint="pass how many clips one batch may hold, at least 1",
        )


def _pack(groups: list[list[int]], batch_size: int) -> list[list[int]]:
    """Greedily packs whole ``groups`` into batches of at most ``batch_size`` indices, in order: a
    group that does not fit closes the current batch first, and a group larger than
    ``batch_size`` gets a batch of its own -- a group is never split."""
    batches: list[list[int]] = []
    batch: list[int] = []
    for group in groups:
        if batch and len(batch) + len(group) > batch_size:
            batches.append(batch)
            batch = []
        batch.extend(group)
    if batch:
        batches.append(batch)
    return batches


class VideoLabelBalanced(Sampler[int]):
    """Draws sample indices of ``source`` so both labels are equally likely, regardless of how
    imbalanced ``source`` actually is.

    ``source`` is a bare :class:`~dfwb.data.index.VideoIndex` (one sample per item), a
    :class:`~dfwb.data.dataset.ClipDataset` (one sample per clip), or a
    :class:`~dfwb.data.dataset.MultiSource` of clip datasets. Each label's total sampling
    weight is split evenly across its members (``1 / count[label]`` each), so
    ``torch.multinomial`` with those weights draws either label with probability 0.5 per draw --
    the standard inverse-frequency balancing a two-label training set needs. One epoch draws
    ``len(source)`` indices, with replacement (the only way to actually balance an imbalanced set).
    """

    def __init__(self, source: VideoIndex | ClipDataset | MultiSource, *, seed: int) -> None:
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
        _check_batch_size("VideoGrouped", batch_size)
        self.dataset = dataset
        self.batch_size = batch_size
        clips_per_item = _clips_per_item(dataset)
        groups = [
            list(range(item_index * clips_per_item, (item_index + 1) * clips_per_item))
            for item_index in range(len(dataset.index.items))
        ]
        self._batches = _pack(groups, batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


class PairGrouped(Sampler[list[int]]):
    """A batch sampler for paired training: a pair's real rows and fake rows always land in the
    same batch, so a pairwise loss always finds a row's partner next to it.

    ``dataset`` is a :class:`~dfwb.data.paired.PairedClipDataset`, or a
    :class:`~dfwb.data.dataset.MultiSource` whose every source is one (each source's pairs are
    offset to its slice of the concatenation). Each epoch shuffles the order of the pairs, seeded
    by ``(seed, epoch)``, then fills batches pair by pair: a pair that would push the current
    batch past ``batch_size`` closes it first, and a pair larger than ``batch_size`` on its own
    gets a batch of its own, never split.

    Raises:
        ConfigError: ``batch_size`` is not positive, or a source is not paired data.
    """

    def __init__(
        self, dataset: PairedClipDataset | MultiSource, batch_size: int, *, seed: int
    ) -> None:
        _check_batch_size("PairGrouped", batch_size)
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0
        self._groups = self._collect_groups(dataset)

    @staticmethod
    def _collect_groups(dataset: PairedClipDataset | MultiSource) -> list[list[int]]:
        sources = dataset.datasets if isinstance(dataset, MultiSource) else [dataset]
        groups: list[list[int]] = []
        offset = 0
        for source in sources:
            if not isinstance(source, PairedClipDataset):
                raise ConfigError(
                    "PairGrouped: every source must be paired data (a PairedClipDataset)",
                    hint="build the training sources with pairs enabled, or use another sampler",
                )
            groups.extend([i + offset for i in group] for group in source.pair_groups())
            offset += len(source)
        return groups

    def set_epoch(self, epoch: int) -> None:
        """Every order drawn after this call comes from ``epoch``'s own seeded sequence."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(_seed_for(self.seed, self._epoch))
        order = torch.randperm(len(self._groups), generator=generator).tolist()
        return iter(_pack([self._groups[i] for i in order], self.batch_size))

    def __len__(self) -> int:
        """The batch count of this sampler's unshuffled order. Every pair of one run has the same
        number of rows (both sides share one clip spec), so shuffling never changes it."""
        return len(_pack(self._groups, self.batch_size))
