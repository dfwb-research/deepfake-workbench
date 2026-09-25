"""``PairedClipDataset``: real/fake clip pairs for pairwise or contrastive plugins.

**Shape chosen: a flat sequence, not one tuple per pair.** ``PairedClipDataset`` wraps two plain
:class:`~dfwb.data.dataset.ClipDataset` instances -- one over the pairs' real items, one over
their fake items, in the same order -- concatenated exactly the way
:class:`~dfwb.data.dataset.MultiSource` already concatenates several sources, and for the same
reason: ``__getitem__`` keeps returning one :class:`~dfwb.data.dataset.ClipSample`, so
:func:`~dfwb.data.collate.collate_clips` collates a batch of them completely unchanged. Every
sample is tagged ``extras["dfwb/pair_id"]`` with its pair's position, so a pairwise loss finds a
row's partner by matching that value within the collated batch -- rows do not need to sit next to
each other for that to work, only to agree on the id.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from torch import Tensor
from torch.utils.data import Dataset

from dfwb.data.clips import ClipSpec
from dfwb.data.dataset import ClipDataset, ClipSample, ClipTransform
from dfwb.data.index import VideoIndex, VideoItem

__all__ = ["PairedClipDataset"]


def _by_key(index: VideoIndex) -> dict[str, VideoItem]:
    """The first item seen for each key. ``pairs`` names real/fake keys with no compression of
    their own (``pairs.jsonl`` predates any particular store), so pairing is by key alone; when a
    key has more than one item in ``index`` (several compressions), the first -- in ``index``'s
    own, already-deterministic order -- is the one a pair resolves to."""
    by_key: dict[str, VideoItem] = {}
    for item in index.items:
        by_key.setdefault(item.key, item)
    return by_key


class PairedClipDataset(Dataset[ClipSample]):
    """Real/fake clip pairs from ``pairs`` (e.g. :meth:`dfwb.protocols.protocol.Protocol.pairs`),
    restricted to keys actually present in ``index``.

    A pair whose real or fake key is not in ``index`` is dropped; :attr:`dropped` counts how many.
    Every surviving pair contributes ``spec.clips_per_mode(train=train)`` clips a side, exactly as
    one :class:`~dfwb.data.dataset.ClipDataset` would for either video alone -- ``transform``,
    ``adapt_chain`` and ``seed`` are passed straight through to the two internal datasets, so a
    pair's clips are seeded, sampled and adapted identically to a non-paired clip of the same
    video would be.
    """

    def __init__(
        self,
        index: VideoIndex,
        pairs: Sequence[tuple[str, str]],
        spec: ClipSpec,
        *,
        train: bool,
        transform: ClipTransform | None = None,
        adapt_chain: Callable[[Tensor], Tensor] | None = None,
        seed: int,
    ) -> None:
        by_key = _by_key(index)
        real_items: list[VideoItem] = []
        fake_items: list[VideoItem] = []
        dropped = 0
        for real_key, fake_key in pairs:
            real_item = by_key.get(real_key)
            fake_item = by_key.get(fake_key)
            if real_item is None or fake_item is None:
                dropped += 1
                continue
            real_items.append(real_item)
            fake_items.append(fake_item)

        self.dropped = dropped
        self._clips_per_pair = spec.clips_per_mode(train=train)

        def _sub_index(items: list[VideoItem]) -> VideoIndex:
            return VideoIndex(items=items, excluded=[], _summaries=[])

        self._real = ClipDataset(
            _sub_index(real_items),
            spec,
            train=train,
            transform=transform,
            adapt_chain=adapt_chain,
            seed=seed,
        )
        self._fake = ClipDataset(
            _sub_index(fake_items),
            spec,
            train=train,
            transform=transform,
            adapt_chain=adapt_chain,
            seed=seed,
        )

    def set_epoch(self, epoch: int) -> None:
        """Reaches both the real and the fake side (see
        :meth:`~dfwb.data.dataset.ClipDataset.set_epoch`)."""
        self._real.set_epoch(epoch)
        self._fake.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._real) + len(self._fake)

    def pair_groups(self) -> list[list[int]]:
        """The dataset indices of each surviving pair, in pair order: that pair's real rows, then
        its fake rows. A batch sampler that keeps a pair together keeps one of these together."""
        n_real = len(self._real)
        size = self._clips_per_pair
        return [
            [*range(p * size, (p + 1) * size), *range(n_real + p * size, n_real + (p + 1) * size)]
            for p in range(n_real // size if size else 0)
        ]

    def __getitem__(self, i: int) -> ClipSample:
        if not 0 <= i < len(self):
            raise IndexError(i)
        if i < len(self._real):
            sample = self._real[i]
            pair_id = i // self._clips_per_pair
        else:
            local = i - len(self._real)
            sample = self._fake[local]
            pair_id = local // self._clips_per_pair
        return replace(sample, extras={**sample.extras, "dfwb/pair_id": pair_id})
