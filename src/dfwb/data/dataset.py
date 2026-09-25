"""``ClipDataset`` and ``MultiSource``: PyTorch datasets over a
:class:`~dfwb.data.index.VideoIndex`.

``ClipDataset`` wraps exactly one video pool -- the join produced for a single training source.
The weighting and bookkeeping a multi-source run needs (several ``data.train:`` entries, each with
its own sampling weight) belongs to the *source*, not to any one video, so it is not something a
single ``ClipDataset`` should have to know about. Building one dataset per source and combining
them with :class:`MultiSource` keeps ``ClipDataset`` indifferent to how many sources a caller is
mixing, and lets ``MultiSource`` own the one thing that genuinely is its job: turning "index i"
into "which source, and which local index", and tagging every sample with the source it came from.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

from dfwb.core.errors import ConfigError
from dfwb.core.records.local import FRAME_FILE
from dfwb.data._images import read_frame
from dfwb.data.clips import ClipSpec, clip_windows_padded
from dfwb.data.index import VideoIndex

if TYPE_CHECKING:
    from dfwb.data.paired import PairedClipDataset

__all__ = ["ClipDataset", "ClipSample", "ClipTransform", "MultiSource"]

_PAIR_ID = "dfwb/pair_id"


class ClipTransform(Protocol):
    """A clip-consistent transform: one call per clip, the same random draw for every frame."""

    def __call__(self, clip: Tensor, *, generator: torch.Generator | None = None) -> Tensor: ...


@dataclass
class ClipSample:
    """One clip: ``clip`` is ``[T, C, H, W]`` float32, and ``frame_indices`` is the source frame
    number (from ``VideoItem.frame_indices``) each position of the clip was decoded from."""

    clip: Tensor
    label: int
    key: str
    dataset: str
    compression: str | None
    clip_index: int
    frame_indices: list[int]
    extras: dict[str, Any] = field(default_factory=dict)


def _torch_seed(seed: int, epoch: int, i: int) -> int:
    """A 64-bit int, deterministic in ``(seed, epoch, i)``, for seeding a transform's generator.

    A plain ``hash()`` of the same string is not usable here: Python randomises string hashing per
    process unless ``PYTHONHASHSEED`` is pinned, so it would give the same clip a different
    transform draw from one run to the next.
    """
    digest = hashlib.sha256(f"{seed}:{epoch}:{i}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class ClipDataset(Dataset[ClipSample]):  # type: ignore[misc, unused-ignore]  # Any w/o torch
    """``__getitem__(i)`` decodes one clip: ``spec.frames`` PNGs from one video's processed store,
    stacked into ``[T, C, H, W]`` float32 in ``[0, 1]``.

    The dataset is laid out one video's clips at a time: sample ``i`` is clip ``i %
    clips_per_mode`` of video ``i // clips_per_mode``, of ``index.items`` in its existing order.
    Eval windows are a pure function of a video's stored frame count, so two eval passes always
    agree. Train windows for sample ``i`` are drawn by a fresh ``random.Random`` seeded from
    ``(seed, epoch, i)`` alone -- never the DataLoader worker id -- so a sample's window does not
    depend on how many workers read it.

    The epoch itself lives in a shared-memory tensor, not a plain attribute: with
    ``persistent_workers=True``, a ``DataLoader``'s worker processes are started once and reused
    across epochs, so a plain ``self.epoch = epoch`` in :meth:`set_epoch` -- run in the main
    process -- would never reach the copy of this dataset each worker already holds. Writing into
    a shared tensor in place, and reading it back with ``.item()``, is visible from every process
    that shares the same underlying storage, workers included, whether they were started by
    ``fork`` or ``spawn``.
    """

    def __init__(
        self,
        index: VideoIndex,
        spec: ClipSpec,
        *,
        train: bool,
        transform: ClipTransform | None = None,
        adapt_chain: Callable[[Tensor], Tensor] | None = None,
        seed: int,
    ) -> None:
        self.index = index
        self.spec = spec
        self.train = train
        self.transform = transform
        self.adapt_chain = adapt_chain
        self.seed = seed
        self._epoch = torch.zeros((), dtype=torch.int64)
        self._epoch.share_memory_()  # type: ignore[no-untyped-call, unused-ignore]
        self._clips_per_video = spec.clips_per_mode(train=train)

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        """Every sample drawn after this call is seeded with ``epoch`` instead -- including by a
        ``DataLoader``'s already-running, persistent workers (see the class docstring)."""
        self._epoch.fill_(epoch)

    def __len__(self) -> int:
        return len(self.index.items) * self._clips_per_video

    def __getitem__(self, i: int) -> ClipSample:
        if not 0 <= i < len(self):
            raise IndexError(i)
        item_index, clip_index = divmod(i, self._clips_per_video)
        item = self.index.items[item_index]
        n = len(item.frame_indices)

        rng = random.Random(f"{self.seed}:{self.epoch}:{i}") if self.train else None
        windows = clip_windows_padded(n, self.spec, train=self.train, rng=rng)
        positions, padded = windows[clip_index]

        frame_numbers = [item.frame_indices[position] for position in positions]
        frames = [
            read_frame(item.video_dir / FRAME_FILE.format(index=number)) for number in frame_numbers
        ]
        clip = torch.stack(frames).to(torch.float32) / 255.0

        if self.transform is not None:
            generator = torch.Generator().manual_seed(_torch_seed(self.seed, self.epoch, i))
            clip = self.transform(clip, generator=generator)
        if self.adapt_chain is not None:
            clip = self.adapt_chain(clip)

        return ClipSample(
            clip=clip,
            label=item.label,
            key=item.key,
            dataset=item.dataset,
            compression=item.compression,
            clip_index=clip_index,
            frame_indices=frame_numbers,
            extras={"dfwb/padded": padded},
        )


class MultiSource(Dataset[ClipSample]):  # type: ignore[misc, unused-ignore]  # Any w/o torch
    """Concatenates several :class:`ClipDataset` sources, tagging every sample with the index of
    the dataset (its position in ``datasets``) it came from, in ``extras["dfwb/source_id"]``.

    A source may also be a :class:`~dfwb.data.paired.PairedClipDataset` (one per training source
    when a run trains on real/fake pairs): it is concatenated exactly the same way, and its
    ``extras["dfwb/pair_id"]`` values are offset by the number of pairs in the sources before it,
    so pair ids never collide between sources that share a batch."""

    def __init__(
        self, datasets: Sequence[ClipDataset | PairedClipDataset], weights: Sequence[float]
    ) -> None:
        datasets = list(datasets)
        weights = list(weights)
        if len(weights) != len(datasets):
            raise ConfigError(
                f"MultiSource: {len(weights)} weights for {len(datasets)} datasets",
                hint="pass exactly one weight per dataset, in the same order",
            )
        if any(weight <= 0 for weight in weights):
            raise ConfigError(
                f"MultiSource: every weight must be positive, got {weights}",
                hint="drop a source you do not want, rather than weighting it to zero or below",
            )
        self.datasets = datasets
        total = sum(weights)
        self.weights = tuple(weight / total for weight in weights)
        lengths = [len(dataset) for dataset in datasets]
        self._offsets = list(itertools.accumulate(lengths, initial=0))
        pair_counts = [getattr(dataset, "pair_count", 0) for dataset in datasets]
        self._pair_offsets = list(itertools.accumulate(pair_counts, initial=0))

    def __len__(self) -> int:
        return self._offsets[-1]

    def source_of(self, i: int) -> int:
        """Which dataset -- its position in ``datasets`` -- sample ``i`` of the concatenation
        comes from."""
        if not 0 <= i < len(self):
            raise IndexError(i)
        return bisect.bisect_right(self._offsets, i) - 1

    def __getitem__(self, i: int) -> ClipSample:
        source = self.source_of(i)
        sample = self.datasets[source][i - self._offsets[source]]
        extras = {**sample.extras, "dfwb/source_id": source}
        if _PAIR_ID in extras:
            extras[_PAIR_ID] += self._pair_offsets[source]
        return replace(sample, extras=extras)
