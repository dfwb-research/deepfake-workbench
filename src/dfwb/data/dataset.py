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
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from dfwb.core.errors import ConfigError
from dfwb.core.records.local import FRAME_FILE
from dfwb.data._images import CorruptFrameError, read_frame
from dfwb.data.clips import ClipSpec, clip_windows_padded
from dfwb.data.index import VideoIndex, VideoItem

if TYPE_CHECKING:
    from dfwb.data.paired import PairedClipDataset

__all__ = ["ClipDataset", "ClipSample", "ClipTransform", "MultiSource"]

_log = logging.getLogger(__name__)

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

    A corrupt stored frame (one Pillow cannot decode) is handled by ``repair_corrupt_frames``, an
    explicit flag independent of ``train``: off (:mod:`dfwb.score.harness`'s own datasets, and
    nothing else) lets the error propagate, so a caller grouping clips by video can catch it and
    blame the right video itself; on (every training and validation source :mod:`dfwb.train`
    builds) instead repeats the clip's nearest still-good frame in the corrupt frame's place, logs
    a warning naming the file, and reports how many frames this happened to in
    ``ClipSample.extras["dfwb/repaired_frames"]`` (summed across a batch by whoever trains or
    validates with it -- never a counter kept here, which a multi-worker ``DataLoader`` could only
    update racily). When every frame of a clip is corrupt (nothing left to repeat from):
    ``train=True`` still raises -- a video this broken should stop a training run, not be quietly
    dropped from it -- but ``train=False`` (a validation source, with repair on) instead makes
    :meth:`__getitem__` return ``None`` for that sample, so :func:`~dfwb.data.collate.collate_clips`
    can drop it from the batch instead of the whole batch raising; a caller counts the drop from
    ``extras["dfwb/videos_skipped"]``, which :func:`~dfwb.data.collate.collate_clips` sets when it
    drops one. A stored frame whose *size* does not match ``expected_frame_size`` is never treated
    this way, whatever ``repair_corrupt_frames``/``train`` say: it always raises, since a wrongly
    sized store is the wrong store, not a one-off corrupt file.
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
        expected_frame_size: int | None = None,
        repair_corrupt_frames: bool = False,
    ) -> None:
        self.index = index
        self.spec = spec
        self.train = train
        self.transform = transform
        self.adapt_chain = adapt_chain
        self.seed = seed
        self.expected_frame_size = expected_frame_size
        self.repair_corrupt_frames = repair_corrupt_frames
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

    def video_key(self, i: int) -> tuple[str, str, str | None]:
        """The ``(dataset, key, compression)`` sample ``i`` belongs to -- from ``index.items``
        alone, so it is known even when decoding ``i``'s frames would raise (a corrupt stored
        frame, in eval mode). A caller that must blame the right video for a batch whose very
        fetch failed (:mod:`dfwb.score.harness`) asks this instead of reading it off a batch that
        was never actually built."""
        if not 0 <= i < len(self):
            raise IndexError(i)
        item_index, _ = divmod(i, self._clips_per_video)
        item = self.index.items[item_index]
        return (item.dataset, item.key, item.compression)

    def _read_clip_frames(
        self, item: VideoItem, frame_numbers: Sequence[int]
    ) -> tuple[list[Tensor], int] | None:
        """``frame_numbers`` of ``item``, decoded, and how many were repaired; ``None`` to signal
        that this whole sample must be dropped (only possible when ``repair_corrupt_frames`` and
        not ``train`` -- see the class docstring for every other case)."""
        paths = [item.video_dir / FRAME_FILE.format(index=number) for number in frame_numbers]
        if not self.repair_corrupt_frames:
            frames = [read_frame(path, expected_size=self.expected_frame_size) for path in paths]
            return frames, 0

        decoded: list[Tensor | None] = [None] * len(paths)
        corrupt: list[int] = []
        for position, path in enumerate(paths):
            try:
                decoded[position] = read_frame(path, expected_size=self.expected_frame_size)
            except CorruptFrameError as exc:
                corrupt.append(position)
                _log.warning("%s: corrupt stored frame skipped and repeated: %s", path, exc)
        if not corrupt:
            return cast(list[Tensor], decoded), 0
        good = [position for position in range(len(paths)) if position not in corrupt]
        if not good:
            if self.train:
                # Nothing in this clip decoded at all, and this is a real training source: there
                # is no neighbour left to repeat, so the original decode failure is the most
                # useful thing to raise rather than train on a synthetic clip.
                read_frame(paths[corrupt[0]], expected_size=self.expected_frame_size)
            return None
        for position in corrupt:
            nearest = min(good, key=lambda g: abs(g - position))
            decoded[position] = decoded[nearest]
        return cast(list[Tensor], decoded), len(corrupt)

    def __getitem__(self, i: int) -> ClipSample | None:  # type: ignore[override]
        if not 0 <= i < len(self):
            raise IndexError(i)
        item_index, clip_index = divmod(i, self._clips_per_video)
        item = self.index.items[item_index]
        n = len(item.frame_indices)

        rng = random.Random(f"{self.seed}:{self.epoch}:{i}") if self.train else None
        windows = clip_windows_padded(n, self.spec, train=self.train, rng=rng)
        positions, padded = windows[clip_index]

        frame_numbers = [item.frame_indices[position] for position in positions]
        result = self._read_clip_frames(item, frame_numbers)
        if result is None:
            return None
        frames, repaired = result
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
            extras={"dfwb/padded": padded, "dfwb/repaired_frames": repaired},
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
        # MultiSource only ever wraps training sources, whose ClipDataset.__getitem__ never
        # returns None (that is a validation-only signal -- see the class's own docstring); this
        # narrows the type mypy sees ClipDataset's shared __getitem__ signature as returning.
        assert sample is not None
        extras = {**sample.extras, "dfwb/source_id": source}
        if _PAIR_ID in extras:
            extras[_PAIR_ID] += self._pair_offsets[source]
        return replace(sample, extras=extras)
