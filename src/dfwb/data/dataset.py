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
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast

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

__all__ = ["ClipDataset", "ClipSample", "ClipTransform", "MultiSource", "SkippedVideo"]

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


class SkippedVideo(NamedTuple):
    """What :meth:`ClipDataset.__getitem__` returns, in place of a :class:`ClipSample`, for a
    sample of a video none of whose stored frames can be read (only when it repairs corrupt
    frames at all): :func:`~dfwb.data.collate.collate_clips` drops it from its batch and records
    the video in ``extras["dfwb/videos_skipped"]``, so the run can count each video once.

    A named tuple rather than a dataclass, so a batch carrying it can be moved between devices
    like any other nested collection."""

    dataset: str
    key: str
    compression: str | None


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

    A corrupt stored frame (one Pillow cannot decode, or a missing file) is handled by
    ``repair_corrupt_frames``, an explicit flag independent of ``train``: off
    (:mod:`dfwb.score.harness`'s own datasets, and nothing else) lets the error propagate, so a
    caller grouping clips by video can catch it and blame the right video itself; on (every
    training and validation source :mod:`dfwb.train` builds) instead repairs it, logs a warning
    naming the file, and lists the source frame numbers repaired in
    ``ClipSample.extras["dfwb/repaired_frames"]`` (so whoever trains or validates with the sample
    can count each ``(video, frame)`` once -- never a counter kept here, which a multi-worker
    ``DataLoader`` could only update racily). The repair repeats, in the corrupt frame's place,
    the nearest good frame of the same clip; a clip with no good frame of its own (always the
    case for a one-frame clip) reads the nearest good stored frame of the same video instead,
    searching outwards from the corrupt one (a tie goes to the earlier frame). Only when no stored
    frame of the video can be read at all does :meth:`__getitem__` return a :class:`SkippedVideo`
    for the sample, with a warning, in training and validation alike:
    :func:`~dfwb.data.collate.collate_clips` drops it from the batch and lists the video in
    ``extras["dfwb/videos_skipped"]``. A stored frame whose *size* does not match
    ``expected_frame_size``, or one with more than 8 bits a channel, is never treated this way,
    whatever ``repair_corrupt_frames``/``train`` say: it always raises, since it means the wrong
    store, not a one-off corrupt file.
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

    def _read(self, item: VideoItem, position: int) -> Tensor:
        """Stored frame ``position`` (an index into ``item.frame_indices``) of ``item``.

        Raises:
            CorruptFrameError: It is missing or cannot be decoded.
        """
        path = item.video_dir / FRAME_FILE.format(index=item.frame_indices[position])
        return read_frame(path, expected_size=self.expected_frame_size)

    def _nearest_good(
        self, item: VideoItem, position: int, known: dict[int, Tensor | None]
    ) -> tuple[int, Tensor] | None:
        """The stored frame of ``item`` nearest to ``position`` that can be read, searching
        outwards (the earlier of two equally near frames first); ``None`` when none can.
        ``known`` holds the frames already tried (``None`` for one that failed) and is updated."""
        n = len(item.frame_indices)
        for distance in range(1, n):
            for candidate in (position - distance, position + distance):
                if not 0 <= candidate < n:
                    continue
                if candidate not in known:
                    try:
                        known[candidate] = self._read(item, candidate)
                    except CorruptFrameError:
                        known[candidate] = None
                frame = known[candidate]
                if frame is not None:
                    return candidate, frame
        return None

    def _read_clip_frames(
        self, item: VideoItem, positions: Sequence[int]
    ) -> tuple[list[Tensor], list[int]] | None:
        """The frames at ``positions`` of ``item``, decoded, and the source frame numbers that
        were repaired; ``None`` when no stored frame of the video can be read (only possible
        when ``repair_corrupt_frames`` -- see the class docstring for every other case)."""
        if not self.repair_corrupt_frames:
            return [self._read(item, position) for position in positions], []

        known: dict[int, Tensor | None] = {}
        failures: dict[int, CorruptFrameError] = {}
        for position in dict.fromkeys(positions):  # a padded clip may repeat a position
            try:
                known[position] = self._read(item, position)
            except CorruptFrameError as exc:
                known[position] = None
                failures[position] = exc
        if not failures:
            return [cast(Tensor, known[position]) for position in positions], []

        in_clip = [position for position in dict.fromkeys(positions) if known[position] is not None]
        replacement: dict[int, tuple[int, Tensor]] = {}
        for position, error in failures.items():
            if in_clip:
                nearest = min(in_clip, key=lambda good: (abs(good - position), good))
                found: tuple[int, Tensor] | None = (nearest, cast(Tensor, known[nearest]))
            else:
                found = self._nearest_good(item, position, known)
            if found is None:
                _log.warning(
                    "%s: no stored frame of this video can be read (%s); it is skipped",
                    item.video_dir,
                    error.message,
                )
                return None
            replacement[position] = found
            _log.warning(
                "%s; repeated stored frame %d of the same video in its place",
                error.message,
                item.frame_indices[found[0]],
            )
        frames = [
            replacement[position][1] if position in replacement else cast(Tensor, known[position])
            for position in positions
        ]
        repaired = sorted(item.frame_indices[position] for position in failures)
        return frames, repaired

    def __getitem__(self, i: int) -> ClipSample | SkippedVideo:  # type: ignore[override, unused-ignore]
        if not 0 <= i < len(self):
            raise IndexError(i)
        item_index, clip_index = divmod(i, self._clips_per_video)
        item = self.index.items[item_index]
        n = len(item.frame_indices)

        rng = random.Random(f"{self.seed}:{self.epoch}:{i}") if self.train else None
        windows = clip_windows_padded(n, self.spec, train=self.train, rng=rng)
        positions, padded = windows[clip_index]

        frame_numbers = [item.frame_indices[position] for position in positions]
        result = self._read_clip_frames(item, positions)
        if result is None:
            return SkippedVideo(item.dataset, item.key, item.compression)
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

    def __getitem__(self, i: int) -> ClipSample | SkippedVideo:  # type: ignore[override, unused-ignore]
        source = self.source_of(i)
        sample = self.datasets[source][i - self._offsets[source]]
        if isinstance(sample, SkippedVideo):
            return sample  # collate_clips drops it and records the video
        extras = {**sample.extras, "dfwb/source_id": source}
        if _PAIR_ID in extras:
            extras[_PAIR_ID] += self._pair_offsets[source]
        return replace(sample, extras=extras)
