"""``VideoLabelBalanced``, ``SourceBalanced`` and ``VideoGrouped``: seeded, deterministic-per-epoch
sampling over :mod:`dfwb.data` datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from dfwb.core.errors import ConfigError
from dfwb.data.clips import ClipSpec, ClipsPerVideo
from dfwb.data.dataset import ClipDataset, MultiSource
from dfwb.data.index import VideoIndex, VideoItem
from dfwb.data.samplers import SourceBalanced, VideoGrouped, VideoLabelBalanced


def _bare_item(label: int, key: str) -> VideoItem:
    """A ``VideoItem`` with no real frames on disk -- fine for the samplers, which only ever look
    at ``.label`` and index arithmetic, never decode a clip."""
    return VideoItem(
        source=0,
        dataset="toy",
        key=key,
        compression=None,
        label=label,
        label_key="L",
        method="M",
        video_dir=Path("/does-not-exist"),
        frame_indices=[0],
    )


def _bare_index(labels: Sequence[int]) -> VideoIndex:
    items = [_bare_item(label, key=f"v{i}") for i, label in enumerate(labels)]
    return VideoIndex(items=items, excluded=[], _summaries=[])


def _write_frames(dir_path: Path, n: int) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(dir_path / f"frame_{i:06d}.png"), np.zeros((4, 4, 3), dtype=np.uint8))


def _real_item(tmp_path: Path, *, key: str, label: int, n_frames: int = 5) -> VideoItem:
    video_dir = tmp_path / key.replace("/", "_")
    _write_frames(video_dir, n_frames)
    return VideoItem(
        source=0,
        dataset="toy",
        key=key,
        compression=None,
        label=label,
        label_key="L",
        method="M",
        video_dir=video_dir,
        frame_indices=list(range(n_frames)),
    )


def _clip_dataset(
    items: list[VideoItem], *, clips_per_video: int, train: bool = False
) -> ClipDataset:
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    spec = ClipSpec(
        frames=1,
        sampling="uniform",
        clips_per_video=ClipsPerVideo(train=clips_per_video, eval=clips_per_video),
    )
    return ClipDataset(index, spec, train=train, seed=0)


def _bare_clip_dataset(n_items: int, *, prefix: str, clips_per_video: int = 1) -> ClipDataset:
    """A :class:`ClipDataset` over ``n_items`` videos with no real frames on disk -- fine for a
    sampler test, which only ever needs ``len(dataset)``, never to decode a clip. Lets the
    weighted-sampling tests below use dataset sizes large enough (hundreds to a thousand) for the
    drawn proportions to actually converge, without writing that many PNGs to disk."""
    items = [_bare_item(0, key=f"{prefix}{i}") for i in range(n_items)]
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    spec = ClipSpec(
        frames=1,
        sampling="uniform",
        clips_per_video=ClipsPerVideo(train=clips_per_video, eval=clips_per_video),
    )
    return ClipDataset(index, spec, train=False, seed=0)


# --------------------------------------------------------------------------- VideoLabelBalanced


def test_video_label_balanced_is_within_5_percent_over_1000_draws():
    labels = [0] * 900 + [1] * 100
    index = _bare_index(labels)
    sampler = VideoLabelBalanced(index, seed=0)
    sampler.set_epoch(0)

    draws = list(sampler)
    assert len(draws) == 1000

    drawn_labels = [index.items[i].label for i in draws]
    fraction_fake = sum(drawn_labels) / len(drawn_labels)
    assert abs(fraction_fake - 0.5) <= 0.05


def test_video_label_balanced_is_within_5_percent_for_a_severely_imbalanced_set():
    labels = [0] * 980 + [1] * 20
    index = _bare_index(labels)
    sampler = VideoLabelBalanced(index, seed=7)
    sampler.set_epoch(0)

    drawn_labels = [index.items[i].label for i in sampler]
    fraction_fake = sum(drawn_labels) / len(drawn_labels)
    assert abs(fraction_fake - 0.5) <= 0.05


def test_video_label_balanced_len_matches_source_length():
    index = _bare_index([0] * 30 + [1] * 10)
    sampler = VideoLabelBalanced(index, seed=0)
    assert len(sampler) == 40
    assert len(list(sampler)) == 40


def test_video_label_balanced_is_deterministic_given_seed_and_epoch():
    index = _bare_index([0] * 50 + [1] * 50)
    a = VideoLabelBalanced(index, seed=42)
    b = VideoLabelBalanced(index, seed=42)
    a.set_epoch(3)
    b.set_epoch(3)
    assert list(a) == list(b)


def test_video_label_balanced_changes_with_a_different_epoch():
    index = _bare_index([0] * 50 + [1] * 50)
    sampler = VideoLabelBalanced(index, seed=42)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second


def test_video_label_balanced_changes_with_a_different_seed():
    index = _bare_index([0] * 50 + [1] * 50)
    a = list(VideoLabelBalanced(index, seed=1))
    b = list(VideoLabelBalanced(index, seed=2))
    assert a != b


def test_video_label_balanced_from_a_clip_dataset_expands_labels_per_clip(tmp_path):
    items = [
        _real_item(tmp_path, key="r0", label=0),
        _real_item(tmp_path, key="r1", label=0),
        _real_item(tmp_path, key="f0", label=1),
    ]
    dataset = _clip_dataset(items, clips_per_video=4)  # length 12: 8 label-0, 4 label-1

    sampler = VideoLabelBalanced(dataset, seed=0)
    sampler.set_epoch(0)

    assert len(sampler) == 12
    draws = list(sampler)
    assert all(0 <= i < len(dataset) for i in draws)
    labels = [dataset.index.items[i // 4].label for i in draws]
    assert 0 in labels
    assert 1 in labels


# -------------------------------------------------------------------------------- SourceBalanced


def test_source_balanced_respects_weights_not_sizes():
    big = _bare_clip_dataset(900, prefix="big")
    small = _bare_clip_dataset(100, prefix="small")
    multi = MultiSource([big, small], weights=[1.0, 1.0])  # equal weight, 9x size difference

    sampler = SourceBalanced(multi, seed=0)
    sampler.set_epoch(0)
    draws = list(sampler)

    assert len(draws) == len(multi)
    from_big = sum(1 for i in draws if i < len(big))
    fraction_big = from_big / len(draws)
    assert abs(fraction_big - 0.5) <= 0.05


def test_source_balanced_respects_unequal_weights():
    a = _bare_clip_dataset(500, prefix="a")
    b = _bare_clip_dataset(500, prefix="b")
    multi = MultiSource([a, b], weights=[1.0, 3.0])  # 25% / 75%

    sampler = SourceBalanced(multi, seed=0)
    sampler.set_epoch(0)
    draws = list(sampler)

    from_a = sum(1 for i in draws if i < len(a))
    fraction_a = from_a / len(draws)
    assert abs(fraction_a - 0.25) <= 0.05


def test_source_balanced_indices_are_always_in_range(tmp_path):
    a = _clip_dataset(
        [_real_item(tmp_path, key=f"a{i}", label=0) for i in range(3)], clips_per_video=2
    )
    b = _clip_dataset(
        [_real_item(tmp_path, key=f"b{i}", label=0) for i in range(5)], clips_per_video=1
    )
    multi = MultiSource([a, b], weights=[2.0, 1.0])
    sampler = SourceBalanced(multi, seed=1)
    sampler.set_epoch(0)
    draws = list(sampler)
    assert all(0 <= i < len(multi) for i in draws)


def test_source_balanced_is_deterministic_given_seed_and_epoch(tmp_path):
    a = _clip_dataset(
        [_real_item(tmp_path, key=f"a{i}", label=0) for i in range(4)], clips_per_video=1
    )
    b = _clip_dataset(
        [_real_item(tmp_path, key=f"b{i}", label=0) for i in range(4)], clips_per_video=1
    )
    multi = MultiSource([a, b], weights=[1.0, 1.0])
    x = SourceBalanced(multi, seed=5)
    y = SourceBalanced(multi, seed=5)
    x.set_epoch(2)
    y.set_epoch(2)
    assert list(x) == list(y)


def test_source_balanced_changes_with_a_different_epoch(tmp_path):
    a = _clip_dataset(
        [_real_item(tmp_path, key=f"a{i}", label=0) for i in range(10)], clips_per_video=1
    )
    b = _clip_dataset(
        [_real_item(tmp_path, key=f"b{i}", label=0) for i in range(10)], clips_per_video=1
    )
    multi = MultiSource([a, b], weights=[1.0, 1.0])
    sampler = SourceBalanced(multi, seed=9)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second


# ---------------------------------------------------------------------------------- VideoGrouped


def test_video_grouped_never_splits_a_videos_clips_across_batches(tmp_path):
    items = [_real_item(tmp_path, key=f"v{i}", label=i % 2) for i in range(5)]
    dataset = _clip_dataset(items, clips_per_video=3)  # length 15, groups of 3

    sampler = VideoGrouped(dataset, batch_size=4)
    batches = list(sampler)

    # every video's group of 3 consecutive indices stays fully inside exactly one batch.
    for video_index in range(5):
        group = set(range(video_index * 3, video_index * 3 + 3))
        containing = [batch for batch in batches if group & set(batch)]
        assert len(containing) == 1
        assert group.issubset(set(containing[0]))


def test_video_grouped_covers_every_index_exactly_once(tmp_path):
    items = [_real_item(tmp_path, key=f"v{i}", label=0) for i in range(7)]
    dataset = _clip_dataset(items, clips_per_video=2)

    sampler = VideoGrouped(dataset, batch_size=3)
    all_indices = [i for batch in sampler for i in batch]

    assert sorted(all_indices) == list(range(len(dataset)))


def test_video_grouped_len_matches_number_of_batches(tmp_path):
    items = [_real_item(tmp_path, key=f"v{i}", label=0) for i in range(6)]
    dataset = _clip_dataset(items, clips_per_video=2)
    sampler = VideoGrouped(dataset, batch_size=4)
    assert len(sampler) == len(list(sampler))


def test_video_grouped_oversized_video_gets_its_own_batch(tmp_path):
    items = [_real_item(tmp_path, key=f"v{i}", label=0) for i in range(2)]
    dataset = _clip_dataset(items, clips_per_video=5)  # each video alone exceeds batch_size

    sampler = VideoGrouped(dataset, batch_size=3)
    batches = list(sampler)

    assert len(batches) == 2
    assert batches[0] == list(range(0, 5))
    assert batches[1] == list(range(5, 10))


def test_video_grouped_rejects_a_non_positive_batch_size(tmp_path):
    items = [_real_item(tmp_path, key="v0", label=0)]
    dataset = _clip_dataset(items, clips_per_video=1)
    with pytest.raises(ConfigError, match="positive"):
        VideoGrouped(dataset, batch_size=0)


def test_video_grouped_with_no_items_yields_no_batches(tmp_path):
    dataset = _clip_dataset([], clips_per_video=1)
    sampler = VideoGrouped(dataset, batch_size=4)
    assert list(sampler) == []
    assert len(sampler) == 0


def test_video_grouped_is_deterministic_across_instances(tmp_path):
    items = [_real_item(tmp_path, key=f"v{i}", label=0) for i in range(9)]
    dataset = _clip_dataset(items, clips_per_video=2)
    a = list(VideoGrouped(dataset, batch_size=5))
    b = list(VideoGrouped(dataset, batch_size=5))
    assert a == b
