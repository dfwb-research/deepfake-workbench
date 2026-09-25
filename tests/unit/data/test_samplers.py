"""``VideoLabelBalanced``, ``SourceBalanced``, ``VideoGrouped`` and ``PairGrouped``: seeded,
deterministic-per-epoch sampling over :mod:`dfwb.data` datasets."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from dfwb.core.errors import ConfigError
from dfwb.data.clips import ClipSpec, ClipsPerVideo
from dfwb.data.dataset import ClipDataset, MultiSource
from dfwb.data.index import VideoIndex, VideoItem
from dfwb.data.paired import PairedClipDataset
from dfwb.data.samplers import (
    PairGrouped,
    SourceBalanced,
    VideoGrouped,
    VideoLabelBalanced,
    epoch_seed,
)


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


def test_video_label_balanced_accepts_a_multisource_of_clip_datasets():
    reals = ClipDataset(_bare_index([0] * 900), _spec(1), train=False, seed=0)
    fakes = ClipDataset(_bare_index([1] * 100), _spec(1), train=False, seed=0)
    multi = MultiSource([reals, fakes], weights=[1.0, 1.0])

    sampler = VideoLabelBalanced(multi, seed=0)
    draws = list(sampler)

    assert len(draws) == len(multi)
    fake_share = sum(1 for i in draws if i >= len(reals)) / len(draws)
    assert abs(fake_share - 0.5) < 0.05


def test_video_label_balanced_refuses_paired_data_inside_a_multisource():
    paired = _bare_paired(3)
    multi = MultiSource([paired], weights=[1.0])
    with pytest.raises(ConfigError, match="pair"):
        VideoLabelBalanced(multi, seed=0)


# ----------------------------------------------------------------------------------- PairGrouped


def _spec(clips_per_video: int) -> ClipSpec:
    return ClipSpec(
        frames=1,
        sampling="uniform",
        clips_per_video=ClipsPerVideo(train=clips_per_video, eval=clips_per_video),
    )


def _pairs_index(
    n_pairs: int, *, prefix: str = "", tmp_path: Path | None = None
) -> tuple[VideoIndex, list[tuple[str, str]]]:
    items: list[VideoItem] = []
    pairs: list[tuple[str, str]] = []
    for i in range(n_pairs):
        real, fake = f"{prefix}REAL/{i}", f"{prefix}FAKE/{i}"
        if tmp_path is None:
            items += [_bare_item(0, key=real), _bare_item(1, key=fake)]
        else:
            items += [
                _real_item(tmp_path, key=real, label=0, n_frames=1),
                _real_item(tmp_path, key=fake, label=1, n_frames=1),
            ]
        pairs.append((real, fake))
    return VideoIndex(items=items, excluded=[], _summaries=[]), pairs


def _bare_paired(n_pairs: int, *, clips_per_video: int = 1) -> PairedClipDataset:
    """A :class:`PairedClipDataset` with no frames on disk: the sampler only needs its layout."""
    index, pairs = _pairs_index(n_pairs)
    return PairedClipDataset(index, pairs, _spec(clips_per_video), train=True, seed=0)


def test_pair_groups_lists_each_pairs_real_rows_then_fake_rows():
    paired = _bare_paired(3, clips_per_video=2)
    # 3 pairs * 2 clips: real rows 0..5, fake rows 6..11, pair-major within each side.
    assert paired.pair_groups() == [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]]


def test_pair_grouped_keeps_each_pairs_real_and_fake_rows_in_one_batch(tmp_path):
    index, pairs = _pairs_index(5, tmp_path=tmp_path)
    paired = PairedClipDataset(index, pairs, _spec(1), train=True, seed=0)

    sampler = PairGrouped(paired, batch_size=4, seed=0)
    batches = list(sampler)

    assert sorted(i for batch in batches for i in batch) == list(range(len(paired)))
    assert [len(batch) for batch in batches] == [4, 4, 2]  # filled pair by pair
    for batch in batches:
        labels_by_pair: dict[int, set[int]] = {}
        for i in batch:
            sample = paired[i]
            labels_by_pair.setdefault(sample.extras["dfwb/pair_id"], set()).add(sample.label)
        assert all(labels == {0, 1} for labels in labels_by_pair.values())


def test_pair_grouped_is_seeded_per_epoch():
    paired = _bare_paired(10)
    sampler = PairGrouped(paired, batch_size=4, seed=3)

    first = list(sampler)
    assert list(PairGrouped(paired, batch_size=4, seed=3)) == first
    sampler.set_epoch(1)
    second = list(sampler)
    sampler.set_epoch(0)

    assert second != first
    assert list(sampler) == first
    assert list(PairGrouped(paired, batch_size=4, seed=4)) != first


def test_pair_grouped_gives_an_oversized_pair_its_own_batch():
    paired = _bare_paired(3, clips_per_video=3)  # one pair is 6 rows, more than a batch of 4

    batches = list(PairGrouped(paired, batch_size=4, seed=0))

    assert len(batches) == 3
    assert sorted(sorted(batch) for batch in batches) == sorted(
        sorted(group) for group in paired.pair_groups()
    )


def test_pair_grouped_len_matches_number_of_batches():
    paired = _bare_paired(7)
    sampler = PairGrouped(paired, batch_size=6, seed=0)
    assert len(sampler) == len(list(sampler)) == 3


def test_pair_grouped_over_a_multisource_offsets_each_sources_pairs():
    index_a, pairs_a = _pairs_index(2, prefix="a")
    index_b, pairs_b = _pairs_index(3, prefix="b")
    paired_a = PairedClipDataset(index_a, pairs_a, _spec(1), train=True, seed=0)
    paired_b = PairedClipDataset(index_b, pairs_b, _spec(1), train=True, seed=0)
    multi = MultiSource([paired_a, paired_b], weights=[1.0, 1.0])

    batches = list(PairGrouped(multi, batch_size=2, seed=0))

    offset = len(paired_a)
    expected = [sorted(g) for g in paired_a.pair_groups()] + [
        sorted(i + offset for i in g) for g in paired_b.pair_groups()
    ]
    assert sorted(sorted(batch) for batch in batches) == sorted(expected)


def test_pair_grouped_refuses_unpaired_data_and_a_non_positive_batch_size(tmp_path):
    plain = _clip_dataset([_real_item(tmp_path, key="v0", label=0)], clips_per_video=1)
    with pytest.raises(ConfigError, match="paired"):
        PairGrouped(MultiSource([plain], weights=[1.0]), batch_size=2, seed=0)
    with pytest.raises(ConfigError, match="positive"):
        PairGrouped(_bare_paired(1), batch_size=0, seed=0)


def test_pair_grouped_never_splits_a_pair_over_random_shapes():
    rng = random.Random(0)
    for _ in range(60):
        clips = rng.randint(1, 3)
        sources = [_bare_paired_prefixed(rng.randint(1, 9), clips, f"s{j}") for j in range(3)]
        sources = sources[: rng.randint(1, 3)]
        multi = MultiSource(sources, weights=[1.0] * len(sources))
        batch_size = rng.randint(1, 13)
        sampler = PairGrouped(multi, batch_size, seed=rng.randint(0, 99))
        groups: list[frozenset[int]] = []
        offset = 0
        for source in sources:
            groups += [frozenset(i + offset for i in g) for g in source.pair_groups()]
            offset += len(source)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            assert len(batches) == len(sampler)
            assert sorted(i for b in batches for i in b) == list(range(len(multi)))
            batch_of = {i: n for n, b in enumerate(batches) for i in b}
            for group in groups:
                assert len({batch_of[i] for i in group}) == 1
            for batch in batches:
                assert len(batch) <= batch_size or len(batch) == 2 * clips


def _bare_paired_prefixed(n_pairs: int, clips_per_video: int, prefix: str) -> PairedClipDataset:
    index, pairs = _pairs_index(n_pairs, prefix=prefix)
    return PairedClipDataset(index, pairs, _spec(clips_per_video), train=True, seed=0)


# ------------------------------------------------------------------------------- epoch_seed


def test_epoch_seed_is_a_stable_64_bit_function_of_seed_and_epoch():
    assert epoch_seed(3, 1) == epoch_seed(3, 1)
    assert len({epoch_seed(3, 1), epoch_seed(3, 2), epoch_seed(4, 1)}) == 3
    assert 0 <= epoch_seed(3, 1) < 2**64
    # every seeded sampler draws from exactly this generator seed
    sampler = PairGrouped(_bare_paired(4), batch_size=2, seed=3)
    sampler.set_epoch(1)
    order = torch.randperm(4, generator=torch.Generator().manual_seed(epoch_seed(3, 1))).tolist()
    groups = _bare_paired(4).pair_groups()
    assert list(sampler) == [groups[i] for i in order]
