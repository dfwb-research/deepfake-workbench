"""``ClipDataset`` and ``MultiSource``: decoded clips, seeding, and source tagging.

Every fixture video here is hand-built next to a tiny store, exactly as :mod:`dfwb.data.index`
itself would read one -- no dependency on ``dfwb.data.index``'s own join, since these tests care
about what happens *after* the join, given a :class:`~dfwb.data.index.VideoIndex` that already has
its items.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from torch.utils.data import DataLoader

from dfwb.core.errors import ConfigError
from dfwb.data.clips import ClipSpec, ClipsPerVideo
from dfwb.data.dataset import ClipDataset, MultiSource, _torch_seed
from dfwb.data.index import VideoIndex, VideoItem


def _write_frames(dir_path: Path, numbers: Sequence[int]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        value = number % 256
        image = np.full((4, 4, 3), value, dtype=np.uint8)
        cv2.imwrite(str(dir_path / f"frame_{number:06d}.png"), image)


def _video_item(
    video_dir: Path,
    *,
    source: int = 0,
    key: str = "video",
    n_frames: int = 5,
    label: int = 0,
    frame_numbers: Sequence[int] | None = None,
) -> VideoItem:
    frame_numbers = list(frame_numbers) if frame_numbers is not None else list(range(n_frames))
    _write_frames(video_dir, frame_numbers)
    return VideoItem(
        source=source,
        dataset="toy",
        key=key,
        compression=None,
        label=label,
        label_key="L",
        method="M",
        video_dir=video_dir,
        frame_indices=frame_numbers,
    )


def _video_index(items: list[VideoItem]) -> VideoIndex:
    return VideoIndex(items=items, excluded=[], _summaries=[])


# -------------------------------------------------------------------------------------- shapes


@pytest.mark.parametrize("frames", [1, 8])
def test_clip_shape_is_t_c_h_w_float32(tmp_path, frames):
    item = _video_item(tmp_path / "v", n_frames=20)
    index = _video_index([item])
    spec = ClipSpec(
        frames=frames, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=2)
    )
    dataset = ClipDataset(index, spec, train=False, seed=0)

    sample = dataset[0]
    assert sample.clip.shape == (frames, 3, 4, 4)
    assert sample.clip.dtype == torch.float32
    assert sample.clip.min() >= 0.0
    assert sample.clip.max() <= 1.0


def test_dataset_length_is_clips_per_video_times_item_count(tmp_path):
    items = [
        _video_item(tmp_path / "a", key="a", n_frames=10),
        _video_item(tmp_path / "b", key="b", n_frames=10),
        _video_item(tmp_path / "c", key="c", n_frames=10),
    ]
    index = _video_index(items)
    spec = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=5, eval=3)
    )
    assert len(ClipDataset(index, spec, train=False, seed=0)) == 3 * 3
    assert len(ClipDataset(index, spec, train=True, seed=0)) == 3 * 5


def test_getitem_out_of_range_raises_index_error(tmp_path):
    item = _video_item(tmp_path / "v", n_frames=5)
    index = _video_index([item])
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))
    dataset = ClipDataset(index, spec, train=False, seed=0)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


# --------------------------------------------------------------------------------------- decode


def test_png_decode_matches_cv2_imread_converted_to_rgb(tmp_path):
    video_dir = tmp_path / "v"
    video_dir.mkdir()
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    image[:, :, 0] = 40  # B
    image[:, :, 1] = 120  # G
    image[:, :, 2] = 200  # R
    frame_path = video_dir / "frame_000000.png"
    cv2.imwrite(str(frame_path), image)
    item = VideoItem(
        source=0,
        dataset="toy",
        key="v",
        compression=None,
        label=0,
        label_key="L",
        method="M",
        video_dir=video_dir,
        frame_indices=[0],
    )
    index = _video_index([item])
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))
    dataset = ClipDataset(index, spec, train=False, seed=0)

    sample = dataset[0]

    expected_bgr = cv2.imread(str(frame_path))
    expected_rgb = cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB)
    expected = torch.from_numpy(expected_rgb).permute(2, 0, 1).to(torch.float32) / 255.0
    torch.testing.assert_close(sample.clip[0], expected)


@pytest.mark.parametrize("shape", [(32, 32, 3), (7, 13, 3), (64, 48, 3)])
def test_png_decoding_is_pixel_identical_to_torchvisions_decoder(tmp_path, shape):
    # stored frames used to be read with torchvision's (now deprecated) decode_png; Pillow must
    # give exactly the same pixels back
    io = pytest.importorskip("torchvision.io")
    if not hasattr(io, "decode_png"):
        pytest.skip("this torchvision no longer has its own PNG decoder to compare with")
    video_dir = tmp_path / "v"
    video_dir.mkdir()
    image = np.random.default_rng(sum(shape)).integers(0, 256, size=shape, dtype=np.uint8)
    frame_path = video_dir / "frame_000000.png"
    cv2.imwrite(str(frame_path), image)
    item = VideoItem(0, "toy", "v", None, 0, "L", "M", video_dir, [0])
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))

    sample = ClipDataset(_video_index([item]), spec, train=False, seed=0)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        expected = io.decode_png(io.read_file(str(frame_path)))
    assert torch.equal(sample.clip[0], expected.to(torch.float32) / 255.0)


def test_decoding_a_clip_raises_no_deprecation_warning(tmp_path):
    item = _video_item(tmp_path / "v", n_frames=3)
    spec = ClipSpec(frames=2, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))
    dataset = ClipDataset(_video_index([item]), spec, train=False, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        dataset[0]


def test_frame_indices_are_source_frame_numbers_not_positions(tmp_path):
    # non-contiguous, so a position and its stored frame number never coincide by accident.
    source_numbers = [10, 15, 20, 25, 30]
    item = _video_item(tmp_path / "v", frame_numbers=source_numbers)
    index = _video_index([item])
    spec = ClipSpec(
        frames=3, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=1), stride=1
    )
    dataset = ClipDataset(index, spec, train=False, seed=0)

    sample = dataset[0]

    # n=5, T=3, stride=1: max_start = 2; eval c=1 -> linspace(0, 2, 1) = [0]; positions [0, 1, 2].
    assert sample.frame_indices == [10, 15, 20]
    assert sample.frame_indices != [0, 1, 2]


# --------------------------------------------------------------------------------------- padding


def test_short_video_is_padded_and_repeats_its_last_frame(tmp_path):
    item = _video_item(tmp_path / "v", n_frames=2)
    index = _video_index([item])
    spec = ClipSpec(
        frames=4, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=1)
    )
    dataset = ClipDataset(index, spec, train=False, seed=0)

    sample = dataset[0]
    assert sample.extras["dfwb/padded"] is True
    assert sample.frame_indices == [0, 1, 1, 1]


def test_long_enough_video_is_not_padded(tmp_path):
    item = _video_item(tmp_path / "v", n_frames=10)
    index = _video_index([item])
    spec = ClipSpec(
        frames=3, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=2)
    )
    dataset = ClipDataset(index, spec, train=False, seed=0)

    for i in range(len(dataset)):
        assert dataset[i].extras["dfwb/padded"] is False


# ---------------------------------------------------------------------------- eval determinism


def test_eval_passes_identical(tmp_path):
    items = [
        _video_item(tmp_path / "a", key="a", n_frames=10),
        _video_item(tmp_path / "b", key="b", n_frames=2),  # short: exercises padding too
    ]
    index = _video_index(items)
    spec = ClipSpec(
        frames=4, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=3)
    )
    dataset = ClipDataset(index, spec, train=False, seed=123)

    first = [dataset[i] for i in range(len(dataset))]
    second = [dataset[i] for i in range(len(dataset))]

    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a.clip, b.clip)
        assert a.frame_indices == b.frame_indices
        assert a.label == b.label
        assert a.key == b.key
        assert a.extras == b.extras


# --------------------------------------------------------------------------- train seeding


def test_two_datasets_same_seed_and_epoch_give_identical_windows(tmp_path):
    items = [
        _video_item(tmp_path / "a", key="a", n_frames=10),
        _video_item(tmp_path / "b", key="b", n_frames=3),
    ]
    index = _video_index(items)
    spec = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=4, eval=1)
    )

    ds_a = ClipDataset(index, spec, train=True, seed=42)
    ds_b = ClipDataset(index, spec, train=True, seed=42)

    for i in range(len(ds_a)):
        sample_a, sample_b = ds_a[i], ds_b[i]
        assert sample_a.frame_indices == sample_b.frame_indices
        assert torch.equal(sample_a.clip, sample_b.clip)


def test_a_different_seed_or_epoch_can_change_train_windows(tmp_path):
    item = _video_item(tmp_path / "v", n_frames=50)
    index = _video_index([item])
    spec = ClipSpec(
        frames=3, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=1)
    )

    baseline = ClipDataset(index, spec, train=True, seed=1)[0].frame_indices

    other_seed = ClipDataset(index, spec, train=True, seed=2)[0].frame_indices
    ds_epoch = ClipDataset(index, spec, train=True, seed=1)
    ds_epoch.set_epoch(1)
    other_epoch = ds_epoch[0].frame_indices

    assert other_seed != baseline or other_epoch != baseline


def test_num_workers_does_not_change_train_windows(tmp_path):
    items = [
        _video_item(tmp_path / "a", key="a", n_frames=6),
        _video_item(tmp_path / "b", key="b", n_frames=3),
    ]
    index = _video_index(items)
    spec = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=3, eval=1)
    )

    def _collect(num_workers: int) -> list[list[int]]:
        dataset = ClipDataset(index, spec, train=True, seed=7)
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers)
        return [sample.frame_indices for sample in loader]

    assert _collect(0) == _collect(2)


def test_uniform_train_frames_change_each_epoch_and_eval_frames_never_do(tmp_path):
    items = [_video_item(tmp_path / "a", key="a", n_frames=32)]
    index = _video_index(items)
    spec = ClipSpec(frames=4, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))

    def _frames(train: bool, epoch: int) -> list[int]:
        dataset = ClipDataset(index, spec, train=train, seed=3)
        dataset.set_epoch(epoch)
        return dataset[0].frame_indices

    train_draws = {tuple(_frames(True, epoch)) for epoch in range(5)}
    assert len(train_draws) > 1
    assert all(len(set(draw)) == 4 for draw in train_draws)
    assert len({tuple(_frames(False, epoch)) for epoch in range(5)}) == 1


def test_dataloader_num_workers_zero_and_two_agree_in_eval_mode_too(tmp_path):
    items = [
        _video_item(tmp_path / "a", key="a", n_frames=10),
        _video_item(tmp_path / "b", key="b", n_frames=2),
    ]
    index = _video_index(items)
    spec = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=2)
    )

    def _collect(num_workers: int) -> list[list[int]]:
        dataset = ClipDataset(index, spec, train=False, seed=0)
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers)
        return [sample.frame_indices for sample in loader]

    assert _collect(0) == _collect(2)


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_set_epoch_reaches_persistent_workers(tmp_path, context):
    # persistent_workers=True starts worker processes once and reuses them across epochs, so this
    # is the scenario a plain `self.epoch = epoch` attribute cannot reach: the workers' own copy
    # of the dataset was pickled/forked once, at loader start, and never updated again.
    items = [
        _video_item(tmp_path / "a", key="a", n_frames=20),
        _video_item(tmp_path / "b", key="b", n_frames=15),
    ]
    index = _video_index(items)
    spec = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=3, eval=1)
    )
    seed = 11

    dataset = ClipDataset(index, spec, train=True, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context=context,
    )

    per_epoch: dict[int, list[list[int]]] = {}
    for epoch in range(3):
        dataset.set_epoch(epoch)
        per_epoch[epoch] = [sample.frame_indices for sample in loader]

    # windows differ between epochs: each epoch reseeds every sample's rng.
    assert per_epoch[0] != per_epoch[1]
    assert per_epoch[1] != per_epoch[2]

    # and, for a given epoch, match a fresh in-process dataset's windows exactly.
    for epoch, expected_windows in per_epoch.items():
        reference = ClipDataset(index, spec, train=True, seed=seed)
        reference.set_epoch(epoch)
        assert [reference[i].frame_indices for i in range(len(reference))] == expected_windows


def _adapted_train_dataset(tmp_path: Path) -> ClipDataset:
    """A train dataset with every kind of transform and every adaptation step, so each one of
    them has to cross a process boundary when a loader starts its workers."""
    from dfwb.core.config.schema import ComponentSpec
    from dfwb.core.detector import InputSpec
    from dfwb.core.records.local import (
        BackendSpec,
        CropSpec,
        DecodeSpec,
        ExtrasSpec,
        ProcessingProfile,
        SamplingSpec,
        TrackSpec,
    )
    from dfwb.data.adapt import adapt
    from dfwb.data.transforms import build_transforms

    rng = np.random.default_rng(0)
    items = []
    for v in range(3):
        video_dir = tmp_path / f"v{v}"
        video_dir.mkdir()
        for number in range(6):
            image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
            cv2.imwrite(str(video_dir / f"frame_{number:06d}.png"), image)
        items.append(
            VideoItem(
                source=0,
                dataset="toy",
                key=f"v{v}",
                compression=None,
                label=v % 2,
                label_key="L",
                method="M",
                video_dir=video_dir,
                frame_indices=list(range(6)),
            )
        )
    profile = ProcessingProfile(
        id="toy-face",
        backend=BackendSpec(name="insightface"),
        track=TrackSpec(iou=0.5, strategy="greedy"),
        crop=CropSpec(scale=1.3, size=16, square=True, align="none"),
        sampling=SamplingSpec(mode="uniform", frames=6),
        decode=DecodeSpec(library="opencv", color="rgb"),
        extras=ExtrasSpec(landmarks=False, mesh=False, masks=False),
    )
    spec = InputSpec(
        crop_scale=1.2,
        size=(12, 12),
        color="bgr",
        value_range=(-1.0, 1.0),
        mean=(0.1, 0.2, 0.3),
        std=(0.5, 0.6, 0.7),
    )
    transform = build_transforms(
        [
            ComponentSpec(name="hflip", p=0.5),
            ComponentSpec(name="random-resized-crop", size=16, scale=(0.5, 1.0)),
            ComponentSpec(name="color-jitter", brightness=0.2, hue=0.05),
            ComponentSpec(name="grayscale", p=0.2),
            ComponentSpec(name="gaussian-blur", kernel_size=3),
            ComponentSpec(name="gaussian-noise", std=0.02),
            ComponentSpec(name="jpeg", quality=(30, 90)),
        ]
    )
    clip_spec = ClipSpec(
        frames=2, sampling="uniform", clips_per_video=ClipsPerVideo(train=2, eval=1)
    )
    return ClipDataset(
        _video_index(items),
        clip_spec,
        train=True,
        transform=transform,
        adapt_chain=adapt(spec, profile).chain,
        seed=3,
    )


@pytest.mark.parametrize("context", ["forkserver", "spawn"])
def test_an_adapted_dataset_loads_in_forkserver_and_spawn_workers(tmp_path, context):
    # spawn and forkserver (the default start method on Python 3.14) pickle the dataset into
    # every worker: the transforms and the adaptation chain have to survive that, and the batches
    # have to come out exactly as a loader without workers makes them.
    from dfwb.data.collate import collate_clips

    dataset = _adapted_train_dataset(tmp_path)

    def _batches(**options: object) -> list[torch.Tensor]:
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_clips, **options)
        return [batch.clips for batch in loader]

    expected = _batches(num_workers=0)
    got = _batches(num_workers=2, multiprocessing_context=context)

    assert len(got) == len(expected) == 3
    for mine, theirs in zip(got, expected, strict=True):
        assert mine.shape == (2, 2, 3, 12, 12)
        torch.testing.assert_close(mine, theirs)


# -------------------------------------------------------------------------- transform / adapt


def test_transform_receives_a_generator_seeded_from_seed_epoch_and_index(tmp_path):
    item = _video_item(tmp_path / "v", n_frames=4)
    index = _video_index([item])
    spec = ClipSpec(frames=2, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))

    seeds_seen = []

    def transform(clip, *, generator=None):
        seeds_seen.append(generator.initial_seed() if generator is not None else None)
        return clip

    dataset = ClipDataset(index, spec, train=False, transform=transform, seed=5)
    dataset[0]

    assert seeds_seen == [_torch_seed(5, 0, 0)]


def test_transform_then_adapt_chain_run_in_order(tmp_path):
    item = _video_item(tmp_path / "v", n_frames=2)
    index = _video_index([item])
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))

    order = []

    def transform(clip, *, generator=None):
        order.append("transform")
        return clip

    def adapt(clip):
        order.append("adapt")
        return torch.ones_like(clip)

    dataset = ClipDataset(index, spec, train=False, transform=transform, adapt_chain=adapt, seed=0)
    sample = dataset[0]

    assert order == ["transform", "adapt"]
    assert torch.equal(sample.clip, torch.ones_like(sample.clip))


def test_no_transform_or_adapt_chain_leaves_clip_untouched(tmp_path):
    item = _video_item(tmp_path / "v", n_frames=2)
    index = _video_index([item])
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=1))
    dataset = ClipDataset(index, spec, train=False, seed=0)
    sample = dataset[0]
    assert sample.clip.shape == (1, 3, 4, 4)


# --------------------------------------------------------------------------------- MultiSource


def _build(tmp_path, name, n_videos, n_frames):
    items = [
        _video_item(tmp_path / f"{name}{i}", key=f"{name}{i}", n_frames=n_frames)
        for i in range(n_videos)
    ]
    index = _video_index(items)
    spec = ClipSpec(frames=1, sampling="uniform", clips_per_video=ClipsPerVideo(train=1, eval=2))
    return ClipDataset(index, spec, train=False, seed=0)


def test_multi_source_concatenates_normalises_weights_and_tags_source_id(tmp_path):
    ds1 = _build(tmp_path, "s1", n_videos=2, n_frames=5)  # length 4
    ds2 = _build(tmp_path, "s2", n_videos=1, n_frames=5)  # length 2

    multi = MultiSource([ds1, ds2], weights=[1.0, 3.0])

    assert len(multi) == len(ds1) + len(ds2)
    assert multi.weights == pytest.approx((0.25, 0.75))
    assert multi.source_of(0) == 0
    assert multi.source_of(len(ds1) - 1) == 0
    assert multi.source_of(len(ds1)) == 1
    assert multi.source_of(len(multi) - 1) == 1

    first = multi[0]
    assert first.extras["dfwb/source_id"] == 0
    assert first.key == ds1[0].key

    second = multi[len(ds1)]
    assert second.extras["dfwb/source_id"] == 1
    assert second.key == ds2[0].key
    assert torch.equal(second.clip, ds2[0].clip)


def test_multi_source_out_of_range_source_of_raises_index_error(tmp_path):
    ds1 = _build(tmp_path, "s1", n_videos=1, n_frames=5)
    multi = MultiSource([ds1], weights=[1.0])
    with pytest.raises(IndexError):
        multi.source_of(len(multi))


def test_multi_source_rejects_mismatched_weight_count(tmp_path):
    ds1 = _build(tmp_path, "s1", n_videos=1, n_frames=5)
    ds2 = _build(tmp_path, "s2", n_videos=1, n_frames=5)
    with pytest.raises(ConfigError, match="weights"):
        MultiSource([ds1, ds2], weights=[1.0])


@pytest.mark.parametrize("bad_weight", [0.0, -1.0])
def test_multi_source_rejects_a_non_positive_weight(tmp_path, bad_weight):
    ds1 = _build(tmp_path, "s1", n_videos=1, n_frames=5)
    ds2 = _build(tmp_path, "s2", n_videos=1, n_frames=5)
    with pytest.raises(ConfigError, match="positive"):
        MultiSource([ds1, ds2], weights=[1.0, bad_weight])
