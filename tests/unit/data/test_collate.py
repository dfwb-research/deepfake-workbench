"""``collate_clips``: ``ClipSample`` -> ``ClipBatch``, shapes for T=1 and T=8, and extras
pass-through."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from dfwb.data.collate import collate_clips
from dfwb.data.dataset import ClipSample, SkippedVideo


def _sample(
    *,
    key: str = "v",
    label: int | None = 0,
    frames: int = 1,
    clip_index: int = 0,
    dataset: str = "toy",
    compression: str | None = None,
    extras: dict | None = None,
) -> ClipSample:
    return ClipSample(
        clip=torch.rand(frames, 3, 4, 4),
        label=label,  # type: ignore[arg-type]
        key=key,
        dataset=dataset,
        compression=compression,
        clip_index=clip_index,
        frame_indices=list(range(frames)),
        extras=extras or {},
    )


@pytest.mark.parametrize("frames", [1, 8])
def test_collate_shapes(frames):
    samples = [_sample(key=f"v{i}", frames=frames, clip_index=i) for i in range(3)]

    batch = collate_clips(samples)

    assert batch.clips.shape == (3, frames, 3, 4, 4)
    assert batch.frame_indices.shape == (3, frames)
    assert batch.keys == ["v0", "v1", "v2"]
    assert batch.dataset_ids == ["toy", "toy", "toy"]
    assert batch.compressions == [None, None, None]
    assert batch.clip_index.tolist() == [0, 1, 2]
    assert batch.clip_index.dtype == torch.long
    assert batch.frame_indices.dtype == torch.long


def test_clips_stack_in_sample_order():
    samples = [_sample(key="a"), _sample(key="b"), _sample(key="c")]
    batch = collate_clips(samples)
    for i, sample in enumerate(samples):
        torch.testing.assert_close(batch.clips[i], sample.clip)


def test_labels_present_when_every_sample_has_one():
    samples = [_sample(label=0), _sample(label=1), _sample(label=1)]
    batch = collate_clips(samples)
    assert batch.labels is not None
    assert batch.labels.tolist() == [0, 1, 1]
    assert batch.labels.dtype == torch.long


def test_labels_is_none_when_any_sample_is_missing_one():
    samples = [_sample(label=0), _sample(label=None), _sample(label=1)]
    batch = collate_clips(samples)
    assert batch.labels is None


def test_compressions_pass_through_including_none():
    samples = [_sample(compression="c23"), _sample(compression=None), _sample(compression="c40")]
    batch = collate_clips(samples)
    assert batch.compressions == ["c23", None, "c40"]


def test_empty_samples_raises_value_error():
    with pytest.raises(ValueError, match="empty"):
        collate_clips([])


# ------------------------------------------------------------------------------ skipped videos


def test_a_skipped_video_is_dropped_and_listed_once():
    bad = SkippedVideo("toy", "bad", None)
    samples = [_sample(key="a"), bad, bad, _sample(key="b")]  # two clips of the same bad video

    batch = collate_clips(samples)

    assert batch.keys == ["a", "b"]
    assert batch.clips.shape[0] == 2
    assert batch.extras["dfwb/videos_skipped"] == [bad]


def test_a_batch_that_skipped_nothing_has_no_skip_entry():
    assert "dfwb/videos_skipped" not in collate_clips([_sample()]).extras


def test_a_batch_of_only_skipped_videos_is_an_empty_batch_that_still_counts_them():
    # e.g. a validation video, alone in its batch, none of whose stored frames can be read
    bad = SkippedVideo("toy", "bad", "c23")

    batch = collate_clips([bad, bad, bad, bad])

    assert batch.keys == []
    assert batch.dataset_ids == []
    assert batch.compressions == []
    assert batch.labels is None
    assert batch.clips.shape == (0,)
    assert batch.clip_index.shape == (0,)
    assert batch.frame_indices.shape == (0, 0)
    assert batch.extras == {"dfwb/videos_skipped": [bad]}


# ------------------------------------------------------------------------------------- extras


def test_extras_tensors_are_stacked():
    samples = [
        _sample(extras={"dfwb/feat": torch.tensor([1.0, 2.0])}),
        _sample(extras={"dfwb/feat": torch.tensor([3.0, 4.0])}),
    ]
    batch = collate_clips(samples)
    assert batch.extras["dfwb/feat"].shape == (2, 2)
    torch.testing.assert_close(batch.extras["dfwb/feat"][0], torch.tensor([1.0, 2.0]))


def test_extras_bools_become_a_bool_tensor():
    samples = [_sample(extras={"dfwb/padded": True}), _sample(extras={"dfwb/padded": False})]
    batch = collate_clips(samples)
    assert batch.extras["dfwb/padded"].dtype == torch.bool
    assert batch.extras["dfwb/padded"].tolist() == [True, False]


def test_extras_ints_become_a_long_tensor():
    samples = [_sample(extras={"dfwb/pair_id": 0}), _sample(extras={"dfwb/pair_id": 1})]
    batch = collate_clips(samples)
    assert batch.extras["dfwb/pair_id"].dtype == torch.long
    assert batch.extras["dfwb/pair_id"].tolist() == [0, 1]


def test_extras_other_values_become_a_list():
    samples = [_sample(extras={"dfwb/note": "a"}), _sample(extras={"dfwb/note": "b"})]
    batch = collate_clips(samples)
    assert batch.extras["dfwb/note"] == ["a", "b"]


def test_extras_key_missing_from_some_samples_falls_back_to_a_list():
    samples = [_sample(extras={"dfwb/only_first": 1}), _sample(extras={})]
    batch = collate_clips(samples)
    assert batch.extras["dfwb/only_first"] == [1, None]


def test_extras_with_no_keys_at_all_collates_to_an_empty_dict():
    samples = [_sample(), _sample()]
    batch = collate_clips(samples)
    assert batch.extras == {}
