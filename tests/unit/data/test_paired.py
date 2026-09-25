"""``PairedClipDataset``: real/fake clips tagged with a shared ``extras["dfwb/pair_id"]``, and
pairs whose real or fake member is missing from the index dropped and counted.

The last test wires this up against a real protocol (``Protocol.pairs``) and a real
``VideoIndex.build`` join, the way a training config actually would.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from tests.unit.data.conftest import processed_record, write_store_frames, write_store_index

from dfwb.data.clips import ClipSpec, ClipsPerVideo
from dfwb.data.collate import collate_clips
from dfwb.data.index import SourceSpec, VideoIndex, VideoItem
from dfwb.data.paired import PairedClipDataset
from dfwb.protocols.protocol import load


def _write_frames(dir_path: Path, n: int) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(dir_path / f"frame_{i:06d}.png"), np.zeros((4, 4, 3), dtype=np.uint8))


def _item(tmp_path: Path, *, key: str, label: int, n_frames: int = 4) -> VideoItem:
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


def _spec(*, clips_per_video: int = 1, frames: int = 1) -> ClipSpec:
    return ClipSpec(
        frames=frames,
        sampling="uniform",
        clips_per_video=ClipsPerVideo(train=clips_per_video, eval=clips_per_video),
    )


# --------------------------------------------------------------------------------- pairing itself


def test_paired_dataset_tags_rows_with_a_shared_pair_id_and_collates(tmp_path):
    items = [
        _item(tmp_path, key="REAL/0", label=0),
        _item(tmp_path, key="FAKE/0", label=1),
        _item(tmp_path, key="REAL/1", label=0),
        _item(tmp_path, key="FAKE/1", label=1),
    ]
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    pairs = [("REAL/0", "FAKE/0"), ("REAL/1", "FAKE/1")]

    dataset = PairedClipDataset(index, pairs, _spec(), train=False, seed=0)
    assert dataset.dropped == 0
    assert len(dataset) == 4  # 2 pairs * 1 clip/video * 2 sides

    samples = [dataset[i] for i in range(len(dataset))]
    batch = collate_clips(samples)

    assert batch.extras["dfwb/pair_id"].tolist() == [0, 1, 0, 1]
    assert batch.labels is not None
    assert batch.labels.tolist() == [0, 0, 1, 1]
    assert batch.keys == ["REAL/0", "REAL/1", "FAKE/0", "FAKE/1"]

    # each pair_id appears exactly twice, once for each label -- real and fake share it.
    by_pair: dict[int, set[int]] = {}
    for pair_id, label in zip(
        batch.extras["dfwb/pair_id"].tolist(), batch.labels.tolist(), strict=True
    ):
        by_pair.setdefault(pair_id, set()).add(label)
    assert by_pair == {0: {0, 1}, 1: {0, 1}}


def test_multiple_clips_per_video_share_the_same_pair_id(tmp_path):
    items = [
        _item(tmp_path, key="REAL/0", label=0, n_frames=10),
        _item(tmp_path, key="FAKE/0", label=1, n_frames=10),
    ]
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    dataset = PairedClipDataset(
        index, [("REAL/0", "FAKE/0")], _spec(clips_per_video=3), train=False, seed=0
    )

    assert len(dataset) == 6  # 1 pair * 3 clips/video * 2 sides
    pair_ids = {dataset[i].extras["dfwb/pair_id"] for i in range(len(dataset))}
    assert pair_ids == {0}


# ------------------------------------------------------------------------------------- dropping


def test_a_pair_missing_its_fake_member_is_dropped_and_counted(tmp_path):
    items = [_item(tmp_path, key="REAL/0", label=0)]
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    pairs = [("REAL/0", "FAKE/missing")]

    dataset = PairedClipDataset(index, pairs, _spec(), train=False, seed=0)

    assert dataset.dropped == 1
    assert len(dataset) == 0


def test_a_pair_missing_its_real_member_is_dropped_and_counted(tmp_path):
    items = [_item(tmp_path, key="FAKE/0", label=1)]
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    pairs = [("REAL/missing", "FAKE/0")]

    dataset = PairedClipDataset(index, pairs, _spec(), train=False, seed=0)

    assert dataset.dropped == 1
    assert len(dataset) == 0


def test_only_valid_pairs_are_kept_when_some_are_dropped(tmp_path):
    items = [
        _item(tmp_path, key="REAL/0", label=0),
        _item(tmp_path, key="FAKE/0", label=1),
        _item(tmp_path, key="REAL/1", label=0),
        # FAKE/1 never makes it into the index.
    ]
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    pairs = [("REAL/0", "FAKE/0"), ("REAL/1", "FAKE/1")]

    dataset = PairedClipDataset(index, pairs, _spec(), train=False, seed=0)

    assert dataset.dropped == 1
    assert len(dataset) == 2  # only the REAL/0<->FAKE/0 pair survives
    samples = [dataset[i] for i in range(len(dataset))]
    assert {s.key for s in samples} == {"REAL/0", "FAKE/0"}
    assert {s.extras["dfwb/pair_id"] for s in samples} == {0}


# ------------------------------------------------------------------------------------- epoch


def test_set_epoch_reaches_both_the_real_and_the_fake_side(tmp_path):
    items = [
        _item(tmp_path, key="REAL/0", label=0, n_frames=50),
        _item(tmp_path, key="FAKE/0", label=1, n_frames=50),
    ]
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    spec = ClipSpec(
        frames=3, sampling="consecutive", clips_per_video=ClipsPerVideo(train=1, eval=1)
    )
    dataset = PairedClipDataset(index, [("REAL/0", "FAKE/0")], spec, train=True, seed=1)

    epoch0 = [dataset[i].frame_indices for i in range(len(dataset))]
    dataset.set_epoch(1)
    epoch1 = [dataset[i].frame_indices for i in range(len(dataset))]

    assert epoch0 != epoch1


def test_two_instances_same_seed_are_deterministic(tmp_path):
    items = [
        _item(tmp_path, key="REAL/0", label=0, n_frames=20),
        _item(tmp_path, key="FAKE/0", label=1, n_frames=20),
    ]
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    spec = ClipSpec(
        frames=2, sampling="consecutive", clips_per_video=ClipsPerVideo(train=2, eval=1)
    )

    a = PairedClipDataset(index, [("REAL/0", "FAKE/0")], spec, train=True, seed=3)
    b = PairedClipDataset(index, [("REAL/0", "FAKE/0")], spec, train=True, seed=3)

    for i in range(len(a)):
        sa, sb = a[i], b[i]
        assert sa.frame_indices == sb.frame_indices
        assert torch.equal(sa.clip, sb.clip)


def test_index_out_of_range_raises_index_error(tmp_path):
    items = [
        _item(tmp_path, key="REAL/0", label=0),
        _item(tmp_path, key="FAKE/0", label=1),
    ]
    index = VideoIndex(items=items, excluded=[], _summaries=[])
    dataset = PairedClipDataset(index, [("REAL/0", "FAKE/0")], _spec(), train=False, seed=0)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


# --------------------------------------------------------------------------------- integration


def test_paired_dataset_from_a_real_protocols_pairs_and_videoindex_join(tmp_path, toyone_pack):
    work_root = tmp_path / "work"
    profile = "toy-profile"
    store_dir = work_root / "toyone" / "processed" / profile
    real_record = processed_record("REAL/r1", compression="c23")
    fake_record = processed_record("FAKE_A/a1")
    write_store_index(store_dir, [real_record, fake_record])
    write_store_frames(store_dir, real_record)
    write_store_frames(store_dir, fake_record)

    protocol = load("toyone-pack:toyone/official", work_root=work_root)
    pairs = protocol.pairs(split="train")
    # FAKE_A/a1<->REAL/r1 (both train), and FAKE_B/b4<->REAL/r4 (r4 unassigned, falls back to
    # b4's split=train, but FAKE_B is label-excluded under "binary" and REAL/r4 is never in the
    # scheme at all) -- so exactly one of the two pairs can possibly resolve.
    assert ("REAL/r1", "FAKE_A/a1") in pairs

    index = VideoIndex.build(
        [SourceSpec("toyone-pack:toyone/official", "train")],
        profile=profile,
        labels="binary",
        work_root=work_root,
    )

    dataset = PairedClipDataset(index, pairs, _spec(), train=False, seed=0)

    assert dataset.dropped == len(pairs) - 1
    assert len(dataset) == 2  # the one surviving pair, 1 clip/video, 2 sides
    samples = [dataset[i] for i in range(len(dataset))]
    assert {s.key for s in samples} == {"REAL/r1", "FAKE_A/a1"}
    assert {s.extras["dfwb/pair_id"] for s in samples} == {0}
