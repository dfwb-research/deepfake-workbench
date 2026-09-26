"""``ProtocolDataModule``: sources joined from protocols and a processed store, adapted to the
detector's input spec, and served through the samplers ``data.loader`` names."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("lightning")

import torch
from tests.unit.train._toy import (
    DATASET,
    PROFILE,
    make_parts,
    toy_config,
    toy_profile,
    toy_source,
    write_toy_store,
)

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError, ContractError
from dfwb.data.dataset import ClipDataset, MultiSource
from dfwb.data.index import VideoIndex
from dfwb.data.paired import PairedClipDataset
from dfwb.data.samplers import (
    PairGrouped,
    SourceBalanced,
    VideoGrouped,
    VideoLabelBalanced,
    epoch_seed,
)
from dfwb.train.datamodule import BALANCE_MODES, ProtocolDataModule, source_names

TINY_INPUT = InputSpec(size=(64, 64), value_range=(0.0, 1.0), mean=None, std=None)


def _datamodule(config, work_root, **options) -> ProtocolDataModule:
    return ProtocolDataModule(
        config.data, input_spec=TINY_INPUT, work_root=work_root, seed=0, **options
    )


# ------------------------------------------------------------------------------------ setup


def test_source_names_come_from_the_protocols_alone(toy_work_root, monkeypatch):
    entries = [toy_source("val", **{"attrs.group": "a"}), toy_source("val"), toy_source("val")]
    config = toy_config(data={"val": entries})
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")
    expected = [source.name for source in datamodule.val_sources]

    def _refuse(*args, **kwargs):
        raise AssertionError("a processed store was read")

    monkeypatch.setattr(VideoIndex, "build", _refuse)
    assert source_names(config.data.val, work_root=toy_work_root) == expected
    assert expected == [f"{DATASET}-official-a", f"{DATASET}-official", f"{DATASET}-official-2"]


def test_setup_joins_each_source_and_names_it(toy_work_root):
    config = toy_config(
        data={"val": [toy_source("val", **{"attrs.group": "a"}), toy_source("val")]}
    )
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")

    (train,) = datamodule.train_sources
    assert train.name == f"{DATASET}-official"
    assert len(train.index.items) == 16
    assert train.profile.id == PROFILE
    assert [source.name for source in datamodule.val_sources] == [
        f"{DATASET}-official-a",
        f"{DATASET}-official",
    ]
    assert isinstance(datamodule.train_dataset, MultiSource)
    assert len(datamodule.train_dataset) == 16 * 2  # 16 videos, 2 train clips each


def test_duplicate_source_names_get_a_suffix(toy_work_root):
    config = toy_config(data={"val": [toy_source("val"), toy_source("val")]})
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")
    assert [s.name for s in datamodule.val_sources] == [
        f"{DATASET}-official",
        f"{DATASET}-official-2",
    ]


def test_setup_is_idempotent(toy_work_root):
    datamodule = _datamodule(toy_config(), toy_work_root)
    datamodule.setup("fit")
    first = datamodule.train_dataset
    datamodule.setup("validate")
    assert datamodule.train_dataset is first


def test_normalize_in_transforms_fails_at_setup_before_any_data_loads(tmp_path):
    config = toy_config(
        data={"transforms": {"train": [{"name": "normalize", "mean": [0.5], "std": [0.5]}]}}
    )
    # nothing exists under this work root: loading any data at all would fail differently.
    datamodule = _datamodule(config, tmp_path / "nothing-here")
    with pytest.raises(ConfigError, match="normalize"):
        datamodule.setup("fit")


def test_adapt_refusal_propagates(tmp_path, toy_pack):
    work_root = tmp_path / "work"
    write_toy_store(work_root, profile=toy_profile(backend="center", scale=1.0))
    datamodule = _datamodule(toy_config(), work_root)
    with pytest.raises(ContractError, match="full-frame"):
        datamodule.setup("fit")


def test_an_adapt_refusal_lists_the_shipped_profiles_it_is_given(tmp_path, toy_pack):
    work_root = tmp_path / "work"
    write_toy_store(work_root, profile=toy_profile(backend="center", scale=1.0))
    shipped = [toy_profile(backend="insightface", scale=1.3, size=64)]
    datamodule = _datamodule(toy_config(), work_root, shipped_profiles=shipped)
    with pytest.raises(ContractError, match=f"would serve it: {PROFILE}"):
        datamodule.setup("fit")


def test_allow_mismatch_proceeds_and_records_it(tmp_path, toy_pack):
    work_root = tmp_path / "work"
    write_toy_store(work_root, profile=toy_profile(backend="center", scale=1.0))
    datamodule = _datamodule(toy_config(), work_root, allow_mismatch=True)
    datamodule.setup("fit")
    assert datamodule.train_sources[0].adaptation.mismatch is True


def test_a_store_without_a_profile_file_is_a_contract_error(toy_work_root):
    for profile_file in (toy_work_root / DATASET / "processed").glob("*/profile.json"):
        profile_file.unlink()
    datamodule = _datamodule(toy_config(), toy_work_root)
    with pytest.raises(ContractError, match="profile"):
        datamodule.setup("fit")


def test_a_mis_sized_stored_frame_is_a_clear_error_during_training(toy_work_root):
    # Every frame of one video is re-sized (not just one): the clip window's random draw would
    # otherwise sometimes miss the corrupted file and make this test flaky.
    profile = toy_profile()
    frame_dir = toy_work_root / DATASET / "processed" / profile.profile_id() / "REAL" / "r00" / "_"
    for frame in frame_dir.glob("frame_*.png"):
        cv2.imwrite(str(frame), np.zeros((16, 16, 3), dtype=np.uint8))  # profile.crop.size is 32

    datamodule = _datamodule(toy_config(), toy_work_root)
    datamodule.setup("fit")

    # 48 clips / batch_size 8: no trailing partial batch is dropped, so a full pass is guaranteed
    # to visit every video, including the one just corrupted -- unlike train mode's usual
    # corrupt-*frame* tolerance (see test_dataset.py), a size mismatch always raises.
    with pytest.raises(ContractError, match="not 32x32"):
        for _ in datamodule.train_dataloader():
            pass


# ---------------------------------------------------------------------------------- loaders


def test_train_batches_are_adapted_clips_tagged_with_their_source(toy_work_root):
    datamodule = _datamodule(toy_config(), toy_work_root)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    assert batch.clips.shape == (8, 1, 3, 64, 64)  # stored at 32 px, resized by adapt()
    assert batch.labels is not None
    assert batch.extras["dfwb/source_id"].tolist() == [0] * 8


def test_transforms_apply_to_train_only(toy_work_root):
    config = toy_config(data={"transforms": {"train": [{"name": "hflip", "p": 1.0}]}})
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")
    train_multi = datamodule.train_dataset
    assert train_multi is not None

    # the train source and an untransformed train dataset over the same videos, seed and epoch
    # draw the same frames, so the only difference is the transform.
    train_sample = train_multi[0]
    same_video = ClipDataset(
        datamodule.train_sources[0].index,
        datamodule.train_sources[0].dataset.spec,
        train=True,
        adapt_chain=datamodule.train_sources[0].adaptation.chain,
        seed=0,
    )[0]
    torch.testing.assert_close(train_sample.clip, same_video.clip.flip(-1))

    val_sample = datamodule.val_sources[0].dataset[0]
    assert val_sample.clip.shape == train_sample.clip.shape
    val_batch = next(iter(datamodule.val_dataloader()[0]))
    assert torch.equal(val_batch.clips[0], val_sample.clip)


@pytest.mark.parametrize(
    ("balance", "mode", "expected"),
    [
        (None, "none", "shuffle"),
        ("none", "none", "shuffle"),
        ("video-label", "video-label", VideoLabelBalanced),
        ("source", "source", SourceBalanced),
    ],
)
def test_balance_chooses_the_train_sampler(toy_work_root, balance, mode, expected):
    config = toy_config(data={"loader": {"batch_size": 8, "num_workers": 0, "balance": balance}})
    datamodule = _datamodule(config, toy_work_root)
    assert datamodule.balance == mode
    datamodule.setup("fit")
    sampler = datamodule.train_dataloader().sampler
    if expected == "shuffle":
        assert not isinstance(sampler, (VideoLabelBalanced, SourceBalanced))
        assert sorted(sampler) == list(range(len(datamodule.train_dataset)))
    else:
        assert isinstance(sampler, expected)


def test_source_weights_drive_source_balancing(toy_work_root):
    group_a = {**toy_source("train", **{"attrs.group": "a"}), "weight": 3.0}
    group_b = toy_source("train", **{"attrs.group": "b"})
    config = toy_config(
        data={
            "train": [group_a, group_b],
            "loader": {"batch_size": 8, "num_workers": 0, "balance": "source"},
        }
    )
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")
    assert datamodule.train_dataset.weights == pytest.approx((0.75, 0.25))
    sampler = datamodule.train_dataloader().sampler
    n_a = len(datamodule.train_dataset.datasets[0])
    draws = []
    for epoch in range(20):
        sampler.set_epoch(epoch)
        draws += list(sampler)
    share_a = sum(index < n_a for index in draws) / len(draws)
    assert share_a == pytest.approx(0.75, abs=0.05)


def test_a_source_weight_without_source_balancing_is_warned_about(toy_work_root, caplog):
    config = toy_config(data={"train": [{**toy_source("train"), "weight": 2.0}]})
    datamodule = _datamodule(config, toy_work_root)
    with caplog.at_level("WARNING"):
        datamodule.setup("fit")
    assert "data.train[0].weight" in caplog.text
    assert "balance: source" in caplog.text


def test_an_unknown_balance_is_a_config_error_naming_video_label(toy_work_root):
    # The config schema already refuses it; a data section built without validation still is.
    config = toy_config()
    loader = config.data.loader.model_copy(update={"balance": "video-labl"})
    data = config.data.model_copy(update={"loader": loader})
    with pytest.raises(ConfigError, match=r"data\.loader\.balance") as caught:
        ProtocolDataModule(data, input_spec=TINY_INPUT, work_root=toy_work_root, seed=0)
    assert "did you mean 'video-label'" in caught.value.message
    assert "video-label" in caught.value.hint
    assert BALANCE_MODES == ("none", "video-label", "source")


def test_the_plain_shuffle_is_seeded_by_seed_and_epoch(toy_work_root):
    datamodule = _datamodule(toy_config(), toy_work_root)
    datamodule.setup("fit")
    sampler = datamodule.train_dataloader().sampler
    datamodule.set_epoch(2)
    n = len(datamodule.train_dataset)
    expected = torch.randperm(n, generator=torch.Generator().manual_seed(epoch_seed(0, 2)))
    assert list(sampler) == expected.tolist()


def test_the_train_loader_drops_a_trailing_partial_batch(toy_work_root):
    # 32 train clips at batch 31: the trailing batch of one is dropped, never trained on (a
    # batch-norm head cannot train on a single sample).
    config = toy_config(data={"loader": {"batch_size": 31, "num_workers": 0}})
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    assert [len(batch.keys) for batch in loader] == [31]
    assert len(loader) == 1


def test_a_train_set_smaller_than_one_batch_is_a_config_error(toy_work_root):
    config = toy_config(data={"loader": {"batch_size": 33, "num_workers": 0}})
    datamodule = _datamodule(config, toy_work_root)
    with pytest.raises(ConfigError, match=r"data\.loader\.batch_size.*32"):
        datamodule.setup("fit")


def test_val_loaders_group_each_video_without_shuffling(toy_work_root):
    config = toy_config(data={"loader": {"batch_size": 3, "num_workers": 0}})
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")
    (loader,) = datamodule.val_dataloader()

    assert isinstance(loader.batch_sampler, VideoGrouped)
    batches = list(loader)
    for batch in batches:
        keys = batch.keys
        assert all(keys.count(key) == 2 for key in keys)  # both eval clips of a video together
    assert [b.keys for b in loader] == [b.keys for b in batches]  # fixed order, no shuffle


def test_train_workers_persist_and_validation_workers_do_not(toy_work_root):
    # validation workers are started for each validation pass and stop after it, so several
    # validation sources never hold num_workers idle processes each while training runs
    config = toy_config(
        data={
            "loader": {"batch_size": 8, "num_workers": 2},
            "val": [toy_source("val"), toy_source("val", **{"attrs.group": "a"})],
        }
    )
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")
    train = datamodule.train_dataloader()
    assert (train.num_workers, train.persistent_workers) == (2, True)
    for val in datamodule.val_dataloader():
        assert (val.num_workers, val.persistent_workers) == (2, False)

    single = _datamodule(toy_config(), toy_work_root)
    single.setup("fit")
    assert single.train_dataloader().persistent_workers is False


@pytest.mark.parametrize(("device", "pinned"), [("cuda", True), ("cpu", False)])
def test_loaders_pin_memory_only_for_a_cuda_device(toy_work_root, device, pinned):
    from types import SimpleNamespace

    datamodule = _datamodule(toy_config(), toy_work_root)
    datamodule.setup("fit")
    datamodule.trainer = SimpleNamespace(
        current_epoch=0, strategy=SimpleNamespace(root_device=torch.device(device))
    )
    assert datamodule.train_dataloader().pin_memory is pinned
    assert all(loader.pin_memory is pinned for loader in datamodule.val_dataloader())


def test_loaders_do_not_pin_memory_without_a_trainer(toy_work_root):
    datamodule = _datamodule(toy_config(), toy_work_root)
    datamodule.setup("fit")
    assert datamodule.train_dataloader().pin_memory is False


@pytest.mark.parametrize("pairs", [False, True], ids=["plain", "pairs"])
def test_every_dataset_pickles_for_loader_workers(toy_work_root, pairs):
    # spawn and forkserver workers (forkserver is Python 3.14's default) receive the dataset,
    # its transforms and its adaptation chain pickled
    from multiprocessing.reduction import ForkingPickler

    import torch.multiprocessing  # registers torch's tensor reductions with ForkingPickler

    config = toy_config(
        data={
            "transforms": {"train": [{"name": "hflip", "p": 0.5}, {"name": "jpeg", "quality": 50}]}
        }
    )
    datamodule = ProtocolDataModule(
        config.data,
        input_spec=InputSpec(size=(24, 24), mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        work_root=toy_work_root,
        seed=0,
        pairs=pairs,
    )
    datamodule.setup("fit")
    datasets = [datamodule.train_dataset, *(source.dataset for source in datamodule.val_sources)]
    for dataset in datasets:
        restored = ForkingPickler.loads(ForkingPickler.dumps(dataset))
        assert len(restored) == len(dataset)
        torch.testing.assert_close(restored[0].clip, dataset[0].clip)


@pytest.mark.parametrize("context", ["forkserver", "spawn"])
def test_the_train_loader_runs_in_forkserver_and_spawn_workers(toy_work_root, context):
    config = toy_config(data={"loader": {"batch_size": 8, "num_workers": 2}})
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    loader.multiprocessing_context = context
    batches = list(loader)
    assert [len(batch.keys) for batch in batches] == [8] * 4


def test_set_epoch_reaches_the_train_datasets_and_sampler(toy_work_root):
    config = toy_config(
        data={"loader": {"batch_size": 8, "num_workers": 0, "balance": "video-label"}}
    )
    datamodule = _datamodule(config, toy_work_root)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    epoch0 = list(loader.sampler)

    datamodule.set_epoch(3)

    assert all(d.epoch == 3 for d in datamodule.train_dataset.datasets)
    assert list(loader.sampler) != epoch0
    reference = VideoLabelBalanced(datamodule.train_dataset, seed=0)
    reference.set_epoch(3)
    assert list(loader.sampler) == list(reference)


# ------------------------------------------------------------------------------------ pairs


def test_pairs_build_paired_sources_and_keep_each_pair_in_one_batch(toy_work_root):
    config = toy_config(data={"loader": {"batch_size": 4, "num_workers": 0}})
    datamodule = _datamodule(config, toy_work_root, pairs=True)
    datamodule.setup("fit")

    assert isinstance(datamodule.train_sources[0].dataset, PairedClipDataset)
    assert isinstance(datamodule.val_sources[0].dataset, ClipDataset)  # val is never paired
    loader = datamodule.train_dataloader()
    assert isinstance(loader.batch_sampler, PairGrouped)

    seen = 0
    for batch in loader:
        seen += len(batch.keys)
        by_pair: dict[int, set[int]] = {}
        for pair_id, label in zip(
            batch.extras["dfwb/pair_id"].tolist(), batch.labels.tolist(), strict=True
        ):
            by_pair.setdefault(pair_id, set()).add(label)
        assert all(labels == {0, 1} for labels in by_pair.values())
    assert seen == len(datamodule.train_dataset) == 8 * 2 * 2  # 8 pairs, 2 sides, 2 clips


def test_pairs_keep_a_short_last_batch(toy_work_root):
    # pair batches always hold a real and a fake row, so nothing needs dropping: 8 pairs of 4
    # rows at batch 12 are three pairs, three pairs, then the last two.
    config = toy_config(data={"loader": {"batch_size": 12, "num_workers": 0}})
    datamodule = _datamodule(config, toy_work_root, pairs=True)
    datamodule.setup("fit")
    assert sorted(len(batch.keys) for batch in datamodule.train_dataloader()) == [8, 12, 12]


def test_pairs_with_no_surviving_pair_are_a_config_error(toy_work_root):
    # only reals in the training split: every pair loses its fake side.
    config = toy_config(data={"train": [toy_source("train", task="REAL")]})
    datamodule = _datamodule(config, toy_work_root, pairs=True)
    with pytest.raises(ConfigError, match="pair"):
        datamodule.setup("fit")


def test_pairs_cannot_be_combined_with_balance(toy_work_root):
    config = toy_config(
        data={"loader": {"batch_size": 4, "num_workers": 0, "balance": "video-label"}}
    )
    with pytest.raises(ConfigError, match="pairs"):
        _datamodule(config, toy_work_root, pairs=True)


# -------------------------------------------------------------------------- epochs in a fit


def test_each_train_epoch_start_sets_the_epoch(tmp_path, toy_work_root):
    from lightning.pytorch.callbacks import Callback
    from tests.unit.train._toy import fit_toy

    seen: list[int] = []

    class _Record(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            seen.append(trainer.datamodule.train_dataset.datasets[0].epoch)

    fit_toy(
        tmp_path / "run",
        toy_config(train={"max_epochs": 3}),
        work_root=toy_work_root,
        callbacks=[_Record()],
    )
    assert seen == [0, 1, 2]


def test_parts_share_the_detectors_input_spec(toy_work_root):
    detector, datamodule, _ = make_parts(toy_config(), work_root=toy_work_root)
    assert datamodule.input_spec == detector.meta.input
