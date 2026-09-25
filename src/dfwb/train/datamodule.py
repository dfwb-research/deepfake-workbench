"""``ProtocolDataModule``: the ``data:`` section of a training config, as Lightning data loaders.

Each ``data.train``/``data.val`` entry becomes one :class:`SourceData`: its protocol split joined
with the processed store (:class:`~dfwb.data.index.VideoIndex`), the input adaptation from that
store's processing profile to the detector's own input spec (:func:`~dfwb.data.adapt.adapt`), and
a clip dataset over the result. Training sources are concatenated into one
:class:`~dfwb.data.dataset.MultiSource`, sampled as ``data.loader.balance`` says; every validation
source keeps a loader of its own, grouped by video and never shuffled, so its clips can be
aggregated per video and scored per source.

Everything seeded here -- train clip windows, transforms, the sampler's draws -- is a function of
``(seed, epoch)``, and :meth:`ProtocolDataModule.set_epoch` moves every one of them to a new epoch
at once, so re-running an epoch (after resuming, say) draws exactly the same batches again.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader, Sampler

from dfwb.core.config.schema import DataSection, DataSource
from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError, ContractError, did_you_mean
from dfwb.core.records.local import ProcessingProfile
from dfwb.data.adapt import AdaptResult, adapt, available_profiles
from dfwb.data.clips import ClipSpec
from dfwb.data.collate import collate_clips
from dfwb.data.dataset import ClipDataset, ClipSample, ClipTransform, MultiSource
from dfwb.data.index import SourceSpec, VideoIndex
from dfwb.data.paired import PairedClipDataset
from dfwb.data.samplers import (
    PairGrouped,
    SourceBalanced,
    VideoGrouped,
    VideoLabelBalanced,
    epoch_seed,
)
from dfwb.data.transforms import build_transforms
from dfwb.protocols.protocol import Protocol as LoadedProtocol
from dfwb.protocols.protocol import load

__all__ = ["BALANCE_MODES", "ProtocolDataModule", "SourceData"]

_log = logging.getLogger(__name__)

#: ``data.loader.balance``: ``none`` (a plain seeded shuffle), ``video-label`` (real and fake
#: videos equally likely, :class:`~dfwb.data.samplers.VideoLabelBalanced`) or ``source`` (each
#: training source weighted equally, :class:`~dfwb.data.samplers.SourceBalanced`). Unset means
#: ``none``.
BALANCE_MODES = ("none", "video-label", "source")

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._+-]+")


@dataclass(frozen=True)
class SourceData:
    """One configured source, joined and ready to load.

    ``name`` is unique among the sources of its role (train or val) and safe as a file name: it
    names the source's logged metrics (``val/<name>/<metric>``) and its validation score file.
    """

    name: str
    entry: DataSource
    protocol: LoadedProtocol
    index: VideoIndex
    profile: ProcessingProfile
    adaptation: AdaptResult
    dataset: ClipDataset | PairedClipDataset


class _EpochSampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


class _EpochShuffle(Sampler[int]):
    """Every index of a dataset once per epoch, in an order seeded by ``(seed, epoch)``."""

    def __init__(self, length: int, *, seed: int) -> None:
        self.length = length
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(epoch_seed(self.seed, self._epoch))
        return iter(torch.randperm(self.length, generator=generator).tolist())


def _balance_mode(value: str | None, *, pairs: bool) -> str:
    mode = "none" if value is None else value
    if mode not in BALANCE_MODES:
        raise ConfigError(
            f"data.loader.balance: {value!r} is not one of {list(BALANCE_MODES)}"
            f"{did_you_mean(str(value), BALANCE_MODES)}",
            hint="balance is none, video-label or source",
        )
    if pairs and mode != "none":
        raise ConfigError(
            f"data.loader.balance: {value!r} cannot be combined with training on pairs",
            hint="pairs are already one real and one fake each, batched pair by pair; "
            "drop the balance setting",
        )
    return mode


def _where_part(value: Any) -> str:
    if isinstance(value, list):
        return "+".join(str(v) for v in value)
    return str(value)


def _base_name(protocol: LoadedProtocol, entry: DataSource) -> str:
    parts = [protocol.dataset, protocol.scheme]
    parts.extend(_where_part(entry.where[key]) for key in sorted(entry.where))
    return _UNSAFE_NAME.sub("_", "-".join(parts))


def _unique_names(bases: Sequence[str]) -> list[str]:
    """``bases``, with ``-2``, ``-3``, ... appended to the second and later repeats of a name."""
    seen: dict[str, int] = {}
    names: list[str] = []
    for base in bases:
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return names


class ProtocolDataModule(L.LightningDataModule):
    """Builds the training and validation data of one run from a config's ``data:`` section.

    Args:
        data: The resolved ``data:`` section.
        input_spec: The detector's input spec (``detector.meta.input``): every source's clips are
            adapted to it from its store's processing profile.
        work_root: Where the processed stores (and any materialized protocols) live.
        seed: The run's seed, for train clip windows, transforms and the train sampler.
        pairs: Train on real/fake pairs from each training protocol's pair list, batched so a
            pair's real and fake rows always share a batch. Validation is never paired.
        allow_mismatch: Let :func:`~dfwb.data.adapt.adapt` proceed past an input it would
            otherwise refuse (a crop-kind mismatch, or a wider crop than the store kept); each
            source records it in ``adaptation.mismatch``.

    Raises:
        ConfigError: ``data.loader.balance`` is unknown, or is combined with ``pairs``.
    """

    def __init__(
        self,
        data: DataSection,
        *,
        input_spec: InputSpec,
        work_root: Path,
        seed: int,
        pairs: bool = False,
        allow_mismatch: bool = False,
    ) -> None:
        super().__init__()
        self.data = data
        self.input_spec = input_spec
        self.work_root = Path(work_root)
        self.seed = seed
        self.pairs = pairs
        self.allow_mismatch = allow_mismatch
        self.balance = _balance_mode(data.loader.balance, pairs=pairs)
        self.train_sources: list[SourceData] = []
        self.val_sources: list[SourceData] = []
        self.train_dataset: MultiSource | None = None
        self._train_samplers: list[_EpochSampler] = []
        self._epoch = 0
        self._is_setup = False

    @staticmethod
    def attached_to(trainer: L.Trainer) -> ProtocolDataModule | None:
        """The ``ProtocolDataModule`` ``trainer`` is fitting with, if that is what it has
        (Lightning attaches a data module to its trainer only once ``fit`` starts)."""
        datamodule = getattr(trainer, "datamodule", None)
        return datamodule if isinstance(datamodule, ProtocolDataModule) else None

    # ------------------------------------------------------------------------------- setup

    def setup(self, stage: str | None = None) -> None:
        """Join, adapt and build every source (once; later calls do nothing).

        The train transforms are built first, so a transform list that can never work (one that
        names ``normalize``, say) fails before any protocol or store is read.

        Raises:
            ConfigError: A transform is unknown or not allowed; a source's protocol, split or
                processed store cannot be resolved; or the training data cannot fill one batch
                (one pair, when training on pairs).
            ContractError: A store has no readable ``profile.json``, or its profile cannot serve
                the detector's input spec (see :func:`~dfwb.data.adapt.adapt`).
        """
        if self._is_setup:
            return
        transform = (
            build_transforms(self.data.transforms.train) if self.data.transforms.train else None
        )
        spec = ClipSpec.from_config(self.data.clip)
        self.train_sources = self._build_sources(self.data.train, spec, transform, train=True)
        self.val_sources = self._build_sources(self.data.val, spec, None, train=False)
        self.train_dataset = MultiSource(
            [source.dataset for source in self.train_sources],
            weights=[1.0] * len(self.train_sources),
        )
        self._check_train_size(len(self.train_dataset))
        self._is_setup = True

    def _check_train_size(self, n_clips: int) -> None:
        """Unpaired training drops a trailing short batch, so it needs at least one full batch;
        paired training needs at least one pair."""
        batch_size = self.data.loader.batch_size
        if self.pairs and n_clips == 0:
            raise ConfigError(
                "data.train: no real/fake pair has both of its videos in the joined split",
                hint="check the training protocols' pairs and that both sides are processed",
            )
        if not self.pairs and n_clips < batch_size:
            raise ConfigError(
                f"data.loader.batch_size: {batch_size} is more than the {n_clips} training "
                "clip(s) available, so not one full batch can be drawn",
                hint="lower data.loader.batch_size, or add training data",
            )

    def _build_sources(
        self,
        entries: Sequence[DataSource],
        spec: ClipSpec,
        transform: ClipTransform | None,
        *,
        train: bool,
    ) -> list[SourceData]:
        built = [self._build_source(entry, spec, transform, train=train) for entry in entries]
        names = _unique_names([source.name for source in built])
        return [replace(source, name=name) for source, name in zip(built, names, strict=True)]

    def _build_source(
        self,
        entry: DataSource,
        spec: ClipSpec,
        transform: ClipTransform | None,
        *,
        train: bool,
    ) -> SourceData:
        """One source, named by its protocol and ``where`` alone (made unique by the caller)."""
        protocol = load(entry.protocol, work_root=self.work_root)
        index = VideoIndex.build(
            [SourceSpec(entry.protocol, entry.split, entry.where or None)],
            profile=self.data.processing,
            labels=self.data.labels,
            work_root=self.work_root,
        )
        profile_id = index.summary()["sources"][0]["profile_id"]
        candidates = available_profiles(self.work_root, protocol.dataset)
        profile = next((p for p in candidates if p.profile_id() == profile_id), None)
        if profile is None:
            raise ContractError(
                f"{protocol.ref}: the processed store {profile_id!r} has no readable profile.json",
                hint="re-run preprocessing for this dataset and profile, which writes it",
            )
        adaptation = adapt(
            self.input_spec, profile, allow_mismatch=self.allow_mismatch, candidates=candidates
        )
        if adaptation.mismatch:
            _log.warning("%s: input mismatch allowed: %s", protocol.ref, adaptation.reason)

        dataset: ClipDataset | PairedClipDataset
        if train and self.pairs:
            dataset = PairedClipDataset(
                index,
                protocol.pairs(split=entry.split),
                spec,
                train=True,
                transform=transform,
                adapt_chain=adaptation.chain,
                seed=self.seed,
            )
            if dataset.dropped:
                _log.info(
                    "%s: %d pair(s) dropped (a member is not in the joined split)",
                    protocol.ref,
                    dataset.dropped,
                )
        else:
            dataset = ClipDataset(
                index,
                spec,
                train=train,
                transform=transform,
                adapt_chain=adaptation.chain,
                seed=self.seed,
            )
        return SourceData(
            _base_name(protocol, entry), entry, protocol, index, profile, adaptation, dataset
        )

    # ----------------------------------------------------------------------------- loaders

    def _loader_options(self) -> dict[str, Any]:
        workers = self.data.loader.num_workers
        return {
            "collate_fn": collate_clips,
            "num_workers": workers,
            "persistent_workers": workers > 0,
        }

    def _require_setup(self) -> MultiSource:
        if not self._is_setup or self.train_dataset is None:
            raise RuntimeError("ProtocolDataModule: call setup() before asking for a loader")
        return self.train_dataset

    def train_dataloader(self) -> DataLoader[ClipSample]:
        """The training loader, already at the trainer's current epoch (so a resumed run
        continues with the draws of the epoch it resumes at)."""
        multi = self._require_setup()
        batch_size = self.data.loader.batch_size
        loader: DataLoader[ClipSample]
        if self.pairs:
            batch_sampler = PairGrouped(multi, batch_size, seed=self.seed)
            self._train_samplers = [batch_sampler]
            loader = DataLoader(multi, batch_sampler=batch_sampler, **self._loader_options())
        else:
            sampler: VideoLabelBalanced | SourceBalanced | _EpochShuffle
            if self.balance == "video-label":
                sampler = VideoLabelBalanced(multi, seed=self.seed)
            elif self.balance == "source":
                sampler = SourceBalanced(multi, seed=self.seed)
            else:
                sampler = _EpochShuffle(len(multi), seed=self.seed)
            self._train_samplers = [sampler]
            # A trailing short batch is dropped: a batch of one cannot train a batch-norm layer,
            # and the sampler's next epoch draws those clips again anyway. (Pair batches always
            # hold at least one real and one fake row, so they keep theirs.)
            loader = DataLoader(
                multi,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=True,
                **self._loader_options(),
            )
        trainer = self.trainer
        self.set_epoch(trainer.current_epoch if trainer is not None else self._epoch)
        return loader

    def val_dataloader(self) -> list[DataLoader[ClipSample]]:
        """One loader per validation source, in config order: every clip of a video in one
        batch, in a fixed order."""
        self._require_setup()
        batch_size = self.data.loader.batch_size
        loaders: list[DataLoader[ClipSample]] = []
        for source in self.val_sources:
            dataset = source.dataset
            if not isinstance(dataset, ClipDataset):  # validation is never built paired
                raise RuntimeError(f"{source.name}: a validation source must be a ClipDataset")
            loaders.append(
                DataLoader(
                    dataset,
                    batch_sampler=VideoGrouped(dataset, batch_size),
                    **self._loader_options(),
                )
            )
        return loaders

    @property
    def epoch(self) -> int:
        """The epoch the training data was last moved to."""
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        """Move every training dataset and the training sampler to ``epoch``."""
        self._epoch = epoch
        if self.train_dataset is not None:
            for dataset in self.train_dataset.datasets:
                dataset.set_epoch(epoch)
        for sampler in self._train_samplers:
            sampler.set_epoch(epoch)
