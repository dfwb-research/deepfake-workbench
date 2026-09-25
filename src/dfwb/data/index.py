"""``VideoIndex``: the join of a protocol split with a processed face store.

:meth:`VideoIndex.build` reads each source's protocol split (:mod:`dfwb.protocols`), matches every
split record against the processed store of a face profile, applies a label mapping, and records
what happened to every video the split named -- not just the ones that made it in.

This module stays inside the ``core``/``protocols`` layer boundary on purpose (import-linter's
layer contract puts ``data`` beside ``preprocess``, not above it, so it may never import
``dfwb.preprocess``):

- a store's ``index.jsonl`` is just ``ProcessedRecord`` rows, so it is read directly with the same
  ``read_jsonl`` every other record file uses, keeping only the latest row per ``(key,
  compression)`` -- the same rule the store itself applies when it decides which redo won;
- a store's root directory is already named after the profile id it holds
  (``<slug>-<hash8>``), so resolving a ``profile`` name to a store directory never needs to read
  the store's ``profile.json`` or load a full processing profile at all -- the directory name
  *is* the profile id.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import PackProvenance, ProcessedRecord, VideoRecord, read_jsonl
from dfwb.protocols.protocol import LabelMapping, Protocol, load

__all__ = ["SourceSpec", "VideoIndex", "VideoItem"]

_log = logging.getLogger(__name__)

_INDEX_FILE = "index.jsonl"
_PROVENANCE_FILE = "PROVENANCE.json"
_INVENTORY_META_FILE = "inventory.meta.json"

Key = tuple[str, str | None]
ExcludedEntry = tuple[int, str, str | None, str]


@dataclass(frozen=True)
class SourceSpec:
    """One entry of a data config's sources: a protocol split, plus a training weight."""

    protocol: str
    split: str
    where: Mapping[str, Any] | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class VideoItem:
    """One usable video: a protocol record joined with its processed frames.

    ``source`` is the 0-based position of the originating :class:`SourceSpec` in the sequence
    passed to :meth:`VideoIndex.build`, so items from different sources (even the same protocol,
    e.g. two ``where`` filters) stay distinguishable.
    """

    source: int
    dataset: str
    key: str
    compression: str | None
    label: int
    label_key: str
    method: str
    video_dir: Path
    frame_indices: list[int]


def _read_processed_index(index_path: Path) -> dict[Key, ProcessedRecord]:
    """The latest row per ``(key, compression)`` of a store's ``index.jsonl``.

    ``{}`` when the file does not exist yet -- a store that has not started, or has not reached a
    video, behaves exactly like one where every remaining video is simply not processed.
    """
    if not index_path.is_file():
        return {}
    latest: dict[Key, ProcessedRecord] = {}
    for record in read_jsonl(index_path, ProcessedRecord):
        latest[(record.key, record.compression)] = record
    return latest


def _resolve_store_dir(work_root: Path, dataset: str, profile: str) -> Path:
    """The processed store directory for ``profile``: a full profile id, or its slug.

    A full id names its store directory directly. A slug is matched against every
    ``processed/<slug>-*`` directory; there must be exactly one.

    Raises:
        ConfigError: neither an exact ``processed/<profile>`` directory exists, nor does exactly
            one ``processed/<profile>-*`` directory.
    """
    processed_dir = work_root / dataset / "processed"
    exact = processed_dir / profile
    if exact.is_dir():
        return exact

    existing = (
        sorted(p for p in processed_dir.iterdir() if p.is_dir()) if processed_dir.is_dir() else []
    )
    candidates = [p for p in existing if p.name.startswith(f"{profile}-")]
    if len(candidates) == 1:
        return candidates[0]

    detail = (
        f"found: {', '.join(p.name for p in existing)}"
        if existing
        else "nothing has been processed there yet"
    )
    raise ConfigError(
        f"{dataset}: no single processed store for profile {profile!r} under {processed_dir} "
        f"({detail})",
        hint="pass the exact profile id, or process this dataset with that profile first",
    )


def _read_pack_builder_version(provenance_path: Path) -> str | None:
    """The inventory-builder version a pack's dataset was built against, or ``None`` if unset."""
    if not provenance_path.is_file():
        return None
    provenance = PackProvenance.model_validate_json(provenance_path.read_text("utf-8"))
    return provenance.builder.get("version")


def _read_store_builder_version(inventory_meta_path: Path) -> str | None:
    """The inventory-builder version an inventory was built with, or ``None`` if there is none."""
    if not inventory_meta_path.is_file():
        return None
    meta = json.loads(inventory_meta_path.read_text("utf-8"))
    builder = meta.get("builder")
    version = builder.get("version") if isinstance(builder, dict) else None
    return version if isinstance(version, str) else None


def _check_version(protocol: Protocol, work_root: Path) -> dict[str, str | None]:
    """Compare the pack's recorded builder version against the local inventory's.

    A mismatch is logged as a warning and still returned, so a caller's summary records it; when
    either side has nothing recorded (most often a local inventory that was never built), there is
    nothing to compare, so the result is ``"unknown"`` and nothing is logged.
    """
    dataset_dir = protocol.pack.dataset_dir(protocol.dataset)
    pack_version = _read_pack_builder_version(dataset_dir / _PROVENANCE_FILE)
    store_version = _read_store_builder_version(work_root / protocol.dataset / _INVENTORY_META_FILE)

    if pack_version is None or store_version is None:
        status = "unknown"
    elif pack_version == store_version:
        status = "match"
    else:
        status = "mismatch"
        _log.warning(
            "%s: pack was built against inventory builder version %s, but %s's local inventory "
            "is now at %s",
            protocol.ref,
            pack_version,
            protocol.dataset,
            store_version,
        )
    return {"status": status, "pack": pack_version, "store": store_version}


def _join_record(
    record: VideoRecord,
    *,
    source: int,
    dataset: str,
    store_dir: Path,
    processed: Mapping[Key, ProcessedRecord],
    label_mapping: LabelMapping,
    mapping_name: str,
) -> VideoItem | str:
    """A :class:`VideoItem` for a usable ``record``, or the reason it is excluded instead.

    Raises:
        ContractError: ``label_mapping`` maps ``record.label_key`` to something other than an int
            class id or ``"exclude"`` -- this join only ever produces int labels.
    """
    processed_record = processed.get((record.key, record.compression))
    if processed_record is None:
        return "not-processed"
    if processed_record.status != "ok":
        return f"processing-failed:{processed_record.status}"

    raw_label = label_mapping(record.label_key)
    if raw_label == "exclude":
        return "label-excluded"
    if not isinstance(raw_label, int):
        raise ContractError(
            f"{mapping_name!r} maps {record.label_key!r} to {raw_label!r}, which is neither an "
            "int class id nor 'exclude'",
            hint="VideoIndex needs an int-valued label mapping, such as 'binary'",
        )

    video_dir = store_dir / processed_record.relpath
    if processed_record.n_frames == 0 or not video_dir.is_dir():
        return "no-frames"

    return VideoItem(
        source=source,
        dataset=dataset,
        key=record.key,
        compression=record.compression,
        label=raw_label,
        label_key=record.label_key,
        method=record.method,
        video_dir=video_dir,
        frame_indices=list(processed_record.frame_indices),
    )


@dataclass(frozen=True)
class VideoIndex:
    """The join of one or more protocol splits with their processed stores.

    Construct through :meth:`build`, never directly. ``items`` are every usable video; ``excluded``
    is every split record that did not make it in, so nothing simply disappears -- a store that is
    still catching up with its protocol just trains on fewer videos, with the gap counted and
    explained rather than hidden.
    """

    items: list[VideoItem]
    excluded: list[ExcludedEntry]
    _summaries: list[dict[str, Any]] = field(repr=False, compare=False)

    @classmethod
    def build(
        cls,
        sources: Sequence[SourceSpec],
        *,
        profile: str,
        labels: str,
        work_root: Path,
    ) -> VideoIndex:
        """Join every source's protocol split with its processed store.

        For each source, in order: load the protocol (``work_root`` backs any recipe scheme that
        needs materializing), read its split records (filtered by ``split`` and ``where``),
        resolve the ``profile`` store for the protocol's dataset, and classify every record --
        see :func:`_join_record` for the exact rule. ``items`` end up sorted by ``(source index,
        key, compression)``, so two builds of the same inputs always agree byte for byte.

        Raises:
            ConfigError: ``profile`` resolves to zero or several store directories for a source's
                dataset.
            ContractError: a source's label mapping maps a label key to something other than an
                int class id or ``"exclude"``.
        """
        items: list[VideoItem] = []
        excluded: list[ExcludedEntry] = []
        summaries: list[dict[str, Any]] = []

        for index, source in enumerate(sources):
            protocol = load(source.protocol, work_root=work_root)
            label_mapping = protocol.labels(labels)
            records = protocol.records(split=source.split, where=source.where)
            store_dir = _resolve_store_dir(work_root, protocol.dataset, profile)
            processed = _read_processed_index(store_dir / _INDEX_FILE)

            excluded_counts: Counter[str] = Counter()
            label_counts: Counter[int] = Counter()

            for record in records:
                result = _join_record(
                    record,
                    source=index,
                    dataset=protocol.dataset,
                    store_dir=store_dir,
                    processed=processed,
                    label_mapping=label_mapping,
                    mapping_name=labels,
                )
                if isinstance(result, VideoItem):
                    items.append(result)
                    label_counts[result.label] += 1
                else:
                    excluded.append((index, record.key, record.compression, result))
                    excluded_counts[result] += 1

            summaries.append(
                {
                    "protocol": {
                        "id": protocol.ref,
                        "split": source.split,
                        "where": dict(source.where or {}),
                        "pack": protocol.pack.name,
                        "pack_version": protocol.pack_version,
                        "scheme_sha256": protocol.sha256,
                    },
                    "profile_id": store_dir.name,
                    "counts": {
                        "in_split": len(records),
                        "included": len(records) - sum(excluded_counts.values()),
                        "excluded": dict(sorted(excluded_counts.items())),
                    },
                    "labels": {str(label): count for label, count in sorted(label_counts.items())},
                    "version_check": _check_version(protocol, work_root),
                }
            )

        items.sort(key=lambda item: (item.source, item.key, item.compression or ""))
        excluded.sort(key=lambda entry: (entry[0], entry[1], entry[2] or ""))
        return cls(items, excluded, summaries)

    def summary(self) -> dict[str, Any]:
        """A JSON-safe summary, per source: what a training run's data record stores."""
        return {"sources": copy.deepcopy(self._summaries)}
