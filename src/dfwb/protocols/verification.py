"""Verify a local inventory against a protocol pack (contract C3a/C3b): coverage reports.

:func:`verify` joins ``$DFWB_WORK_ROOT/<dataset>/inventory.jsonl`` (C3b) to the pack's full
``videos.jsonl.gz`` (every ``VideoRecord`` of the dataset, not just this scheme's assigned rows --
so a video this scheme leaves unassigned is never reported ``extra`` just because it sits
outside the current scheme) on ``(key, compression)``, and buckets the result. :func:`write_report`
persists the outcome so training and scoring can embed its summary in run metadata.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dfwb.core.errors import ConfigError, did_you_mean
from dfwb.core.paths import require_root, resolve_roots
from dfwb.core.records import InventoryRecord, VideoRecord, read_jsonl
from dfwb.protocols._rawdata import check_outside_datasets_roots
from dfwb.protocols.protocol import load
from dfwb.protocols.refs import ProtocolRef

__all__ = ["CoverageReport", "verify", "write_report"]

_MAX_SAMPLES = 20
_RELEASE_MISMATCH_THRESHOLD = 10
_BUCKETS = ("have", "missing", "missing_requested", "extra", "label_mismatch")


def _task_of(key: str) -> str:
    return key.split("/", 1)[0]


def _sample(key: str, compression: str | None) -> str:
    return f"{key}|{compression or ''}"


@dataclass(frozen=True)
class CoverageReport:
    """The result of joining a local inventory to a protocol's pack videos.

    ``counts``/``samples`` share the same five keys: ``have``, ``missing``, ``missing_requested``
    (the part of ``missing`` whose scheme split was requested), ``extra`` and ``label_mismatch``.
    ``samples`` holds up to 20 sorted ``key|compression`` strings per bucket.
    """

    dataset: str
    scheme: str
    pack: str
    pack_version: str
    scheme_sha256: str
    requested_splits: tuple[str, ...]
    counts: dict[str, int]
    samples: dict[str, list[str]]
    warnings: list[str]

    @property
    def exit_code(self) -> int:
        """4 if any ``label_mismatch``, else 3 if any ``missing_requested``, else 0."""
        if self.counts.get("label_mismatch", 0) > 0:
            return 4
        if self.counts.get("missing_requested", 0) > 0:
            return 3
        return 0

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "scheme": self.scheme,
            "pack": self.pack,
            "pack_version": self.pack_version,
            "scheme_sha256": self.scheme_sha256,
            "requested_splits": list(self.requested_splits),
            "counts": self.counts,
            "samples": self.samples,
            "warnings": self.warnings,
            "exit_code": self.exit_code,
        }


def verify(
    ref: str | ProtocolRef,
    *,
    inventory: Path | None = None,
    splits: Sequence[str] | None = None,
    work_root: Path | None = None,
) -> CoverageReport:
    """Join the local inventory to ``ref``'s pack videos and bucket the result.

    The default inventory is ``<work_root>/<dataset>/inventory.jsonl`` (``work_root`` defaults to
    the resolved ``work`` root, as :func:`~dfwb.protocols.protocol.load` does). The default
    ``requested_splits`` is every split this scheme assigns, except ``exclude``.

    Raises:
        ConfigError: no inventory file exists at the resolved path, or ``splits`` names a split
            this scheme does not assign.
        UnknownKeyError, ContractError: as raised by :func:`~dfwb.protocols.protocol.load`.
    """
    resolved_work_root = (
        work_root if work_root is not None else require_root("work", resolve_roots())
    )
    protocol = load(ref, work_root=resolved_work_root)

    split_by_key = {(row.key, row.compression): row.split for row in protocol.split_rows()}
    scheme_splits = frozenset(split_by_key.values())
    if splits is not None:
        for s in splits:
            if s not in scheme_splits:
                raise ConfigError(
                    f"{protocol.ref}: split {s!r} is not in this scheme"
                    f"{did_you_mean(s, scheme_splits)}",
                    hint="splits in this scheme: " + ", ".join(sorted(scheme_splits)),
                )

    inventory_path = (
        inventory
        if inventory is not None
        else resolved_work_root / protocol.dataset / "inventory.jsonl"
    )
    if not inventory_path.is_file():
        raise ConfigError(
            f"{protocol.dataset}: no inventory at {inventory_path}",
            hint=f"run: dfwb inventory build {protocol.dataset}",
        )

    pack_videos = read_jsonl(protocol._videos_path, VideoRecord)
    inventory_rows = read_jsonl(inventory_path, InventoryRecord)
    on_disk = {(r.key, r.compression): r for r in inventory_rows}

    requested = frozenset(splits) if splits is not None else scheme_splits - {"exclude"}

    counts = dict.fromkeys(_BUCKETS, 0)
    samples: dict[str, list[str]] = {bucket: [] for bucket in _BUCKETS}
    missing_by_task: dict[str, int] = {}
    extra_by_task: dict[str, int] = {}

    pack_keys: set[tuple[str, str | None]] = set()
    for video in pack_videos:
        key_pair = (video.key, video.compression)
        pack_keys.add(key_pair)
        local = on_disk.get(key_pair)
        text = _sample(video.key, video.compression)
        if local is None:
            counts["missing"] += 1
            samples["missing"].append(text)
            task = _task_of(video.key)
            missing_by_task[task] = missing_by_task.get(task, 0) + 1
            if split_by_key.get(key_pair) in requested:
                counts["missing_requested"] += 1
                samples["missing_requested"].append(text)
        elif local.label_key != video.label_key or local.method != video.method:
            counts["label_mismatch"] += 1
            samples["label_mismatch"].append(text)
        else:
            counts["have"] += 1
            samples["have"].append(text)

    for key_pair, record in on_disk.items():
        if key_pair not in pack_keys:
            counts["extra"] += 1
            samples["extra"].append(_sample(record.key, record.compression))
            task = _task_of(record.key)
            extra_by_task[task] = extra_by_task.get(task, 0) + 1

    for bucket in samples:
        samples[bucket] = sorted(samples[bucket])[:_MAX_SAMPLES]

    warnings: list[str] = []
    for task in sorted(set(missing_by_task) & set(extra_by_task)):
        m, x = missing_by_task[task], extra_by_task[task]
        if m >= _RELEASE_MISMATCH_THRESHOLD and x >= _RELEASE_MISMATCH_THRESHOLD:
            warnings.append(
                f"{task}: {m} missing and {x} extra — the local copy may be a different "
                "upstream release (see the card's 'release')"
            )

    return CoverageReport(
        dataset=protocol.dataset,
        scheme=protocol.scheme,
        pack=protocol.pack.name,
        pack_version=protocol.pack_version,
        scheme_sha256=protocol.sha256,
        requested_splits=tuple(sorted(requested)),
        counts=counts,
        samples=samples,
        warnings=warnings,
    )


def write_report(
    report: CoverageReport, work_root: Path, *, datasets_roots: Sequence[Path] | None = None
) -> Path:
    """Write ``<work_root>/<dataset>/verify/<scheme>.json`` atomically, as sorted JSON.

    Raises:
        ConfigError: the report would land inside a datasets root (``datasets_roots``, default
            the resolved ones): raw data is never written to.
    """
    target = work_root / report.dataset / "verify" / f"{report.scheme}.json"
    check_outside_datasets_roots(target, datasets_roots, what="the verify report")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target
