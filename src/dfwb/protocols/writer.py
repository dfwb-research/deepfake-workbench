"""Write one dataset of a protocol pack (contract C3a), byte for byte reproducibly.

The same inputs always give the same bytes, whatever order they arrive in: records and rows
are sorted, gzip members carry mtime 0, YAML and JSON keys are sorted, and nothing holds a
timestamp. Each file is written atomically (a hidden sibling, then a rename).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml

from dfwb.core.errors import ContractError
from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PackProvenance,
    PairRecord,
    SchemeCard,
    SplitRow,
    VideoRecord,
    assert_no_absolute_paths,
    records_sha256,
    split_sha256,
    write_jsonl,
    write_split_tsv,
)
from dfwb.protocols.rules import Assignment

__all__ = ["rows_from_assignment", "scheme_card_for", "write_dataset_files"]


def rows_from_assignment(assignment: Assignment) -> list[SplitRow]:
    """The split rows of an assignment, sorted by ``(key, compression or "")``."""
    items = sorted(assignment.items(), key=lambda item: (item[0][0], item[0][1] or ""))
    return [SplitRow(key, compression, split) for (key, compression), split in items]


def scheme_card_for(
    rows: Iterable[SplitRow],
    *,
    kind: Literal["official", "derived", "subset"],
    rule: str,
    source: str | None = None,
    params: Mapping[str, Any] | None = None,
    rationale: str | None = None,
) -> SchemeCard:
    """A scheme card for ``rows``, with its ``sha256`` and per-split ``counts`` filled in."""
    rows = list(rows)
    counts = Counter(row.split for row in rows)
    return SchemeCard(
        kind=kind,
        source=source,
        rule=rule,
        sha256=split_sha256(rows),
        rationale=rationale,
        counts=dict(sorted(counts.items())),
        params=dict(params or {}),
    )


def _dump_yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=True, allow_unicode=True)


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _check_schemes(schemes: Mapping[str, list[SplitRow]], card: DatasetCard) -> None:
    for name, rows in sorted(schemes.items()):
        scheme = card.schemes.get(name)
        if scheme is None:
            raise ContractError(
                f"{card.id}: scheme {name!r} has split rows but is not in the dataset card",
                hint="add its SchemeCard (scheme_card_for) to the card's schemes",
            )
        sha256 = split_sha256(rows)
        if sha256 != scheme.sha256:
            raise ContractError(
                f"{card.id}/{name}: split rows hash to {sha256}, the card says {scheme.sha256}",
                hint="build the scheme card from the same rows (scheme_card_for)",
            )


def _check_key_list_hashes(
    videos: Sequence[VideoRecord], pairs: Sequence[PairRecord], card: DatasetCard
) -> None:
    """A card that records its video or pair list's hash must record the lists being written."""
    for field, records in (("videos_sha256", videos), ("pairs_sha256", set(pairs))):
        published = getattr(card, field)
        if published is None:
            continue
        sha256 = records_sha256(records)
        if sha256 != published:
            raise ContractError(
                f"{card.id}: the card's {field} is {published}, but the list being written "
                f"hashes to {sha256}",
                hint="build the card's hashes from the same lists (records_sha256)",
            )


def _check_unique(videos: Sequence[VideoRecord], dataset: str) -> None:
    seen: set[tuple[str, str | None]] = set()
    for video in videos:
        ident = (video.key, video.compression)
        if ident in seen:
            raise ContractError(
                f"{dataset}: video {video.key!r} (compression {video.compression!r}) is listed "
                "twice",
                hint="(key, compression) must be unique; check the inventory",
            )
        seen.add(ident)


def write_dataset_files(
    out: Path,
    *,
    videos: Sequence[VideoRecord],
    schemes: Mapping[str, list[SplitRow]],
    pairs: Sequence[PairRecord],
    card: DatasetCard,
    labels: LabelVocab,
    provenance: PackProvenance,
    notice: str,
) -> None:
    """Write ``out/`` as one dataset of a protocol pack.

    Files: ``dataset.yaml`` and ``labels.yaml`` (sorted keys, Unicode kept), ``videos.jsonl.gz``
    (sorted by ``(key, compression or "")``), ``splits/<scheme>.tsv.gz`` for each scheme in
    ``schemes``, ``pairs.jsonl.gz`` (only when there are pairs; sorted by
    ``(real_key, fake_key)``), ``PROVENANCE.json`` (sorted keys, indent 2) and ``NOTICE.md``.

    A scheme of the card may ship without split rows (a recipe); an existing split file for it, or
    for a scheme no longer in ``schemes``, is removed, and so is a ``pairs.jsonl.gz`` when there
    are no pairs, so ``out/`` always reflects exactly these inputs.

    Raises:
        ContractError: a scheme's rows are not in the card or do not hash to its ``sha256``; the
            card's ``videos_sha256`` or ``pairs_sha256``, when set, is not the hash of ``videos``
            or ``pairs``; a ``(key, compression)`` repeats; or any value looks like an absolute
            path. All of these are checked before any file is written, except absolute paths
            inside ``videos`` and ``pairs``, which are checked as each of those files is written.
    """
    _check_schemes(schemes, card)
    _check_unique(videos, card.id)
    _check_key_list_hashes(videos, pairs, card)
    card_data = card.model_dump(mode="json", by_alias=True)
    labels_data = labels.model_dump(mode="json", by_alias=True)
    provenance_data = provenance.model_dump(mode="json")
    assert_no_absolute_paths(card_data, where="dataset.yaml")
    assert_no_absolute_paths(labels_data, where="labels.yaml")
    assert_no_absolute_paths(provenance_data, where="PROVENANCE.json")
    assert_no_absolute_paths(notice, where="NOTICE.md")

    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out / "videos.jsonl.gz", sorted(videos, key=lambda v: (v.key, v.compression or "")))
    for name, rows in sorted(schemes.items()):
        write_split_tsv(splits_dir / f"{name}.tsv.gz", rows)
    for stale in sorted(splits_dir.glob("*.tsv.gz")):
        if stale.name.removesuffix(".tsv.gz") not in schemes:
            stale.unlink()

    pairs_path = out / "pairs.jsonl.gz"
    if pairs:
        write_jsonl(pairs_path, sorted(set(pairs), key=lambda p: (p.real_key, p.fake_key, p.rule)))
    else:
        pairs_path.unlink(missing_ok=True)

    _write_text(
        out / "PROVENANCE.json",
        json.dumps(provenance_data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
    )
    _write_text(out / "NOTICE.md", notice)
    _write_text(out / "labels.yaml", _dump_yaml(labels_data))
    # The card goes last: an interrupted rewrite leaves the previous card, whose hashes no longer
    # match the new split files, so load() reports drift instead of serving a half-written pack.
    _write_text(out / "dataset.yaml", _dump_yaml(card_data))
