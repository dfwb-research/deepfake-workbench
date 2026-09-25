"""Pack integrity checks for pack authors (contract C3a): ``dfwb protocols lint``.

:func:`lint_pack` walks a protocol pack directory on disk -- the pack a plugin will eventually
publish, not one that has to be installed first -- and re-derives every fact a card claims about
itself (scheme hashes, counts, cross references between videos, splits, pairs and labels) instead
of trusting it. Nothing here mutates the pack; every problem becomes one :class:`LintIssue` rather
than an exception, so one author can see everything wrong in a single run.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from dfwb.core.errors import ContractError
from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PackCard,
    PackProvenance,
    PairRecord,
    VideoRecord,
    assert_no_absolute_paths,
    read_jsonl,
    read_split_tsv,
    split_sha256,
)
from dfwb.protocols._yaml import read_model

__all__ = ["LintIssue", "lint_pack"]


@dataclass(frozen=True)
class LintIssue:
    """One problem found in a pack: ``severity`` ("error"/"warning"), ``where`` and ``message``."""

    severity: Literal["error", "warning"]
    where: str
    message: str


# Local-machine leakage a pack must never carry, beyond what ``assert_no_absolute_paths`` already
# catches from a value that starts (or follows a separator) with a path root: a mid-sentence
# mention, or a Windows-style drive path, which that check does not look for. The directory names
# are substituted into the pattern at runtime, not spelled out next to their slashes here, so this
# file's own source text never contains the very path shape it is built to look for.
_LOCAL_ROOTS = ("home", "mnt")
_LEAK_RE = re.compile("|".join(f"/{root}/" for root in _LOCAL_ROOTS) + "|" + re.escape("C:\\"))


def _leak_issue(value: Any, where: str) -> LintIssue | None:
    try:
        assert_no_absolute_paths(value, where=where)
    except ContractError as exc:
        return LintIssue("error", where, exc.message)
    text = value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)
    match = _LEAK_RE.search(text)
    if match is not None:
        return LintIssue(
            "error",
            where,
            f"{match.group()!r} looks like a local path or hostname, never publish it",
        )
    return None


def _read_or_issue[M: BaseModel](
    path: Path, model: type[M], where: str, issues: list[LintIssue]
) -> M | None:
    if not path.is_file():
        issues.append(LintIssue("error", where, "file is missing"))
        return None
    try:
        return read_model(path, model)
    except ContractError as exc:
        issues.append(LintIssue("error", where, exc.message))
        return None


def _lint_pack_card(root: Path, issues: list[LintIssue]) -> PackCard | None:
    card = _read_or_issue(root / "pack.yaml", PackCard, "pack.yaml", issues)
    if card is not None:
        leak = _leak_issue(card.model_dump(mode="json", by_alias=True), "pack.yaml")
        if leak is not None:
            issues.append(leak)
    return card


def _dataset_dirs_present(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "dataset.yaml").is_file())


def _check_listing(root: Path, card: PackCard, present: list[str], issues: list[LintIssue]) -> None:
    declared = set(card.datasets) | set(card.withheld)
    for dataset_id in sorted(declared - set(present)):
        issues.append(
            LintIssue(
                "error",
                "pack.yaml",
                f"dataset {dataset_id!r} is listed but its directory is missing",
            )
        )
    for dataset_id in sorted(set(present) - declared):
        issues.append(
            LintIssue(
                "error",
                "pack.yaml",
                f"dataset directory {dataset_id!r} is present but not listed in pack.yaml",
            )
        )


def _lint_videos(
    dataset_dir: Path, dataset_id: str, labels: LabelVocab | None, issues: list[LintIssue]
) -> list[VideoRecord]:
    where = f"{dataset_id}/videos.jsonl.gz"
    path = dataset_dir / "videos.jsonl.gz"
    if not path.is_file():
        issues.append(LintIssue("error", where, "file is missing"))
        return []
    try:
        videos = read_jsonl(path, VideoRecord, strict=True)
    except ContractError as exc:
        issues.append(LintIssue("error", where, exc.message))
        return []

    seen: set[tuple[str, str | None]] = set()
    for video in videos:
        key = (video.key, video.compression)
        if key in seen:
            issues.append(
                LintIssue(
                    "error",
                    where,
                    f"video {video.key!r} (compression {video.compression!r}) is listed more "
                    "than once",
                )
            )
        seen.add(key)
        if labels is not None and video.label_key not in labels.vocab:
            issues.append(
                LintIssue(
                    "error",
                    where,
                    f"video {video.key!r} uses label_key {video.label_key!r}, which is not in "
                    "labels.yaml",
                )
            )
    return videos


def _lint_schemes(
    dataset_dir: Path,
    dataset_id: str,
    card: DatasetCard,
    video_keys: set[tuple[str, str | None]],
    issues: list[LintIssue],
) -> None:
    for scheme_name, scheme_card in sorted(card.schemes.items()):
        where = f"{dataset_id}/splits/{scheme_name}.tsv.gz"
        split_path = dataset_dir / "splits" / f"{scheme_name}.tsv.gz"
        if not split_path.is_file():
            if card.distribution == "recipe":
                continue
            issues.append(LintIssue("error", where, "scheme has no split file"))
            continue
        try:
            rows = read_split_tsv(split_path)
        except ContractError as exc:
            issues.append(LintIssue("error", where, exc.message))
            continue

        sha256 = split_sha256(rows)
        if sha256 != scheme_card.sha256:
            issues.append(
                LintIssue(
                    "error",
                    where,
                    f"split rows hash to {sha256}, the card says {scheme_card.sha256}",
                )
            )
        elif scheme_card.counts is not None:
            counts = dict(sorted(Counter(row.split for row in rows).items()))
            if counts != scheme_card.counts:
                issues.append(
                    LintIssue(
                        "error",
                        where,
                        f"split counts {counts} do not match the card's {scheme_card.counts}",
                    )
                )

        for row in rows:
            if (row.key, row.compression) not in video_keys:
                issues.append(
                    LintIssue(
                        "error",
                        where,
                        f"split row {row.key!r} (compression {row.compression!r}) is not in "
                        "videos.jsonl.gz",
                    )
                )


def _lint_pairs(
    dataset_dir: Path, dataset_id: str, video_keys: set[str], issues: list[LintIssue]
) -> None:
    where = f"{dataset_id}/pairs.jsonl.gz"
    path = dataset_dir / "pairs.jsonl.gz"
    if not path.is_file():
        return
    try:
        pairs = read_jsonl(path, PairRecord, strict=True)
    except ContractError as exc:
        issues.append(LintIssue("error", where, exc.message))
        return
    for pair in pairs:
        for key, role in ((pair.fake_key, "fake"), (pair.real_key, "real")):
            if key not in video_keys:
                issues.append(
                    LintIssue(
                        "error",
                        where,
                        f"pair references {role} key {key!r}, not in videos.jsonl.gz",
                    )
                )


def _lint_dataset(root: Path, dataset_id: str, *, release: bool, issues: list[LintIssue]) -> None:
    dataset_dir = root / dataset_id
    card = _read_or_issue(
        dataset_dir / "dataset.yaml", DatasetCard, f"{dataset_id}/dataset.yaml", issues
    )
    labels = _read_or_issue(
        dataset_dir / "labels.yaml", LabelVocab, f"{dataset_id}/labels.yaml", issues
    )

    if card is not None:
        card_where = f"{dataset_id}/dataset.yaml"
        leak = _leak_issue(card.model_dump(mode="json", by_alias=True), card_where)
        if leak is not None:
            issues.append(leak)
        if card.distribution == "undecided":
            issues.append(
                LintIssue(
                    "error" if release else "warning", card_where, "distribution is undecided"
                )
            )

    if labels is not None:
        leak = _leak_issue(
            labels.model_dump(mode="json", by_alias=True), f"{dataset_id}/labels.yaml"
        )
        if leak is not None:
            issues.append(leak)

    videos = _lint_videos(dataset_dir, dataset_id, labels, issues)
    video_keys = {(v.key, v.compression) for v in videos}
    video_key_only = {v.key for v in videos}

    if card is not None:
        _lint_schemes(dataset_dir, dataset_id, card, video_keys, issues)

    _lint_pairs(dataset_dir, dataset_id, video_key_only, issues)

    notice_path = dataset_dir / "NOTICE.md"
    if not notice_path.is_file():
        issues.append(LintIssue("error", f"{dataset_id}/NOTICE.md", "file is missing"))
    else:
        leak = _leak_issue(notice_path.read_text("utf-8"), f"{dataset_id}/NOTICE.md")
        if leak is not None:
            issues.append(leak)

    provenance_path = dataset_dir / "PROVENANCE.json"
    provenance = _read_or_issue(
        provenance_path, PackProvenance, f"{dataset_id}/PROVENANCE.json", issues
    )
    if provenance is not None:
        leak = _leak_issue(provenance.model_dump(mode="json"), f"{dataset_id}/PROVENANCE.json")
        if leak is not None:
            issues.append(leak)


def lint_pack(root: Path, *, release: bool = False) -> list[LintIssue]:
    """Check a protocol pack directory for the problems a pack author needs to fix before release.

    Reads ``root/pack.yaml`` and every ``<dataset_id>/`` directory it (or a directory actually
    present) names, re-deriving each fact a card claims -- a scheme's ``sha256`` and ``counts``,
    that every split row and pair endpoint names a real video, that every video's ``label_key`` is
    in its ``labels.yaml`` -- and reports a mismatch as an error rather than raising. A dataset
    card left ``distribution: undecided`` is a warning normally, and an error when ``release`` is
    set (the check a release build runs, since an undecided dataset must not ship). Nothing is
    written.
    """
    issues: list[LintIssue] = []
    pack_card = _lint_pack_card(root, issues)
    present = _dataset_dirs_present(root)

    if pack_card is not None:
        _check_listing(root, pack_card, present, issues)

    for dataset_id in present:
        _lint_dataset(root, dataset_id, release=release, issues=issues)

    return issues
