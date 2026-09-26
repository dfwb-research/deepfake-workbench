"""Pack integrity checks for pack authors (contract C3a): ``dfwb protocols lint``.

:func:`lint_pack` walks a protocol pack directory on disk -- the pack a plugin will eventually
publish, not one that has to be installed first -- and re-derives every fact a card claims about
itself (scheme hashes, counts, cross references between videos, splits, pairs and labels) instead
of trusting it. Nothing here mutates the pack; every problem becomes one :class:`LintIssue` rather
than an exception, so one author can see everything wrong in a single run.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter
from collections.abc import Sequence
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
    records_sha256,
    split_sha256,
)
from dfwb.protocols._yaml import read_model
from dfwb.protocols.materialization import RULES

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


def _row_leaks(rows: Sequence[Any], noun: str, where: str) -> LintIssue | None:
    """One issue covering every row of a record file that leaks a local path, naming the first."""
    first: tuple[str, LintIssue] | None = None
    count = 0
    for row in rows:
        leak = _leak_issue(dataclasses.asdict(row), where)
        if leak is not None:
            count += 1
            if first is None:
                first = (str(getattr(row, "key", None) or getattr(row, "fake_key", "")), leak)
    if first is None:
        return None
    key, leak = first
    return LintIssue(
        "error",
        where,
        f"{count} {noun} row(s) hold a local path or hostname; the first, {key!r}: {leak.message}",
    )


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
    leak = _row_leaks(videos, "video", where)
    if leak is not None:
        issues.append(leak)
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

        repeated = Counter((row.key, row.compression) for row in rows)
        for (key, compression), times in sorted(
            repeated.items(), key=lambda item: (item[0][0], item[0][1] or "")
        ):
            if times > 1:
                issues.append(
                    LintIssue(
                        "error",
                        where,
                        f"split row {key!r} (compression {compression!r}) is listed more than "
                        f"once ({times} times)",
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


def _check_videos_hash(
    videos: list[VideoRecord], dataset_id: str, card: DatasetCard | None, issues: list[LintIssue]
) -> None:
    """Shipped videos must hash to the card's ``videos_sha256``, when the card records one."""
    if card is None or card.videos_sha256 is None:
        return
    sha256 = records_sha256(videos)
    if sha256 != card.videos_sha256:
        issues.append(
            LintIssue(
                "error",
                f"{dataset_id}/videos.jsonl.gz",
                f"the videos hash to {sha256}, the card's videos_sha256 is {card.videos_sha256}",
            )
        )


def _check_pairs_card(
    pairs: list[PairRecord], where: str, card: DatasetCard | None, issues: list[LintIssue]
) -> None:
    """Shipped pairs must hash to the card's ``pairs_sha256`` and record its ``pairing_rule``."""
    if card is None:
        return
    if card.pairs_sha256 is not None:
        sha256 = records_sha256(set(pairs))
        if sha256 != card.pairs_sha256:
            issues.append(
                LintIssue(
                    "error",
                    where,
                    f"the pairs hash to {sha256}, the card's pairs_sha256 is {card.pairs_sha256}",
                )
            )
    if card.pairing_rule is not None:
        for rule in sorted({pair.rule for pair in pairs} - {card.pairing_rule}):
            issues.append(
                LintIssue(
                    "error",
                    where,
                    f"the pairs record the rule {rule!r}, the card's pairing_rule is "
                    f"{card.pairing_rule!r}",
                )
            )


def _lint_pairs(
    dataset_dir: Path,
    dataset_id: str,
    video_keys: set[str],
    card: DatasetCard | None,
    issues: list[LintIssue],
) -> None:
    where = f"{dataset_id}/pairs.jsonl.gz"
    path = dataset_dir / "pairs.jsonl.gz"
    if not path.is_file():
        _check_pairs_card([], where, card, issues)
        return
    try:
        pairs = read_jsonl(path, PairRecord, strict=True)
    except ContractError as exc:
        issues.append(LintIssue("error", where, exc.message))
        return
    _check_pairs_card(pairs, where, card, issues)
    leak = _row_leaks(pairs, "pair", where)
    if leak is not None:
        issues.append(leak)
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


def _key_list_files(dataset_dir: Path) -> list[str]:
    """The key lists a dataset folder ships besides ``videos.jsonl.gz``, as relative paths."""
    found = ["pairs.jsonl.gz"] if (dataset_dir / "pairs.jsonl.gz").is_file() else []
    splits = dataset_dir / "splits"
    if splits.is_dir():
        found += sorted(f"splits/{path.name}" for path in splits.iterdir() if path.is_file())
    return found


def _lint_key_free_recipe(
    dataset_dir: Path, dataset_id: str, card: DatasetCard, issues: list[LintIssue]
) -> None:
    """A recipe that ships no ``videos.jsonl.gz``: its card must hold all materialising needs."""
    card_where = f"{dataset_id}/dataset.yaml"
    for name in _key_list_files(dataset_dir):
        issues.append(
            LintIssue(
                "error",
                f"{dataset_id}/{name}",
                "a recipe dataset that ships no videos.jsonl.gz ships no other key list either: "
                "remove this file, or ship videos.jsonl.gz too",
            )
        )
    missing = [field for field in ("videos_sha256", "pairs_sha256") if getattr(card, field) is None]
    if missing:
        issues.append(
            LintIssue(
                "error",
                card_where,
                f"a recipe dataset without its key lists needs {' and '.join(missing)} in its "
                "card, so dfwb protocols materialize can check the lists it rebuilds; rebuild "
                "the dataset with dfwb protocols build, which records them",
            )
        )
    for name, scheme in sorted(card.schemes.items()):
        if scheme.rule not in RULES:
            issues.append(
                LintIssue(
                    "error",
                    card_where,
                    f"scheme {name!r} has rule {scheme.rule!r}, which dfwb protocols materialize "
                    f"cannot recompute (rules: {', '.join(RULES)})",
                )
            )
    if card.pairs_sha256 not in (None, records_sha256(())) and card.pairing_rule is None:
        issues.append(
            LintIssue(
                "error",
                card_where,
                "pairs_sha256 is the hash of a non-empty pair list, but the card names no "
                "pairing_rule to rebuild it with",
            )
        )


def _lint_dataset(
    root: Path, dataset_id: str, *, release: bool, withheld: bool, issues: list[LintIssue]
) -> None:
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
        # A withheld dataset is not published, which is what an undecided one needs.
        if card.distribution == "undecided" and not withheld:
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

    key_free = not (dataset_dir / "videos.jsonl.gz").is_file()
    if card is not None and card.distribution == "recipe" and key_free:
        _lint_key_free_recipe(dataset_dir, dataset_id, card, issues)
    else:
        videos = _lint_videos(dataset_dir, dataset_id, labels, issues)
        _check_videos_hash(videos, dataset_id, card, issues)
        video_keys = {(v.key, v.compression) for v in videos}
        video_key_only = {v.key for v in videos}

        if card is not None:
            _lint_schemes(dataset_dir, dataset_id, card, video_keys, issues)

        _lint_pairs(dataset_dir, dataset_id, video_key_only, card, issues)

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
    the card's ``videos_sha256``, ``pairs_sha256`` and ``pairing_rule`` when it records them,
    that every split row and pair endpoint names a real video, that no video or split row is
    listed twice, that every video's ``label_key`` is in its ``labels.yaml``, and that no card,
    notice, video or pair holds a local path -- and reports a mismatch as an error rather than
    raising. A ``recipe`` dataset that ships no ``videos.jsonl.gz`` ships no key list at all: it
    must ship no pairs or split files either, and its card must hold everything ``dfwb protocols
    materialize`` rebuilds the lists with (``videos_sha256``, ``pairs_sha256``, a recomputable
    rule for every scheme, and a ``pairing_rule`` when there are pairs). A published dataset
    whose card is left ``distribution: undecided`` is a warning normally, and an error when
    ``release`` is set (the check a release build runs, since an undecided dataset must not
    ship); a dataset listed under ``withheld`` is not published, so being undecided is no issue
    there. Nothing is written.
    """
    issues: list[LintIssue] = []
    pack_card = _lint_pack_card(root, issues)
    present = _dataset_dirs_present(root)

    if pack_card is not None:
        _check_listing(root, pack_card, present, issues)

    withheld = set(pack_card.withheld) if pack_card is not None else set()
    for dataset_id in present:
        _lint_dataset(
            root, dataset_id, release=release, withheld=dataset_id in withheld, issues=issues
        )

    return issues
