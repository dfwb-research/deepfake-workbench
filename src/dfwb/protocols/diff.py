"""Membership changes between two pack versions (contract C3a): ``dfwb protocols diff``.

:func:`diff_packs` compares two pack directories dataset by dataset and scheme by scheme, and
turns what changed into the SemVer bump a release must carry: a scheme whose membership or split
assignment changed, or a label mapping whose resolved values changed, breaks comparability with
past results and needs a major release; a new dataset, scheme or label mapping only adds to what
is published and needs a minor release; anything else (card text, a hash-stable rewrite) needs only
a patch release. This mirrors the pack contract itself, never a plugin's or a dataset's internals:
both packs are read straight off disk, exactly as a pack author would have them checked out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PackCard,
    SplitRow,
    read_split_tsv,
)
from dfwb.protocols._yaml import read_card, read_labels, read_model

__all__ = ["DiffResult", "SchemeDiff", "bump_rank", "diff_packs", "version_bump"]

Bump = Literal["major", "minor", "patch"]

_BUMP_RANK: dict[str, int] = {"none": 0, "patch": 1, "minor": 2, "major": 3}


def bump_rank(bump: str) -> int:
    """How breaking a bump level is: ``major`` (3) > ``minor`` (2) > ``patch`` (1) > ``none`` (0).

    Used to compare a pack's actual ``pack.yaml`` version change against the bump the changes
    require: a release is valid only when its actual rank is at least the required one.
    """
    return _BUMP_RANK[bump]


@dataclass(frozen=True)
class SchemeDiff:
    """How one scheme's rows changed between the two packs (all zero when ``status`` is "same")."""

    dataset: str
    scheme: str
    added: int
    removed: int
    moved: int
    status: Literal["new", "removed", "changed", "same"]


@dataclass(frozen=True)
class DiffResult:
    """The full comparison: every scheme touched, every label mapping changed, and the bump."""

    schemes: list[SchemeDiff]
    labels_changed: list[str]
    required_bump: Bump


def version_bump(old: str, new: str) -> Literal["major", "minor", "patch", "none"]:
    """The highest-order SemVer component that differs between two ``X.Y.Z`` versions."""
    old_parts = tuple(int(part) for part in old.split("."))
    new_parts = tuple(int(part) for part in new.split("."))
    names: tuple[Literal["major", "minor", "patch"], ...] = ("major", "minor", "patch")
    for index, name in enumerate(names):
        if old_parts[index] != new_parts[index]:
            return name
    return "none"


def _dataset_ids(card: PackCard) -> set[str]:
    return set(card.datasets) | set(card.withheld)


def _rows_or_empty(root: Path, dataset_id: str, scheme_name: str) -> list[SplitRow]:
    path = root / dataset_id / "splits" / f"{scheme_name}.tsv.gz"
    if not path.is_file():
        return []
    return read_split_tsv(path)


def _row_diff(old_rows: list[SplitRow], new_rows: list[SplitRow]) -> tuple[int, int, int]:
    old_map = {(r.key, r.compression): r.split for r in old_rows}
    new_map = {(r.key, r.compression): r.split for r in new_rows}
    added = sum(1 for key in new_map if key not in old_map)
    removed = sum(1 for key in old_map if key not in new_map)
    moved = sum(1 for key in old_map.keys() & new_map.keys() if old_map[key] != new_map[key])
    return added, removed, moved


def _resolve_mapping(labels: LabelVocab, name: str) -> dict[str, object]:
    spec = labels.mappings[name]
    values: dict[str, object] = {}
    for label_key, attrs in labels.vocab.items():
        values[label_key] = spec.override.get(label_key, attrs.get(spec.from_))
    return values


def _diff_schemes(
    old: Path,
    new: Path,
    dataset_id: str,
    old_card: DatasetCard | None,
    new_card: DatasetCard | None,
) -> tuple[list[SchemeDiff], bool, bool]:
    """One :class:`SchemeDiff` per scheme in either card, plus whether it forces major/minor."""
    old_schemes = old_card.schemes if old_card is not None else {}
    new_schemes = new_card.schemes if new_card is not None else {}
    diffs: list[SchemeDiff] = []
    major = False
    minor = False

    for scheme_name in sorted(set(old_schemes) | set(new_schemes)):
        old_scheme = old_schemes.get(scheme_name)
        new_scheme = new_schemes.get(scheme_name)

        if old_scheme is None:
            new_rows = _rows_or_empty(new, dataset_id, scheme_name)
            diffs.append(SchemeDiff(dataset_id, scheme_name, len(new_rows), 0, 0, "new"))
            minor = True
            continue
        if new_scheme is None:
            old_rows = _rows_or_empty(old, dataset_id, scheme_name)
            diffs.append(SchemeDiff(dataset_id, scheme_name, 0, len(old_rows), 0, "removed"))
            major = True
            continue
        if old_scheme.sha256 == new_scheme.sha256:
            diffs.append(SchemeDiff(dataset_id, scheme_name, 0, 0, 0, "same"))
            continue

        old_rows = _rows_or_empty(old, dataset_id, scheme_name)
        new_rows = _rows_or_empty(new, dataset_id, scheme_name)
        added, removed, moved = _row_diff(old_rows, new_rows)
        diffs.append(SchemeDiff(dataset_id, scheme_name, added, removed, moved, "changed"))
        major = True

    return diffs, major, minor


def _diff_labels(
    dataset_id: str, old_labels: LabelVocab | None, new_labels: LabelVocab | None
) -> tuple[list[str], bool, bool]:
    """Identifiers ``<dataset>/<mapping>`` whose resolved values changed, plus major/minor flags."""
    changed: list[str] = []
    major = False
    minor = False
    if old_labels is None and new_labels is None:
        return changed, major, minor

    old_names = set(old_labels.mappings) if old_labels is not None else set()
    new_names = set(new_labels.mappings) if new_labels is not None else set()

    for name in sorted(new_names & old_names):
        assert old_labels is not None  # both sets are non-empty only when their vocab is too
        assert new_labels is not None
        if _resolve_mapping(old_labels, name) != _resolve_mapping(new_labels, name):
            changed.append(f"{dataset_id}/{name}")
            major = True

    for name in sorted(new_names - old_names):
        changed.append(f"{dataset_id}/{name}")
        minor = True

    for name in sorted(old_names - new_names):
        changed.append(f"{dataset_id}/{name}")
        major = True

    return changed, major, minor


def diff_packs(old: Path, new: Path) -> DiffResult:
    """Compare two pack directories and report the scheme, label and version-bump differences.

    Every dataset present in either pack is compared: a dataset only in ``new`` contributes a
    "new" :class:`SchemeDiff` per scheme (a minor change); a dataset only in ``old`` contributes a
    "removed" one per scheme (a major change, since something published no longer is). Within a
    dataset present in both, schemes are matched by name and compared by their card ``sha256``
    first -- an unchanged hash short-circuits straight to "same" without touching the split files
    -- and, when it changed, by their actual rows, to report how many were added, removed or moved
    to a different split. Label mappings are compared by their fully resolved ``label_key -> value``
    table (vocab entries plus overrides), not just the mapping's own spec, so a mapping whose
    source attribute values changed is still caught.
    """
    old_card = read_model(old / "pack.yaml", PackCard)
    new_card = read_model(new / "pack.yaml", PackCard)
    dataset_ids = sorted(_dataset_ids(old_card) | _dataset_ids(new_card))

    schemes: list[SchemeDiff] = []
    labels_changed: list[str] = []
    major = bool(_dataset_ids(old_card) - _dataset_ids(new_card))
    minor = bool(_dataset_ids(new_card) - _dataset_ids(old_card))

    for dataset_id in dataset_ids:
        old_dataset_card = read_card(old / dataset_id) if (old / dataset_id).is_dir() else None
        new_dataset_card = read_card(new / dataset_id) if (new / dataset_id).is_dir() else None

        scheme_diffs, scheme_major, scheme_minor = _diff_schemes(
            old, new, dataset_id, old_dataset_card, new_dataset_card
        )
        schemes.extend(scheme_diffs)
        major = major or scheme_major
        minor = minor or scheme_minor

        old_labels = read_labels(old / dataset_id) if old_dataset_card is not None else None
        new_labels = read_labels(new / dataset_id) if new_dataset_card is not None else None
        label_changes, labels_major, labels_minor = _diff_labels(dataset_id, old_labels, new_labels)
        labels_changed.extend(label_changes)
        major = major or labels_major
        minor = minor or labels_minor

    required_bump: Bump = "major" if major else "minor" if minor else "patch"
    return DiffResult(schemes=schemes, labels_changed=labels_changed, required_bump=required_bump)
