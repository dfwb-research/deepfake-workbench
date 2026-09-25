"""Membership changes between two pack versions (contract C3a): ``dfwb protocols diff``.

:func:`diff_packs` compares two pack directories dataset by dataset and scheme by scheme, and
turns what changed into the SemVer bump a release must carry: a scheme whose membership or split
assignment changed, a video whose label or method changed, or a label mapping whose resolved
values changed, breaks comparability with past results and needs a major release; a new dataset,
scheme or label mapping only adds to what is published and needs a minor release; anything else
(card text, a hash-stable rewrite) needs only a patch release; and a pack whose files are all
unchanged (only its version may differ) needs none. This mirrors the pack contract itself, never a
plugin's or a dataset's internals: both packs are read straight off disk, exactly as a pack author
would have them checked out.
"""

from __future__ import annotations

import filecmp
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dfwb.core.errors import ContractError
from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PackCard,
    SplitRow,
    VideoRecord,
    read_jsonl,
    read_split_tsv,
)
from dfwb.protocols._yaml import read_card, read_labels, read_model

__all__ = ["DiffResult", "SchemeDiff", "bump_rank", "diff_packs", "version_bump"]

Bump = Literal["major", "minor", "patch", "none"]

_BUMP_RANK: dict[str, int] = {"none": 0, "patch": 1, "minor": 2, "major": 3}


def bump_rank(bump: str) -> int:
    """How breaking a bump level is: ``major`` (3) > ``minor`` (2) > ``patch`` (1) > ``none`` (0).

    Used to compare the bump a pack author claims for a release against the bump the changes
    require: the claim holds only when its rank is at least the required one.
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
    """The full comparison: every scheme touched, every label change, addition, and the bump.

    ``relabelled`` lists, sorted, every video present in both packs whose ``label_key`` or
    ``method`` changed, as ``<dataset>/<key>|<compression>`` (the compression empty when there
    is none).
    """

    schemes: list[SchemeDiff]
    labels_changed: list[str]
    labels_added: list[str]
    required_bump: Bump
    relabelled: list[str] = field(default_factory=list)


# ``MAJOR.MINOR.PATCH``, with an optional SemVer pre-release suffix (``-rc1``, ``-alpha.2``): the
# suffix is accepted but plays no part in the comparison, since it says nothing about the numeric
# core two packs are actually ordered by.
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?$")


def _parse_semver_core(value: str) -> tuple[int, int, int]:
    match = _SEMVER_RE.match(value)
    if match is None:
        raise ContractError(
            f"pack.yaml version {value!r} is not MAJOR.MINOR.PATCH",
            hint="use plain SemVer, e.g. 1.2.3 or 1.2.3-rc1",
        )
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def version_bump(old: str, new: str) -> Literal["major", "minor", "patch", "none"]:
    """The highest-order SemVer component that differs between two ``X.Y.Z[-pre]`` versions.

    The pre-release suffix, if any, is accepted but ignored: two versions with the same numeric
    core (``1.0.0`` and ``1.0.0-rc1``, or two packs at exactly the same version) give ``"none"``,
    a bump level below ``"patch"``.

    Raises:
        ContractError: either version is not ``MAJOR.MINOR.PATCH`` (with an optional pre-release
            suffix), or ``new``'s numeric core is lower than ``old``'s (a downgrade).
    """
    old_core = _parse_semver_core(old)
    new_core = _parse_semver_core(new)
    if new_core < old_core:
        raise ContractError(
            f"pack.yaml version went from {old!r} to {new!r}, which is a downgrade",
            hint="the new pack's version must be the same as, or later than, the old pack's",
        )
    names: tuple[Literal["major", "minor", "patch"], ...] = ("major", "minor", "patch")
    for index, name in enumerate(names):
        if old_core[index] != new_core[index]:
            return name
    return "none"


def _dataset_ids(card: PackCard) -> set[str]:
    """Every dataset id this pack knows about, published or withheld."""
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


def _relabelled(old: Path, new: Path, dataset_id: str) -> list[str]:
    """Videos in both packs, joined on ``(key, compression)``, whose label or method changed."""
    old_path = old / dataset_id / "videos.jsonl.gz"
    new_path = new / dataset_id / "videos.jsonl.gz"
    if not old_path.is_file() or not new_path.is_file():
        return []
    if filecmp.cmp(old_path, new_path, shallow=False):
        return []
    before = {(v.key, v.compression): v for v in read_jsonl(old_path, VideoRecord)}
    changed: list[str] = []
    for video in read_jsonl(new_path, VideoRecord):
        previous = before.get((video.key, video.compression))
        if previous is not None and (
            previous.label_key != video.label_key or previous.method != video.method
        ):
            changed.append(f"{dataset_id}/{video.key}|{video.compression or ''}")
    return sorted(changed)


def _files(dataset_dir: Path) -> dict[str, Path]:
    """Every file of a dataset folder by relative path, leaving out hidden and cache entries."""
    if not dataset_dir.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in dataset_dir.rglob("*"):
        relative = path.relative_to(dataset_dir)
        if path.is_file() and not any(
            part.startswith(".") or part == "__pycache__" for part in relative.parts
        ):
            found[relative.as_posix()] = path
    return found


def _content_changed(
    old: Path, new: Path, old_card: PackCard, new_card: PackCard, dataset_ids: Collection[str]
) -> bool:
    """Whether anything but the version differs: the pack card, or any dataset file's bytes."""
    if old_card.model_copy(update={"version": ""}) != new_card.model_copy(update={"version": ""}):
        return True
    for dataset_id in dataset_ids:
        old_files, new_files = _files(old / dataset_id), _files(new / dataset_id)
        if old_files.keys() != new_files.keys():
            return True
        for name, path in old_files.items():
            if not filecmp.cmp(path, new_files[name], shallow=False):
                return True
    return False


def _resolve_mapping(
    labels: LabelVocab, name: str, keys: Collection[str] | None = None
) -> dict[str, object]:
    """``label_key -> value`` for this mapping, restricted to ``keys`` (default: every key)."""
    spec = labels.mappings[name]
    wanted = labels.vocab if keys is None else {k: labels.vocab[k] for k in keys}
    values: dict[str, object] = {}
    for label_key, attrs in wanted.items():
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
) -> tuple[list[str], list[str], bool, bool]:
    """``<dataset>/<mapping>`` identifiers split into changed (major) and added (minor).

    Growing the vocab with a wholly new ``label_key`` changes what every mapping built over it
    resolves to (it gains one more entry), but that is an addition, not a change: a mapping counts
    as *changed* only when a ``label_key`` present in **both** vocabs now resolves to a different
    value, or when the new vocab dropped a ``label_key`` the old one had. A wholly new mapping name
    is likewise an addition, not a change.
    """
    changed: list[str] = []
    added: list[str] = []
    major = False
    minor = False
    if old_labels is None and new_labels is None:
        return changed, added, major, minor

    old_names = set(old_labels.mappings) if old_labels is not None else set()
    new_names = set(new_labels.mappings) if new_labels is not None else set()
    old_vocab_keys = set(old_labels.vocab) if old_labels is not None else set()
    new_vocab_keys = set(new_labels.vocab) if new_labels is not None else set()
    shared_keys = old_vocab_keys & new_vocab_keys
    vocab_shrank = bool(old_vocab_keys - new_vocab_keys)
    vocab_grew = bool(new_vocab_keys - old_vocab_keys)

    for name in sorted(new_names & old_names):
        assert old_labels is not None  # both sets are non-empty only when their vocab is too
        assert new_labels is not None
        old_shared = _resolve_mapping(old_labels, name, shared_keys)
        new_shared = _resolve_mapping(new_labels, name, shared_keys)
        if old_shared != new_shared or vocab_shrank:
            changed.append(f"{dataset_id}/{name}")
            major = True
        elif vocab_grew:
            added.append(f"{dataset_id}/{name}")
            minor = True

    for name in sorted(new_names - old_names):
        added.append(f"{dataset_id}/{name}")
        minor = True

    for name in sorted(old_names - new_names):
        changed.append(f"{dataset_id}/{name}")
        major = True

    return changed, added, major, minor


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
    source attribute values changed is still caught -- but only over the ``label_key``s the two
    vocabs share, so growing the vocab is an addition (minor), not a change (major).

    Separately from scheme membership, a dataset id moving from ``withheld`` to ``datasets`` --
    deciding to publish it, even with byte-identical scheme data -- is at least a minor change (it
    is new to anyone installing the pack); moving the other way, or dropping it outright, having
    been published before, is a major change (something that was public no longer is).

    The videos of a dataset in both packs are joined on ``(key, compression)``: a video whose
    ``label_key`` or ``method`` changed is a major change even when every scheme is the same, and
    is listed in :attr:`DiffResult.relabelled`. With nothing major or minor, any other change to a
    pack file (or to ``pack.yaml`` beyond its version) needs a patch release; none at all needs no
    release (``"none"``).
    """
    old_card = read_model(old / "pack.yaml", PackCard)
    new_card = read_model(new / "pack.yaml", PackCard)
    old_all, new_all = _dataset_ids(old_card), _dataset_ids(new_card)
    old_published, new_published = set(old_card.datasets), set(new_card.datasets)
    dataset_ids = sorted(old_all | new_all)

    schemes: list[SchemeDiff] = []
    labels_changed: list[str] = []
    labels_added: list[str] = []
    relabelled: list[str] = []
    major = bool(old_all - new_all) or bool(old_published - new_published)
    minor = bool(new_all - old_all) or bool(new_published - old_published)

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
        label_changed, label_added, labels_major, labels_minor = _diff_labels(
            dataset_id, old_labels, new_labels
        )
        labels_changed.extend(label_changed)
        labels_added.extend(label_added)
        major = major or labels_major
        minor = minor or labels_minor

        relabelled.extend(_relabelled(old, new, dataset_id))

    major = major or bool(relabelled)
    required_bump: Bump
    if major:
        required_bump = "major"
    elif minor:
        required_bump = "minor"
    elif _content_changed(old, new, old_card, new_card, dataset_ids):
        required_bump = "patch"
    else:
        required_bump = "none"
    return DiffResult(
        schemes=schemes,
        labels_changed=labels_changed,
        labels_added=labels_added,
        required_bump=required_bump,
        relabelled=relabelled,
    )
