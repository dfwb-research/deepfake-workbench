"""Tests for ``dfwb.protocols.diff``: membership changes between two pack versions."""

from __future__ import annotations

from pathlib import Path

import yaml

from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PackCard,
    PackProvenance,
    SplitRow,
    VideoRecord,
)
from dfwb.core.records.protocol import LabelMappingSpec, LicenseInfo
from dfwb.protocols._yaml import read_labels
from dfwb.protocols.diff import bump_rank, diff_packs, version_bump
from dfwb.protocols.writer import scheme_card_for, write_dataset_files

DATASET_ID = "diffds"


def _videos() -> list[VideoRecord]:
    return [
        VideoRecord("REAL/r1", None, "DIFFDS-REAL", "original", identity="r1"),
        VideoRecord("REAL/r2", None, "DIFFDS-REAL", "original", identity="r2"),
        VideoRecord("REAL/r3", None, "DIFFDS-REAL", "original", identity="r3"),
        VideoRecord("FAKE/f1", None, "DIFFDS-FAKE", "FakeA", identity="f1", target_id="r1"),
        VideoRecord("FAKE/f2", None, "DIFFDS-FAKE", "FakeA", identity="f2", target_id="r2"),
        VideoRecord("FAKE/f3", None, "DIFFDS-FAKE", "FakeA", identity="f3", target_id="r3"),
    ]


def _labels() -> LabelVocab:
    return LabelVocab(
        vocab={
            "DIFFDS-REAL": {"binary": 0, "family": "real", "task": "REAL"},
            "DIFFDS-FAKE": {"binary": 1, "family": "face-swap", "task": "FAKE"},
        },
        mappings={"binary": LabelMappingSpec(from_="binary")},
    )


def _official_rows() -> list[SplitRow]:
    return [
        SplitRow("REAL/r1", None, "train"),
        SplitRow("REAL/r2", None, "val"),
        SplitRow("REAL/r3", None, "test"),
        SplitRow("FAKE/f1", None, "train"),
        SplitRow("FAKE/f2", None, "val"),
        SplitRow("FAKE/f3", None, "test"),
    ]


def _all_test_rows() -> list[SplitRow]:
    return [SplitRow(v.key, v.compression, "test") for v in _videos()]


def _official_only() -> dict[str, dict[str, list[SplitRow]]]:
    return {DATASET_ID: {"official": _official_rows()}}


def _write_dataset(
    dataset_dir: Path,
    dataset_id: str,
    schemes: dict[str, list[SplitRow]],
    *,
    mappings: dict[str, LabelMappingSpec] | None = None,
) -> None:
    scheme_cards = {
        name: scheme_card_for(rows, kind="official" if name == "official" else "subset", rule=name)
        for name, rows in schemes.items()
    }
    card = DatasetCard(
        id=dataset_id,
        name=dataset_id,
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only; never media",
        distribution="list",
        modalities=["video"],
        key_rule="<task>/<stem>",
        schemes=scheme_cards,
        default_scheme=next(iter(scheme_cards)),
    )
    labels = _labels()
    if mappings is not None:
        labels = labels.model_copy(update={"mappings": mappings})
    provenance = PackProvenance(
        builder={"id": dataset_id, "version": "1"},
        dfwb="0.1.0",
        source_listing_sha256="0" * 64,
        rules={name: {"rule": name, "params": {}} for name in schemes},
    )
    notice = f"# {dataset_id}\n\nSynthetic fixture; never real media.\n"
    write_dataset_files(
        dataset_dir,
        videos=_videos(),
        schemes=schemes,
        pairs=[],
        card=card,
        labels=labels,
        provenance=provenance,
        notice=notice,
    )


def _write_pack(
    root: Path,
    name: str,
    version: str,
    dataset_schemes: dict[str, dict[str, list[SplitRow]]],
    *,
    mappings: dict[str, dict[str, LabelMappingSpec]] | None = None,
) -> Path:
    mappings = mappings or {}
    for dataset_id, schemes in dataset_schemes.items():
        _write_dataset(root / dataset_id, dataset_id, schemes, mappings=mappings.get(dataset_id))
    card = PackCard(schema_version=1, name=name, version=version, datasets=sorted(dataset_schemes))
    root.mkdir(parents=True, exist_ok=True)
    (root / "pack.yaml").write_text(
        yaml.safe_dump(card.model_dump(mode="json", by_alias=True), sort_keys=True)
    )
    return root


# -------------------------------------------------------------------------------------------
# The three scenarios named by the versioning contract.
# -------------------------------------------------------------------------------------------


def test_identical_packs_require_a_patch_bump(tmp_path):
    old = _write_pack(tmp_path / "old", "pack", "1.0.0", _official_only())
    new = _write_pack(tmp_path / "new", "pack", "1.0.1", _official_only())

    result = diff_packs(old, new)

    assert result.required_bump == "patch"
    assert result.labels_changed == []
    scheme_rows = [
        (s.dataset, s.scheme, s.status, s.added, s.removed, s.moved) for s in result.schemes
    ]
    assert scheme_rows == [(DATASET_ID, "official", "same", 0, 0, 0)]


def test_added_scheme_requires_a_minor_bump(tmp_path):
    old = _write_pack(tmp_path / "old", "pack", "1.0.0", _official_only())
    new = _write_pack(
        tmp_path / "new",
        "pack",
        "1.1.0",
        {DATASET_ID: {"official": _official_rows(), "all-test": _all_test_rows()}},
    )

    result = diff_packs(old, new)

    assert result.required_bump == "minor"
    statuses = {s.scheme: s.status for s in result.schemes}
    assert statuses == {"official": "same", "all-test": "new"}
    added_scheme = next(s for s in result.schemes if s.scheme == "all-test")
    assert (added_scheme.added, added_scheme.removed, added_scheme.moved) == (6, 0, 0)


def test_moved_video_requires_a_major_bump_with_moved_count_one(tmp_path):
    old_rows = _official_rows()
    new_rows = [SplitRow("REAL/r1", None, "test") if r.key == "REAL/r1" else r for r in old_rows]
    old = _write_pack(tmp_path / "old", "pack", "1.0.0", {DATASET_ID: {"official": old_rows}})
    new = _write_pack(tmp_path / "new", "pack", "1.1.0", {DATASET_ID: {"official": new_rows}})

    result = diff_packs(old, new)

    assert result.required_bump == "major"
    (scheme_diff,) = result.schemes
    assert scheme_diff.status == "changed"
    assert (scheme_diff.added, scheme_diff.removed, scheme_diff.moved) == (0, 0, 1)


# -------------------------------------------------------------------------------------------
# Dataset and label mapping changes.
# -------------------------------------------------------------------------------------------


def test_new_dataset_requires_a_minor_bump(tmp_path):
    old = _write_pack(tmp_path / "old", "pack", "1.0.0", {})
    new = _write_pack(tmp_path / "new", "pack", "1.1.0", _official_only())

    result = diff_packs(old, new)

    assert result.required_bump == "minor"
    (scheme_diff,) = result.schemes
    assert scheme_diff.status == "new"


def test_removed_scheme_requires_a_major_bump(tmp_path):
    old = _write_pack(
        tmp_path / "old",
        "pack",
        "1.0.0",
        {DATASET_ID: {"official": _official_rows(), "all-test": _all_test_rows()}},
    )
    new = _write_pack(tmp_path / "new", "pack", "2.0.0", _official_only())

    result = diff_packs(old, new)

    assert result.required_bump == "major"
    statuses = {s.scheme: s.status for s in result.schemes}
    assert statuses["all-test"] == "removed"
    assert statuses["official"] == "same"


def test_new_label_mapping_requires_a_minor_bump(tmp_path):
    old = _write_pack(tmp_path / "old", "pack", "1.0.0", _official_only())
    new_mappings = {
        "binary": LabelMappingSpec(from_="binary"),
        "family": LabelMappingSpec(from_="family"),
    }
    new = _write_pack(
        tmp_path / "new", "pack", "1.1.0", _official_only(), mappings={DATASET_ID: new_mappings}
    )

    result = diff_packs(old, new)

    assert result.required_bump == "minor"
    assert result.labels_changed == [f"{DATASET_ID}/family"]


def test_removed_label_mapping_requires_a_major_bump(tmp_path):
    old_mappings = {
        "binary": LabelMappingSpec(from_="binary"),
        "family": LabelMappingSpec(from_="family"),
    }
    old = _write_pack(
        tmp_path / "old", "pack", "1.0.0", _official_only(), mappings={DATASET_ID: old_mappings}
    )
    new = _write_pack(tmp_path / "new", "pack", "2.0.0", _official_only())

    result = diff_packs(old, new)

    assert result.required_bump == "major"
    assert result.labels_changed == [f"{DATASET_ID}/family"]


def test_new_recipe_scheme_reports_zero_added_rows(tmp_path):
    # A scheme may ship without a split file (only its rule and hash are published); diff still
    # reports it as "new" for an unpublished dataset, just with nothing to count.
    old = _write_pack(tmp_path / "old", "pack", "1.0.0", {})

    dataset_dir = tmp_path / "new" / DATASET_ID
    rows = _official_rows()
    scheme_cards = {
        "official": scheme_card_for(rows, kind="official", rule="official"),
        "all-test": scheme_card_for([], kind="subset", rule="all-test"),
    }
    card = DatasetCard(
        id=DATASET_ID,
        name=DATASET_ID,
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only; never media",
        distribution="recipe",
        modalities=["video"],
        key_rule="<task>/<stem>",
        schemes=scheme_cards,
        default_scheme="official",
    )
    provenance = PackProvenance(
        builder={"id": DATASET_ID, "version": "1"},
        dfwb="0.1.0",
        source_listing_sha256="0" * 64,
        rules={
            "official": {"rule": "official", "params": {}},
            "all-test": {"rule": "all-test", "params": {}},
        },
    )
    write_dataset_files(
        dataset_dir,
        videos=_videos(),
        schemes={"official": rows},  # "all-test" ships without a list: a recipe
        pairs=[],
        card=card,
        labels=_labels(),
        provenance=provenance,
        notice=f"# {DATASET_ID}\n\nSynthetic fixture; never real media.\n",
    )
    new_pack_card = PackCard(schema_version=1, name="pack", version="1.1.0", datasets=[DATASET_ID])
    (tmp_path / "new" / "pack.yaml").write_text(
        yaml.safe_dump(new_pack_card.model_dump(mode="json", by_alias=True), sort_keys=True)
    )

    result = diff_packs(old, tmp_path / "new")

    all_test_diff = next(s for s in result.schemes if s.scheme == "all-test")
    assert all_test_diff.status == "new"
    assert (all_test_diff.added, all_test_diff.removed, all_test_diff.moved) == (0, 0, 0)


def test_changed_label_mapping_requires_a_major_bump(tmp_path):
    old = _write_pack(tmp_path / "old", "pack", "1.0.0", _official_only())
    new = _write_pack(tmp_path / "new", "pack", "1.1.0", _official_only())
    dataset_dir = tmp_path / "new" / DATASET_ID
    labels = read_labels(dataset_dir)
    mutated_vocab = {**labels.vocab, "DIFFDS-FAKE": {**labels.vocab["DIFFDS-FAKE"], "binary": 0}}
    mutated = labels.model_copy(update={"vocab": mutated_vocab})
    (dataset_dir / "labels.yaml").write_text(
        yaml.safe_dump(mutated.model_dump(mode="json", by_alias=True), sort_keys=True)
    )

    result = diff_packs(old, new)

    assert result.required_bump == "major"
    assert result.labels_changed == [f"{DATASET_ID}/binary"]


# -------------------------------------------------------------------------------------------
# version_bump: the plain SemVer component comparison used to gate ``--expect-bump``.
# -------------------------------------------------------------------------------------------


def test_version_bump_reports_the_highest_differing_component():
    assert version_bump("1.0.0", "2.0.0") == "major"
    assert version_bump("1.0.0", "1.1.0") == "minor"
    assert version_bump("1.0.0", "1.0.1") == "patch"
    assert version_bump("1.0.0", "1.0.0") == "none"


def test_bump_rank_orders_major_over_minor_over_patch_over_none():
    assert bump_rank("major") > bump_rank("minor") > bump_rank("patch") > bump_rank("none")
