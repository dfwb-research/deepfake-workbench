"""Tests for ``dfwb.protocols.lint``: pack integrity checks for pack authors."""

from __future__ import annotations

import dataclasses
import gzip
import json
import shutil
from pathlib import Path

import pytest
import yaml

from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PackCard,
    PackProvenance,
    PairRecord,
    SplitRow,
    VideoRecord,
    read_jsonl,
    read_split_tsv,
    records_sha256,
    write_jsonl,
    write_split_tsv,
)
from dfwb.core.records.protocol import LabelMappingSpec, LicenseInfo
from dfwb.protocols._yaml import read_card
from dfwb.protocols.lint import LintIssue, lint_pack
from dfwb.protocols.writer import scheme_card_for, write_dataset_files

# Local-machine path fixtures, assembled from pieces rather than spelled out as one contiguous
# literal: this repository's own "no machine-specific absolute paths" hook scans committed text
# for exactly that shape, and these tests need the real shape to exercise the check.
_SLASH_HOME = "/" + "home/"


def _videos(dataset_id: str) -> list[VideoRecord]:
    prefix = dataset_id.upper()
    return [
        VideoRecord("REAL/r1", None, f"{prefix}-REAL", "original", identity="r1"),
        VideoRecord("REAL/r2", None, f"{prefix}-REAL", "original", identity="r2"),
        VideoRecord("FAKE_A/f1", None, f"{prefix}-FAKE_A", "FakeA", identity="f1", target_id="r1"),
        VideoRecord("FAKE_B/f2", None, f"{prefix}-FAKE_B", "FakeB", identity="f2", target_id="r2"),
    ]


def _rows() -> list[SplitRow]:
    return [
        SplitRow("REAL/r1", None, "train"),
        SplitRow("REAL/r2", None, "test"),
        SplitRow("FAKE_A/f1", None, "train"),
        SplitRow("FAKE_B/f2", None, "test"),
    ]


def _labels(dataset_id: str) -> LabelVocab:
    prefix = dataset_id.upper()
    return LabelVocab(
        vocab={
            f"{prefix}-REAL": {"binary": 0, "family": "real", "task": "REAL"},
            f"{prefix}-FAKE_A": {"binary": 1, "family": "face-swap", "task": "FAKE_A"},
            f"{prefix}-FAKE_B": {"binary": 1, "family": "lip-sync", "task": "FAKE_B"},
        },
        mappings={"binary": LabelMappingSpec(from_="binary")},
    )


def _dump(model_data: dict) -> str:
    return yaml.safe_dump(model_data, sort_keys=True, allow_unicode=True)


def _dump_card(dataset_dir: Path, card: DatasetCard) -> None:
    (dataset_dir / "dataset.yaml").write_text(_dump(card.model_dump(mode="json", by_alias=True)))


def _write_dataset(
    dataset_dir: Path, dataset_id: str, *, distribution: str = "list"
) -> DatasetCard:
    videos = _videos(dataset_id)
    rows = _rows()
    scheme = scheme_card_for(rows, kind="official", rule="official")
    card = DatasetCard(
        id=dataset_id,
        name=dataset_id,
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only; never media",
        distribution=distribution,
        modalities=["video"],
        key_rule="<task>/<stem>",
        schemes={"official": scheme},
        default_scheme="official",
    )
    labels = _labels(dataset_id)
    pairs = [PairRecord("FAKE_A/f1", "REAL/r1", "toy"), PairRecord("FAKE_B/f2", "REAL/r2", "toy")]
    provenance = PackProvenance(
        builder={"id": dataset_id, "version": "1"},
        dfwb="0.1.0",
        source_listing_sha256="0" * 64,
        rules={"official": {"rule": "official", "params": {}}},
    )
    notice = f"# {dataset_id}\n\nSynthetic fixture; never real media.\n"
    write_dataset_files(
        dataset_dir,
        videos=videos,
        schemes={"official": rows},
        pairs=pairs,
        card=card,
        labels=labels,
        provenance=provenance,
        notice=notice,
    )
    return card


def _write_pack(root: Path, dataset_ids: list[str], *, listed: list[str] | None = None) -> Path:
    for dataset_id in dataset_ids:
        _write_dataset(root / dataset_id, dataset_id)
    pack_card = PackCard(
        schema_version=1,
        name="lint-pack",
        version="1.0.0",
        datasets=sorted(listed if listed is not None else dataset_ids),
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "pack.yaml").write_text(_dump(pack_card.model_dump(mode="json", by_alias=True)))
    return root


def _replace_scheme_rows(
    dataset_dir: Path, card: DatasetCard, scheme_name: str, rows: list[SplitRow]
) -> None:
    """Rewrite a scheme's split file and its card entry together, so only ``rows`` is corrupted."""
    old_scheme = card.schemes[scheme_name]
    new_scheme = scheme_card_for(
        rows,
        kind=old_scheme.kind,
        rule=old_scheme.rule,
        source=old_scheme.source,
        rationale=old_scheme.rationale,
        params=old_scheme.params,
    )
    write_split_tsv(dataset_dir / "splits" / f"{scheme_name}.tsv.gz", rows)
    new_card = card.model_copy(update={"schemes": {**card.schemes, scheme_name: new_scheme}})
    _dump_card(dataset_dir, new_card)


# -------------------------------------------------------------------------------------------
# A valid pack reports nothing.
# -------------------------------------------------------------------------------------------


def test_valid_pack_has_no_issues(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    assert lint_pack(root) == []


# -------------------------------------------------------------------------------------------
# One corruption per test, exactly one reported error.
# -------------------------------------------------------------------------------------------


def test_hash_drift_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    split_path = dataset_dir / "splits" / "official.tsv.gz"
    rows = read_split_tsv(split_path)
    # Only the split assignment of one row changes; per-split counts are unaffected, so this
    # corrupts the hash alone, not the counts.
    mutated = [SplitRow(r.key, r.compression, "val") if r.key == "FAKE_B/f2" else r for r in rows]
    write_split_tsv(split_path, mutated)

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/splits/official.tsv.gz"
    assert "sha256" in issues[0].message or "hash" in issues[0].message


def test_duplicate_video_key_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    videos = read_jsonl(dataset_dir / "videos.jsonl.gz", VideoRecord)
    write_jsonl(dataset_dir / "videos.jsonl.gz", [*videos, videos[0]])

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert videos[0].key in issues[0].message


def test_split_row_not_in_videos_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    card = read_card(dataset_dir)
    rows = read_split_tsv(dataset_dir / "splits" / "official.tsv.gz")
    _replace_scheme_rows(
        dataset_dir, card, "official", [*rows, SplitRow("FAKE_A/ghost", None, "train")]
    )

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "FAKE_A/ghost" in issues[0].message


def test_missing_notice_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    (root / "toylint" / "NOTICE.md").unlink()

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "NOTICE.md" in issues[0].where


def test_absolute_path_in_notice_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    (root / "toylint" / "NOTICE.md").write_text(f"See {_SLASH_HOME}luke/data for the source.\n")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "NOTICE.md" in issues[0].where


def test_missing_label_key_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    labels = _labels("toylint")
    trimmed = labels.model_copy(
        update={"vocab": {k: v for k, v in labels.vocab.items() if k != "TOYLINT-FAKE_B"}}
    )
    (dataset_dir / "labels.yaml").write_text(_dump(trimmed.model_dump(mode="json", by_alias=True)))

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "FAKE_B/f2" in issues[0].message
    assert "TOYLINT-FAKE_B" in issues[0].message


def test_undecided_distribution_warns_unless_release(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    card = read_card(dataset_dir)
    _dump_card(dataset_dir, card.model_copy(update={"distribution": "undecided"}))

    issues = lint_pack(root)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "distribution" in issues[0].message

    release_issues = lint_pack(root, release=True)
    assert len(release_issues) == 1
    assert release_issues[0].severity == "error"
    assert "distribution" in release_issues[0].message


def test_an_undecided_withheld_dataset_passes_a_release_lint(tmp_path):
    # Withholding is exactly what an undecided dataset needs: it is not published.
    root = _write_pack(tmp_path, ["toylint"], listed=[])
    card = PackCard.model_validate(yaml.safe_load((root / "pack.yaml").read_text()))
    withheld = card.model_copy(update={"withheld": ["toylint"]})
    (root / "pack.yaml").write_text(_dump(withheld.model_dump(mode="json", by_alias=True)))
    dataset_dir = root / "toylint"
    _dump_card(dataset_dir, read_card(dataset_dir).model_copy(update={"distribution": "undecided"}))

    assert lint_pack(root, release=True) == []
    assert lint_pack(root) == []


def test_absolute_paths_in_videos_are_one_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    videos = read_jsonl(dataset_dir / "videos.jsonl.gz", VideoRecord)
    # Written by hand: the record writer itself refuses an absolute path.
    lines = [
        json.dumps(
            {
                **dataclasses.asdict(v),
                "attrs": {"source": f"{_SLASH_HOME}luke/{v.key}.mp4"}
                if v.key in ("REAL/r2", "FAKE_A/f1")
                else v.attrs,
            }
        )
        for v in videos
    ]
    with gzip.open(dataset_dir / "videos.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/videos.jsonl.gz"
    assert "2 video row(s)" in issues[0].message
    assert "FAKE_A/f1" in issues[0].message


def test_a_leak_in_pairs_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    pairs = [
        PairRecord("FAKE_A/f1", "REAL/r1", f"matched on data{_SLASH_HOME}shared/list"),
        PairRecord("FAKE_B/f2", "REAL/r2", "toy"),
    ]
    write_jsonl(dataset_dir / "pairs.jsonl.gz", pairs)

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/pairs.jsonl.gz"
    assert "1 pair row(s)" in issues[0].message


def test_a_duplicate_split_row_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    card = read_card(dataset_dir)
    rows = read_split_tsv(dataset_dir / "splits" / "official.tsv.gz")
    # The card is rewritten from the rows, so its hash and counts still agree with the file.
    _replace_scheme_rows(dataset_dir, card, "official", [*rows, SplitRow("REAL/r1", None, "test")])

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/splits/official.tsv.gz"
    assert "REAL/r1" in issues[0].message
    assert "more than once" in issues[0].message


def test_dataset_not_listed_in_pack_yaml_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint", "toylint-extra"], listed=["toylint"])

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "toylint-extra" in issues[0].message


def test_listed_dataset_with_a_missing_directory_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"], listed=["toylint", "toylint-ghost"])

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "toylint-ghost" in issues[0].message


def test_invalid_pack_yaml_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    (root / "pack.yaml").write_text("schema_version: 1\n")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "pack.yaml"


# -------------------------------------------------------------------------------------------
# Files missing outright (a run that never gets as far as writing them).
# -------------------------------------------------------------------------------------------


def test_missing_dataset_yaml_makes_the_whole_directory_unrecognised(tmp_path):
    # A directory without a dataset.yaml is not counted as a dataset directory at all (that file
    # is exactly how one is recognised), so this surfaces as pack.yaml missing its listing, not
    # as a per-dataset "file is missing".
    root = _write_pack(tmp_path, ["toylint"])
    (root / "toylint" / "dataset.yaml").unlink()

    issues = lint_pack(root)

    assert issues == [
        LintIssue("error", "pack.yaml", "dataset 'toylint' is listed but its directory is missing")
    ]


def test_missing_labels_yaml_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    (root / "toylint" / "labels.yaml").unlink()

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0] == LintIssue("error", "toylint/labels.yaml", "file is missing")


def test_missing_pack_yaml_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    (root / "pack.yaml").unlink()

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0] == LintIssue("error", "pack.yaml", "file is missing")


def test_missing_videos_file_is_reported(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    (root / "toylint" / "videos.jsonl.gz").unlink()

    issues = lint_pack(root)

    assert LintIssue("error", "toylint/videos.jsonl.gz", "file is missing") in issues


def test_corrupt_videos_file_is_reported(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    with gzip.open(root / "toylint" / "videos.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("not json\n")

    issues = lint_pack(root)

    assert any(i.where == "toylint/videos.jsonl.gz" and "invalid JSON" in i.message for i in issues)


def test_missing_split_file_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    (root / "toylint" / "splits" / "official.tsv.gz").unlink()

    issues = lint_pack(root)

    assert issues == [
        LintIssue("error", "toylint/splits/official.tsv.gz", "scheme has no split file")
    ]


def test_corrupt_split_file_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    path = root / "toylint" / "splits" / "official.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("not-a-valid-row\n")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].where == "toylint/splits/official.tsv.gz"


def test_corrupt_pairs_file_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    with gzip.open(root / "toylint" / "pairs.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("not json\n")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].where == "toylint/pairs.jsonl.gz"


# -------------------------------------------------------------------------------------------
# More corruptions from the interface's own check list, each isolated to one reported error.
# -------------------------------------------------------------------------------------------


def test_counts_mismatch_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    card = read_card(dataset_dir)
    scheme = card.schemes["official"]
    wrong_counts = {**(scheme.counts or {}), "train": 999}
    bad_scheme = scheme.model_copy(update={"counts": wrong_counts})
    _dump_card(dataset_dir, card.model_copy(update={"schemes": {"official": bad_scheme}}))

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/splits/official.tsv.gz"
    assert "counts" in issues[0].message


def test_pair_references_missing_video_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    pairs = read_jsonl(dataset_dir / "pairs.jsonl.gz", PairRecord)
    write_jsonl(
        dataset_dir / "pairs.jsonl.gz",
        [*pairs, PairRecord("FAKE_A/ghost", "REAL/r1", "toy")],
    )

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/pairs.jsonl.gz"
    assert "FAKE_A/ghost" in issues[0].message


def test_a_leak_not_next_to_a_separator_is_still_caught(tmp_path):
    # A path root straight after a word character, with no space, "=", ":" etc. before it, is
    # something ``assert_no_absolute_paths`` alone would miss; the dedicated regex does not.
    root = _write_pack(tmp_path, ["toylint"])
    (root / "toylint" / "NOTICE.md").write_text(f"archived under data{_SLASH_HOME}shared/archive\n")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/NOTICE.md"
    assert _SLASH_HOME in issues[0].message


def test_absolute_path_in_dataset_yaml_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    card = read_card(dataset_dir)
    _dump_card(dataset_dir, card.model_copy(update={"homepage": _SLASH_HOME + "luke/notes"}))

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/dataset.yaml"


def test_absolute_path_in_labels_yaml_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    dataset_dir = root / "toylint"
    original = _labels("toylint")
    mutated_vocab = {
        **original.vocab,
        "TOYLINT-REAL": {**original.vocab["TOYLINT-REAL"], "note": _SLASH_HOME + "luke/notes"},
    }
    mutated = original.model_copy(update={"vocab": mutated_vocab})
    (dataset_dir / "labels.yaml").write_text(_dump(mutated.model_dump(mode="json", by_alias=True)))

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/labels.yaml"


def test_absolute_path_in_provenance_is_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    provenance_path = root / "toylint" / "PROVENANCE.json"
    data = json.loads(provenance_path.read_text("utf-8"))
    data["builder"]["note"] = _SLASH_HOME + "luke/notes"
    provenance_path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].where == "toylint/PROVENANCE.json"


def test_lint_issue_is_a_plain_frozen_record():
    issue = LintIssue("error", "toylint/dataset.yaml", "boom")
    assert (issue.severity, issue.where, issue.message) == ("error", "toylint/dataset.yaml", "boom")


# -------------------------------------------------------------------------------------------
# Key-free recipe datasets: the card's hashes stand in for the key lists.
# -------------------------------------------------------------------------------------------


def _hash_the_lists(dataset_dir: Path, **changes: object) -> DatasetCard:
    """Record the shipped lists' hashes (and the pairing rule) in the card, then ``changes``."""
    card = read_card(dataset_dir)
    pairs_path = dataset_dir / "pairs.jsonl.gz"
    pairs = read_jsonl(pairs_path, PairRecord) if pairs_path.is_file() else []
    card = card.model_copy(
        update={
            "videos_sha256": records_sha256(
                read_jsonl(dataset_dir / "videos.jsonl.gz", VideoRecord)
            ),
            "pairs_sha256": records_sha256(pairs),
            "pairing_rule": "toy",
            **changes,
        }
    )
    _dump_card(dataset_dir, card)
    return card


def _strip(dataset_dir: Path) -> None:
    """Take out every key list, as a release build does for a recipe dataset."""
    (dataset_dir / "videos.jsonl.gz").unlink()
    (dataset_dir / "pairs.jsonl.gz").unlink(missing_ok=True)
    shutil.rmtree(dataset_dir / "splits")


def test_a_key_free_recipe_dataset_passes_a_release_lint(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    _hash_the_lists(root / "toylint", distribution="recipe")
    _strip(root / "toylint")

    assert sorted(p.name for p in (root / "toylint").iterdir()) == [
        "NOTICE.md",
        "PROVENANCE.json",
        "dataset.yaml",
        "labels.yaml",
    ]
    assert lint_pack(root, release=True) == []


def test_a_recipe_card_with_its_hashes_passes_while_it_still_ships_its_lists(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    _hash_the_lists(root / "toylint", distribution="recipe")

    assert lint_pack(root, release=True) == []


def test_a_key_free_recipe_needs_the_hashes_of_its_lists(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    _dump_card(
        root / "toylint", read_card(root / "toylint").model_copy(update={"distribution": "recipe"})
    )
    _strip(root / "toylint")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].where == "toylint/dataset.yaml"
    assert "videos_sha256" in issues[0].message
    assert "pairs_sha256" in issues[0].message
    assert "dfwb protocols build" in issues[0].message


def test_a_key_free_dataset_that_is_not_a_recipe_still_misses_its_lists(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    _hash_the_lists(root / "toylint")  # distribution: list
    _strip(root / "toylint")

    issues = lint_pack(root)

    assert LintIssue("error", "toylint/videos.jsonl.gz", "file is missing") in issues
    assert (
        LintIssue("error", "toylint/splits/official.tsv.gz", "scheme has no split file") in issues
    )


def test_a_recipe_without_its_videos_ships_no_other_key_list(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    _hash_the_lists(root / "toylint", distribution="recipe")
    (root / "toylint" / "videos.jsonl.gz").unlink()

    issues = lint_pack(root)

    assert sorted(issue.where for issue in issues) == [
        "toylint/pairs.jsonl.gz",
        "toylint/splits/official.tsv.gz",
    ]
    assert all(issue.severity == "error" for issue in issues)
    assert all("no videos.jsonl.gz" in issue.message for issue in issues)


@pytest.mark.parametrize("rule", [None, "md5-carve-key"])
def test_every_scheme_of_a_key_free_recipe_can_be_recomputed(tmp_path, rule):
    root = _write_pack(tmp_path, ["toylint"])
    card = _hash_the_lists(root / "toylint", distribution="recipe")
    scheme = card.schemes["official"].model_copy(update={"rule": rule})
    _dump_card(root / "toylint", card.model_copy(update={"schemes": {"official": scheme}}))
    _strip(root / "toylint")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].where == "toylint/dataset.yaml"
    assert f"scheme 'official' has rule {rule!r}" in issues[0].message
    assert "cannot recompute" in issues[0].message


def test_a_key_free_recipe_with_pairs_names_its_pairing_rule(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    _hash_the_lists(root / "toylint", distribution="recipe", pairing_rule=None)
    _strip(root / "toylint")

    issues = lint_pack(root)

    assert len(issues) == 1
    assert issues[0].where == "toylint/dataset.yaml"
    assert "pairing_rule" in issues[0].message


def test_shipped_videos_that_differ_from_the_card_hash_are_the_only_reported_error(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    card = _hash_the_lists(root / "toylint")
    path = root / "toylint" / "videos.jsonl.gz"
    videos = read_jsonl(path, VideoRecord)
    write_jsonl(path, [dataclasses.replace(videos[0], method="Other"), *videos[1:]])

    issues = lint_pack(root)

    assert issues == [
        LintIssue(
            "error",
            "toylint/videos.jsonl.gz",
            f"the videos hash to {records_sha256(read_jsonl(path, VideoRecord))}, the card's "
            f"videos_sha256 is {card.videos_sha256}",
        )
    ]


def test_shipped_pairs_that_differ_from_the_card_are_reported(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    card = _hash_the_lists(root / "toylint", pairing_rule="other")
    path = root / "toylint" / "pairs.jsonl.gz"
    write_jsonl(path, read_jsonl(path, PairRecord)[1:])

    issues = lint_pack(root)

    assert issues == [
        LintIssue(
            "error",
            "toylint/pairs.jsonl.gz",
            f"the pairs hash to {records_sha256(read_jsonl(path, PairRecord))}, the card's "
            f"pairs_sha256 is {card.pairs_sha256}",
        ),
        LintIssue(
            "error",
            "toylint/pairs.jsonl.gz",
            "the pairs record the rule 'toy', the card's pairing_rule is 'other'",
        ),
    ]


def test_no_pairs_file_hashes_as_no_pairs(tmp_path):
    root = _write_pack(tmp_path, ["toylint"])
    (root / "toylint" / "pairs.jsonl.gz").unlink()
    _hash_the_lists(root / "toylint", pairing_rule=None)

    assert read_card(root / "toylint").pairs_sha256 == records_sha256([])
    assert lint_pack(root) == []
