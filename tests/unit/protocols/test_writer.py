"""The pack writer: split rows and scheme cards from an assignment, and byte-reproducible files."""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from dfwb.core.errors import ContractError
from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    PackProvenance,
    PairRecord,
    SplitRow,
    VideoRecord,
    read_jsonl,
    read_split_tsv,
    records_sha256,
    split_sha256,
)
from dfwb.core.records.protocol import LabelMappingSpec, LicenseInfo
from dfwb.protocols.protocol import load
from dfwb.protocols.rules import (
    assign_72_14_14,
    assign_all_test,
    assign_official,
    resolve_pairs,
    task_of,
)
from dfwb.protocols.writer import rows_from_assignment, scheme_card_for, write_dataset_files

DATASET_ID = "toyrt"


def _videos() -> list[VideoRecord]:
    videos: list[VideoRecord] = []
    for i in range(8):
        for compression in ("c23", "c40"):
            videos.append(
                VideoRecord(
                    f"REAL/r{i}",
                    compression,
                    "TOYRT-REAL",
                    "original",
                    identity=f"id{i}",
                    attrs={"lang": "en" if i % 2 else "fr"},
                )
            )
    for i in range(8):
        videos.append(
            VideoRecord(
                f"FS_A/f{i}",
                None,
                "TOYRT-FS_A",
                "FakeA",
                identity=f"id{i}",
                target_id=f"r{i}",
                source_id=f"r{(i + 1) % 8}",
                attrs={"lang": "en" if i % 2 else "fr"},
            )
        )
    return videos


def _is_real(record: VideoRecord) -> bool:
    return task_of(record.key) == "REAL"


def _official() -> dict[str, str]:
    split_of = {0: "train", 1: "train", 2: "train", 3: "val", 4: "test", 5: "test"}
    official = {f"REAL/r{i}": s for i, s in split_of.items()}
    official |= {f"FS_A/f{i}": s for i, s in split_of.items()}
    official["FS_A/f6"] = "train"  # its real, REAL/r6, is left unassigned
    return official


def _inputs(videos: list[VideoRecord]) -> dict:
    schemes = {
        "official": rows_from_assignment(assign_official(videos, _official())),
        "ident-72-14-14": rows_from_assignment(assign_72_14_14(videos)),
        "all-test": rows_from_assignment(assign_all_test(videos)),
    }
    pairs = resolve_pairs(
        videos, is_real=_is_real, candidates=lambda f: f.target_id, fanout_cap=None, rule="toy"
    )
    cards = {
        "official": scheme_card_for(schemes["official"], kind="official", rule="official"),
        "ident-72-14-14": scheme_card_for(
            schemes["ident-72-14-14"],
            kind="derived",
            rule="ident-72-14-14",
            rationale="no official split; identity-disjoint md5 carve (72/14/14)",
        ),
        "all-test": scheme_card_for(schemes["all-test"], kind="subset", rule="all-test"),
    }
    card = DatasetCard(
        id=DATASET_ID,
        name="Toy Ünïcode",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only; never media",
        modalities=["video"],
        compressions=["c23", "c40"],
        key_rule="<task>/<stem>",
        schemes=cards,
        default_scheme="official",
    )
    labels = LabelVocab(
        vocab={
            "TOYRT-REAL": {"binary": 0, "family": "real", "task": "REAL"},
            "TOYRT-FS_A": {"binary": 1, "family": "face-swap", "task": "FS_A"},
        },
        mappings={
            "binary": LabelMappingSpec(from_="binary"),
            "family": LabelMappingSpec(from_="family"),
        },
    )
    provenance = PackProvenance(
        builder={"id": DATASET_ID, "version": "1"},
        dfwb="0.1.0",
        source_listing_sha256="0" * 64,
        rules={name: {"rule": c.rule, "params": c.params} for name, c in cards.items()},
    )
    return {
        "videos": videos,
        "schemes": schemes,
        "pairs": pairs,
        "card": card,
        "labels": labels,
        "provenance": provenance,
        "notice": "# Toy Ünïcode\n\nOwned by nobody. DFWB never distributes media.\n",
    }


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------------------------
# Rows and scheme cards
# ---------------------------------------------------------------------------------------------


def test_rows_from_assignment_is_sorted_by_key_then_compression():
    assignment = {
        ("b", None): "test",
        ("a", "c40"): "train",
        ("a", None): "val",
        ("a", "c23"): "train",
    }
    assert rows_from_assignment(assignment) == [
        SplitRow("a", None, "val"),
        SplitRow("a", "c23", "train"),
        SplitRow("a", "c40", "train"),
        SplitRow("b", None, "test"),
    ]


def test_scheme_card_for_fills_hash_and_counts():
    rows = rows_from_assignment(assign_72_14_14(_videos()))
    card = scheme_card_for(
        rows,
        kind="derived",
        rule="ident-72-14-14",
        source="md5 carve",
        params={"ratios": [72, 14, 14]},
        rationale="no official split",
    )
    assert card.sha256 == split_sha256(rows)
    assert card.sha256 == scheme_card_for(list(reversed(rows)), kind="derived", rule="x").sha256
    expected = {s: sum(1 for r in rows if r.split == s) for s in {r.split for r in rows}}
    assert card.counts == expected
    assert sum(card.counts.values()) == len(rows)
    assert (card.kind, card.rule, card.source, card.rationale) == (
        "derived",
        "ident-72-14-14",
        "md5 carve",
        "no official split",
    )
    assert card.params == {"ratios": [72, 14, 14]}


def test_scheme_card_for_defaults():
    card = scheme_card_for([], kind="subset", rule="all-test")
    assert card.counts == {}
    assert card.params == {}
    assert card.source is None
    assert card.rationale is None
    assert card.sha256 == split_sha256([])


# ---------------------------------------------------------------------------------------------
# write_dataset_files
# ---------------------------------------------------------------------------------------------


def test_write_dataset_files_is_byte_reproducible(tmp_path):
    first = _inputs(_videos())
    write_dataset_files(tmp_path / "a", **first)

    # The same content, handed over in a different order.
    second = _inputs(list(reversed(_videos())))
    second["schemes"] = {
        name: list(reversed(rows)) for name, rows in reversed(second["schemes"].items())
    }
    second["pairs"] = list(reversed(second["pairs"]))
    write_dataset_files(tmp_path / "b", **second)
    write_dataset_files(tmp_path / "a", **first)  # and a rewrite in place

    a, b = _files(tmp_path / "a"), _files(tmp_path / "b")
    assert sorted(a) == [
        "NOTICE.md",
        "PROVENANCE.json",
        "dataset.yaml",
        "labels.yaml",
        "pairs.jsonl.gz",
        "splits/all-test.tsv.gz",
        "splits/ident-72-14-14.tsv.gz",
        "splits/official.tsv.gz",
        "videos.jsonl.gz",
    ]
    assert a == b
    for name, data in a.items():
        if name.endswith(".gz"):
            assert data[4:8] == b"\x00\x00\x00\x00", f"{name}: gzip mtime is not 0"


def test_written_files_are_canonical(tmp_path):
    inputs = _inputs(_videos())
    write_dataset_files(tmp_path, **inputs)

    dataset_yaml = (tmp_path / "dataset.yaml").read_text("utf-8")
    assert "Toy Ünïcode" in dataset_yaml  # allow_unicode
    top_level = [line.split(":")[0] for line in dataset_yaml.splitlines() if line[:1].isalpha()]
    assert top_level == sorted(top_level)
    assert yaml.safe_load(dataset_yaml) == inputs["card"].model_dump(mode="json", by_alias=True)

    labels = yaml.safe_load((tmp_path / "labels.yaml").read_text("utf-8"))
    assert labels["mappings"]["binary"] == {"from": "binary", "override": {}}

    provenance = (tmp_path / "PROVENANCE.json").read_text("utf-8")
    assert provenance == (
        json.dumps(inputs["provenance"].model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
    )
    assert PackProvenance.model_validate_json(provenance) == inputs["provenance"]

    assert (tmp_path / "NOTICE.md").read_text("utf-8") == inputs["notice"]

    videos = read_jsonl(tmp_path / "videos.jsonl.gz", VideoRecord)
    assert videos == sorted(inputs["videos"], key=lambda v: (v.key, v.compression or ""))

    pairs = read_jsonl(tmp_path / "pairs.jsonl.gz", PairRecord)
    assert pairs == sorted(inputs["pairs"], key=lambda p: (p.real_key, p.fake_key))
    assert read_split_tsv(tmp_path / "splits" / "official.tsv.gz") == inputs["schemes"]["official"]

    assert not [p.name for p in tmp_path.rglob(".*")]  # no temporary files left behind


def test_pairs_file_only_when_pairs_exist_and_stale_files_are_removed(tmp_path):
    write_dataset_files(tmp_path, **_inputs(_videos()))
    assert (tmp_path / "pairs.jsonl.gz").is_file()

    smaller = _inputs(_videos())
    smaller["pairs"] = []
    del smaller["schemes"]["all-test"]  # a card scheme may ship without a list (a recipe)
    write_dataset_files(tmp_path, **smaller)

    assert not (tmp_path / "pairs.jsonl.gz").exists()
    assert sorted(p.name for p in (tmp_path / "splits").iterdir()) == [
        "ident-72-14-14.tsv.gz",
        "official.tsv.gz",
    ]


def test_a_scheme_must_match_its_card(tmp_path):
    inputs = _inputs(_videos())
    inputs["schemes"]["all-test"] = inputs["schemes"]["all-test"][1:]
    with pytest.raises(ContractError, match="all-test"):
        write_dataset_files(tmp_path / "out", **inputs)
    assert not (tmp_path / "out").exists()  # checked before anything is written


@pytest.mark.parametrize("field", ["videos_sha256", "pairs_sha256"])
def test_the_card_key_list_hashes_must_match_what_is_written(tmp_path, field):
    inputs = _inputs(_videos())
    card = inputs["card"]
    right = {
        "videos_sha256": records_sha256(inputs["videos"]),
        "pairs_sha256": records_sha256(inputs["pairs"]),
    }
    inputs["card"] = card.model_copy(update={**right, field: "0" * 64})
    with pytest.raises(ContractError, match=field):
        write_dataset_files(tmp_path / "out", **inputs)
    assert not (tmp_path / "out").exists()  # checked before anything is written

    inputs["card"] = card.model_copy(update=right)
    write_dataset_files(tmp_path / "out", **inputs)
    assert (tmp_path / "out" / "dataset.yaml").is_file()


def test_a_scheme_must_be_in_the_card(tmp_path):
    inputs = _inputs(_videos())
    inputs["schemes"]["benchmark"] = inputs["schemes"]["all-test"]
    with pytest.raises(ContractError, match="benchmark"):
        write_dataset_files(tmp_path / "out", **inputs)
    assert not (tmp_path / "out").exists()


def test_duplicate_videos_are_rejected(tmp_path):
    inputs = _inputs(_videos())
    inputs["videos"] = [*inputs["videos"], inputs["videos"][0]]
    with pytest.raises(ContractError, match="REAL/r0"):
        write_dataset_files(tmp_path / "out", **inputs)
    assert not (tmp_path / "out").exists()


def test_an_absolute_path_in_the_notice_is_rejected(tmp_path):
    inputs = _inputs(_videos())
    inputs["notice"] = "Data lives at /" + "srv/datasets/toy\n"
    with pytest.raises(ContractError, match="absolute path"):
        write_dataset_files(tmp_path / "out", **inputs)
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------------------------
# Round trip through load()
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def written_pack(fixture_packs: Callable[..., dict[str, Path]]) -> dict:
    inputs = _inputs(_videos())

    def _build(dataset_dir: Path, dataset_id: str) -> None:
        assert dataset_id == DATASET_ID
        write_dataset_files(dataset_dir, **inputs)

    fixture_packs({"rt-pack": {DATASET_ID: {}}}, builders={"rt-pack": {DATASET_ID: _build}})
    return inputs


def test_written_pack_round_trips_through_load(written_pack):
    videos = sorted(written_pack["videos"], key=lambda v: (v.key, v.compression or ""))
    protocol = load(f"{DATASET_ID}/all-test")
    assert protocol.records() == videos
    assert protocol.sha256 == written_pack["card"].schemes["all-test"].sha256
    assert protocol.labels("binary")("TOYRT-FS_A") == 1

    carve = load(f"{DATASET_ID}/ident-72-14-14")
    val_rows = {
        (r.key, r.compression)
        for r in written_pack["schemes"]["ident-72-14-14"]
        if r.split == "val"
    }
    assert {(v.key, v.compression) for v in carve.records(split="val")} == val_rows

    official = load(DATASET_ID)
    assert official.scheme == "official"
    assert {v.key for v in official.records(split="test")} == {
        "REAL/r4",
        "REAL/r5",
        "FS_A/f4",
        "FS_A/f5",
    }


def test_round_trip_pairs_take_the_split_of_the_real_else_the_fake(written_pack):
    # Ported: a pair's split is its real's split, or its fake's when the real is unassigned.
    protocol = load(f"{DATASET_ID}/official")
    assert set(protocol.pairs(split="train")) == {
        ("REAL/r0", "FS_A/f0"),
        ("REAL/r1", "FS_A/f1"),
        ("REAL/r2", "FS_A/f2"),
        ("REAL/r6", "FS_A/f6"),  # REAL/r6 is unassigned: the fake's split decides
    }
    assert set(protocol.pairs(split="test")) == {("REAL/r4", "FS_A/f4"), ("REAL/r5", "FS_A/f5")}
    assert len(protocol.pairs()) == 8


def test_gzip_members_are_plain_gzip(tmp_path):
    write_dataset_files(tmp_path, **_inputs(_videos()))
    with gzip.open(tmp_path / "splits" / "all-test.tsv.gz", "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    assert lines == sorted(lines)
    assert all(line.endswith("\ttest") for line in lines)
