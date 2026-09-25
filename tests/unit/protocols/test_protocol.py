"""Tests for the ``Protocol`` object, ``load`` and ``list_protocols``."""

from __future__ import annotations

import gzip
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.unit.protocols.conftest import write_toyone_dataset

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError
from dfwb.core.records import SplitRow, read_split_tsv, write_split_tsv
from dfwb.protocols.protocol import LabelMapping, Protocol, list_protocols, load


def _rewrite_lines(path: Path, transform: Callable[[list[str]], list[str]]) -> None:
    """Decompress ``path`` (gzipped JSONL), apply ``transform`` to its lines, and recompress it.

    Used to inject a malformed row into an otherwise-valid ``videos.jsonl.gz`` fixture, to pin
    ``records()``'s per-line error handling (fix round 1: it must match io.py's ``iter_jsonl``,
    not lose the line number or raise a raw ``KeyError``/``TypeError``).
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    lines = transform(lines)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def test_load_defaults_to_the_default_scheme(toyone_pack):
    protocol = load("toyone")
    assert isinstance(protocol, Protocol)
    assert protocol.dataset == "toyone"
    assert protocol.scheme == "official"
    assert protocol.ref == "toyone/official"
    assert protocol.pack.name == "toyone-pack"
    assert protocol.pack_version == "1.0.0"
    assert protocol.sha256 == protocol.scheme_card.sha256
    assert len(protocol.split_rows()) == 14  # 16 rows minus REAL/r4's two unassigned rows


def test_records_by_split_and_where(toyone_pack):
    protocol = load("toyone/official")

    train = protocol.records(split="train")
    assert {r.key for r in train} == {
        "REAL/r1",
        "FAKE_A/a1",
        "FAKE_A/a4",
        "FAKE_B/b1",
        "FAKE_B/b4",
    }
    assert sum(1 for r in train if r.key == "REAL/r1") == 2  # c23 and c40

    subset = protocol.records(where={"compression": "c23", "task": ["REAL", "FAKE_A"]})
    assert {(r.key, r.compression) for r in subset} == {
        ("REAL/r1", "c23"),
        ("REAL/r2", "c23"),
        ("REAL/r3", "c23"),
    }  # FAKE_A has no compression variants, so it never matches "c23"; r4 is unassigned

    by_lang = protocol.records(where={"attrs.lang": "fr"})
    assert {r.key for r in by_lang} == {
        "REAL/r2",
        "FAKE_A/a2",
        "FAKE_B/b2",
        "FAKE_A/a4",
        "FAKE_B/b4",
    }  # REAL/r4 is also "fr" but unassigned in "official", so it is absent


def test_records_reports_a_row_with_no_key(toyone_pack):
    def _drop_key(lines: list[str]) -> list[str]:
        first = json.loads(lines[0])
        del first["key"]
        lines[0] = json.dumps(first)
        return lines

    _rewrite_lines(toyone_pack / "videos.jsonl.gz", _drop_key)

    protocol = load("toyone/official")
    with pytest.raises(ContractError, match=r"videos\.jsonl\.gz:1: missing 'key'"):
        protocol.records()


def test_records_reports_an_unexpected_field_on_a_matched_row(toyone_pack):
    def _add_bogus_field(lines: list[str]) -> list[str]:
        for i, line in enumerate(lines):
            row = json.loads(line)
            if row["key"] == "REAL/r1" and row.get("compression") == "c23":
                row["bogus"] = 1
                lines[i] = json.dumps(row)
                return lines
        raise AssertionError("REAL/r1 (c23) row not found in the fixture")

    _rewrite_lines(toyone_pack / "videos.jsonl.gz", _add_bogus_field)

    protocol = load("toyone/official")
    with pytest.raises(
        ContractError,
        match=r"videos\.jsonl\.gz:\d+: not a valid VideoRecord: unexpected field\(s\) \['bogus'\]",
    ):
        protocol.records()  # REAL/r1 (c23) is assigned in "official", so it reaches construction


def test_records_reports_a_corrupt_json_line(toyone_pack):
    def _corrupt_first_line(lines: list[str]) -> list[str]:
        lines[0] = "not json"
        return lines

    _rewrite_lines(toyone_pack / "videos.jsonl.gz", _corrupt_first_line)

    protocol = load("toyone/official")
    with pytest.raises(ContractError, match=r"videos\.jsonl\.gz:1: invalid JSON"):
        protocol.records()


def test_where_rejects_unknown_fields_with_suggestion(toyone_pack):
    protocol = load("toyone/official")
    with pytest.raises(ConfigError, match="did you mean 'compression'"):
        protocol.records(where={"compresion": "c23"})
    with pytest.raises(ConfigError):
        protocol.records(where={"attrs.nope": 1})


@pytest.mark.parametrize("split", ["tset", ["train", "tset"], ("tset",), {"tset"}])
def test_a_misspelt_split_is_a_config_error_with_a_suggestion(toyone_pack, split):
    protocol = load("toyone/official")
    with pytest.raises(ConfigError, match="did you mean 'test'") as info:
        protocol.records(split=split)
    assert "train, val, test, exclude" in info.value.hint
    with pytest.raises(ConfigError, match="did you mean 'test'"):
        protocol.pairs(split="tset")


def test_a_split_the_scheme_does_not_assign_is_empty_not_an_error(toyone_pack):
    assert load("toyone/all-test").records(split="val") == []
    assert load("toyone/official").records(split="exclude") == []


@pytest.mark.parametrize(
    "tasks",
    [("REAL", "FAKE_A"), {"REAL", "FAKE_A"}, frozenset({"REAL", "FAKE_A"})],
    ids=["tuple", "set", "frozenset"],
)
def test_a_where_sequence_or_set_means_membership(toyone_pack, tasks):
    protocol = load("toyone/official")
    as_list = protocol.records(where={"compression": "c23", "task": ["REAL", "FAKE_A"]})
    assert as_list
    assert protocol.records(where={"compression": "c23", "task": tasks}) == as_list


def test_labels_mapping_and_unknown_mapping(toyone_pack):
    protocol = load("toyone/official")

    binary = protocol.labels("binary")
    assert isinstance(binary, LabelMapping)
    assert binary.name == "binary"
    assert binary("TOYONE-REAL") == 0
    assert binary("TOYONE-FAKE_A") == 1
    assert binary("TOYONE-FAKE_B") == "exclude"  # override
    with pytest.raises(ContractError):
        binary("nope-key")

    av = protocol.labels("audiovisual-binary")
    assert av("TOYONE-FAKE_B") == 0
    assert av("TOYONE-FAKE_A") == 1

    family = protocol.labels("family")
    assert family("TOYONE-FAKE_A") == "face-swap"

    with pytest.raises(UnknownKeyError, match="did you mean 'binary'"):
        protocol.labels("binry")


def test_pairs_by_split(toyone_pack):
    protocol = load("toyone/official")

    all_pairs = protocol.pairs()
    assert set(all_pairs) == {
        ("REAL/r1", "FAKE_A/a1"),
        ("REAL/r2", "FAKE_A/a2"),
        ("REAL/r3", "FAKE_B/b3"),
        ("REAL/r4", "FAKE_B/b4"),
    }

    train_pairs = protocol.pairs(split="train")
    # r1 is train -> included; r2 is val -> excluded; r3 is test -> excluded;
    # r4 is unassigned -> falls back to its fake (b4), which is train -> included
    assert set(train_pairs) == {("REAL/r1", "FAKE_A/a1"), ("REAL/r4", "FAKE_B/b4")}

    val_pairs = protocol.pairs(split="val")
    assert set(val_pairs) == {("REAL/r2", "FAKE_A/a2")}


def test_pairs_is_empty_when_the_pack_has_no_pairs_file(toyone_pack):
    (toyone_pack / "pairs.jsonl.gz").unlink()
    protocol = load("toyone/official")
    assert protocol.pairs() == []
    assert protocol.pairs(split="train") == []


def test_summary(toyone_pack):
    protocol = load("toyone/official")
    summary = protocol.summary()
    assert summary["ref"] == "toyone/official"
    assert summary["dataset"] == "toyone"
    assert summary["scheme"] == "official"
    assert summary["pack"] == "toyone-pack"
    assert summary["pack_version"] == "1.0.0"
    assert summary["sha256"] == protocol.sha256
    assert summary["kind"] == "official"
    assert summary["counts"] == {"train": 6, "val": 4, "test": 4}


def test_pin_by_version_and_hash_prefix(toyone_pack):
    good_version = load("toyone/official@1.0.0")
    assert good_version.ref == "toyone/official"

    good_hash = load(f"toyone/official@{good_version.sha256[:8]}")
    assert good_hash.sha256 == good_version.sha256

    with pytest.raises(ContractError, match=r"pinned @9\.9\.9 but installed"):
        load("toyone/official@9.9.9")

    with pytest.raises(ContractError, match="pinned @deadbeef but installed"):
        load("toyone/official@deadbeef")


def test_split_file_hash_drift_is_a_contract_error(toyone_pack):
    split_path = toyone_pack / "splits" / "official.tsv.gz"
    rows = read_split_tsv(split_path)
    first = rows[0]
    flipped = "val" if first.split != "val" else "test"
    mutated = [SplitRow(first.key, first.compression, flipped), *rows[1:]]
    write_split_tsv(split_path, mutated)

    with pytest.raises(ContractError, match="hash"):
        load("toyone/official")


def test_unknown_scheme_suggests(toyone_pack):
    with pytest.raises(UnknownKeyError, match="did you mean 'official'"):
        load("toyone/officail")


def test_recipe_scheme_needs_materialize(toyone_pack, tmp_path):
    (toyone_pack / "splits" / "official.tsv.gz").unlink()
    work_root = tmp_path / "empty-work"

    with pytest.raises(ContractError) as info:
        load("toyone/official", work_root=work_root)
    assert "run: dfwb protocols materialize toyone/official" in info.value.hint


def test_recipe_scheme_reads_materialized_files(toyone_pack, tmp_path):
    # A materialized copy under the work root satisfies a missing pack split file.
    materialized = tmp_path / "work" / "toyone" / "materialized"
    (materialized / "splits").mkdir(parents=True)
    rows = read_split_tsv(toyone_pack / "splits" / "official.tsv.gz")
    write_split_tsv(materialized / "splits" / "official.tsv.gz", rows)
    shutil.copy(toyone_pack / "videos.jsonl.gz", materialized / "videos.jsonl.gz")
    (toyone_pack / "splits" / "official.tsv.gz").unlink()

    protocol = load("toyone/official", work_root=tmp_path / "work")
    assert protocol.records(split="train")


def test_package_exports_are_lazy(toyone_pack):
    import dfwb.protocols as protocols_pkg
    from dfwb.protocols.refs import ProtocolRef as RealProtocolRef
    from dfwb.protocols.refs import parse_ref as real_parse_ref

    assert protocols_pkg.Protocol is Protocol
    assert protocols_pkg.ProtocolRef is RealProtocolRef
    assert protocols_pkg.parse_ref is real_parse_ref
    assert protocols_pkg.load("toyone").ref == "toyone/official"
    assert any(r.dataset_id == "toyone" for r in protocols_pkg.list())
    with pytest.raises(AttributeError):
        _ = protocols_pkg.nope


def test_verify_and_materialize_stay_functions_once_their_modules_are_imported():
    # Importing a submodule binds its name on the package, so a module named like the function it
    # holds would replace that exported function. Theirs are named differently.
    import types

    import dfwb.protocols
    import dfwb.protocols.materialization
    import dfwb.protocols.verification
    from dfwb.protocols import materialize, verify

    for exported in (dfwb.protocols.verify, dfwb.protocols.materialize, verify, materialize):
        assert callable(exported)
        assert isinstance(exported, types.FunctionType)
        assert not isinstance(exported, types.ModuleType)
    assert dfwb.protocols.verify is dfwb.protocols.verification.verify
    assert dfwb.protocols.materialize is dfwb.protocols.materialization.materialize


def test_list_protocols_marks_default_and_broken(fixture_packs):
    roots = fixture_packs(
        {"toyone-pack": {"toyone": {}}, "broken": {"toytwo": {}}},
        builders={"toyone-pack": {"toyone": write_toyone_dataset}},
    )
    (roots["broken"] / "pack.yaml").write_text("schema_version: 99\n")

    rows = list_protocols()
    by_scheme = {(r.dataset_id, r.scheme): r for r in rows if r.broken is None}

    assert by_scheme[("toyone", "official")].default is True
    assert by_scheme[("toyone", "official")].pack == "toyone-pack"
    assert by_scheme[("toyone", "official")].counts == {"train": 6, "val": 4, "test": 4}
    assert by_scheme[("toyone", "all-test")].default is False
    assert by_scheme[("toyone", "all-test")].counts == {"test": 16}

    broken_rows = [r for r in rows if r.broken is not None]
    assert len(broken_rows) == 1
    assert broken_rows[0].pack == "broken"
    assert "schema_version" in (broken_rows[0].broken or "")


def test_a_broken_dataset_card_is_one_broken_row_not_a_failure(fixture_packs):
    # One bad dataset.yaml never hides the other datasets of its pack, nor other packs.
    roots = fixture_packs(
        {"toyone-pack": {"toyone": {}}, "mixed": {"good": {}, "bad": {}}},
        builders={"toyone-pack": {"toyone": write_toyone_dataset}},
    )
    (roots["mixed"] / "bad" / "dataset.yaml").write_text("id: [\n")

    rows = list_protocols()

    healthy = {(r.pack, r.dataset_id, r.scheme) for r in rows if r.broken is None}
    assert ("toyone-pack", "toyone", "official") in healthy
    assert ("mixed", "good", "official") in healthy
    (broken,) = [r for r in rows if r.broken is not None]
    assert (broken.pack, broken.dataset_id, broken.scheme) == ("mixed", "bad", "")
    assert broken.version == "1.0.0"
    assert "invalid YAML" in (broken.broken or "")
    assert load("toyone").dataset == "toyone"
    assert load("good").dataset == "good"


def test_a_pre_release_pack_version_pins(tmp_path):
    # The built-in toyfake pack is versioned with dfwb, a pre-release; a pin compares versions,
    # so an equivalent spelling of the same version pins too.
    protocol = load("toyfake/official", work_root=tmp_path)
    version = protocol.pack_version
    assert load(f"toyfake/official@{version}", work_root=tmp_path).ref == "toyfake/official"
    pre = re.fullmatch(r"(\d+\.\d+\.\d+)(a|b|rc)(\d+)", version)
    assert pre, f"toyfake is expected to carry a pre-release version, got {version}"
    release, kind, number = pre.groups()
    spelled = f"{release}-{ {'a': 'alpha', 'b': 'beta', 'rc': 'pre'}[kind] }.{number}"
    assert load(f"toyfake/official@{spelled}", work_root=tmp_path).pack_version == version
    with pytest.raises(ContractError, match=r"pinned @0\.1\.0a1"):
        load("toyfake/official@0.1.0a1", work_root=tmp_path)
