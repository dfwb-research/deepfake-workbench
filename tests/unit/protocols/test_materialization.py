"""Tests for ``dfwb.protocols.materialization``: recompute a recipe scheme from a local inventory.

The pack is built from the synthetic ``packdemo`` dataset of the pack-building tests, then a split
file is deleted: the pack then describes that scheme only by its rule, parameters and hash, as a
pack for a dataset whose terms forbid redistributing key lists would.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest
from tests.unit.preprocess.test_packbuild import PACK_NAME, setup_packdemo
from tests.unit.protocols.conftest import make_pack, register_packs

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError
from dfwb.core.records import InventoryRecord, read_jsonl, read_split_tsv, write_jsonl
from dfwb.preprocess.inventory.runner import build_inventory, get_builder
from dfwb.preprocess.packbuild import add_to_pack_yaml, build_dataset
from dfwb.protocols.materialization import (
    MaterializeResult,
    assign_rule,
    benchmark_params,
    materialize,
    needs_official,
    rule_needs_official,
)
from dfwb.protocols.protocol import load
from dfwb.protocols.rules import BenchmarkSpec, assign_benchmark

_SCHEMES = ("official", "ident-72-14-14", "all-test", "benchmark")


@pytest.fixture
def built(monkeypatch, tmp_path):
    """The packdemo dataset built into a registered pack, with its inventory."""
    paths = setup_packdemo(monkeypatch, tmp_path)
    build_dataset("packdemo", out=paths["pack"] / "packdemo")
    add_to_pack_yaml(paths["pack"], "packdemo")
    return {**paths, "dataset": paths["pack"] / "packdemo"}


def _official(built):
    records = read_jsonl(built["inventory"], InventoryRecord)
    return get_builder("packdemo").official_splits(built["raw"] / "PackDemo", records)


def test_materialize_is_exported_lazily():
    # A fresh interpreter: the package names it without importing the module up front.
    code = (
        "import sys, dfwb.protocols as p; "
        "assert 'dfwb.protocols.materialization' not in sys.modules; "
        "f = p.materialize; print(f.__module__, f.__name__, 'materialize' in p.__all__)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.split() == ["dfwb.protocols.materialization", "materialize", "True"]


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_materialize_matches_published_hash(built, scheme):
    ref = f"packdemo/{scheme}"
    published = load(ref, work_root=built["work"])
    rows, records = published.split_rows(), published.records()
    (built["dataset"] / "splits" / f"{scheme}.tsv.gz").unlink()
    official = _official(built) if needs_official(ref) else None

    result = materialize(
        ref, inventory=built["inventory"], official=official, work_root=built["work"]
    )

    materialized = built["work"] / "packdemo" / "materialized"
    assert result == MaterializeResult(
        ref=ref,
        path=materialized / "splits" / f"{scheme}.tsv.gz",
        sha256=published.sha256,
        matched=True,
    )
    assert (materialized / "videos.jsonl.gz").is_file()
    served = load(ref, work_root=built["work"])
    assert sorted(served.split_rows(), key=_row_order) == sorted(rows, key=_row_order)
    assert served.records() == records
    assert read_split_tsv(result.path) == sorted(rows, key=_row_order)


def test_materialize_defaults_to_the_card_default_scheme(built):
    (built["dataset"] / "splits" / "official.tsv.gz").unlink()

    result = materialize(
        "packdemo", inventory=built["inventory"], official=_official(built), work_root=built["work"]
    )

    assert result.ref == "packdemo/official"
    assert result.matched is True


def test_materialize_mismatch_leaves_nothing_behind(built, tmp_path):
    (built["dataset"] / "splits" / "ident-72-14-14.tsv.gz").unlink()
    rows = read_jsonl(built["inventory"], InventoryRecord)
    short = tmp_path / "short.jsonl"
    write_jsonl(short, rows[1:])

    with pytest.raises(ContractError) as info:
        materialize(
            "packdemo/ident-72-14-14", inventory=short, official=None, work_root=built["work"]
        )

    assert info.value.message.startswith("packdemo/ident-72-14-14: materialized hash ")
    assert " != published " in info.value.message
    assert info.value.hint == (
        "your local copy differs from the release the pack describes; run dfwb protocols verify"
    )
    materialized = built["work"] / "packdemo" / "materialized"
    assert not materialized.exists() or not any(p.is_file() for p in materialized.rglob("*"))


def test_a_rule_needing_the_official_split_without_one_is_a_config_error(built):
    with pytest.raises(ConfigError) as info:
        materialize(
            "packdemo/official",
            inventory=built["inventory"],
            official=None,
            work_root=built["work"],
        )
    assert "official split" in info.value.message
    assert "dfwb protocols materialize" in info.value.hint


def test_needs_official_follows_the_scheme_card(built):
    assert needs_official("packdemo/official") is True
    assert needs_official("packdemo/benchmark") is True  # drawn from the official test
    assert needs_official("packdemo/ident-72-14-14") is False
    assert needs_official("packdemo/all-test") is False
    assert needs_official("packdemo") is True  # the default scheme


def test_rule_needs_official():
    assert rule_needs_official("official", {}) is True
    assert rule_needs_official("official+ident-80-20", {"policy": "test-only-official"}) is True
    assert rule_needs_official("benchmark", {"pool": "official-test"}) is True
    assert rule_needs_official("benchmark", {"pool": "all"}) is False
    assert rule_needs_official("ident-72-14-14", {}) is False
    assert rule_needs_official(None, {}) is False


def test_unknown_scheme_is_an_unknown_key_error(built):
    with pytest.raises(UnknownKeyError) as info:
        materialize(
            "packdemo/ident-72-14",
            inventory=built["inventory"],
            official=None,
            work_root=built["work"],
        )
    assert "did you mean 'ident-72-14-14'" in info.value.message


def test_missing_inventory_is_a_config_error(built, tmp_path):
    with pytest.raises(ConfigError) as info:
        materialize(
            "packdemo/all-test",
            inventory=tmp_path / "none.jsonl",
            official=None,
            work_root=built["work"],
        )
    assert info.value.hint == "run: dfwb inventory build packdemo"


def test_a_pin_is_checked(built):
    with pytest.raises(ContractError) as info:
        materialize(
            "packdemo/all-test@9.9.9",
            inventory=built["inventory"],
            official=None,
            work_root=built["work"],
        )
    assert "pinned @9.9.9" in info.value.message


def test_a_scheme_without_a_recomputable_rule_is_a_contract_error(monkeypatch, tmp_path):
    # The fixture pack's official scheme records no rule at all.
    root = make_pack(tmp_path, "plain-pack", {"plain": {}})
    register_packs(monkeypatch, {"plain-pack": root})
    inventory = tmp_path / "inventory.jsonl"
    write_jsonl(inventory, [])

    with pytest.raises(ContractError) as info:
        materialize("plain/official", inventory=inventory, official={}, work_root=tmp_path / "w")
    assert "cannot be recomputed" in info.value.message


def test_assign_rule_rejects_an_unknown_rule_and_incomplete_benchmark_params():
    with pytest.raises(ContractError):
        assign_rule("shuffle", {}, [], official=None, is_real=lambda r: True)
    with pytest.raises(ContractError) as info:
        assign_rule("benchmark", {"k_fake": 2}, [], official=None, is_real=lambda r: True)
    assert "task_order" in info.value.message


def test_benchmark_params_reproduce_the_benchmark_rule(built):
    records = read_jsonl(built["inventory"], InventoryRecord)
    builder = get_builder("packdemo")
    spec = BenchmarkSpec(k_fake=3, strata=("task",), k_real_cap=1, exclude_tasks=("FS_B",), seed=7)
    params = benchmark_params(spec, task_order=["REAL", "FS_A", "FS_B"], pool="all")

    direct = assign_benchmark(
        records, spec=spec, is_real=builder.is_real, task_rank=builder.task_rank(), pool_keys=None
    )
    via_params = assign_rule("benchmark", params, records, official=None, is_real=builder.is_real)

    assert via_params == direct
    assert params == {
        "k_fake": 3,
        "strata": ["task"],
        "k_real_cap": 1,
        "exclude_tasks": ["FS_B"],
        "seed": 7,
        "task_order": ["REAL", "FS_A", "FS_B"],
        "pool": "all",
    }


def test_materialize_does_not_touch_the_pack(built):
    (built["dataset"] / "splits" / "all-test.tsv.gz").unlink()
    before = sorted(p.name for p in built["dataset"].rglob("*"))

    materialize(
        "packdemo/all-test", inventory=built["inventory"], official=None, work_root=built["work"]
    )

    assert sorted(p.name for p in built["dataset"].rglob("*")) == before


def _row_order(row) -> tuple[str, str]:
    return (row.key, row.compression or "")


def test_the_pack_is_the_registered_one(built):
    assert load("packdemo/official", work_root=built["work"]).pack.name == PACK_NAME


def test_materialize_an_official_plus_carve_recipe(built):
    build_inventory("packdemo-carve")
    inventory = built["work"] / "packdemo-carve" / "inventory.jsonl"
    build_dataset("packdemo-carve", out=built["pack"] / "packdemo-carve")
    add_to_pack_yaml(built["pack"], "packdemo-carve")
    ref = "packdemo-carve/official+ident-80-20"
    published = load(ref, work_root=built["work"])
    (built["pack"] / "packdemo-carve" / "splits" / "official+ident-80-20.tsv.gz").unlink()
    builder = get_builder("packdemo-carve")
    records = read_jsonl(inventory, InventoryRecord)
    official = builder.official_splits(built["raw"] / "PackDemo", records)

    result = materialize(ref, inventory=inventory, official=official, work_root=built["work"])

    assert result.sha256 == published.sha256
    assert load(ref, work_root=built["work"]).split_rows() == published.split_rows()


def test_an_inventory_listing_a_video_twice_is_a_contract_error(built, tmp_path):
    rows = read_jsonl(built["inventory"], InventoryRecord)
    doubled = tmp_path / "doubled.jsonl"
    write_jsonl(doubled, [*rows, rows[0]])

    with pytest.raises(ContractError) as info:
        materialize("packdemo/all-test", inventory=doubled, official=None, work_root=built["work"])
    assert "listed twice" in info.value.message


def test_a_benchmark_over_an_unknown_label_is_a_contract_error(built, tmp_path):
    rows = read_jsonl(built["inventory"], InventoryRecord)
    relabelled = tmp_path / "relabelled.jsonl"
    # A video of the official test, so the benchmark draw has to ask whether it is real.
    write_jsonl(
        relabelled,
        [
            dataclasses.replace(r, label_key="PD-NOPE") if r.key == "FS_A/p08_p09" else r
            for r in rows
        ],
    )

    with pytest.raises(ContractError) as info:
        materialize(
            "packdemo/benchmark",
            inventory=relabelled,
            official=_official(built),
            work_root=built["work"],
        )
    assert "'PD-NOPE'" in info.value.message


def test_benchmark_pools_are_checked():
    with pytest.raises(ContractError):
        benchmark_params(BenchmarkSpec(k_fake=1), task_order=["REAL"], pool="some")  # type: ignore[arg-type]
    params = {"k_fake": 1, "task_order": ["REAL"], "pool": "some"}
    with pytest.raises(ContractError) as info:
        assign_rule("benchmark", params, [], official=None, is_real=lambda r: True)
    assert "unknown benchmark pool 'some'" in info.value.message


def test_materialize_never_replaces_different_videos(built, tmp_path):
    for scheme in ("all-test", "ident-72-14-14"):
        (built["dataset"] / "splits" / f"{scheme}.tsv.gz").unlink()
    materialize(
        "packdemo/all-test", inventory=built["inventory"], official=None, work_root=built["work"]
    )
    materialized = built["work"] / "packdemo" / "materialized"
    videos = materialized / "videos.jsonl.gz"
    split = materialized / "splits" / "all-test.tsv.gz"
    before = (videos.read_bytes(), videos.stat().st_mtime_ns, split.read_bytes())

    # The same inventory again: the videos are left exactly as they are.
    materialize(
        "packdemo/ident-72-14-14",
        inventory=built["inventory"],
        official=None,
        work_root=built["work"],
    )
    assert (videos.read_bytes(), videos.stat().st_mtime_ns, split.read_bytes()) == before

    # An inventory whose split rows still hash right but whose records differ is refused.
    rows = read_jsonl(built["inventory"], InventoryRecord)
    changed = tmp_path / "changed.jsonl"
    write_jsonl(changed, [dataclasses.replace(rows[0], method="Other"), *rows[1:]])
    with pytest.raises(ContractError) as info:
        materialize("packdemo/all-test", inventory=changed, official=None, work_root=built["work"])

    assert str(videos) in info.value.message
    assert "delete" in info.value.hint
    assert "same inventory" in info.value.hint
    assert (videos.read_bytes(), videos.stat().st_mtime_ns, split.read_bytes()) == before
