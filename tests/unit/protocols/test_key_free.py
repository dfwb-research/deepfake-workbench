"""Key-free recipe datasets: a pack that ships no key list, rebuilt and checked from an inventory.

The packdemo dataset of the pack-building tests is built into a registered pack as a list
dataset, then turned into a recipe the way a release build does: its card says
``distribution: recipe`` and its ``videos.jsonl.gz``, ``pairs.jsonl.gz`` and split files are taken
out, so the pack holds only ``dataset.yaml``, ``labels.yaml``, ``NOTICE.md`` and
``PROVENANCE.json``. ``materialize_dataset`` then rebuilds every list from the local inventory.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
import yaml
from tests.unit.preprocess.test_packbuild import PACK_NAME, setup_packdemo

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError
from dfwb.core.records import InventoryRecord, PairRecord, read_jsonl, write_jsonl
from dfwb.preprocess.inventory.runner import get_builder
from dfwb.preprocess.packbuild import _pairs, add_to_pack_yaml, build_dataset
from dfwb.protocols.materialization import (
    DatasetMaterialization,
    materialize,
    materialize_dataset,
    ships_key_lists,
)
from dfwb.protocols.protocol import load

_SCHEMES = ("all-test", "benchmark", "ident-72-14-14", "official")
_KEY_LISTS = ("videos.jsonl.gz", "pairs.jsonl.gz", *(f"splits/{s}.tsv.gz" for s in _SCHEMES))


def _files(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _set_card(dataset_dir: Path, **changes: object) -> None:
    path = dataset_dir / "dataset.yaml"
    card = yaml.safe_load(path.read_text("utf-8"))
    card.update(changes)
    path.write_text(yaml.safe_dump(card, sort_keys=True), "utf-8")


def _served(ref: str, work_root: Path) -> dict[str, object]:
    """Everything a loaded protocol serves, read now (the lists are read lazily otherwise)."""
    protocol = load(ref, work_root=work_root)
    return {
        "sha256": protocol.sha256,
        "rows": protocol.split_rows(),
        "records": protocol.records(),
        "pairs": protocol.pairs(),
        "test pairs": protocol.pairs("test"),
        "labels": protocol.labels("binary"),
    }


@pytest.fixture
def recipe(monkeypatch, tmp_path):
    """packdemo built as a list dataset, then stripped to a key-free recipe."""
    paths = setup_packdemo(monkeypatch, tmp_path)
    dataset = paths["pack"] / "packdemo"
    build_dataset("packdemo", out=dataset)
    add_to_pack_yaml(paths["pack"], "packdemo")
    lists = {name: data for name, data in _files(dataset).items() if name in _KEY_LISTS}
    published = {scheme: _served(f"packdemo/{scheme}", paths["work"]) for scheme in _SCHEMES}
    _set_card(dataset, distribution="recipe")
    for name in ("videos.jsonl.gz", "pairs.jsonl.gz"):
        (dataset / name).unlink()
    shutil.rmtree(dataset / "splits")
    return {**paths, "dataset": dataset, "lists": lists, "published": published}


def _official(recipe):
    records = read_jsonl(recipe["inventory"], InventoryRecord)
    return get_builder("packdemo").official_splits(recipe["raw"] / "PackDemo", records)


def _builder_pairs(inventory: Path) -> list[PairRecord]:
    return _pairs(get_builder("packdemo"), read_jsonl(inventory, InventoryRecord))


def _materialize(recipe, *, inventory=None, pairs=None, ref="packdemo", **options):
    inventory = recipe["inventory"] if inventory is None else inventory
    return materialize_dataset(
        ref,
        inventory=inventory,
        official=options.pop("official", _official(recipe)),
        pairs=_builder_pairs(inventory) if pairs is None else pairs,
        work_root=options.pop("work_root", recipe["work"]),
        **options,
    )


def test_the_stripped_dataset_ships_no_key_list(recipe):
    assert sorted(_files(recipe["dataset"])) == [
        "NOTICE.md",
        "PROVENANCE.json",
        "dataset.yaml",
        "labels.yaml",
    ]
    assert ships_key_lists("packdemo") is False
    assert ships_key_lists(f"{PACK_NAME}:packdemo/official") is False


def test_a_list_dataset_ships_its_key_lists(monkeypatch, tmp_path):
    paths = setup_packdemo(monkeypatch, tmp_path)
    build_dataset("packdemo", out=paths["pack"] / "packdemo")
    add_to_pack_yaml(paths["pack"], "packdemo")
    assert ships_key_lists("packdemo") is True


def test_materialize_dataset_rebuilds_every_list_byte_for_byte(recipe):
    result = _materialize(recipe)

    materialized = recipe["work"] / "packdemo" / "materialized"
    card = yaml.safe_load((recipe["dataset"] / "dataset.yaml").read_text("utf-8"))
    schemes = {name: scheme["sha256"] for name, scheme in card["schemes"].items()}
    assert result == DatasetMaterialization(
        dataset="packdemo",
        path=materialized,
        videos_sha256=card["videos_sha256"],
        pairs_sha256=card["pairs_sha256"],
        schemes=schemes,
        n_videos=30,
        n_pairs=20,
    )
    files = _files(materialized)
    assert {name: data for name, data in files.items() if name != "hashes.json"} == recipe["lists"]
    assert json.loads(files["hashes.json"]) == {
        "videos_sha256": card["videos_sha256"],
        "pairs_sha256": card["pairs_sha256"],
        "schemes": schemes,
    }


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_the_loader_serves_the_materialized_copy(recipe, scheme):
    _materialize(recipe)

    served = _served(f"packdemo/{scheme}", recipe["work"])
    assert served == recipe["published"][scheme]
    assert served["pairs"]  # packdemo pairs every fake with its target's real


@pytest.mark.parametrize(
    ("ref", "command"),
    [
        ("packdemo/official", "packdemo"),
        ("packdemo", "packdemo"),
        (f"{PACK_NAME}:packdemo/all-test", f"{PACK_NAME}:packdemo"),
    ],
)
def test_the_loader_names_the_command_when_nothing_is_materialized(recipe, ref, command):
    with pytest.raises(ContractError) as info:
        load(ref, work_root=recipe["work"])

    assert "recipe dataset" in info.value.message
    assert "no key list" in info.value.message
    assert info.value.hint == f"run: dfwb protocols materialize {command}"


def test_a_partly_missing_materialization_counts_as_none(recipe):
    _materialize(recipe)
    (recipe["work"] / "packdemo" / "materialized" / "hashes.json").unlink()

    with pytest.raises(ContractError) as info:
        load("packdemo/official", work_root=recipe["work"])
    assert info.value.hint == "run: dfwb protocols materialize packdemo"


def test_a_key_free_dataset_that_is_not_a_recipe_is_a_broken_pack(recipe):
    _set_card(recipe["dataset"], distribution="list")

    with pytest.raises(ContractError) as info:
        load("packdemo/official", work_root=recipe["work"])
    assert "videos.jsonl.gz" in info.value.message
    assert "reinstall" in info.value.hint


@pytest.mark.parametrize("field", ["videos_sha256", "pairs_sha256"])
def test_a_pack_upgrade_since_materializing_asks_to_materialize_again(recipe, field):
    # A new pack version relabels a video, say: the scheme hashes are the same, the lists' not.
    _materialize(recipe)
    _set_card(recipe["dataset"], **{field: "0" * 64})

    with pytest.raises(ContractError) as info:
        load("packdemo/official", work_root=recipe["work"])
    assert field in info.value.message
    assert info.value.hint == (
        "the pack changed since it was materialised; run: dfwb protocols materialize packdemo"
    )


def test_a_materialized_split_that_drifted_asks_to_materialize_again(recipe):
    _materialize(recipe)
    split = recipe["work"] / "packdemo" / "materialized" / "splits" / "all-test.tsv.gz"
    shutil.copyfile(
        recipe["work"] / "packdemo" / "materialized" / "splits" / "official.tsv.gz", split
    )

    with pytest.raises(ContractError) as info:
        load("packdemo/all-test", work_root=recipe["work"])
    assert "materialized split" in info.value.message
    assert info.value.hint.endswith("run: dfwb protocols materialize packdemo")


def _tree(work: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(work).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(work.rglob("*"))
        if path.is_file()
    }


def test_a_relabelled_video_fails_the_videos_hash_and_writes_nothing(recipe, tmp_path):
    rows = read_jsonl(recipe["inventory"], InventoryRecord)
    tampered = tmp_path / "tampered.jsonl"
    write_jsonl(tampered, [dataclasses.replace(rows[0], method="Other"), *rows[1:]])
    before = _tree(recipe["work"])

    with pytest.raises(ContractError) as info:
        _materialize(recipe, inventory=tampered)

    assert info.value.message.startswith("packdemo: 1 of 6 published hashes differ")
    assert "videos: " in info.value.message
    assert "official" not in info.value.message
    assert "dfwb inventory build packdemo" in info.value.hint
    assert _tree(recipe["work"]) == before


def test_a_missing_video_names_every_hash_it_breaks(recipe, tmp_path):
    rows = read_jsonl(recipe["inventory"], InventoryRecord)
    short = tmp_path / "short.jsonl"
    write_jsonl(short, [row for row in rows if row.key != "REAL/p00"])
    before = _tree(recipe["work"])

    with pytest.raises(ContractError) as info:
        _materialize(recipe, inventory=short)

    message = info.value.message
    for part in ("videos: ", "all-test: ", "ident-72-14-14: ", "official: ", "pairs: "):
        assert part in message
    assert _tree(recipe["work"]) == before


def test_a_mismatch_leaves_an_earlier_materialization_exactly_as_it_was(recipe, tmp_path):
    _materialize(recipe)
    before = _tree(recipe["work"])
    rows = read_jsonl(recipe["inventory"], InventoryRecord)
    tampered = tmp_path / "tampered.jsonl"
    write_jsonl(tampered, rows[1:])

    with pytest.raises(ContractError):
        _materialize(recipe, inventory=tampered)
    assert _tree(recipe["work"]) == before


def test_materializing_again_replaces_the_whole_folder(recipe):
    _materialize(recipe)
    materialized = recipe["work"] / "packdemo" / "materialized"
    (materialized / "splits" / "stale.tsv.gz").write_bytes(b"left from an older pack")

    _materialize(recipe)

    assert not (materialized / "splits" / "stale.tsv.gz").exists()
    assert sorted(p.name for p in recipe["work"].joinpath("packdemo").iterdir()) == [
        "inventory.jsonl",
        "inventory.meta.json",
        "materialized",
    ]


def test_pairs_drawn_by_another_rule_are_refused(recipe):
    pairs = [dataclasses.replace(p, rule="other") for p in _builder_pairs(recipe["inventory"])]

    with pytest.raises(ContractError) as info:
        _materialize(recipe, pairs=pairs)
    assert "'other'" in info.value.message
    assert "'target-id'" in info.value.message
    assert not (recipe["work"] / "packdemo" / "materialized").exists()


@pytest.mark.parametrize("field", ["videos_sha256", "pairs_sha256"])
def test_a_card_without_the_list_hashes_cannot_be_materialized(recipe, field):
    _set_card(recipe["dataset"], **{field: None})

    with pytest.raises(ContractError) as info:
        _materialize(recipe)
    assert field in info.value.message
    assert "dfwb protocols build" in info.value.hint


def test_a_scheme_whose_rule_cannot_be_recomputed_is_refused(recipe):
    card = yaml.safe_load((recipe["dataset"] / "dataset.yaml").read_text("utf-8"))
    card["schemes"]["all-test"]["rule"] = "hand-picked"
    _set_card(recipe["dataset"], schemes=card["schemes"])

    with pytest.raises(ContractError) as info:
        _materialize(recipe)
    assert "'hand-picked'" in info.value.message


def test_the_official_split_is_needed_when_a_rule_reads_it(recipe):
    with pytest.raises(ConfigError) as info:
        _materialize(recipe, official=None)
    assert "official split" in info.value.message


def test_a_missing_inventory_is_a_config_error(recipe, tmp_path):
    with pytest.raises(ConfigError) as info:
        _materialize(recipe, inventory=tmp_path / "none.jsonl", pairs=[])
    assert info.value.hint == "run: dfwb inventory build packdemo"


def test_it_never_writes_under_a_datasets_root(recipe):
    inside = recipe["raw"] / "dfwb-work"  # the fixture's DFWB_DATASETS_ROOT is recipe["raw"]
    with pytest.raises(ConfigError):
        _materialize(recipe, work_root=inside)
    assert not inside.exists()


def test_a_scheme_and_pin_in_the_reference_are_checked(recipe):
    with pytest.raises(UnknownKeyError):
        _materialize(recipe, ref="packdemo/nope")
    with pytest.raises(ContractError) as info:
        _materialize(recipe, ref="packdemo/official@9.9.9")
    assert "pinned @9.9.9" in info.value.message
    assert _materialize(recipe, ref="packdemo/official@1.0.0").dataset == "packdemo"


def test_one_scheme_alone_is_never_materialized_for_a_dataset_without_key_lists(recipe):
    with pytest.raises(ConfigError) as info:
        materialize(
            "packdemo/all-test",
            inventory=recipe["inventory"],
            official=None,
            work_root=recipe["work"],
        )

    assert "every list" in info.value.message
    assert "materialize_dataset" in info.value.hint
    assert "dfwb protocols materialize packdemo" in info.value.hint
    assert not (recipe["work"] / "packdemo" / "materialized").exists()
