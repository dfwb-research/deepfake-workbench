"""Tests for ``dfwb.preprocess.inventory.runner``: building, writing and reading inventories."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.unit.preprocess.inventory._demo import (
    DemoBuilder,
    install,
    make_demo_tree,
)

from dfwb import __version__
from dfwb.core.errors import (
    ConfigError,
    ContractError,
    InstallationError,
    PluginError,
    UnknownKeyError,
)
from dfwb.core.paths import resolve_roots
from dfwb.core.records import InventoryRecord, read_jsonl
from dfwb.preprocess.inventory.runner import (
    InventoryResult,
    build_inventory,
    folder_status,
    get_builder,
    inventory_path,
    read_inventory,
)

_MODULE = "tests.unit.preprocess.inventory._demo"


@pytest.fixture
def env(monkeypatch, tmp_path):
    """An isolated environment: no config files, no overrides, a datasets root and a work root."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    for name in list(os.environ):
        if name.startswith("DFWB_"):
            monkeypatch.delenv(name)
    raw, work = tmp_path / "raw", tmp_path / "work"
    raw.mkdir()
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(raw))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work))
    return raw, work


def test_build_writes_a_sorted_inventory_and_a_meta_without_paths(env, monkeypatch, tmp_path):
    raw, work = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo")

    result = build_inventory("demo")

    assert result == InventoryResult(
        dataset_id="demo",
        path=work / "demo" / "inventory.jsonl",
        count=8,
        by_task={"REAL": 4, "FS_SWAP": 4},
        dataset_dir=raw / "Demo",
        location_source="root 1",
    )
    records = read_jsonl(result.path, InventoryRecord)
    assert [(r.key, r.compression) for r in records] == [
        ("FS_SWAP/000_001", "c23"),
        ("FS_SWAP/000_001", "c40"),
        ("FS_SWAP/001_000", "c23"),
        ("FS_SWAP/001_000", "c40"),
        ("REAL/000", "c23"),
        ("REAL/000", "c40"),
        ("REAL/001", "c23"),
        ("REAL/001", "c40"),
    ]
    assert records[0].relpath == "swapped/c23/000_001.mp4"

    meta_text = (work / "demo" / "inventory.meta.json").read_text("utf-8")
    assert json.loads(meta_text) == {
        "builder": {"id": "demo", "version": "1"},
        "dfwb": __version__,
        "count": 8,
        "compressions": ["c23", "c40"],
        "by_task": {"REAL": 4, "FS_SWAP": 4},
        "location_source": "root 1",
    }
    assert str(tmp_path) not in meta_text
    assert str(tmp_path) not in result.path.read_text("utf-8")


def test_rebuilding_gives_identical_bytes(env, monkeypatch):
    raw, work = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo")
    first = build_inventory("demo")
    before = (first.path.read_bytes(), (work / "demo" / "inventory.meta.json").read_bytes())
    build_inventory("demo")
    after = (first.path.read_bytes(), (work / "demo" / "inventory.meta.json").read_bytes())
    assert before == after


def test_compressions_filter_and_count_by_task(env, monkeypatch):
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", fakes=())
    result = build_inventory("demo", compressions=["c40"])
    assert result.count == 2
    assert result.by_task == {"REAL": 2, "FS_SWAP": 0}
    with pytest.raises(ConfigError, match="did you mean 'c23'"):
        build_inventory("demo", compressions=["c32"])


def test_a_later_root_and_an_override_are_named_in_the_source(env, monkeypatch, tmp_path):
    raw, _ = env
    install(monkeypatch)
    second = tmp_path / "second"
    make_demo_tree(second / "Demo", compressions=("c23",))
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")
    assert build_inventory("demo").location_source == "root 2"

    elsewhere = make_demo_tree(tmp_path / "anywhere" / "renamed", compressions=("c23",))
    monkeypatch.setenv("DFWB_DATASET_DEMO", str(elsewhere))
    result = build_inventory("demo")
    assert result.location_source == "override (env: DFWB_DATASET_DEMO)"
    assert result.dataset_dir == elsewhere


def test_an_override_from_a_config_file_names_only_the_file(env, monkeypatch, tmp_path):
    install(monkeypatch)
    folder = make_demo_tree(tmp_path / "mine", compressions=("c23",))
    (tmp_path / "dfwb.toml").write_text(f'[datasets]\ndemo = "{folder}"\n')
    result = build_inventory("demo")
    assert result.location_source == "override (dfwb.toml)"
    meta = (result.path.parent / "inventory.meta.json").read_text("utf-8")
    assert str(tmp_path) not in meta


def test_root_wins_over_the_roots_search(env, monkeypatch, tmp_path):
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", compressions=("c23",))
    chosen = make_demo_tree(tmp_path / "chosen", reals=("007",), fakes=(), compressions=("c40",))
    result = build_inventory("demo", root=chosen)
    assert result.count == 1
    assert result.dataset_dir == chosen
    assert result.location_source == "--root"
    assert [r.key for r in read_inventory("demo", result.path.parent.parent)] == ["REAL/007"]


def test_root_must_be_a_directory(env, monkeypatch, tmp_path):
    install(monkeypatch)
    with pytest.raises(ConfigError, match="not a directory"):
        build_inventory("demo", root=tmp_path / "missing")


def test_a_dataset_that_is_not_found_is_a_config_error(env, monkeypatch):
    install(monkeypatch)
    with pytest.raises(ConfigError, match="was not found in any datasets root") as info:
        build_inventory("demo")
    assert "DFWB_DATASET_DEMO" in info.value.hint


def test_duplicate_key_and_compression_is_a_contract_error(env, monkeypatch):
    raw, _ = env
    install(monkeypatch, {"dup": (f"{_MODULE}:DuplicatingBuilder", "Dup", "Demo")})
    make_demo_tree(raw / "Demo")
    with pytest.raises(ContractError, match=r"duplicate record 'REAL/000' \(compression 'c23'\)"):
        build_inventory("dup")
    assert not (env[1] / "dup" / "inventory.jsonl").exists()


def test_a_key_without_a_task_is_a_contract_error(env, monkeypatch):
    raw, _ = env
    install(monkeypatch, {"badkey": (f"{_MODULE}:BadKeyBuilder", "Bad", "Demo")})
    (raw / "Demo").mkdir()
    with pytest.raises(ContractError, match="'no-slash'"):
        build_inventory("badkey")


def test_a_key_with_an_unknown_task_is_a_contract_error(env, monkeypatch):
    raw, _ = env
    install(monkeypatch, {"unknowntask": (f"{_MODULE}:UnknownTaskBuilder", "Bad", "Demo")})
    (raw / "Demo").mkdir()
    with pytest.raises(ContractError, match=r"task 'NOPE'.*REAL, FS_SWAP"):
        build_inventory("unknowntask")


def test_probe_is_not_available_yet(env, monkeypatch):
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo")
    with pytest.raises(InstallationError) as info:
        build_inventory("demo", probe=True)
    assert "[preprocess]" in info.value.hint


def test_jobs_must_be_positive(env, monkeypatch):
    install(monkeypatch)
    with pytest.raises(ConfigError, match="jobs"):
        build_inventory("demo", jobs=0)


def test_the_inventory_is_never_written_inside_the_dataset_folder(env, monkeypatch):
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", compressions=("c23",))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(raw / "Demo" / "work"))
    with pytest.raises(ConfigError, match="inside the dataset folder"):
        build_inventory("demo")
    assert not (raw / "Demo" / "work").exists()


def test_missing_inventory_gives_the_build_hint(tmp_path):
    with pytest.raises(ConfigError, match="no inventory") as info:
        read_inventory("demo", tmp_path)
    assert info.value.hint == "run: dfwb inventory build demo"


def test_read_inventory_round_trips(env, monkeypatch):
    raw, work = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo")
    build_inventory("demo")
    records = read_inventory("demo", work)
    assert len(records) == 8
    assert records[0].builder.id == "demo"


def test_inventory_path():
    assert inventory_path("ffpp", Path("work")) == Path("work") / "ffpp" / "inventory.jsonl"


def test_get_builder_resolves_through_the_registry(monkeypatch):
    install(monkeypatch, {**_demo(), "notabuilder": (f"{_MODULE}:NotABuilder", "No", "No")})
    assert isinstance(get_builder("demo"), DemoBuilder)
    with pytest.raises(UnknownKeyError, match="did you mean 'demo'"):
        get_builder("dmeo")
    with pytest.raises(PluginError, match="BaseBuilder"):
        get_builder("notabuilder")


def _demo() -> dict[str, tuple[str, str, str]]:
    return {"demo": (f"{_MODULE}:DemoBuilder", "Demo", "Demo")}


def test_folder_status_never_raises(env, monkeypatch, tmp_path):
    raw, _ = env
    make_demo_tree(raw / "Demo", compressions=("c23",))
    second = tmp_path / "second"
    (second / "Demo").mkdir(parents=True)
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")
    roots = resolve_roots()

    found = folder_status("demo", "Demo", roots, {})
    assert found.location is not None
    assert found.location.path == raw / "Demo"
    assert found.source == "root 1"
    assert found.problem is None
    assert found.location.also_found == (second / "Demo",)

    missing = folder_status("other", "Other", roots, {})
    assert (missing.location, missing.source) == (None, "not found")
    assert "was not found" in (missing.problem or "")

    bad = folder_status("demo", "Demo", roots, {"demo": (tmp_path / "nowhere", "env: X")})
    assert (bad.location, bad.source) == (None, "not found")
    assert "is not a directory" in (bad.problem or "")

    unnamed = folder_status("anon", None, roots, {})
    assert (unnamed.location, unnamed.source) == (None, "not found")

    override = folder_status("anon", None, roots, {"anon": (raw / "Demo", "env: Y")})
    assert override.source == "override (env: Y)"
