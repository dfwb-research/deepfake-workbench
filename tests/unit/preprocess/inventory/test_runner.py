"""Tests for ``dfwb.preprocess.inventory.runner``: building, writing and reading inventories."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterator, Sequence
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
from dfwb.core.paths import locate_dataset, resolve_roots
from dfwb.core.records import BuilderRef, InventoryRecord, read_jsonl
from dfwb.preprocess.inventory.base import BaseBuilder, TaskSpec
from dfwb.preprocess.inventory.runner import (
    InventoryResult,
    _root_source,  # a private fallback-text check
    build_inventory,
    collect_records,
    dataset_copies,
    folder_copies,
    folder_status,
    get_builder,
    inventory_path,
    read_inventory,
    resolve_video_path,
)

_MODULE = "tests.unit.preprocess.inventory._demo"
_HERE = "tests.unit.preprocess.inventory.test_runner"


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
        location_source={"c23": "root 1", "c40": "root 1", "metadata": "root 1"},
        copies=(raw / "Demo",),
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
        "location_source": {"c23": "root 1", "c40": "root 1", "metadata": "root 1"},
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
    result = build_inventory("demo")
    assert result.location_source == {"c23": "root 2", "metadata": "root 2"}

    elsewhere = make_demo_tree(tmp_path / "anywhere" / "renamed", compressions=("c23",))
    monkeypatch.setenv("DFWB_DATASET_DEMO", str(elsewhere))
    result = build_inventory("demo")
    override_source = "override (env: DFWB_DATASET_DEMO)"
    assert result.location_source == {"c23": override_source, "metadata": override_source}
    assert result.dataset_dir == elsewhere


def test_an_override_from_a_config_file_names_only_the_file(env, monkeypatch, tmp_path):
    install(monkeypatch)
    folder = make_demo_tree(tmp_path / "mine", compressions=("c23",))
    (tmp_path / "dfwb.toml").write_text(f'[datasets]\ndemo = "{folder}"\n')
    result = build_inventory("demo")
    assert result.location_source == {
        "c23": "override (dfwb.toml)",
        "metadata": "override (dfwb.toml)",
    }
    meta = (result.path.parent / "inventory.meta.json").read_text("utf-8")
    assert str(tmp_path) not in meta


def test_a_dataset_split_across_roots_by_compression_is_merged(env, monkeypatch, tmp_path):
    # root 1 has only c23 originals; root 2 has c40 originals and every fake.
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", reals=("000", "001"), fakes=(), compressions=("c23",))
    second = tmp_path / "second"
    make_demo_tree(second / "Demo", reals=("000", "001"), compressions=("c40",))
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    result = build_inventory("demo")

    assert result.count == 6  # 2 c23 reals (root 1) + 2 c40 reals + 2 c40 fakes (root 2)
    assert result.location_source == {"c23": "root 1", "c40": "root 2", "metadata": "root 1"}
    records = read_inventory("demo", result.path.parent.parent)
    c23_real = next(r for r in records if r.key == "REAL/000" and r.compression == "c23")
    assert (raw / "Demo" / c23_real.relpath).is_file()
    c40_real = next(r for r in records if r.key == "REAL/000" and r.compression == "c40")
    assert (second / "Demo" / c40_real.relpath).is_file()


def test_falls_back_to_the_first_root_when_no_copy_has_any_video(
    env, monkeypatch, tmp_path, caplog
):
    raw, _ = env
    install(monkeypatch)
    (raw / "Demo").mkdir()
    second = tmp_path / "second"
    (second / "Demo").mkdir(parents=True)
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    with caplog.at_level("WARNING", logger="dfwb.preprocess.inventory.runner"):
        result = build_inventory("demo")

    assert result.dataset_dir == raw / "Demo"
    assert result.location_source == {"metadata": "root 1"}
    assert result.count == 0
    assert "no copy of the dataset folder has any video" in caplog.text
    assert "root 1" in caplog.text
    assert "root 2" in caplog.text


def test_an_override_without_any_video_is_used_with_a_warning(env, monkeypatch, tmp_path, caplog):
    install(monkeypatch)
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    monkeypatch.setenv("DFWB_DATASET_DEMO", str(empty))

    with caplog.at_level("WARNING", logger="dfwb.preprocess.inventory.runner"):
        result = build_inventory("demo")

    assert result.dataset_dir == empty
    assert result.location_source == {"metadata": "override (env: DFWB_DATASET_DEMO)"}
    assert result.count == 0
    assert "does not have the expected raw layout" in caplog.text
    assert "originals" in caplog.text


def test_root_flag_without_any_video_is_used_with_a_warning(env, monkeypatch, tmp_path, caplog):
    install(monkeypatch)
    empty = tmp_path / "chosen"
    empty.mkdir()

    with caplog.at_level("WARNING", logger="dfwb.preprocess.inventory.runner"):
        result = build_inventory("demo", root=empty)

    assert result.dataset_dir == empty
    assert result.location_source == {"metadata": "--root"}
    assert result.count == 0
    assert "does not have the expected raw layout" in caplog.text
    assert "originals" in caplog.text


def test_a_real_builder_picks_the_root_with_the_raw_videos_over_frames(env, monkeypatch, tmp_path):
    # Uses the real, built-in ffpp builder (no install()).
    raw, _ = env
    processed = raw / "FaceForensics++" / "original_content" / "youtube" / "c23" / "frames"
    processed.mkdir(parents=True)
    (processed / "000_0001.png").touch()

    second = tmp_path / "second"
    real_dir = second / "FaceForensics++" / "original_content" / "YouTube" / "c23" / "videos"
    real_dir.mkdir(parents=True)
    for stem in ("000", "001"):
        (real_dir / f"{stem}.mp4").touch()
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    result = build_inventory("ffpp", compressions=["c23"])

    assert result.dataset_dir == second / "FaceForensics++"
    assert result.location_source == {"c23": "root 2", "metadata": "root 2"}
    assert result.count == 2
    assert result.by_task["REAL"] == 2


def test_ffpp_merges_compressions_split_across_roots_and_reads_metadata_from_its_copy(
    env, monkeypatch, tmp_path
):
    # root 1: c40 videos only, no official split files. root 2: c23 videos and the official
    # split files.
    raw, _ = env
    c40_dir = raw / "FaceForensics++" / "original_content" / "YouTube" / "c40" / "videos"
    c40_dir.mkdir(parents=True)
    (c40_dir / "000.mp4").touch()

    second = tmp_path / "second"
    c23_dir = second / "FaceForensics++" / "original_content" / "YouTube" / "c23" / "videos"
    c23_dir.mkdir(parents=True)
    (c23_dir / "000.mp4").touch()
    official = second / "FaceForensics++" / ".official_files" / "splits"
    official.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (official / f"{split}.json").write_text("[]")
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    result = build_inventory("ffpp")

    assert result.count == 2
    assert result.location_source["c23"] == "root 2"
    assert result.location_source["c40"] == "root 1"
    assert result.location_source["metadata"] == "root 2"  # holds every official split file
    assert result.dataset_dir == second / "FaceForensics++"


def test_compression_provenance_lists_every_source_when_tasks_disagree(env, monkeypatch, tmp_path):
    # root 1 has REAL/c23 only; root 2 has FS_SWAP/c23 only: both tasks want c23, from two
    # different copies.
    raw, _ = env
    install(monkeypatch)
    (raw / "Demo" / "originals" / "c23").mkdir(parents=True)
    (raw / "Demo" / "originals" / "c23" / "000.mp4").touch()
    second = tmp_path / "second"
    (second / "Demo" / "swapped" / "c23").mkdir(parents=True)
    (second / "Demo" / "swapped" / "c23" / "000_001.mp4").touch()
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    result = build_inventory("demo", compressions=["c23"])

    assert result.count == 2
    assert result.location_source["c23"] == "root 1, root 2"


def test_warns_when_a_requested_compression_is_found_in_no_copy(env, monkeypatch, caplog):
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", compressions=("c23",))  # no c40 anywhere

    with caplog.at_level("WARNING", logger="dfwb.preprocess.inventory.runner"):
        result = build_inventory("demo", compressions=["c23", "c40"])

    assert result.count == 4  # only the c23 records
    assert "c40" in caplog.text
    assert "found in no copy" in caplog.text


def test_warns_when_a_later_copy_is_shadowed(env, monkeypatch, tmp_path, caplog):
    # Both roots hold REAL/c23; root 1 wins and root 2's copy is never scanned.
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", reals=("000",), fakes=(), compressions=("c23",))
    second = tmp_path / "second"
    make_demo_tree(second / "Demo", reals=("000",), fakes=(), compressions=("c23",))
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    with caplog.at_level("WARNING", logger="dfwb.preprocess.inventory.runner"):
        result = build_inventory("demo")

    assert result.count == 1
    assert "REAL/c23" in caplog.text
    assert "root 1" in caplog.text
    assert "root 2" in caplog.text
    assert "never scanned" in caplog.text


def test_root_source_falls_back_when_the_path_matches_no_root(env, monkeypatch, tmp_path):
    roots = resolve_roots()
    assert _root_source(tmp_path / "elsewhere" / "Demo", "Demo", roots) == "a datasets root"


class _MetadataDrivenDemo(DemoBuilder):
    """A builder whose layout is a metadata file, not a set of video directories."""

    dataset_id = "metadriven"

    def layout_present(self, folder: Path) -> bool:
        return (folder / "manifest.json").is_file()


def test_dataset_copies_uses_the_builder_s_layout_present_override(env, monkeypatch, tmp_path):
    raw, _ = env
    install(monkeypatch, {"metadriven": (f"{_HERE}:_MetadataDrivenDemo", "MetaDriven", "Demo")})
    # root 1 has real video files but no manifest: layout_present() says False regardless.
    make_demo_tree(raw / "Demo", compressions=("c23",))
    second = tmp_path / "second"
    # root 2 has the manifest but no videos at all: layout_present() says True regardless.
    (second / "Demo").mkdir(parents=True)
    (second / "Demo" / "manifest.json").touch()
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")
    roots = resolve_roots()
    location = locate_dataset("metadriven", "Demo", roots, overrides={})

    copies = dataset_copies(_MetadataDrivenDemo(), location, roots)

    assert [(c.source, c.has_layout) for c in copies] == [("root 1", False), ("root 2", True)]


def test_metadata_copy_and_no_videos_warning_respect_layout_present_override(
    env, monkeypatch, tmp_path, caplog
):
    raw, _ = env
    install(monkeypatch, {"metadriven": (f"{_HERE}:_MetadataDrivenDemo", "MetaDriven", "Demo")})
    (raw / "Demo").mkdir(parents=True)  # no manifest, no videos
    second = tmp_path / "second"
    (second / "Demo").mkdir(parents=True)
    (second / "Demo" / "manifest.json").touch()  # layout_present() True, though no videos either
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    with caplog.at_level("WARNING", logger="dfwb.preprocess.inventory.runner"):
        result = build_inventory("metadriven")

    assert result.dataset_dir == second / "Demo"
    assert "no copy of the dataset folder has any video" not in caplog.text


def test_folder_copies_finds_every_root_holding_the_folder(env, monkeypatch, tmp_path):
    raw, _ = env
    (raw / "FaceForensics++").mkdir()
    middle = tmp_path / "middle"
    middle.mkdir()  # no FaceForensics++ here
    second = tmp_path / "second"
    (second / "FaceForensics++").mkdir(parents=True)
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{middle}:{second}")
    roots = resolve_roots()

    assert folder_copies("FaceForensics++", roots) == [
        raw / "FaceForensics++",
        second / "FaceForensics++",
    ]
    assert folder_copies("Nonexistent", roots) == []


def test_resolve_video_path_follows_record_folder_independent_of_its_own_dataset(
    env, monkeypatch, tmp_path
):
    # The sibling folder is split by compression exactly like a dataset's own folder can be:
    # root 1 holds a c40 copy, root 3 holds the c23 copy the record actually needs.
    raw, _ = env
    c40_dir = raw / "FaceForensics++" / "original_content" / "YouTube" / "c40" / "videos"
    c40_dir.mkdir(parents=True)
    (c40_dir / "000.mp4").touch()
    middle = tmp_path / "middle"
    middle.mkdir()
    third = tmp_path / "third"
    c23_dir = third / "FaceForensics++" / "original_content" / "YouTube" / "c23" / "videos"
    c23_dir.mkdir(parents=True)
    (c23_dir / "000.mp4").touch()
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{middle}:{third}")
    roots = resolve_roots()

    record = InventoryRecord(
        key="REAL/000",
        compression="c23",
        label_key="THB-REAL",
        method="original",
        relpath="original_content/YouTube/c23/videos/000.mp4",
        builder=BuilderRef("thb", "1"),
        folder="FaceForensics++",
    )

    resolved = resolve_video_path(record, (), datasets_roots=roots)

    assert resolved == c23_dir / "000.mp4"


def test_resolve_video_path_requires_datasets_roots_for_a_sibling_folder_record():
    record = InventoryRecord(
        key="REAL/000",
        compression=None,
        label_key="THB-REAL",
        method="original",
        relpath="x/000.mp4",
        builder=BuilderRef("thb", "1"),
        folder="FaceForensics++",
    )
    with pytest.raises(ConfigError, match="datasets roots"):
        resolve_video_path(record, ())


def test_the_copy_choice_is_computed_once_and_reused_for_discover_and_provenance(
    env, monkeypatch, tmp_path
):
    # Both copies hold both tasks (at their own compression), so every (task, compression)
    # combination has exactly one match -- none is left for discover() to recompute on the spot.
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", compressions=("c23",))
    second = tmp_path / "second"
    make_demo_tree(second / "Demo", compressions=("c40",))
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    calls: list[tuple[str, str | None]] = []
    original = BaseBuilder.video_copy_for

    def counting(self, copies, task, compression):
        calls.append((task.abbr, compression))
        return original(self, copies, task, compression)

    monkeypatch.setattr(BaseBuilder, "video_copy_for", counting)

    result = build_inventory("demo")

    assert result.count == 8
    # DemoBuilder: 2 tasks x 2 known compressions = 4 combinations, each probed exactly once:
    # discover() and the provenance report share the one choice the runner computed.
    assert len(calls) == 4
    assert len(set(calls)) == 4


def test_root_wins_over_the_roots_search(env, monkeypatch, tmp_path):
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", compressions=("c23",))
    chosen = make_demo_tree(tmp_path / "chosen", reals=("007",), fakes=(), compressions=("c40",))
    result = build_inventory("demo", root=chosen)
    assert result.count == 1
    assert result.dataset_dir == chosen
    assert result.location_source == {"c40": "--root", "metadata": "--root"}
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


def test_the_inventory_is_never_written_under_a_datasets_root(env, monkeypatch, tmp_path):
    raw, _ = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", compressions=("c23",))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(raw / "work"))
    with pytest.raises(ConfigError, match="inside the datasets root"):
        build_inventory("demo")
    # the same holds when the dataset folder itself is given with --root, outside any root
    chosen = make_demo_tree(tmp_path / "chosen", compressions=("c23",))
    with pytest.raises(ConfigError, match="inside the datasets root"):
        build_inventory("demo", root=chosen)
    assert not (raw / "work").exists()


def test_dataset_copies_annotates_each_with_its_populated_video_dirs(env, monkeypatch, tmp_path):
    raw, _ = env
    install(monkeypatch)
    (raw / "Demo" / "originals" / "c23").mkdir(parents=True)  # a placeholder, no video
    second = tmp_path / "second"
    make_demo_tree(second / "Demo", compressions=("c23",))
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")
    roots = resolve_roots()
    location = locate_dataset("demo", "Demo", roots, overrides={})

    copies = dataset_copies(DemoBuilder(), location, roots)

    assert [(c.path, c.source, c.has_layout) for c in copies] == [
        (raw / "Demo", "root 1", False),
        (second / "Demo", "root 2", True),
    ]
    assert copies[1].video_dirs == ("originals/c23", "swapped/c23")


def test_dataset_copies_of_an_override_is_a_single_copy(env, monkeypatch, tmp_path):
    install(monkeypatch)
    folder = make_demo_tree(tmp_path / "mine", compressions=("c23",))
    monkeypatch.setenv("DFWB_DATASET_DEMO", str(folder))
    roots = resolve_roots()
    location = locate_dataset("demo", "Demo", roots, overrides={"demo": (folder, "env: X")})

    copies = dataset_copies(DemoBuilder(), location, roots)

    assert len(copies) == 1
    assert copies[0].path == folder
    assert copies[0].source == "override (env: X)"
    assert copies[0].has_layout


def test_resolve_video_path_uses_the_first_copy_that_has_the_file(tmp_path):
    copy_a, copy_b = tmp_path / "a", tmp_path / "b"
    (copy_b / "originals" / "c23").mkdir(parents=True)
    (copy_b / "originals" / "c23" / "000.mp4").touch()
    record = InventoryRecord(
        key="REAL/000",
        compression="c23",
        label_key="DEMO-REAL",
        method="original",
        relpath="originals/c23/000.mp4",
        builder=BuilderRef("demo", "1"),
    )
    assert resolve_video_path(record, (copy_a, copy_b)) == copy_b / "originals" / "c23" / "000.mp4"


def test_resolve_video_path_raises_when_no_copy_has_the_file(tmp_path):
    record = InventoryRecord(
        key="REAL/000",
        compression="c23",
        label_key="DEMO-REAL",
        method="original",
        relpath="originals/c23/000.mp4",
        builder=BuilderRef("demo", "1"),
    )
    with pytest.raises(ConfigError, match=r"REAL/000.*was not found in any copy"):
        resolve_video_path(record, (tmp_path / "a", tmp_path / "b"))


def _frame_dir_record(folder: str | None = None) -> InventoryRecord:
    """A record whose relpath is a directory of frames, not a video file."""
    return InventoryRecord(
        key="REAL/real_test_1_2",
        compression=None,
        label_key="DEMO-REAL",
        method="original",
        relpath="frames/real_test_1_2",
        builder=BuilderRef("demo", "1"),
        folder=folder,
    )


def test_resolve_video_path_resolves_a_frame_directory(tmp_path):
    copy_a, copy_b = tmp_path / "a", tmp_path / "b"
    (copy_a / "frames").mkdir(parents=True)  # the parent only: not this record's directory
    (copy_b / "frames" / "real_test_1_2").mkdir(parents=True)
    (copy_b / "frames" / "real_test_1_2" / "000000.png").touch()
    assert resolve_video_path(_frame_dir_record(), (copy_a, copy_b)) == (
        copy_b / "frames" / "real_test_1_2"
    )
    with pytest.raises(ConfigError, match=r"real_test_1_2.*was not found in any copy"):
        resolve_video_path(_frame_dir_record(), (copy_a,))


def test_resolve_video_path_resolves_a_frame_directory_in_a_sibling_folder(
    env, monkeypatch, tmp_path
):
    raw, _ = env
    (raw / "Sibling" / "frames").mkdir(parents=True)  # the folder, without the directory
    second = tmp_path / "second"
    (second / "Sibling" / "frames" / "real_test_1_2").mkdir(parents=True)
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")
    roots = resolve_roots()

    resolved = resolve_video_path(_frame_dir_record("Sibling"), (), datasets_roots=roots)

    assert resolved == second / "Sibling" / "frames" / "real_test_1_2"


# ------------------------------------------------------------------------------ collect_records


class _Preparing(DemoBuilder):
    """Loads a (pretend) metadata file once per build, then discovers from it."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.known: set[str] = set()

    def prepare(self, root: Path) -> None:
        self.calls.append("prepare")
        listing = root / "listing.txt"
        self.known = set(listing.read_text().split()) if listing.is_file() else set()

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        self.calls.append("discover")
        for record in super().discover(root, compressions=compressions):
            if record.key.partition("/")[2] in self.known:
                yield record


def test_prepare_runs_once_before_discover(tmp_path):
    make_demo_tree(tmp_path, compressions=("c23",))
    (tmp_path / "listing.txt").write_text("000 000_001\n")
    builder = _Preparing()
    records = collect_records(builder, tmp_path)
    assert builder.calls == ["prepare", "discover"]
    assert [r.key for r in records] == ["FS_SWAP/000_001", "REAL/000"]
    assert collect_records(_Preparing(), tmp_path / "empty") == []


class _IgnoresCompressions(DemoBuilder):
    """Overrides discover and never looks at ``compressions``."""

    def __init__(self) -> None:
        self.received: list[Sequence[str] | None] = []

    def discover(
        self, root: Path, *, compressions: Sequence[str] | None = None
    ) -> Iterator[InventoryRecord]:
        self.received.append(compressions)
        return iter(())


def test_compressions_are_validated_even_when_discover_is_overridden(tmp_path):
    builder = _IgnoresCompressions()
    with pytest.raises(ConfigError, match=r"unknown compression 'c32' \(did you mean 'c23'"):
        collect_records(builder, tmp_path, compressions=["c32"])
    assert builder.received == []
    collect_records(builder, tmp_path, compressions=["c40", "c23", "c40"])
    collect_records(builder, tmp_path)
    assert builder.received == [["c23", "c40"], None]


class _WrongLabel(DemoBuilder):
    """Stamps every record with another task's label key."""

    def record_for_video(
        self, task: TaskSpec, path: Path, relpath: str, compression: str | None
    ) -> InventoryRecord | None:
        record = self.record(task, path.stem, relpath, compression)
        return dataclasses.replace(record, label_key="DEMO-FS_SWAP")


def test_a_label_key_that_does_not_match_the_task_is_a_contract_error(tmp_path):
    make_demo_tree(tmp_path, fakes=(), compressions=("c23",))
    with pytest.raises(ContractError, match=r"'REAL/000'.*'DEMO-FS_SWAP'.*'DEMO-REAL'"):
        collect_records(_WrongLabel(), tmp_path)
