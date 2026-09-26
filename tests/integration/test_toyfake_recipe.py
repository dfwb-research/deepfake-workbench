"""toyfake as a key-free recipe dataset, round-tripped through the ``dfwb`` command.

``protocols build`` writes toyfake as a list dataset; it is then decided a recipe (its card says
``distribution: recipe``, and a rebuild rewrites its notice) and stripped the way a release build
strips one (its videos, pairs and split files are taken out, leaving only ``dataset.yaml``,
``labels.yaml``, ``NOTICE.md`` and ``PROVENANCE.json``), and that pack takes the place of the
built-in toyfake pack. From there:
``protocols lint --release`` passes, a tampered inventory fails ``protocols materialize`` with exit
4 and writes nothing, the real inventory materializes every list byte for byte, and ``protocols
verify``, ``train``, ``score`` and ``eval`` all read the materialized copy.

Commands run in-process, so the stripped pack can be registered in place of the built-in one.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
import yaml
from tests._dfwb_cli import run_dfwb

import dfwb._builtins
from dfwb.core.records import InventoryRecord, read_jsonl, write_jsonl
from dfwb.protocols.protocol import load

DATASET = "toyfake"
REF = f"{DATASET}/official"
VIDEOS = 200  # dfwb datasets synth toyfake's default tree
BENCHMARK = 36  # toyfake/benchmark: every video of it is test
_KEY_LISTS = (".jsonl.gz", ".tsv.gz")


def _files(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _stamped(folder: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(folder).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def recipe(capsys, monkeypatch, tmp_path):
    """A synthesised toyfake tree and inventory, and toyfake built, then stripped to a recipe."""
    monkeypatch.chdir(tmp_path)
    for name in list(os.environ):
        if name.startswith("DFWB_"):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    datasets, work = tmp_path / "datasets", tmp_path / "work"
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(datasets))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work))
    monkeypatch.setenv("DFWB_RUNS_ROOT", str(tmp_path / "runs"))

    # The pack that takes the built-in toyfake pack's place: an importable package holding it.
    package = tmp_path / "dfwb_toyfake_recipe_fixture"
    pack = package / "pack"
    pack.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (pack / "pack.yaml").write_text(
        f"schema_version: 1\nname: {DATASET}\nversion: 1.0.0\ndatasets: []\n", "utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        dfwb._builtins,
        "PROTOCOL_PACKS",
        ((DATASET, f"{package.name}:pack", "toyfake stripped to a recipe dataset"),),
    )

    def run(*args: str):
        return run_dfwb(capsys, *args)

    assert run("datasets", "synth", DATASET, "--out", str(datasets), "--no-media").code == 0
    assert run("inventory", "build", DATASET).code == 0
    out = pack / DATASET
    built = run("protocols", "build", DATASET, "--out", str(out), "--update-pack-yaml")
    assert built.code == 0, built.err
    lists = {name: data for name, data in _files(out).items() if name.endswith(_KEY_LISTS)}
    assert sorted(lists) == [
        "pairs.jsonl.gz",
        "splits/all-test.tsv.gz",
        "splits/benchmark.tsv.gz",
        "splits/ident-72-14-14.tsv.gz",
        "splits/official.tsv.gz",
        "videos.jsonl.gz",
    ]

    # Decided a recipe: a rebuild keeps the decision, and toyfake's own terms notes, and
    # rewrites the notice to match it.
    card = yaml.safe_load((out / "dataset.yaml").read_text("utf-8"))
    assert "may be redistributed" in card["terms"]["notes"]
    card["distribution"] = "recipe"
    (out / "dataset.yaml").write_text(yaml.safe_dump(card, sort_keys=True), "utf-8")
    assert run("protocols", "build", DATASET, "--out", str(out)).code == 0
    for name in lists:
        (out / name).unlink()
    (out / "splits").rmdir()
    return {"run": run, "pack": pack, "work": work, "lists": lists, "tmp": tmp_path}


def _materialize(recipe) -> None:
    result = recipe["run"]("protocols", "materialize", DATASET)
    assert result.code == 0, result.err


def test_a_recipe_lints_fails_on_a_tampered_inventory_then_materializes_and_verifies(recipe):
    run, work = recipe["run"], recipe["work"]
    assert sorted(_files(recipe["pack"] / DATASET)) == [
        "NOTICE.md",
        "PROVENANCE.json",
        "dataset.yaml",
        "labels.yaml",
    ]
    linted = run("protocols", "lint", str(recipe["pack"]), "--release")
    assert linted.code == 0, linted.out + linted.err
    assert linted.out == "no issues found\n"

    # Nothing is materialized yet: loading says which command to run.
    unmaterialized = run("protocols", "verify", REF)
    assert unmaterialized.code == 4
    assert "hint: run: dfwb protocols materialize toyfake\n" in unmaterialized.err

    # One relabelled video: exit 4, and the work root is left exactly as it was.
    rows = read_jsonl(work / DATASET / "inventory.jsonl", InventoryRecord)
    fake = next(row for row in rows if row.key.startswith("BLEND_A/"))
    tampered = recipe["tmp"] / "tampered.jsonl"
    write_jsonl(
        tampered,
        [dataclasses.replace(r, label_key="TOY-BLEND_B") if r is fake else r for r in rows],
    )
    before = _stamped(work)
    failed = run("protocols", "materialize", DATASET, "--inventory", str(tampered))
    assert failed.code == 4
    assert "error: toyfake: " in failed.err
    assert "published hashes differ" in failed.err
    assert "hint: your inventory has the release's videos in the published numbers" in failed.err
    assert _stamped(work) == before

    # A copy that also holds a compression the release does not list still materializes: those
    # rows are no part of the published lists.
    with_c40 = recipe["tmp"] / "with-c40.jsonl"
    write_jsonl(with_c40, [*rows, *(dataclasses.replace(r, compression="c40") for r in rows[:3])])
    extra = run("protocols", "materialize", DATASET, "--inventory", str(with_c40), "--json")
    assert extra.code == 0, extra.err
    assert json.loads(extra.out)["n_videos"] == VIDEOS

    materialized = run("protocols", "materialize", DATASET, "--json")
    assert materialized.code == 0, materialized.err
    result = json.loads(materialized.out)
    assert (result["n_videos"], result["n_pairs"]) == (VIDEOS, 120)
    rebuilt = _files(work / DATASET / "materialized")
    assert {name: data for name, data in rebuilt.items() if name != "hashes.json"} == (
        recipe["lists"]
    )

    verified = run("protocols", "verify", REF, "--json")
    assert verified.code == 0, verified.err
    report = json.loads(verified.out)
    assert report["counts"]["have"] == VIDEOS
    assert report["counts"]["missing"] == report["counts"]["extra"] == 0


def _write_store(work: Path, profile) -> None:
    """A hand-built processed store over every toyfake video, for ``profile``."""
    from tests.unit.data.conftest import processed_record, write_store_frames, write_store_index

    store = work / DATASET / "processed" / profile.profile_id()
    store.mkdir(parents=True)
    payload = {
        "profile": profile.model_dump(mode="json"),
        "sha256": profile.sha256(),
        "profile_id": profile.profile_id(),
        "backend": {"name": profile.backend.name, "version": None, "license": None, "meta": {}},
    }
    (store / "profile.json").write_text(json.dumps(payload), encoding="utf-8")
    videos = read_jsonl(work / DATASET / "inventory.jsonl", InventoryRecord)
    records = [processed_record(video.key, n_frames=2) for video in videos]
    for record in records:
        write_store_frames(store, record, size=profile.crop.size)
    write_store_index(store, records)


def test_train_score_and_eval_read_the_materialized_recipe(recipe, monkeypatch):
    pytest.importorskip("lightning")
    import torch
    from tests.unit.train._toy import toy_config, toy_profile, write_toy_config

    run, work = recipe["run"], recipe["work"]
    _materialize(recipe)
    _write_store(work, toy_profile())
    assert len(load(REF, work_root=work).pairs()) == 120  # the pairs come back too
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # seeding sets it; put it back afterwards
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        config = toy_config(
            data={
                "train": [{"protocol": REF, "split": "train"}],
                "val": [{"protocol": REF, "split": "val"}],
            }
        )
        trained = run("train", "-c", str(write_toy_config(recipe["tmp"], config)), "--json")
        assert trained.code == 0, trained.err
        (entry,) = json.loads(trained.out)
        run_dir = Path(entry["run_dir"])
        data = json.loads((run_dir / "data.json").read_text("utf-8"))
        (train_source,) = data["train"]
        assert train_source["index"]["sources"][0]["counts"]["included"] == 122

        # Another scheme than the one trained on: one materialize call rebuilt every scheme.
        scored = run(
            "score",
            "--detector",
            f"run:{run_dir}",
            "--protocol",
            f"{DATASET}/benchmark",
            "--split",
            "test",
            "--device",
            "cpu",
            "--json",
        )
        assert scored.code == 0, scored.err
        (row,) = json.loads(scored.out)["results"]
        assert row["coverage"] == {"expected": BENCHMARK, "ok": BENCHMARK, "missing": 0, "error": 0}

        # A breakdown by family reads the protocol's labels again, from the score file's meta.
        evaluated = run("eval", row["csv"], "--by", "family", "--json")
        assert evaluated.code == 0, evaluated.err
    finally:
        torch.use_deterministic_algorithms(deterministic)
