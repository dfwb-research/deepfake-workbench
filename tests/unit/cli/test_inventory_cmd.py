import json

import pytest
from tests.unit.preprocess.inventory._demo import install, make_demo_tree


@pytest.fixture
def demo(run, monkeypatch, tmp_path):
    """The demo builder registered, its release under a datasets root, and a work root."""
    install(monkeypatch)
    raw, work = tmp_path / "raw", tmp_path / "work"
    make_demo_tree(raw / "Demo")
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(raw))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work))
    monkeypatch.delenv("DFWB_DATASET_DEMO", raising=False)
    return raw, work


def test_build_then_show_by_task(run, demo):
    raw, work = demo
    built = run("inventory", "build", "demo")
    assert built.code == 0, built.err
    assert built.out.strip() == (
        f"wrote 8 records to {work / 'demo' / 'inventory.jsonl'} "
        f"(dataset folder: {raw / 'Demo'}, from c23: root 1, c40: root 1, metadata: root 1)"
    )

    shown = run("inventory", "show", "demo", "--by", "task", "--json")
    assert shown.code == 0, shown.err
    assert json.loads(shown.out) == {
        "dataset_id": "demo",
        "by": "task",
        "counts": [{"value": "FS_SWAP", "count": 4}, {"value": "REAL", "count": 4}],
        "total": 8,
    }


def test_build_json_and_filters(run, demo, tmp_path):
    raw, work = demo
    result = run("inventory", "build", "demo", "--compressions", "c40", "--json")
    assert result.code == 0, result.err
    assert json.loads(result.out) == {
        "dataset_id": "demo",
        "path": str(work / "demo" / "inventory.jsonl"),
        "count": 4,
        "by_task": {"REAL": 2, "FS_SWAP": 2},
        "dataset_dir": str(raw / "Demo"),
        "location_source": {"c40": "root 1", "metadata": "root 1"},
    }
    chosen = make_demo_tree(tmp_path / "chosen", fakes=(), compressions=("c23",))
    result = run("inventory", "build", "demo", "--root", str(chosen), "--compressions", "c23,")
    assert result.code == 0, result.err
    assert result.out.startswith("wrote 2 records")
    assert "c23: --root" in result.out
    assert "metadata: --root" in result.out


def test_show_by_every_column(run, demo):
    assert run("inventory", "build", "demo").code == 0
    table = run("inventory", "show", "demo")
    assert table.code == 0
    assert table.out.splitlines()[0].split() == ["TASK", "COUNT"]
    assert "total" in table.out

    def counts(by):
        data = json.loads(run("inventory", "show", "demo", "--by", by, "--json").out)
        return {row["value"]: row["count"] for row in data["counts"]}

    assert counts("method") == {"Swapper": 4, "original": 4}
    assert counts("label") == {"DEMO-FS_SWAP": 4, "DEMO-REAL": 4}
    assert counts("compression") == {"c23": 4, "c40": 4}


def test_show_without_an_inventory_hints_at_build(run, demo):
    result = run("inventory", "show", "demo")
    assert result.code == 2
    assert "hint: run: dfwb inventory build demo" in result.err


def test_build_errors_are_reported_with_hints(run, demo):
    unknown = run("inventory", "build", "dmeo")
    assert unknown.code == 2
    assert "did you mean 'demo'" in unknown.err
    typo = run("inventory", "build", "demo", "--compressions", "c32")
    assert typo.code == 2
    assert "did you mean 'c23'" in typo.err
    probe = run("inventory", "build", "demo", "--probe")
    assert probe.code == 5
    assert "[preprocess]" in probe.err
    jobs = run("inventory", "build", "demo", "--jobs", "0")
    assert jobs.code == 2


def test_show_rejects_an_unknown_column(run, demo):
    result = run("inventory", "show", "demo", "--by", "colour")
    assert result.code == 2
