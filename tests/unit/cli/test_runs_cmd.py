"""``dfwb runs list|show``: read run directories back, torch-free, and spot runs of the same
config by their fingerprint."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tests.unit.cli._runs import FP_A, FP_B, fake_run


def _runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "runs"
    first = fake_run(root, "toy", "20260101-000000", 0, FP_A)
    second = fake_run(root, "toy", "20260102-000000", 1, FP_A, completed=False, latest=True)
    other = fake_run(root, "vit", "20260103-000000", 0, FP_B, latest=True)
    return first, second, other


def test_runs_list_json(run, tmp_path):
    first, second, other = _runs(tmp_path)
    result = run("runs", "list", "--json")
    assert result.code == 0, result.err
    rows = json.loads(result.out)
    assert [(r["name"], r["run"]) for r in rows] == [
        ("toy", first.name),
        ("toy", second.name),
        ("vit", other.name),
    ]
    assert rows[0] == {
        "name": "toy",
        "run": first.name,
        "path": str(first),
        "seed": 0,
        "created": "2026-01-01T00:00:00Z",
        "status": "completed",
        "epochs": 2,
        "fingerprint": FP_A,
        "monitor": {
            "key": "val/video_auc",
            "mode": "max",
            "fallback": False,
            "best": 0.75,
            "best_epoch": 1,
        },
        "latest": False,
    }
    assert (rows[1]["status"], rows[1]["epochs"], rows[1]["monitor"]) == ("incomplete", 1, None)
    assert [r["latest"] for r in rows] == [False, True, True]


def test_runs_list_table(run, tmp_path):
    first, second, _ = _runs(tmp_path)
    lines = run("runs", "list").out.splitlines()
    assert lines[0].split() == ["NAME", "RUN", "SEED", "STATUS", "EPOCHS", "BEST", "FINGERPRINT"]
    assert lines[1].split() == [
        "toy",
        first.name,
        "0",
        "completed",
        "2",
        "val/video_auc=0.7500",
        FP_A[:12],
    ]
    assert lines[2].split()[:4] == ["toy", f"{second.name}*", "1", "incomplete"]
    assert run("runs", "list").code == 0


def test_runs_list_root_lists_runs_written_under_another_output_root(run, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    run_dir = fake_run(elsewhere, "custom", "20260104-000000", 3, FP_B, latest=True)
    assert json.loads(run("runs", "list", "--json").out) == []  # not under the runs root
    rows = json.loads(run("runs", "list", "--root", str(elsewhere), "--json").out)
    assert [(row["name"], row["path"]) for row in rows] == [("custom", str(run_dir))]
    table = run("runs", "list", "--root", "elsewhere")
    assert table.code == 0
    assert table.out.splitlines()[1].split()[:2] == ["custom", f"{run_dir.name}*"]


def test_runs_list_with_no_runs(run, tmp_path):
    result = run("runs", "list")
    assert result.code == 0
    assert result.out.startswith("no runs under ")
    assert json.loads(run("runs", "list", "--json").out) == []


def test_runs_list_skips_what_is_not_a_run(run, tmp_path):
    _runs(tmp_path)
    (tmp_path / "runs" / "notes.txt").write_text("x")
    (tmp_path / "runs" / "toy" / "scratch").mkdir()
    rows = json.loads(run("runs", "list", "--json").out)
    assert len(rows) == 3


def test_runs_show_json_names_runs_of_the_same_config(run, tmp_path):
    first, second, _ = _runs(tmp_path)
    result = run("runs", "show", str(first), "--json")
    assert result.code == 0, result.err
    shown = json.loads(result.out)
    assert shown["fingerprint"] == FP_A
    assert shown["same_config"] == [{"path": str(second), "seed": 1, "status": "incomplete"}]
    assert shown["env"]["git"]["commit"] == "0123456789abcdef"
    assert shown["val"] == {"val/loss": 0.5, "val/video_auc": 0.75}
    assert shown["data"] == [
        {
            "role": "train",
            "name": "toyfake-official",
            "protocol": "toyfake/official",
            "split": "train",
            "in_split": 10,
            "included": 9,
        }
    ]


def test_runs_show_finds_runs_by_name_latest_or_path(run, tmp_path, monkeypatch):
    first, second, other = _runs(tmp_path)
    for ref, expected in [
        ("toy", second),
        ("toy/latest", second),
        (f"toy/{first.name}", first),
        (str(other), other),
        (os.path.relpath(first, tmp_path), first),
    ]:
        result = run("runs", "show", ref, "--json")
        assert result.code == 0, (ref, result.err)
        assert json.loads(result.out)["path"] == str(expected), ref


def test_runs_show_unknown_run_suggests_names(run, tmp_path):
    _runs(tmp_path)
    result = run("runs", "show", "tyo")
    assert result.code == 2
    assert "did you mean 'toy'" in result.err


def test_runs_show_text(run, tmp_path):
    first, second, _ = _runs(tmp_path)
    result = run("runs", "show", str(first))
    assert result.code == 0, result.err
    text = result.out
    assert f"fingerprint:  {FP_A}" in text
    assert f"same config:  {second} (seed 1, incomplete)" in text
    assert "monitor:      val/video_auc (max): best 0.7500 after epoch 1 of 2" in text
    assert "train toyfake-official: 9 of 10 videos" in text
    assert "git 0123456 (dirty)" in text
    alone = run("runs", "show", str(second)).out
    assert "status:       incomplete (1 epoch(s) done)" in alone


def test_runs_show_a_directory_that_is_not_a_run(run, tmp_path):
    (tmp_path / "somewhere").mkdir()
    result = run("runs", "show", str(tmp_path / "somewhere"))
    assert result.code == 2
    assert "not a run directory" in result.err
