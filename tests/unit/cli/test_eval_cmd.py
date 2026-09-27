from __future__ import annotations

import json

import pytest
from tests.unit.eval.conftest import make_meta, make_rows

from dfwb.core.records import write_scores

# `import_pack` (used below) is a fixture from tests/unit/conftest.py, found by name.


def _write(tmp_path, name, rows, meta):
    path, _ = write_scores(tmp_path / name, rows, meta)
    return path


@pytest.fixture
def score_file(tmp_path):
    rows = make_rows(20, 20)
    meta = make_meta(coverage={"expected": 40, "ok": 40, "missing": 0, "error": 0})
    return _write(tmp_path, "a.scores.csv", rows, meta)


# --------------------------------------------------------------------------- bare `dfwb eval`


def test_eval_bare_runs_and_prints_a_table(run, score_file):
    result = run("eval", str(score_file), "--metrics", "auc", "--bootstrap", "20")
    assert result.code == 0
    assert "auc" in result.out.lower()


def test_eval_bare_json(run, score_file):
    result = run("eval", str(score_file), "--metrics", "auc", "--bootstrap", "20", "--json")
    assert result.code == 0
    data = json.loads(result.out)
    assert data["tables"]["files"][0]["metrics"]["auc"]["value"] > 0.5
    assert "input_files" in data


def test_eval_bare_low_coverage_exits_3(run, tmp_path):
    from dfwb.core.records import ScoreRow

    rows = [*make_rows(10, 10)]
    rows.append(ScoreRow("d", "missing0", None, 0, None, "missing"))
    meta = make_meta(coverage={"expected": 21, "ok": 20, "missing": 1, "error": 0})
    path = _write(tmp_path, "low.scores.csv", rows, meta)
    result = run("eval", str(path), "--metrics", "auc", "--bootstrap", "10", "--json")
    assert result.code == 3
    data = json.loads(result.out)
    assert data["exit_code"] == 3


def test_eval_bare_directory_expands_score_files(run, tmp_path):
    rows = make_rows(10, 10)
    meta = make_meta(coverage={"expected": 20, "ok": 20, "missing": 0, "error": 0})
    _write(tmp_path, "a.scores.csv", rows, meta)
    result = run("eval", str(tmp_path), "--metrics", "auc", "--bootstrap", "10", "--json")
    assert result.code == 0


def test_eval_bare_unknown_files_is_a_usage_error(run, tmp_path):
    result = run("eval", str(tmp_path / "nope.scores.csv"))
    assert result.code == 2


def test_eval_bare_writes_metrics_json_report_and_plots(run, score_file, tmp_path):
    out_dir = tmp_path / "out"
    result = run(
        "eval", str(score_file), "--metrics", "auc", "--bootstrap", "10", "--out", str(out_dir)
    )
    assert result.code == 0
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.is_file()
    payload = json.loads(metrics_path.read_text())
    assert "input_files" in payload
    assert (out_dir / "report.md").is_file()
    pytest.importorskip("matplotlib")
    plots_dir = out_dir / "plots"
    assert any(plots_dir.glob("*-roc.png"))
    assert any(plots_dir.glob("*-roc.pdf"))


def test_eval_bare_format_csv(run, score_file):
    result = run(
        "eval", str(score_file), "--metrics", "auc", "--bootstrap", "10", "--format", "csv"
    )
    assert result.code == 0
    assert "file" in result.out


def test_eval_bare_format_latex_uses_booktabs(run, score_file):
    result = run(
        "eval", str(score_file), "--metrics", "auc", "--bootstrap", "10", "--format", "latex"
    )
    assert result.code == 0
    assert "\\toprule" in result.out
    assert "\\pm" in result.out


def test_eval_help_lists_subcommands_not_run(run):
    result = run("eval", "--help")
    assert result.code == 0
    assert "compare" in result.out
    assert "calibrate" in result.out
    assert "import" in result.out
    assert " run " not in result.out.lower().replace("\n", " ")


# --------------------------------------------------------------------------- compare


def test_eval_compare_json(run, tmp_path):
    path_a = _write(
        tmp_path,
        "a.scores.csv",
        make_rows(20, 20, fake_score=0.7),
        make_meta(coverage={"expected": 40, "ok": 40, "missing": 0, "error": 0}),
    )
    path_b = _write(
        tmp_path,
        "b.scores.csv",
        make_rows(20, 20, fake_score=0.9),
        make_meta(coverage={"expected": 40, "ok": 40, "missing": 0, "error": 0}),
    )
    result = run(
        "eval",
        "compare",
        str(path_a),
        str(path_b),
        "--metrics",
        "auc",
        "--bootstrap",
        "20",
        "--json",
    )
    assert result.code == 0
    data = json.loads(result.out)
    assert data["comparisons"][0]["n"] == 40


def test_eval_compare_needs_two_files(run, score_file):
    result = run("eval", "compare", str(score_file))
    assert result.code == 2


# --------------------------------------------------------------------------- calibrate


def test_eval_calibrate_writes_a_new_c5_file(run, tmp_path):
    fit_path = _write(
        tmp_path,
        "val.scores.csv",
        make_rows(50, 50),
        make_meta(coverage={"expected": 100, "ok": 100, "missing": 0, "error": 0}),
    )
    apply_path = _write(
        tmp_path,
        "test.scores.csv",
        make_rows(30, 30),
        make_meta(coverage={"expected": 60, "ok": 60, "missing": 0, "error": 0}),
    )
    result = run(
        "eval",
        "calibrate",
        "--fit",
        str(fit_path),
        "--apply",
        str(apply_path),
        "--method",
        "temperature",
        "--json",
    )
    assert result.code == 0
    data = json.loads(result.out)
    from pathlib import Path

    out_csv = Path(data["csv"])
    assert out_csv.is_file()

    from dfwb.core.records import read_scores

    reloaded = read_scores(out_csv)
    assert reloaded.meta.calibration.method == "temperature"


def test_eval_calibrate_unknown_method_is_a_usage_error(run, tmp_path):
    fit_path = _write(
        tmp_path,
        "val.scores.csv",
        make_rows(20, 20),
        make_meta(coverage={"expected": 40, "ok": 40, "missing": 0, "error": 0}),
    )
    apply_path = _write(
        tmp_path,
        "test.scores.csv",
        make_rows(10, 10),
        make_meta(coverage={"expected": 20, "ok": 20, "missing": 0, "error": 0}),
    )
    result = run(
        "eval",
        "calibrate",
        "--fit",
        str(fit_path),
        "--apply",
        str(apply_path),
        "--method",
        "nope",
    )
    assert result.code == 2


# --------------------------------------------------------------------------- import


def test_eval_import_writes_a_new_c5_file(run, import_pack, tmp_path):
    csv_path = tmp_path / "videos.csv"
    csv_path.write_text(
        "video_id,task,label,prediction\n"
        "00001,CDF,0,0.1\n00002,CDF,0,0.2\n00003,CDF,1,0.9\n00004,CDF,1,0.8\n00005,CDF,1,0.7\n"
    )
    out_dir = tmp_path / "out"
    result = run(
        "eval",
        "import",
        str(csv_path),
        "--protocol",
        "cdf/official",
        "--split",
        "test",
        "--map",
        "key={task}/{video_id},score=prediction,label=label",
        "--out",
        str(out_dir),
        "--json",
    )
    assert result.code == 0
    data = json.loads(result.out)
    assert data["coverage"]["ok"] == 5
    from dfwb.core.records import read_scores

    reloaded = read_scores(data["csv"])
    assert len(reloaded.rows) == 5


def test_eval_import_unknown_key_exits_4_with_suggestion(run, import_pack, tmp_path):
    csv_path = tmp_path / "videos.csv"
    csv_path.write_text("key,prediction\nCDF/00001.mp4,0.1\nCDF/00002.mp4,0.2\nCDF/00003.mp4,0.9\n")
    out_dir = tmp_path / "out"
    result = run(
        "eval",
        "import",
        str(csv_path),
        "--protocol",
        "cdf/official",
        "--split",
        "test",
        "--map",
        "key=key,score=prediction",
        "--out",
        str(out_dir),
    )
    assert result.code == 4
    assert "hint: " in result.err
    assert "extension" in result.err


def test_eval_import_label_mismatch_exits_4(run, import_pack, tmp_path):
    csv_path = tmp_path / "videos.csv"
    csv_path.write_text(
        "video_id,task,label,prediction\n"
        "00001,CDF,1,0.1\n00002,CDF,0,0.2\n00003,CDF,1,0.9\n00004,CDF,1,0.8\n00005,CDF,1,0.7\n"
    )
    out_dir = tmp_path / "out"
    result = run(
        "eval",
        "import",
        str(csv_path),
        "--protocol",
        "cdf/official",
        "--split",
        "test",
        "--map",
        "key={task}/{video_id},score=prediction,label=label",
        "--out",
        str(out_dir),
    )
    assert result.code == 4
