"""``dfwb score``: the CLI over :func:`dfwb.score.harness.score` -- grammar, --suite, --frames,
--json, exit codes and hints."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tests.unit.score._toy import PROTOCOL, SUITE, toy_profile, write_toy_store

from dfwb.cli.main import main
from dfwb.core.records import read_scores

# `score_roots` (which requires `scoretoy_pack`) comes from tests/unit/score/conftest.py.


@pytest.fixture
def cli(capsys, monkeypatch, tmp_path, score_roots):
    """Run ``dfwb ARGS...`` in-process, with roots already pointed at the scoretoy fixtures."""
    monkeypatch.chdir(tmp_path)

    def _run(*args: str) -> SimpleNamespace:
        code = main(list(args))
        captured = capsys.readouterr()
        return SimpleNamespace(code=code, out=captured.out, err=captured.err)

    return _run


# ------------------------------------------------------------------------------------- the basics


def test_score_writes_a_c5_file_and_reports_it_as_json(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = cli(
        "score", "--detector", "fake:", "--protocol", PROTOCOL, "--split", "test", "--json"
    )

    assert result.code == 0
    data = json.loads(result.out)
    assert len(data["results"]) == 1
    row = data["results"][0]
    assert row["coverage"] == {"expected": 8, "ok": 8, "missing": 0, "error": 0}
    assert row["cached"] is False
    assert row["frames"] is None
    scored = read_scores(row["csv"])
    assert len(scored.rows) == 8


def test_score_plain_output_is_a_table(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = cli("score", "--detector", "fake:", "--protocol", PROTOCOL, "--split", "test")

    assert result.code == 0
    assert "ok=8/8" in result.out
    assert PROTOCOL in result.out


def test_score_where_filters_the_split(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--where",
        "identity=r00",
        "--json",
    )

    assert result.code == 0
    row = json.loads(result.out)["results"][0]
    assert row["coverage"] == {"expected": 1, "ok": 1, "missing": 0, "error": 0}
    assert row["where"] == {"identity": "r00"}


# --------------------------------------------------------------------------------------- caching


def test_second_run_with_identical_config_is_cached(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))
    args = ["score", "--detector", "fake:", "--protocol", PROTOCOL, "--split", "test", "--json"]

    first = json.loads(cli(*args).out)["results"][0]
    second = json.loads(cli(*args).out)["results"][0]

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["csv"] == first["csv"]


def test_force_recomputes_even_when_cached(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))
    args = ["score", "--detector", "fake:", "--protocol", PROTOCOL, "--split", "test"]

    cli(*args, "--json")
    forced = json.loads(cli(*args, "--force", "--json").out)["results"][0]

    assert forced["cached"] is False


# --------------------------------------------------------------------------------------- --suite


def test_suite_scores_every_entry_one_file_each(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = cli("score", "--detector", "fake:", "--suite", SUITE, "--json")

    assert result.code == 0
    data = json.loads(result.out)
    rows = data["results"]
    assert len(rows) == 2
    assert {r["group"] for r in rows} == {"real", "fake"}
    for row in rows:
        assert row["coverage"] == {"expected": 1, "ok": 1, "missing": 0, "error": 0}
    assert len({r["csv"] for r in rows}) == 2  # one C5 file per entry


def test_suite_combined_with_protocol_is_a_usage_error(cli, score_roots):
    result = cli(
        "score", "--detector", "fake:", "--suite", SUITE, "--protocol", PROTOCOL, "--split", "test"
    )
    assert result.code == 2
    assert "hint: " in result.err


def test_neither_protocol_nor_suite_is_a_usage_error(cli, score_roots):
    result = cli("score", "--detector", "fake:")
    assert result.code == 2
    assert "hint: " in result.err


def test_unknown_suite_name_exits_2_with_a_hint(cli, score_roots):
    result = cli("score", "--detector", "fake:", "--suite", "does-not-exist")
    assert result.code == 2
    assert "hint: " in result.err


# ----------------------------------------------------------------------------- exit codes and hints


def test_missing_detector_is_a_usage_error(cli, score_roots):
    result = cli("score", "--protocol", PROTOCOL, "--split", "test")
    assert result.code == 2
    assert "hint: " in result.err


def test_unknown_detector_scheme_exits_2_with_a_hint(cli, score_roots):
    result = cli("score", "--detector", "bogus:whatever", "--protocol", PROTOCOL, "--split", "test")
    assert result.code == 2
    assert "hint: " in result.err
    assert "fake" in result.err


def test_unknown_aggregate_exits_2_with_a_did_you_mean(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))
    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--aggregate",
        "mean-probs",
    )
    assert result.code == 2
    assert "hint: " in result.err
    assert "mean-prob" in result.err


def test_input_mismatch_exits_4(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-full", backend="center", scale=1.0))

    result = cli("score", "--detector", "fake:", "--protocol", PROTOCOL, "--split", "test")

    assert result.code == 4
    assert "hint: " in result.err


def test_allow_input_mismatch_recovers(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-full", backend="center", scale=1.0))

    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--allow-input-mismatch",
        "--json",
    )

    assert result.code == 0


def test_no_processed_store_exits_2_with_a_hint(cli, score_roots):
    result = cli("score", "--detector", "fake:", "--protocol", PROTOCOL, "--split", "test")
    assert result.code == 2
    assert "hint: " in result.err


# --------------------------------------------------------------------------------------- --frames


def test_frames_writes_a_parquet_file(cli, score_roots):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    write_toy_store(score_roots, toy_profile("toy-face"))

    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--frames",
        "--json",
    )

    assert result.code == 0
    row = json.loads(result.out)["results"][0]
    assert row["frames"] is not None
    table = pq.read_table(row["frames"])
    assert table.num_rows > 0


def test_frames_is_skipped_without_pyarrow(cli, score_roots, monkeypatch):
    import builtins

    write_toy_store(score_roots, toy_profile("toy-face"))
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--frames",
        "--json",
    )

    assert result.code == 0
    row = json.loads(result.out)["results"][0]
    assert row["frames"] is None
    assert "pyarrow" in result.err
