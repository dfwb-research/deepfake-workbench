"""``dfwb score``: the CLI over :func:`dfwb.score.harness.score` -- grammar, --suite, --frames,
--json, exit codes and hints."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from tests._dfwb_cli import run_dfwb
from tests.unit.score._toy import PROTOCOL, SUITE, SUITE_ANY_OF, toy_profile, write_toy_store

from dfwb.core.records import read_scores

# `score_roots` (which requires `scoretoy_pack`) comes from tests/unit/score/conftest.py.


@pytest.fixture
def cli(capsys, monkeypatch, tmp_path, score_roots):
    """Run ``dfwb ARGS...`` in-process, with roots already pointed at the scoretoy fixtures."""
    monkeypatch.chdir(tmp_path)

    def _run(*args: str):
        return run_dfwb(capsys, *args)

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


def test_input_mismatch_names_a_shipped_profile_that_would_serve(cli, score_roots):
    # The CLI (never dfwb.score itself) knows about the face pipeline's shipped profiles, and
    # passes them in: fake:'s default spec (a face crop at scale 1.3) is exactly what the shipped
    # face-256-1.3x-64f profile provides.
    write_toy_store(score_roots, toy_profile("toy-full", backend="center", scale=1.0))

    result = cli("score", "--detector", "fake:", "--protocol", PROTOCOL, "--split", "test")

    assert result.code == 4
    assert "face-256-1.3x-64f" in result.err


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


def test_unknown_device_exits_2_with_a_hint(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))
    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--device",
        "bogus",
    )
    assert result.code == 2
    assert "hint: " in result.err


def test_device_gpu_alias_without_cuda_exits_2_with_a_hint(cli, score_roots):
    """``gpu`` is the same alias ``dfwb train --device`` accepts; without CUDA available it must
    fail cleanly (exit 2, a hint to use --device cpu), not with a raw torch error."""
    write_toy_store(score_roots, toy_profile("toy-face"))
    result = cli(
        "score", "--detector", "fake:", "--protocol", PROTOCOL, "--split", "test", "--device", "gpu"
    )
    assert result.code == 2
    assert "hint: " in result.err
    assert "--device cpu" in result.err


def test_device_cuda_index_without_cuda_exits_2_with_a_hint(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))
    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--device",
        "cuda:0",
    )
    assert result.code == 2
    assert "hint: " in result.err
    assert "--device cpu" in result.err


def test_clips_per_video_zero_is_a_usage_error(cli, score_roots):
    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--clips-per-video",
        "0",
    )
    assert result.code == 2
    assert "hint: " in result.err


def test_clips_per_video_negative_is_a_usage_error(cli, score_roots):
    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--clips-per-video=-1",
    )
    assert result.code == 2
    assert "hint: " in result.err


def test_batch_size_zero_is_a_usage_error(cli, score_roots):
    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--batch-size",
        "0",
    )
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


def test_frames_without_pyarrow_exits_5_before_scoring_anything(cli, score_roots, monkeypatch):
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

    assert result.code == 5
    assert "hint: " in result.err
    assert "deepfake-workbench[eval]" in result.err
    assert result.out == ""  # nothing was ever scored or printed


# ----------------------------------------------------------------------- score -> eval round trips


def test_an_any_of_where_score_file_goes_through_eval(cli, score_roots):
    """Repeating ``--where`` for one key (any of those values) stores a list in the meta; ``dfwb
    eval`` must read and evaluate that file like any other."""
    write_toy_store(score_roots, toy_profile("toy-face"))
    scored = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--where",
        "identity=r00",
        "--where",
        "identity=f00",
        "--json",
    )
    assert scored.code == 0
    (row,) = json.loads(scored.out)["results"]
    assert read_scores(row["csv"]).meta.protocol.where == {"identity": ["f00", "r00"]}

    result = cli("eval", row["csv"], "--metrics", "auc", "--bootstrap", "0", "--json")

    assert result.code == 0, result.err
    data = json.loads(result.out)
    assert data["tables"]["files"][0]["n"] == 2


def test_eval_suite_matches_list_filters_written_in_another_order(cli, score_roots):
    """The suite lists ``identity: [r00, f00]`` and ``method: [swap, original]``; the files were
    scored from ``--where`` values in yet another order, and a score file's meta stores them
    sorted. Every entry must still find its file."""
    write_toy_store(score_roots, toy_profile("toy-face"))
    pair = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--where",
        "identity=f00",
        "--where",
        "identity=r00",
        "--json",
    )
    in_domain = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--where",
        "method=original",
        "--where",
        "method=swap",
        "--json",
    )
    files = [json.loads(r.out)["results"][0]["csv"] for r in (pair, in_domain)]

    result = cli(
        "eval", *files, "--suite", SUITE_ANY_OF, "--metrics", "auc", "--bootstrap", "0", "--json"
    )

    assert result.code == 0, result.err
    suite_rows = {row["group"]: row for row in json.loads(result.out)["tables"]["suite"]}
    assert suite_rows["pair"]["n_entries"] == 1
    assert suite_rows["in-domain"]["n_entries"] == 1


def test_score_suite_then_eval_suite_round_trips_with_list_filters(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))
    scored = cli("score", "--detector", "fake:", "--suite", SUITE_ANY_OF, "--json")
    assert scored.code == 0
    files = [row["csv"] for row in json.loads(scored.out)["results"]]

    result = cli(
        "eval", *files, "--suite", SUITE_ANY_OF, "--metrics", "auc", "--bootstrap", "0", "--json"
    )

    assert result.code == 0, result.err
    suite_rows = {row["group"]: row for row in json.loads(result.out)["tables"]["suite"]}
    assert {group: row["n_entries"] for group, row in suite_rows.items()} == {
        "pair": 1,
        "in-domain": 1,
    }


# ------------------------------------------------------------------------ the command in the meta


def test_the_meta_records_the_command_without_absolute_paths(
    cli, score_roots, tmp_path, monkeypatch
):
    write_toy_store(score_roots, toy_profile("toy-face"))
    out = tmp_path / "elsewhere" / "scores"
    argv = [
        "/opt/venv/bin/dfwb",
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--where",
        "identity=r00",
        "--out",
        str(out),
        "--json",
    ]
    monkeypatch.setattr("sys.argv", argv)

    result = cli(*argv[1:])

    assert result.code == 0, result.err
    (row,) = json.loads(result.out)["results"]
    command = read_scores(row["csv"]).meta.command
    assert command.startswith("dfwb score --detector fake: --protocol ")
    assert "--where identity=r00" in command
    assert "'<abs>/scores'" in command
    assert str(tmp_path) not in command


# ------------------------------------------------------------------------------ --min-coverage


def test_score_exits_3_when_coverage_is_below_min_coverage(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"), skip=["FAKE/f03", "REAL/r03"])

    result = cli(
        "score", "--detector", "fake:", "--protocol", PROTOCOL, "--split", "test", "--json"
    )

    assert result.code == 3
    (row,) = json.loads(result.out)["results"]  # the results are still written and reported
    assert row["coverage"] == {"expected": 8, "ok": 6, "missing": 2, "error": 0}
    assert "0.7500" in result.err
    assert "hint: " in result.err


def test_score_min_coverage_can_be_lowered(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"), skip=["FAKE/f03", "REAL/r03"])

    result = cli(
        "score",
        "--detector",
        "fake:",
        "--protocol",
        PROTOCOL,
        "--split",
        "test",
        "--min-coverage",
        "0.75",
    )

    assert result.code == 0
    assert "ok=6/8" in result.out


def test_score_exits_3_when_every_row_errored(cli, score_roots):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = cli("score", "--detector", "fake:bad=nan", "--protocol", PROTOCOL, "--split", "test")

    assert result.code == 3
    assert "ok=0/8" in result.out
    assert "hint: " in result.err


def test_score_an_empty_split_is_full_coverage(cli, score_roots):
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
        "identity=nobody",
    )

    assert result.code == 0


# ----------------------------------------------------------------------- --suite resolves once


def test_suite_resolves_the_detector_once_for_every_entry(cli, score_roots):
    import uuid

    from tests.unit.score import _toy

    write_toy_store(score_roots, toy_profile("toy-face"))
    spy_id = str(uuid.uuid4())

    result = cli("score", "--detector", f"fake:spy={spy_id}", "--suite", SUITE, "--json")

    assert result.code == 0
    assert len(json.loads(result.out)["results"]) == 2
    assert _toy.SPY_LOADS[spy_id] == 1


def test_an_invalid_device_is_refused_before_the_detector_is_resolved(cli, score_roots):
    import uuid

    from tests.unit.score import _toy

    write_toy_store(score_roots, toy_profile("toy-face"))
    spy_id = str(uuid.uuid4())

    result = cli("score", "--detector", f"fake:spy={spy_id}", "--suite", SUITE, "--device", "tpu")

    assert result.code == 2
    assert spy_id not in _toy.SPY_LOADS
