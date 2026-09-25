"""``dfwb.score.harness.score``: run a detector over a protocol split and write a C5 file."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tests.unit.score._toy import PROTOCOL, toy_profile, write_toy_store

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records.scores import read_scores
from dfwb.score.harness import score


def _score(tmp_path, detector_uri="fake:", **kwargs):
    kwargs.setdefault("protocol", PROTOCOL)
    kwargs.setdefault("split", "test")
    kwargs.setdefault("out", tmp_path / "out")
    return score(detector_uri, **kwargs)


# ------------------------------------------------------------------------------------- the basics


def test_fake_detector_on_a_toy_store_gives_a_valid_c5_file(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path)

    scored = read_scores(result.csv_path)  # checks rows and meta against C5
    assert scored.meta is not None
    assert len(scored.rows) == 8
    assert {row.status for row in scored.rows} == {"ok"}
    for row in scored.rows:
        assert 0.0 <= row.score <= 1.0
        assert row.n_clips == 4
        assert row.n_frames == 4
    assert result.coverage == {"expected": 8, "ok": 8, "missing": 0, "error": 0}
    assert result.cached is False


def test_every_video_of_the_split_gets_exactly_one_row(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    result = _score(tmp_path)
    scored = read_scores(result.csv_path)
    keys = {(row.dataset, row.key) for row in scored.rows}
    expected = {("scoretoy", f"REAL/r{i:02d}") for i in range(4)} | {
        ("scoretoy", f"FAKE/f{i:02d}") for i in range(4)
    }
    assert keys == expected


def test_precision_runs_prediction_under_autocast(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, precision="bf16")

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"ok"}


# --------------------------------------------------------------------------------- missing videos


def test_missing_processed_videos_become_missing_rows(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"), skip=["FAKE/f03", "REAL/r03"])

    result = _score(tmp_path)

    scored = read_scores(result.csv_path)
    by_key = {row.key: row for row in scored.rows}
    assert by_key["FAKE/f03"].status == "missing"
    assert by_key["FAKE/f03"].score is None
    assert by_key["FAKE/f03"].label == 1
    assert by_key["REAL/r03"].status == "missing"
    assert by_key["REAL/r03"].label == 0
    assert result.coverage == {"expected": 8, "ok": 6, "missing": 2, "error": 0}
    assert scored.meta.coverage.model_dump() == result.coverage


# ----------------------------------------------------------------------------------- error videos


def test_detector_error_rows(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    # batch_size == clips_per_video: each video is scored in its own batch, so only FAKE/f00
    # fails; every other video still gets scored.
    result = score(
        "fake:raise=FAKE/f00",
        protocol=PROTOCOL,
        split="test",
        out=tmp_path / "out",
        batch_size=4,
    )

    scored = read_scores(result.csv_path)
    by_key = {row.key: row for row in scored.rows}
    assert by_key["FAKE/f00"].status == "error"
    assert by_key["FAKE/f00"].score is None
    assert by_key["FAKE/f00"].label == 1
    ok_statuses = {key: row.status for key, row in by_key.items() if key != "FAKE/f00"}
    assert set(ok_statuses.values()) == {"ok"}
    assert result.coverage == {"expected": 8, "ok": 7, "missing": 0, "error": 1}


def test_every_video_erroring_leaves_nothing_to_aggregate(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    all_keys = ",".join([f"REAL/r{i:02d}" for i in range(4)] + [f"FAKE/f{i:02d}" for i in range(4)])

    result = _score(tmp_path, f"fake:raise={all_keys}")

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"error"}
    assert result.coverage == {"expected": 8, "ok": 0, "missing": 0, "error": 8}


# --------------------------------------------------------------------------------- profile choice


def test_profile_choice_uses_the_preferred_profile_when_present_locally(score_roots, tmp_path):
    profile_a = toy_profile("toy-a", scale=1.3)
    profile_b = toy_profile("toy-b", scale=1.6)
    write_toy_store(score_roots, profile_a)
    write_toy_store(score_roots, profile_b)

    result = _score(tmp_path, "fake:preferred=toy-b")

    meta = read_scores(result.csv_path).meta
    assert meta.processing_profile.id == profile_b.profile_id()


def test_profile_choice_uses_the_only_compatible_profile(score_roots, tmp_path):
    profile_a = toy_profile("toy-a", scale=1.3)
    profile_b = toy_profile("toy-b", scale=1.6)
    write_toy_store(score_roots, profile_a)
    write_toy_store(score_roots, profile_b)

    # scale=1.5 is compatible only with profile_b (scale 1.6 >= 1.5; profile_a's 1.3 is not).
    result = _score(tmp_path, "fake:scale=1.5")

    meta = read_scores(result.csv_path).meta
    assert meta.processing_profile.id == profile_b.profile_id()


def test_profile_choice_is_ambiguous_without_a_preference(score_roots, tmp_path):
    profile_a = toy_profile("toy-a", scale=1.3)
    profile_b = toy_profile("toy-b", scale=1.6)
    write_toy_store(score_roots, profile_a)
    write_toy_store(score_roots, profile_b)

    with pytest.raises(ConfigError) as info:
        _score(tmp_path, "fake:scale=1.3")
    assert profile_a.profile_id() in info.value.message
    assert profile_b.profile_id() in info.value.message


def test_explicit_profile_overrides_the_automatic_choice(score_roots, tmp_path):
    profile_a = toy_profile("toy-a", scale=1.3)
    profile_b = toy_profile("toy-b", scale=1.6)
    write_toy_store(score_roots, profile_a)
    write_toy_store(score_roots, profile_b)

    result = _score(tmp_path, "fake:scale=1.3", profile=profile_a.profile_id())

    meta = read_scores(result.csv_path).meta
    assert meta.processing_profile.id == profile_a.profile_id()


def test_unknown_explicit_profile_is_refused(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-a"))

    with pytest.raises(ConfigError) as info:
        _score(tmp_path, profile="does-not-exist")
    assert "does-not-exist" in info.value.message


def test_no_local_store_at_all_is_refused(score_roots, tmp_path):
    with pytest.raises(ConfigError) as info:
        _score(tmp_path)
    assert "scoretoy" in info.value.message


def test_preferred_profile_not_installed_falls_back_to_the_only_compatible_one(
    score_roots, tmp_path
):
    profile_a = toy_profile("toy-a", scale=1.3)
    write_toy_store(score_roots, profile_a)

    # spec.preferred_profile names a profile that was never processed locally; the only
    # compatible one installed is still chosen automatically.
    result = _score(tmp_path, "fake:preferred=does-not-exist")

    meta = read_scores(result.csv_path).meta
    assert meta.processing_profile.id == profile_a.profile_id()


def test_a_label_the_mapping_excludes_gets_no_row_at_all(score_roots, tmp_path):
    # FAKE/f00 is both unprocessed *and* excluded by this mapping: it must get no row (neither
    # missing nor ok), not a missing row with a nonsensical label. The rest of FAKE/* is excluded
    # by label alone (still processed); only REAL/* -- unaffected by the override -- gets rows.
    write_toy_store(score_roots, toy_profile("toy-face"), skip=["FAKE/f00"])

    result = _score(tmp_path, labels="binary-exclude-fake")

    scored = read_scores(result.csv_path)
    assert {row.key for row in scored.rows} == {f"REAL/r{i:02d}" for i in range(4)}
    assert {row.status for row in scored.rows} == {"ok"}
    assert {row.label for row in scored.rows} == {0}


# -------------------------------------------------------------------------------- input mismatch


def test_input_mismatch_is_refused_by_default(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-full", backend="center", scale=1.0))

    with pytest.raises(ContractError):
        _score(tmp_path)  # default fake: crop="face", but only a full-frame store exists


def test_allow_input_mismatch_records_it_in_meta(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-full", backend="center", scale=1.0))

    result = _score(tmp_path, allow_input_mismatch=True)

    scored = read_scores(result.csv_path)
    assert scored.meta.input_adaptation.mismatch_override is True
    assert {row.status for row in scored.rows} == {"ok"}


# ---------------------------------------------------------------------------------------- caching


def test_rerunning_with_an_identical_config_is_cached(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    first = _score(tmp_path)
    assert first.cached is False
    second = _score(tmp_path)
    assert second.cached is True
    assert second.csv_path == first.csv_path


def test_force_recomputes_even_when_a_cached_file_exists(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    first = _score(tmp_path)
    second = _score(tmp_path, force=True)
    assert second.cached is False
    assert second.csv_path == first.csv_path


def test_an_unreadable_file_at_the_cache_path_is_recomputed_rather_than_trusted(
    score_roots, tmp_path
):
    write_toy_store(score_roots, toy_profile("toy-face"))

    first = _score(tmp_path)
    first.csv_path.write_text("not a score file", encoding="utf-8")

    second = _score(tmp_path)

    assert second.cached is False
    scored = read_scores(second.csv_path)
    assert {row.status for row in scored.rows} == {"ok"}


def test_a_csv_with_no_meta_file_at_the_cache_path_is_recomputed(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    first = _score(tmp_path)
    first.meta_path.unlink()

    second = _score(tmp_path)

    assert second.cached is False
    assert second.meta_path.is_file()
