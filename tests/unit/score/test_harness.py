"""``dfwb.score.harness.score``: run a detector over a protocol split and write a C5 file."""

from __future__ import annotations

import logging
import uuid

import pytest

pytest.importorskip("torch")

from tests.unit.score import _toy
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


def test_predict_runs_under_torch_inference_mode(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    spy_id = str(uuid.uuid4())

    _score(tmp_path, f"fake:spy={spy_id}")

    seen = _toy.SPY_INFERENCE_MODE[spy_id]
    assert seen  # at least one batch was scored
    assert all(seen)


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


def test_a_failing_batch_holding_several_videos_marks_exactly_those(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    # batch_size=8 (2 videos' worth of clips at clips_per_video=4): VideoGrouped packs greedily
    # in index order (FAKE/* sorts before REAL/*), so FAKE/f00 and FAKE/f01 share the first
    # batch; raising for both fails exactly that batch, and no other.
    result = _score(tmp_path, "fake:raise=FAKE/f00,FAKE/f01", batch_size=8)

    scored = read_scores(result.csv_path)
    by_key = {row.key: row.status for row in scored.rows}
    assert by_key["FAKE/f00"] == "error"
    assert by_key["FAKE/f01"] == "error"
    others = {key: status for key, status in by_key.items() if key not in ("FAKE/f00", "FAKE/f01")}
    assert set(others.values()) == {"ok"}
    assert result.coverage == {"expected": 8, "ok": 6, "missing": 0, "error": 2}


# ------------------------------------------------------------------------------ output validation


def test_a_nan_output_marks_its_videos_error_and_scoring_continues(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:bad=nan", batch_size=4)

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"error"}
    assert result.coverage == {"expected": 8, "ok": 0, "missing": 0, "error": 8}


def test_a_wrong_length_output_marks_its_videos_error(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:bad=length", batch_size=4)

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"error"}


def test_a_wrong_shape_output_marks_its_videos_error(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:bad=shape", batch_size=4)

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"error"}


def test_a_non_tensor_output_marks_its_videos_error(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:bad=nontensor", batch_size=4)

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"error"}


def test_an_out_of_range_output_marks_its_videos_error(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:bad=range", batch_size=4)

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"error"}


def test_a_b1_shaped_score_is_squeezed_and_accepted(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:bad=squeeze", batch_size=4)

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"ok"}
    for row in scored.rows:
        assert 0.0 <= row.score <= 1.0


# ------------------------------------------------------------------------------ aggregation modes


def test_mean_prob_aggregation_matches_a_hand_computed_fixture(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:scripted=0.1,0.3,0.9", clips_per_video=3, aggregate="mean-prob")

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"ok"}
    for row in scored.rows:
        assert row.score == pytest.approx((0.1 + 0.3 + 0.9) / 3, abs=1e-4)


def test_mean_logit_aggregation_matches_a_hand_computed_fixture(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(
        tmp_path, "fake:scripted=0.1,0.3,0.9", clips_per_video=3, aggregate="mean-logit"
    )

    scored = read_scores(result.csv_path)
    # Hand-computed: logit(p) = ln(p / (1 - p)) for each of 0.1, 0.3, 0.9, averaged, then
    # mapped back through the sigmoid -- distinct from the plain mean of 0.1, 0.3, 0.9 above.
    for row in scored.rows:
        assert row.score == pytest.approx(0.42985748800076856, abs=1e-4)


def test_max_aggregation_matches_a_hand_computed_fixture(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:scripted=0.1,0.3,0.9", clips_per_video=3, aggregate="max")

    scored = read_scores(result.csv_path)
    for row in scored.rows:
        assert row.score == pytest.approx(0.9, abs=1e-4)


def test_median_aggregation_matches_a_hand_computed_fixture(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:scripted=0.1,0.3,0.9", clips_per_video=3, aggregate="median")

    scored = read_scores(result.csv_path)
    for row in scored.rows:
        assert row.score == pytest.approx(0.3, abs=1e-4)


def test_n_clips_and_n_frames_reflect_the_requested_clip_count(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path, "fake:frames=2", clips_per_video=3)

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"ok"}
    for row in scored.rows:
        assert row.n_clips == 3
        assert row.n_frames == 6  # 3 clips * 2 frames each


# ------------------------------------------------------------------------- pre-flight validation


def test_unknown_precision_is_refused_with_a_did_you_mean(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    with pytest.raises(ConfigError) as info:
        _score(tmp_path, precision="fp61")
    assert "fp61" in info.value.message
    assert "fp16" in info.value.hint


def test_unknown_aggregate_is_refused_with_a_did_you_mean(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    with pytest.raises(ConfigError) as info:
        _score(tmp_path, aggregate="mean-probs")
    assert "mean-probs" in info.value.message
    assert "mean-prob" in info.value.hint


def test_aggregate_only_accepts_the_four_documented_modes(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    with pytest.raises(ConfigError):
        # dfwb.eval.aggregate itself accepts "vote"; score restricts to the four documented modes.
        _score(tmp_path, aggregate="vote")


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


def test_ambiguity_error_lists_only_the_tied_compatible_profiles(score_roots, tmp_path):
    profile_a = toy_profile("toy-a", scale=1.3)
    profile_b = toy_profile("toy-b", scale=1.6)
    profile_full = toy_profile("toy-full", backend="center", scale=1.0)  # incompatible: full-frame
    write_toy_store(score_roots, profile_a)
    write_toy_store(score_roots, profile_b)
    write_toy_store(score_roots, profile_full)

    with pytest.raises(ConfigError) as info:
        _score(tmp_path, "fake:scale=1.3")  # face crop: both a and b compatible, full is not

    assert profile_a.profile_id() in info.value.message
    assert profile_b.profile_id() in info.value.message
    assert profile_full.profile_id() not in info.value.message


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


def test_several_local_profiles_none_compatible_is_refused(score_roots, tmp_path):
    # Two local stores, neither a face crop: no single one to defer the mismatch decision to.
    write_toy_store(score_roots, toy_profile("toy-full-a", backend="center", scale=1.0))
    write_toy_store(score_roots, toy_profile("toy-full-b", backend="center", scale=1.2))

    with pytest.raises(ConfigError) as info:
        _score(tmp_path)  # default fake: crop="face"
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


def test_input_mismatch_names_a_shipped_profile_that_would_serve(score_roots, tmp_path):
    # No local store is a face crop at all, so the refusal falls back to naming the shipped
    # profiles that would serve -- passed in exactly as the CLI passes them (dfwb.score never
    # imports the face pipeline that owns them).
    write_toy_store(score_roots, toy_profile("toy-full", backend="center", scale=1.0))
    shipped = [toy_profile("shipped-face", backend="insightface", scale=1.3)]

    with pytest.raises(ContractError) as info:
        _score(tmp_path, shipped_profiles=shipped)  # default fake: crop="face", scale=1.3
    assert "shipped-face" in info.value.message


def test_with_no_shipped_profiles_the_refusal_names_none(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-full", backend="center", scale=1.0))

    with pytest.raises(ContractError) as info:
        _score(tmp_path)  # shipped_profiles defaults to ()
    assert "shipped-face" not in info.value.message


# ---------------------------------------------------------------------------------------- caching


def test_rerunning_with_an_identical_config_is_cached(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    first = _score(tmp_path)
    assert first.cached is False
    second = _score(tmp_path)
    assert second.cached is True
    assert second.csv_path == first.csv_path


def test_the_cache_is_checked_before_any_detector_call(score_roots, tmp_path):
    """Everything the cache key needs is known before a single clip is scored, so a cache hit
    makes zero ``predict()`` calls -- not "runs and discards the result"."""
    write_toy_store(score_roots, toy_profile("toy-face"))
    spy_id = str(uuid.uuid4())

    first = _score(tmp_path, f"fake:spy={spy_id}")
    assert first.cached is False
    calls_after_first = _toy.SPY_CALLS[spy_id]
    assert calls_after_first > 0

    second = _score(tmp_path, f"fake:spy={spy_id}")

    assert second.cached is True
    assert _toy.SPY_CALLS[spy_id] == calls_after_first  # not one more call


def test_force_recomputes_and_calls_predict_again(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))
    spy_id = str(uuid.uuid4())

    _score(tmp_path, f"fake:spy={spy_id}")
    calls_after_first = _toy.SPY_CALLS[spy_id]

    result = _score(tmp_path, f"fake:spy={spy_id}", force=True)

    assert result.cached is False
    assert _toy.SPY_CALLS[spy_id] > calls_after_first


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


def test_a_valid_but_mismatched_meta_at_the_cache_path_is_recomputed(score_roots, tmp_path):
    """A readable, otherwise-valid meta at the cache path whose fields do not actually match this
    request (a stale file left by hand, or a hash collision) is recomputed, not trusted -- the
    cache path alone is not proof enough."""
    import json

    write_toy_store(score_roots, toy_profile("toy-face"))

    first = _score(tmp_path)
    payload = json.loads(first.meta_path.read_text("utf-8"))
    payload["processing_profile"]["sha256"] = "0" * 64  # no longer this request's profile
    first.meta_path.write_text(json.dumps(payload), encoding="utf-8")

    second = _score(tmp_path)

    assert second.cached is False
    payload_after = json.loads(second.meta_path.read_text("utf-8"))
    assert payload_after["processing_profile"]["sha256"] != "0" * 64


def test_a_cached_file_with_error_rows_is_retried_not_reused(score_roots, tmp_path, caplog):
    """Re-running score with an identical configuration reuses the cached file only for an
    identical *successful* result: a detector failure is usually transient (an OOM, a flaky
    device fault), so a cache entry with any ``error`` row is retried instead of served."""
    write_toy_store(score_roots, toy_profile("toy-face"))
    spy_id = str(uuid.uuid4())
    detector_uri = f"fake:raise=FAKE/f00&spy={spy_id}"

    first = _score(tmp_path, detector_uri, batch_size=4)
    assert first.cached is False
    assert first.coverage["error"] == 1
    calls_after_first = _toy.SPY_CALLS[spy_id]

    with caplog.at_level(logging.INFO, logger="dfwb"):
        second = _score(tmp_path, detector_uri, batch_size=4)

    assert second.cached is False
    assert second.csv_path == first.csv_path
    assert _toy.SPY_CALLS[spy_id] > calls_after_first  # predict() ran again, not served stale
    assert any("errored" in message for message in caplog.messages)


def test_a_where_matching_no_videos_logs_a_warning(score_roots, tmp_path, caplog):
    write_toy_store(score_roots, toy_profile("toy-face"))

    with caplog.at_level(logging.WARNING, logger="dfwb"):
        result = _score(tmp_path, where={"identity": "does-not-exist"})

    assert result.coverage == {"expected": 0, "ok": 0, "missing": 0, "error": 0}
    assert any("matches no videos" in message for message in caplog.messages)


# ---------------------------------------------------------------------------- cache keys end to end


_CACHE_KEY_VARIATIONS = {
    "clips_per_video": {"clips_per_video": 2},
    "aggregate": {"aggregate": "max"},
    "labels": {"labels": "binary-exclude-fake"},
    "where": {"where": {"identity": "r00"}},
    "split": {"split": "train"},  # the scoretoy scheme assigns every video to "test" alone
}


@pytest.mark.parametrize(
    "changed", sorted(_CACHE_KEY_VARIATIONS), ids=sorted(_CACHE_KEY_VARIATIONS)
)
def test_cache_keys(score_roots, tmp_path, changed):
    """Every component the cache key hashes gives a fresh cache entry through :func:`score`
    itself, one at a time, matching the baseline in everything else."""
    write_toy_store(score_roots, toy_profile("toy-face"))
    baseline = _score(tmp_path)
    assert baseline.cached is False

    changed_result = _score(tmp_path, **_CACHE_KEY_VARIATIONS[changed])

    assert changed_result.cached is False
    assert changed_result.csv_path != baseline.csv_path


def test_cache_keys_detector_differs(score_roots, tmp_path):
    from tests.unit.score._toy import toy_run_profile, write_toy_run

    write_toy_store(score_roots, toy_run_profile())
    run_a = write_toy_run(tmp_path / "runs", name="toy-a", seed=0, fingerprint="fp-a")
    run_b = write_toy_run(tmp_path / "runs", name="toy-b", seed=0, fingerprint="fp-b")

    result_a = score(
        f"run:{run_a}#best", protocol=PROTOCOL, split="test", out=tmp_path / "out", batch_size=4
    )
    result_b = score(
        f"run:{run_b}#best", protocol=PROTOCOL, split="test", out=tmp_path / "out", batch_size=4
    )

    assert result_a.cached is False
    assert result_b.cached is False
    assert result_a.csv_path != result_b.csv_path


def test_cache_keys_profile_differs(score_roots, tmp_path):
    profile_a = toy_profile("toy-a", scale=1.3)
    profile_b = toy_profile("toy-b", scale=1.6)
    write_toy_store(score_roots, profile_a)
    write_toy_store(score_roots, profile_b)

    result_a = _score(tmp_path, "fake:scale=1.3", profile=profile_a.profile_id())
    result_b = _score(tmp_path, "fake:scale=1.3", profile=profile_b.profile_id())

    assert result_a.cached is False
    assert result_b.cached is False
    assert result_a.csv_path != result_b.csv_path


def test_cache_keys_identical_config_is_cached_force_recomputes(score_roots, tmp_path):
    """An identical configuration reuses the cached file, and --force always recomputes even
    then (both already covered individually above by
    ``test_rerunning_with_an_identical_config_is_cached`` and
    ``test_force_recomputes_even_when_a_cached_file_exists``; kept here too as part of the
    ``test_cache_keys`` group, alongside every component that changes the key)."""
    write_toy_store(score_roots, toy_profile("toy-face"))

    first = _score(tmp_path)
    identical = _score(tmp_path)
    forced = _score(tmp_path, force=True)

    assert first.cached is False
    assert identical.cached is True
    assert identical.csv_path == first.csv_path
    assert forced.cached is False
    assert forced.csv_path == first.csv_path


def test_cache_keys_where_membership_list_order_does_not_bust_the_cache(score_roots, tmp_path):
    """``cache_key()`` already hashes a canonical (sorted) ``where``, so an unsorted and a sorted
    call land at the same path -- but a cache *hit* also needs the stored C5 meta itself to be
    canonical, or ``cache_matches()`` compares a canonicalised incoming ``where`` against a
    differently-ordered stored one and wrongly calls it a miss."""
    write_toy_store(score_roots, toy_profile("toy-face"))

    unsorted = _score(tmp_path, where={"identity": ["r00", "f00"]})
    assert unsorted.cached is False

    already_sorted = _score(tmp_path, where={"identity": ["f00", "r00"]})
    assert already_sorted.csv_path == unsorted.csv_path
    assert already_sorted.cached is True

    unsorted_again = _score(tmp_path, where={"identity": ["r00", "f00"]})
    assert unsorted_again.csv_path == unsorted.csv_path
    assert unsorted_again.cached is True

    meta = read_scores(unsorted.csv_path).meta
    assert meta.protocol.where == {"identity": ["f00", "r00"]}  # stored canonical (sorted)


# ---------------------------------------------------- precision, store contents and pack version


def test_the_meta_records_the_resolved_precision(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    default = _score(tmp_path)
    bf16 = _score(tmp_path, precision="bf16")

    assert read_scores(default.csv_path).meta.env["precision"] == "fp32"
    assert read_scores(bf16.csv_path).meta.env["precision"] == "bf16"


def test_an_fp32_file_is_not_reused_for_a_bf16_request(score_roots, tmp_path):
    write_toy_store(score_roots, toy_profile("toy-face"))

    fp32 = _score(tmp_path)
    bf16 = _score(tmp_path, precision="bf16")

    assert bf16.cached is False
    assert bf16.csv_path != fp32.csv_path


def test_no_precision_and_fp32_are_the_same_request(score_roots, tmp_path):
    """Both mean "no autocast": the same scores, so the same cache entry."""
    write_toy_store(score_roots, toy_profile("toy-face"))

    unset = _score(tmp_path)
    explicit = _score(tmp_path, precision="fp32")

    assert explicit.cached is True
    assert explicit.csv_path == unset.csv_path


def test_a_store_that_gained_videos_is_rescored(score_roots, tmp_path):
    """Scoring, then processing the videos that were missing, then scoring again must score the
    new videos, not serve the old file with its missing rows."""
    profile = toy_profile("toy-face")
    write_toy_store(score_roots, profile, skip=["FAKE/f03", "REAL/r03"])
    first = _score(tmp_path)
    assert first.coverage["missing"] == 2

    write_toy_store(score_roots, profile)  # the two videos are processed now
    second = _score(tmp_path)

    assert second.cached is False
    assert second.coverage == {"expected": 8, "ok": 8, "missing": 0, "error": 0}


def test_a_reprocessed_store_is_rescored(score_roots, tmp_path):
    """Re-processing videos (a backend fix, say) keeps the profile's hash but changes what the
    store serves; its index changes, and so does the cache key."""
    profile = toy_profile("toy-face")
    write_toy_store(score_roots, profile)
    first = _score(tmp_path)

    write_toy_store(score_roots, profile, n_frames=6)
    second = _score(tmp_path)

    assert second.cached is False
    assert second.csv_path != first.csv_path


def test_the_meta_records_the_store_index_hash(score_roots, tmp_path):
    from dfwb.core.hashing import sha256_file

    store_dir = write_toy_store(score_roots, toy_profile("toy-face"))

    result = _score(tmp_path)

    meta = read_scores(result.csv_path).meta
    assert meta.env["store_index_sha256"] == sha256_file(store_dir / "index.jsonl")


def test_a_pack_label_release_is_rescored(score_roots, scoretoy_pack, tmp_path):
    """A pack release can fix labels without touching the split file (so the scheme hash stays
    the same); the pack version is part of the cache key, so the old file is not served."""
    import yaml

    write_toy_store(score_roots, toy_profile("toy-face"))
    first = _score(tmp_path)

    pack_yaml = scoretoy_pack.parent / "pack.yaml"
    card = yaml.safe_load(pack_yaml.read_text())
    card["version"] = "1.0.1"
    pack_yaml.write_text(yaml.safe_dump(card, sort_keys=False))
    second = _score(tmp_path)

    assert second.cached is False
    assert second.csv_path != first.csv_path
    assert read_scores(second.csv_path).meta.protocol.pack_version == "1.0.1"
