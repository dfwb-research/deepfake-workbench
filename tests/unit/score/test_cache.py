"""``dfwb.score.cache``: the cache key, the output path it drives, and cache-hit lookup -- entirely
torch-free (it never needs a real detector, dataset or protocol)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from dfwb.core.detector import DetectorMeta, InputSpec
from dfwb.core.records.scores import ScoreMeta, write_scores
from dfwb.score.cache import DetectorIdentity, cache_key, look_up, score_path, slug


def _detector(
    *, name="det", version="1", source="run:fp", contract=(1, 0), checkpoint=None, extra=None
) -> SimpleNamespace:
    meta = DetectorMeta(
        name=name,
        version=version,
        contract_version=contract,
        input=InputSpec(),
        license="MIT",
        weights_license=None,
        citation=None,
        source=source,
    )
    return SimpleNamespace(meta=meta, checkpoint_sha256=checkpoint, fingerprint_extra=extra)


def _identity(detector: SimpleNamespace, *, training_seed=None, cacheable=True) -> DetectorIdentity:
    return DetectorIdentity(
        source=detector.meta.source,
        checkpoint_sha256=detector.checkpoint_sha256,
        training_seed=training_seed,
        fingerprint_extra=detector.fingerprint_extra,
        cacheable=cacheable,
    )


_BASE_KWARGS = {
    "effective_seed": 0,
    "scheme_sha256": "s" * 64,
    "split": "test",
    "where": None,
    "profile_sha256": "p" * 64,
    "aggregate_mode": "mean-prob",
    "clips_per_video": 4,
    "labels": "binary",
    "precision": "fp32",
    "store_index_sha256": "i" * 64,
    "pack_version": "1.0.0",
}


def _key(**overrides):
    detector = overrides.pop("detector", None) or _detector()
    identity = overrides.pop("identity", None) or _identity(detector)
    kwargs = {**_BASE_KWARGS, **overrides}
    return cache_key(detector=detector, identity=identity, **kwargs)


# --------------------------------------------------------------------------------- DetectorIdentity


def test_detector_identity_of_reads_every_field_duck_typed():
    detector = _detector(source="run:fp", checkpoint="c" * 64, extra="mod:factory:h")
    detector.training_seed = 7
    detector.cacheable = False

    identity = DetectorIdentity.of(detector)

    assert identity == DetectorIdentity("run:fp", "c" * 64, 7, "mod:factory:h", False)


def test_detector_identity_of_defaults_missing_attributes():
    detector = _detector()  # no training_seed/cacheable set at all

    identity = DetectorIdentity.of(detector)

    assert identity.training_seed is None
    assert identity.cacheable is True


def test_effective_seed_prefers_the_detectors_own_training_seed():
    identity = DetectorIdentity("s", None, 5, None, True)
    assert identity.effective_seed(99) == 5


def test_effective_seed_falls_back_to_the_requested_seed():
    identity = DetectorIdentity("s", None, None, None, True)
    assert identity.effective_seed(99) == 99


# ------------------------------------------------------------------------------------- cache_key


def test_cache_key_is_deterministic():
    assert _key() == _key()


def test_cache_key_changes_with_detector_name():
    baseline = _key()
    changed = _key(detector=_detector(name="other"))
    assert changed != baseline


def test_cache_key_changes_with_detector_version():
    baseline = _key()
    changed = _key(detector=_detector(version="2"))
    assert changed != baseline


def test_cache_key_changes_with_detector_source():
    baseline = _key()
    changed = _key(detector=_detector(source="run:other-fp"))
    assert changed != baseline


def test_cache_key_changes_with_checkpoint_sha256():
    baseline_detector = _detector(checkpoint="a" * 64)
    changed_detector = _detector(checkpoint="b" * 64)
    baseline = _key(detector=baseline_detector, identity=_identity(baseline_detector))
    changed = _key(detector=changed_detector, identity=_identity(changed_detector))
    assert changed != baseline


def test_cache_key_changes_with_fingerprint_extra():
    baseline_detector = _detector(extra="mod:factory:aaa")
    changed_detector = _detector(extra="mod:factory:bbb")
    baseline = _key(detector=baseline_detector, identity=_identity(baseline_detector))
    changed = _key(detector=changed_detector, identity=_identity(changed_detector))
    assert changed != baseline


def test_cache_key_changes_with_contract_version():
    baseline = _key()
    changed = _key(detector=_detector(contract=(2, 0)))
    assert changed != baseline


def test_cache_key_changes_with_effective_seed():
    baseline = _key()
    changed = _key(effective_seed=1)
    assert changed != baseline


def test_cache_key_changes_with_training_seed_via_effective_seed():
    """A detector's own training seed feeds the key through ``effective_seed`` -- exactly how
    :func:`~dfwb.score.harness.score` calls it -- so two runs that differ only in which seed they
    were trained with still land at different cache paths."""
    detector = _detector()
    seed_a = _identity(detector, training_seed=0)
    seed_b = _identity(detector, training_seed=1)
    baseline = _key(detector=detector, identity=seed_a, effective_seed=seed_a.effective_seed(0))
    changed = _key(detector=detector, identity=seed_b, effective_seed=seed_b.effective_seed(0))
    assert changed != baseline


def test_cache_key_changes_with_scheme_sha256():
    baseline = _key()
    changed = _key(scheme_sha256="t" * 64)
    assert changed != baseline


def test_cache_key_changes_with_split():
    baseline = _key()
    changed = _key(split="train")
    assert changed != baseline


def test_cache_key_changes_with_where():
    baseline = _key()
    changed = _key(where={"compression": "c23"})
    assert changed != baseline


def test_cache_key_changes_with_profile_sha256():
    baseline = _key()
    changed = _key(profile_sha256="q" * 64)
    assert changed != baseline


def test_cache_key_changes_with_aggregation():
    baseline = _key()
    changed = _key(aggregate_mode="max")
    assert changed != baseline


def test_cache_key_changes_with_clips_per_video():
    baseline = _key()
    changed = _key(clips_per_video=8)
    assert changed != baseline


def test_cache_key_changes_with_labels():
    baseline = _key()
    changed = _key(labels="family")
    assert changed != baseline


def test_cache_key_changes_with_precision():
    baseline = _key()
    changed = _key(precision="bf16")
    assert changed != baseline


def test_cache_key_changes_with_the_store_index():
    baseline = _key()
    changed = _key(store_index_sha256="j" * 64)
    assert changed != baseline


def test_cache_key_changes_with_pack_version():
    baseline = _key()
    changed = _key(pack_version="1.0.1")
    assert changed != baseline


def test_cache_key_ignores_where_key_order():
    a = _key(where={"compression": "c23", "identity": "000"})
    b = _key(where={"identity": "000", "compression": "c23"})
    assert a == b


def test_cache_key_ignores_where_membership_list_order():
    """Repeated ``--where key=v`` values are a set of alternatives, not a sequence: asking for
    ``identity in {000, 002}`` in either order must give the same cache entry."""
    a = _key(where={"identity": ["000", "002"]})
    b = _key(where={"identity": ["002", "000"]})
    assert a == b


def test_cache_key_none_and_empty_where_are_the_same():
    assert _key(where=None) == _key(where={})


# -------------------------------------------------------------------------------------- score_path


def test_score_path_uses_the_first_eight_hex_characters_of_the_key():
    key = _key()
    path = score_path(
        Path("/out"),
        detector_name="My Detector",
        protocol_ref="celebdf-v2/official",
        split="test",
        key=key,
    )
    assert path.name == f"test-{key[:8]}.scores.csv"
    assert path.parent.name == "celebdf-v2-official"
    assert path.parent.parent.name == "my-detector"


def test_slug_lower_kebab_cases_arbitrary_text():
    assert slug("My Detector!!") == "my-detector"
    assert slug("") == "x"


# ------------------------------------------------------------------------------------------ look_up


def _meta_payload(**overrides) -> dict:
    payload = {
        "schema": "dfwb.scores/1",
        "detector": {
            "name": "det",
            "version": "1",
            "source": "run:fp",
            "checkpoint_sha256": None,
            "contract_version": [1, 0],
        },
        "protocol": {
            "id": "toyfake/official",
            "split": "test",
            "where": {},
            "pack": "dfwb",
            "pack_version": "0.1.0",
            "scheme_sha256": "a" * 64,
        },
        "labels": "binary",
        "processing_profile": {"id": "prof-abcdef01", "sha256": "b" * 64},
        "input_adaptation": {"derived_crop": False, "mismatch_override": False},
        "aggregation": {"clip_to_video": "mean-prob", "clips_per_video": 4},
        "coverage": {"expected": 1, "ok": 1, "missing": 0, "error": 0},
        "seed": 0,
        "env": {"dfwb": "0.1.0", "precision": "fp32", "store_index_sha256": "i" * 64},
        "git": None,
        "command": None,
        "created": "2026-01-01T00:00:00Z",
    }
    payload.update(overrides)
    return payload


def _write_meta_only(tmp_path, **overrides):
    from dfwb.core.records import ScoreRow

    target = tmp_path / "d" / "p" / "test-abcd1234.scores.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    row = ScoreRow("toyfake", "REAL/r00", None, 0, 0.1, "ok")
    meta = ScoreMeta.model_validate(_meta_payload(**overrides))
    return write_scores(target, [row], meta)


_EXPECTED = {
    "identity": DetectorIdentity("run:fp", None, None, None, True),
    "scheme_sha256": "a" * 64,
    "split": "test",
    "where": None,
    "profile_sha256": "b" * 64,
    "aggregate_mode": "mean-prob",
    "clips_per_video": 4,
    "labels": "binary",
    "precision": "fp32",
    "store_index_sha256": "i" * 64,
    "pack_version": "0.1.0",
}


def test_look_up_returns_a_hit_when_everything_matches(tmp_path):
    csv_path, meta_path = _write_meta_only(tmp_path)

    hit = look_up(csv_path, **_EXPECTED)

    assert hit is not None
    assert hit.csv_path == csv_path
    assert hit.meta_path == meta_path
    assert hit.coverage == {"expected": 1, "ok": 1, "missing": 0, "error": 0}


def test_look_up_returns_none_when_the_file_does_not_exist(tmp_path):
    assert look_up(tmp_path / "nope.scores.csv", **_EXPECTED) is None


def test_look_up_returns_none_for_an_unreadable_file(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    csv_path.write_text("not a score file", encoding="utf-8")

    assert look_up(csv_path, **_EXPECTED) is None


def test_look_up_returns_none_when_the_meta_file_is_missing(tmp_path):
    csv_path, meta_path = _write_meta_only(tmp_path)
    meta_path.unlink()

    assert look_up(csv_path, **_EXPECTED) is None


def test_look_up_returns_none_on_source_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "identity": DetectorIdentity("run:other", None, None, None, True)}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_returns_none_on_checkpoint_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(
        tmp_path,
        detector={
            "name": "det",
            "version": "1",
            "source": "run:fp",
            "checkpoint_sha256": "c" * 64,
            "contract_version": [1, 0],
        },
    )
    assert look_up(csv_path, **_EXPECTED) is None


def test_look_up_returns_none_on_scheme_sha256_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "scheme_sha256": "t" * 64}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_returns_none_on_split_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "split": "train"}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_returns_none_on_profile_sha256_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "profile_sha256": "q" * 64}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_returns_none_on_aggregation_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "aggregate_mode": "max"}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_returns_none_on_clips_per_video_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "clips_per_video": 8}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_returns_none_on_labels_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "labels": "family"}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_returns_none_on_where_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "where": {"compression": "c23"}}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_matches_where_up_to_membership_list_order(tmp_path):
    csv_path, _ = _write_meta_only(
        tmp_path, protocol={**_meta_payload()["protocol"], "where": {"identity": ["000", "002"]}}
    )
    expected = {**_EXPECTED, "where": {"identity": ["002", "000"]}}

    hit = look_up(csv_path, **expected)

    assert hit is not None


def test_look_up_returns_none_and_logs_when_the_cached_file_has_error_rows(tmp_path, caplog):
    import logging

    from dfwb.core.records import ScoreRow

    target = tmp_path / "d" / "p" / "test-abcd1234.scores.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ScoreRow("toyfake", "REAL/r00", None, 0, 0.1, "ok"),
        ScoreRow("toyfake", "FAKE/f00", None, 1, None, "error"),
    ]
    meta = ScoreMeta.model_validate(
        _meta_payload(coverage={"expected": 2, "ok": 1, "missing": 0, "error": 1})
    )
    write_scores(target, rows, meta)

    with caplog.at_level(logging.INFO, logger="dfwb"):
        hit = look_up(target, **_EXPECTED)

    assert hit is None
    assert any("1 video(s) errored" in message for message in caplog.messages)


def test_look_up_treats_a_none_source_identity_as_the_recorded_unknown(tmp_path):
    """A detector with no ``meta.source`` at all is recorded as literally ``"unknown"``
    (:func:`~dfwb.score.writer.assemble_meta`); a lookup with ``identity.source is None`` must
    still match that file, not just one recorded with the literal string ``"unknown"``."""
    csv_path, _ = _write_meta_only(
        tmp_path,
        detector={
            "name": "det",
            "version": "1",
            "source": "unknown",
            "checkpoint_sha256": None,
            "contract_version": [1, 0],
        },
    )
    expected = {**_EXPECTED, "identity": DetectorIdentity(None, None, None, None, True)}

    hit = look_up(csv_path, **expected)

    assert hit is not None


def test_cache_key_and_score_path_round_trip_is_json_serialisable():
    # cache_key builds canonical JSON internally; a quick sanity check that a where mapping with
    # nested-looking values (lists, from repeated --where) does not blow up json.dumps.
    key = _key(where={"identity": ["000", "002"]})
    assert isinstance(key, str)
    assert len(key) == 64
    json.dumps({"key": key})


def test_look_up_matches_a_stored_where_that_was_never_canonicalised(tmp_path):
    """A meta written by something other than the harness (by hand, or an older writer) may hold
    a membership list in any order; the comparison canonicalises both sides."""
    csv_path, _ = _write_meta_only(
        tmp_path, protocol={**_meta_payload()["protocol"], "where": {"identity": ["002", "000"]}}
    )
    expected = {**_EXPECTED, "where": {"identity": ["000", "002"]}}

    assert look_up(csv_path, **expected) is not None


def test_look_up_returns_none_on_precision_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "precision": "bf16"}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_returns_none_when_the_meta_records_no_precision(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path, env={"dfwb": "0.1.0", "store_index_sha256": "i" * 64})
    assert look_up(csv_path, **_EXPECTED) is None


def test_look_up_returns_none_on_store_index_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "store_index_sha256": "j" * 64}
    assert look_up(csv_path, **mismatched) is None


def test_look_up_returns_none_on_pack_version_mismatch(tmp_path):
    csv_path, _ = _write_meta_only(tmp_path)
    mismatched = {**_EXPECTED, "pack_version": "0.1.1"}
    assert look_up(csv_path, **mismatched) is None
