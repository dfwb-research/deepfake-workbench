"""The ``run:`` detector source: resolves ``<dir>[#best|#last]`` to a saved checkpoint."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dfwb.core.config.schema import ComponentSpec, ModelSection
from dfwb.core.errors import ConfigError
from dfwb.core.plugins import get_registry
from dfwb.models import checkpoint
from dfwb.models.detector import build_detector
from dfwb.models.source import load_run


def _model_cfg() -> ModelSection:
    return ModelSection(
        backbone=ComponentSpec(name="tiny-cnn"),
        temporal_pool=ComponentSpec(name="mean"),
        head=ComponentSpec(name="linear"),
    )


def _make_run(tmp_path, tag="best"):
    run_dir = tmp_path / "runs" / "toy" / "20260925-120000-s0"
    checkpoint_dir = run_dir / "checkpoints" / tag
    checkpoint_dir.mkdir(parents=True)
    detector = build_detector(_model_cfg(), source=f"run:{tag}-fingerprint")
    checkpoint.save(checkpoint_dir, detector, _model_cfg())
    name_dir = tmp_path / "runs" / "toy"
    (name_dir / "latest").symlink_to(run_dir.name)
    return run_dir, name_dir


def _make_dangling_latest(tmp_path):
    name_dir = tmp_path / "runs" / "toy"
    name_dir.mkdir(parents=True)
    (name_dir / "latest").symlink_to("20260925-120000-s0-does-not-exist")
    return name_dir


def test_run_source_resolves_a_run_directory_directly(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    detector = load_run(f"{run_dir}#best")
    assert detector.meta.source == "run:best-fingerprint"


def test_run_source_returns_the_detector_in_eval_mode(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    detector = load_run(f"{run_dir}#best")
    assert detector.training is False


def test_run_source_defaults_to_best(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    detector = load_run(str(run_dir))
    assert detector.meta.source == "run:best-fingerprint"


def test_run_source_follows_a_latest_symlink_given_directly(tmp_path):
    _, name_dir = _make_run(tmp_path)
    detector = load_run(f"{name_dir / 'latest'}#best")
    assert detector.meta.source == "run:best-fingerprint"


def test_run_source_follows_a_run_name_directory_with_latest_inside(tmp_path):
    _, name_dir = _make_run(tmp_path)
    detector = load_run(f"{name_dir}#best")
    assert detector.meta.source == "run:best-fingerprint"


def test_run_source_resolves_the_last_tag(tmp_path):
    _, name_dir = _make_run(tmp_path, tag="last")
    detector = load_run(f"{name_dir}#last")
    assert detector.meta.source == "run:last-fingerprint"


def test_run_source_unknown_path_raises_config_error_naming_the_paths_tried(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(ConfigError) as info:
        load_run(str(missing))
    assert "nope" in info.value.message


def test_run_source_rejects_an_unknown_tag(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    with pytest.raises(ConfigError, match="tag"):
        load_run(f"{run_dir}#worst")


def test_run_source_dangling_latest_inside_a_run_name_dir_names_the_missing_target(tmp_path):
    name_dir = _make_dangling_latest(tmp_path)
    with pytest.raises(ConfigError) as info:
        load_run(f"{name_dir}#best")
    assert "20260925-120000-s0-does-not-exist" in info.value.message


def test_run_source_dangling_latest_symlink_given_directly_names_the_missing_target(tmp_path):
    name_dir = _make_dangling_latest(tmp_path)
    with pytest.raises(ConfigError) as info:
        load_run(f"{name_dir / 'latest'}#best")
    assert "20260925-120000-s0-does-not-exist" in info.value.message


def test_registered_as_a_builtin_detector_source():
    entry = get_registry("detector_sources").entry("run")
    assert entry.provider == "dfwb"
    assert entry.requires == ("torch",)
    loader = get_registry("detector_sources").load("run")
    assert loader is load_run


# ------------------------------------------------------------------- checkpoint_sha256 / seed


def test_checkpoint_sha256_equals_the_weights_files_hash(tmp_path):
    from dfwb.core.hashing import sha256_file

    run_dir, _ = _make_run(tmp_path)
    detector = load_run(f"{run_dir}#best")
    expected = sha256_file(checkpoint.weights_path(run_dir / "checkpoints" / "best"))
    assert detector.checkpoint_sha256 == expected


def test_checkpoint_sha256_differs_between_best_and_last(tmp_path):
    run_dir = tmp_path / "runs" / "toy" / "20260925-120000-s0"
    for tag in ("best", "last"):
        checkpoint_dir = run_dir / "checkpoints" / tag
        checkpoint_dir.mkdir(parents=True)
        # Freshly (and separately) initialised weights: never seeded the same way twice, so
        # their sha256 differ, exactly like two real, independently trained checkpoints would.
        detector = build_detector(_model_cfg(), source="run:shared-fingerprint")
        checkpoint.save(checkpoint_dir, detector, _model_cfg())

    best = load_run(f"{run_dir}#best")
    last = load_run(f"{run_dir}#last")
    assert best.checkpoint_sha256 != last.checkpoint_sha256
    assert best.meta.source == last.meta.source == "run:shared-fingerprint"


def test_training_seed_is_read_from_env_json(tmp_path):
    import json

    run_dir, _ = _make_run(tmp_path)
    (run_dir / "env.json").write_text(json.dumps({"seed": 7}), encoding="utf-8")

    detector = load_run(f"{run_dir}#best")

    assert detector.training_seed == 7


def test_training_seed_is_none_without_an_env_json(tmp_path):
    run_dir, _ = _make_run(tmp_path)  # _make_run never writes env.json
    detector = load_run(f"{run_dir}#best")
    assert detector.training_seed is None


def test_training_seed_is_none_for_an_unreadable_env_json(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    (run_dir / "env.json").write_text("not json", encoding="utf-8")
    detector = load_run(f"{run_dir}#best")
    assert detector.training_seed is None
