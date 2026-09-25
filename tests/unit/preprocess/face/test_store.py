"""The processed store: ``profile.json``, ``index.jsonl``, and per-video output directories.

A store is a plain directory under a profile id: one ``profile.json`` recorded on first use, one
append-only ``index.jsonl`` with one row per processed video (the latest row per key wins on
redo), and one output directory per video. Tests here never touch a real datasets root or user
config file: every :class:`Store` is built with an explicit, minimal ``roots`` mapping so the
datasets-root refusal is exercised on purpose, not by accident of the machine running the tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.paths import ResolvedRoot
from dfwb.core.records.local import (
    BackendSpec,
    CropSpec,
    DecodeSpec,
    ExtrasSpec,
    ProcessedRecord,
    ProcessingProfile,
    SamplingSpec,
    TrackSpec,
    TrackStats,
)
from dfwb.preprocess.face.store import Store

_NO_DATASETS_ROOT = {"datasets": ResolvedRoot("datasets", None, "unset", "DFWB_DATASETS_ROOT")}


def _profile(profile_id: str = "toy-test") -> ProcessingProfile:
    return ProcessingProfile(
        id=profile_id,
        backend=BackendSpec(name="center"),
        track=TrackSpec(iou=0.5, strategy="largest-then-iou"),
        crop=CropSpec(scale=1.0, size=32, square=True, align="none"),
        sampling=SamplingSpec(mode="uniform", frames=4),
        decode=DecodeSpec(library="opencv", color="rgb"),
        extras=ExtrasSpec(landmarks=False, mesh=False, masks=False),
    )


def _record(
    key: str = "toy/vid001",
    *,
    compression: str | None = None,
    status: str = "ok",
    n_frames: int = 4,
) -> ProcessedRecord:
    return ProcessedRecord(
        key=key,
        compression=compression,
        status=status,  # type: ignore[arg-type]
        n_frames=n_frames,
        frame_indices=list(range(n_frames)),
        relpath=f"{key}/{compression or '_'}",
        track=TrackStats(mean_confidence=0.9, identity_switch=False),
        reason=None,
    )


def test_index_path_and_video_dir_nest_under_the_key():
    store = Store(Path("/tmp/store-root"), _profile(), roots=_NO_DATASETS_ROOT)
    assert store.index_path == Path("/tmp/store-root/index.jsonl")
    assert store.video_dir(_record("ffpp/vid001", compression="c23")) == Path(
        "/tmp/store-root/ffpp/vid001/c23"
    )
    assert store.video_dir(_record("ffpp/vid002", compression=None)) == Path(
        "/tmp/store-root/ffpp/vid002/_"
    )


def test_append_writes_one_line_per_record_and_flushes(tmp_path):
    store = Store(tmp_path / "store", _profile(), roots=_NO_DATASETS_ROOT)
    store.append(_record("a/1"))
    store.append(_record("a/2", status="no_face", n_frames=0))

    lines = store.index_path.read_text("utf-8").splitlines()
    assert len(lines) == 2
    assert '"key":"a/1"' in lines[0]
    assert '"status":"no_face"' in lines[1]


def test_done_keys_and_records_keep_the_latest_row_per_key_compression(tmp_path):
    store = Store(tmp_path / "store", _profile(), roots=_NO_DATASETS_ROOT)
    store.append(_record("a/1", status="decode_error", n_frames=0))
    store.append(_record("a/1", status="ok", n_frames=4))  # a redo that succeeded
    store.append(_record("a/2", status="ok", n_frames=4))
    store.append(_record("a/2", status="ok", n_frames=2))  # a later, still-ok redo

    records = store.records()
    by_key = {(r.key, r.compression): r for r in records}
    assert len(records) == 2
    assert by_key[("a/1", None)].status == "ok"
    assert by_key[("a/2", None)].n_frames == 2

    assert store.done_keys() == {("a/1", None), ("a/2", None)}


def test_records_and_done_keys_are_empty_before_anything_is_appended(tmp_path):
    store = Store(tmp_path / "store", _profile(), roots=_NO_DATASETS_ROOT)
    assert store.records() == []
    assert store.done_keys() == set()


def test_done_keys_excludes_a_key_whose_latest_row_is_not_ok(tmp_path):
    store = Store(tmp_path / "store", _profile(), roots=_NO_DATASETS_ROOT)
    store.append(_record("a/1", status="ok", n_frames=4))
    store.append(_record("a/1", status="decode_error", n_frames=0))  # redo that then failed

    assert store.done_keys() == set()


def test_write_profile_records_profile_and_backend_meta_on_first_use(tmp_path):
    profile = _profile()
    store = Store(tmp_path / "store", profile, roots=_NO_DATASETS_ROOT)
    store.write_profile({"name": "center", "version": "1", "license": "MIT", "meta": {}})

    import json

    payload = json.loads((tmp_path / "store" / "profile.json").read_text("utf-8"))
    assert payload["sha256"] == profile.sha256()
    assert payload["profile_id"] == profile.profile_id()
    assert payload["profile"] == profile.model_dump(mode="json")
    assert payload["backend"] == {"name": "center", "version": "1", "license": "MIT", "meta": {}}


def test_write_profile_refuses_a_mismatched_profile_hash(tmp_path):
    root = tmp_path / "store"
    store_a = Store(root, _profile("profile-a"), roots=_NO_DATASETS_ROOT)
    store_a.write_profile({"name": "center", "version": "1", "license": "MIT", "meta": {}})

    store_b = Store(root, _profile("profile-b"), roots=_NO_DATASETS_ROOT)
    with pytest.raises(ContractError) as excinfo:
        store_b.write_profile({"name": "center", "version": "1", "license": "MIT", "meta": {}})
    assert _profile("profile-a").profile_id() in str(excinfo.value)
    assert _profile("profile-b").profile_id() in str(excinfo.value)


def test_write_profile_allows_the_same_profile_with_different_backend_meta_but_warns(
    tmp_path, caplog
):
    root = tmp_path / "store"
    profile = _profile()
    store = Store(root, profile, roots=_NO_DATASETS_ROOT)
    store.write_profile({"name": "center", "version": "1", "license": "MIT", "meta": {"a": 1}})

    caplog.clear()
    with caplog.at_level("WARNING"):
        store.write_profile({"name": "center", "version": "1", "license": "MIT", "meta": {"a": 2}})
    assert any("backend" in message.lower() for message in caplog.messages)

    import json

    payload = json.loads((root / "profile.json").read_text("utf-8"))
    # the file recorded on first use is kept; the mismatch is only logged
    assert payload["backend"]["meta"] == {"a": 1}


def test_cleanup_partial_removes_leftover_tmp_directories(tmp_path):
    root = tmp_path / "store"
    good = root / "a" / "1" / "_"
    good.mkdir(parents=True)
    (good / "frame_000000.png").write_bytes(b"not really a png")
    stale = root / "a" / "2" / "_.tmp-12345"
    stale.mkdir(parents=True)
    (stale / "frame_000000.png").write_bytes(b"partial")

    store = Store(root, _profile(), roots=_NO_DATASETS_ROOT)
    store.cleanup_partial()

    assert good.is_dir()
    assert not stale.exists()


def test_cleanup_partial_is_a_no_op_when_the_store_root_does_not_exist_yet(tmp_path):
    store = Store(tmp_path / "never-created", _profile(), roots=_NO_DATASETS_ROOT)
    store.cleanup_partial()  # must not raise


def test_a_store_root_inside_a_datasets_root_is_refused(tmp_path):
    datasets_root = tmp_path / "datasets"
    datasets_root.mkdir()
    roots = {
        "datasets": ResolvedRoot(
            "datasets", datasets_root, "env", "DFWB_DATASETS_ROOT", paths=(datasets_root,)
        )
    }
    with pytest.raises(ConfigError, match="datasets root"):
        Store(datasets_root / "work" / "processed" / "toy-test", _profile(), roots=roots)


def test_a_store_root_outside_every_datasets_root_is_accepted(tmp_path):
    datasets_root = tmp_path / "datasets"
    datasets_root.mkdir()
    roots = {
        "datasets": ResolvedRoot(
            "datasets", datasets_root, "env", "DFWB_DATASETS_ROOT", paths=(datasets_root,)
        )
    }
    Store(tmp_path / "work" / "processed" / "toy-test", _profile(), roots=roots)
