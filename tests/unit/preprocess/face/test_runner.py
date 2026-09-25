"""Running the face pipeline over a dataset: scoping, resume and redo, workers, the licence gate.

Every test builds its own tiny dataset: a handful of short, lossless FFV1 clips under a temporary
datasets root, laid out for the two-task ``Demo`` inventory builder the inventory tests use; an
inventory built from them with the real inventory runner; and a protocol pack whose records name
those same keys (plus two videos this machine does not have). The ``center`` backend and the
shipped ``toy-64-center-8f`` profile keep every run detector-free and quick. The few tests that
need a backend to behave in a particular way register a stand-in of their own, and run it in
this process (``workers=0``), since a spawned worker only sees the backends that are installed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import multiprocessing
import multiprocessing.context
import os
import shutil
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pytest
import yaml
from tests.unit.preprocess.face._failing import FailingBackend, install_plugin
from tests.unit.preprocess.inventory._demo import install
from tests.unit.protocols.conftest import _dump, make_pack

from dfwb.core import licenses, plugins
from dfwb.core.errors import ConfigError, DFWBError, InstallationError, UnknownKeyError
from dfwb.core.plugins import get_registry
from dfwb.core.records import (
    DatasetCard,
    InventoryRecord,
    LabelVocab,
    ProcessedRecord,
    SchemeCard,
    SplitRow,
    VideoRecord,
    read_jsonl,
    write_jsonl,
    write_split_tsv,
)
from dfwb.core.records.protocol import LicenseInfo
from dfwb.preprocess.face import runner
from dfwb.preprocess.face.profiles import load_profile
from dfwb.preprocess.face.runner import RunSummary, run
from dfwb.preprocess.face.store import Store
from dfwb.preprocess.face.types import Face
from dfwb.preprocess.inventory.runner import build_inventory

PROFILE = "toy-64-center-8f"

Key = tuple[str, str | None]

# The Demo release on disk: (directory under the dataset folder, file stems).
LAYOUT = {
    "originals/c23": ("000", "001", "002"),
    "originals/c40": ("000", "001"),
    "swapped/c23": ("000_001", "001_000"),
}
ALL_KEYS: set[Key] = {
    ("REAL/000", "c23"),
    ("REAL/000", "c40"),
    ("REAL/001", "c23"),
    ("REAL/001", "c40"),
    ("REAL/002", "c23"),
    ("FS_SWAP/000_001", "c23"),
    ("FS_SWAP/001_000", "c23"),
}

# The protocol's ``official`` scheme: every inventory video, plus two this machine does not have.
SPLITS: dict[Key, str] = {
    ("REAL/000", "c23"): "train",
    ("REAL/000", "c40"): "train",
    ("REAL/001", "c23"): "test",
    ("REAL/001", "c40"): "test",
    ("REAL/002", "c23"): "exclude",
    ("REAL/003", "c23"): "test",
    ("FS_SWAP/000_001", "c23"): "test",
    ("FS_SWAP/001_000", "c23"): "val",
    ("FS_SWAP/002_000", "c23"): "test",
}


# ------------------------------------------------------------------------------------- fixtures


def _write_clip(path: Path, *, frames: int = 10, dark: bool = False, white: bool = False) -> None:
    """A lossless FFV1 clip, 48 x 32: a circle sliding across a plain background, or all black,
    or all white."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (48, 32))
    assert writer.isOpened(), path
    for index in range(frames):
        frame = np.zeros((32, 48, 3), np.uint8)
        if white:
            frame[:] = 255
        elif not dark:
            frame[:] = 40
            cv2.circle(frame, (4 + 4 * index, 16), 6, (60, 180, 220), -1)
        writer.write(frame)
    writer.release()


def _write_demo_protocol(dataset_dir: Path, dataset_id: str) -> None:
    """The ``demo`` dataset of a protocol pack: one ``official`` scheme, :data:`SPLITS`."""
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits").mkdir()
    videos = []
    for key, compression in SPLITS:
        task, _, stem = key.partition("/")
        method = "original" if task == "REAL" else "Swapper"
        identity = stem.partition("_")[0]
        videos.append(VideoRecord(key, compression, f"DEMO-{task}", method, identity=identity))
    write_jsonl(dataset_dir / "videos.jsonl.gz", videos)
    rows = [SplitRow(key, compression, split) for (key, compression), split in SPLITS.items()]
    sha256 = write_split_tsv(dataset_dir / "splits" / "official.tsv.gz", rows)
    card = DatasetCard(
        id=dataset_id,
        name="Demo",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only",
        modalities=["video"],
        key_rule="<task>/<file stem>",
        schemes={"official": SchemeCard(kind="official", source="fixture", sha256=sha256)},
        default_scheme="official",
    )
    (dataset_dir / "dataset.yaml").write_text(_dump(card))
    labels = LabelVocab(
        vocab={"DEMO-REAL": {"binary": 0}, "DEMO-FS_SWAP": {"binary": 1}},
        mappings={"binary": {"from": "binary"}},
    )
    (dataset_dir / "labels.yaml").write_text(_dump(labels))


@dataclasses.dataclass
class Env:
    raw: Path
    work: Path
    dataset: Path
    tmp: Path


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    """An isolated machine: its own roots, licence store and cache, offline, with the Demo
    dataset on disk, its inventory built, and the Demo builder and protocol pack installed next
    to the built-in components."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    for name in list(os.environ):
        if name.startswith("DFWB_"):
            monkeypatch.delenv(name)
    raw, work = tmp_path / "raw", tmp_path / "work"
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(raw))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work))
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("DFWB_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("DFWB_OFFLINE", "1")
    # Another test may have run the CLI in this process, which stops the dfwb logger from
    # propagating to the root logger that caplog listens on.
    monkeypatch.setattr(logging.getLogger("dfwb"), "propagate", True)

    dataset = raw / "Demo"
    for directory, stems in LAYOUT.items():
        for stem in stems:
            _write_clip(dataset / directory / f"{stem}.avi")

    pack = make_pack(
        tmp_path / "packs",
        "demo-pack",
        {"demo": {}, "other": {}},
        builders={"demo": _write_demo_protocol},
    )
    real_entry_points = plugins._entry_points
    install(monkeypatch, packs={"demo-pack": pack})
    fake_entry_points = plugins._entry_points
    monkeypatch.setattr(
        plugins,
        "_entry_points",
        lambda group: (
            real_entry_points(group)
            if group == plugins.BUILTINS_GROUP
            else fake_entry_points(group)
        ),
    )
    build_inventory("demo")
    return Env(raw=raw, work=work, dataset=dataset, tmp=tmp_path)


def _run(**options: Any) -> RunSummary:
    options.setdefault("profile", PROFILE)
    options.setdefault("workers", 0)
    return run("demo", **options)


def _index_lines(store: Path) -> list[str]:
    return (store / "index.jsonl").read_text("utf-8").splitlines()


def _rows(store: Path) -> list[ProcessedRecord]:
    return read_jsonl(store / "index.jsonl", ProcessedRecord)


def _rows_if_any(store: Path) -> list[ProcessedRecord]:
    return _rows(store) if (store / "index.jsonl").is_file() else []


def _keys(rows: list[ProcessedRecord]) -> list[Key]:
    return sorted(((row.key, row.compression) for row in rows), key=lambda k: (k[0], k[1] or ""))


def _tree(root: Path) -> list[tuple[str, int]]:
    """Every file under ``root`` with its size, and every directory, as a comparable listing."""
    return sorted(
        (path.relative_to(root).as_posix(), path.stat().st_size if path.is_file() else -1)
        for path in root.rglob("*")
    )


def _store_path(work: Path, profile: str = PROFILE) -> Path:
    return work / "demo" / "processed" / load_profile(profile).profile_id()


def _write_profile(tmp: Path, backend: str, **changes: Any) -> str:
    """The toy profile with another backend (and any other top-level changes), as a YAML file."""
    data = load_profile(PROFILE).model_dump(mode="json")
    data["id"] = f"toy-{backend}"
    data["backend"] = {"name": backend}
    data.update(changes)
    path = tmp / f"{backend}.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


# ------------------------------------------------------------------------ stand-in backends


class PickyBackend:
    """The centre square, but only on frames that are not black: a black clip has no face."""

    name = "picky"
    version = "1"
    license = "MIT"
    meta: ClassVar[dict[str, Any]] = {"picky": True}
    see_in_the_dark: ClassVar[bool] = False

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device

    def detect(self, frames: np.ndarray) -> list[list[Face]]:
        faces: list[list[Face]] = []
        for frame in frames:
            height, width = frame.shape[:2]
            side = min(height, width)
            x1, y1 = (width - side) / 2, (height - side) / 2
            face = Face(bbox=(x1, y1, x1 + side, y1 + side), score=1.0)
            faces.append([face] if type(self).see_in_the_dark or frame.mean() > 5 else [])
        return faces


class CountingBackend(PickyBackend):
    """Records every build, ``prepare`` and ``close``, and the device it was built for."""

    name = "counting"
    see_in_the_dark = True
    events: ClassVar[list[str]] = []

    def __init__(self, *, device: str = "cpu") -> None:
        super().__init__(device=device)
        type(self).events.append(f"build {device}")

    def prepare(self) -> None:
        type(self).events.append("prepare")

    def close(self) -> None:
        type(self).events.append("close")


class GatedBackend(PickyBackend):
    """A backend whose weights need a licence acknowledgement, gated like the shipped ones."""

    name = "gated"
    see_in_the_dark = True
    license_gate: ClassVar[str | None] = "gated-weights"
    license_terms: ClassVar[str | None] = "for testing only"

    def __init__(self, *, device: str = "cpu") -> None:
        if not licenses.is_accepted("gated-weights"):
            raise InstallationError(
                "gated-weights: needs a one-time licence acknowledgement",
                hint="re-run with --accept-license",
            )
        super().__init__(device=device)


def _register(key: str, backend: type) -> None:
    get_registry("face_backends").register(key, summary=f"a {key} stand-in")(backend)


@pytest.fixture
def picky(env: Env, monkeypatch: pytest.MonkeyPatch) -> str:
    """The Demo dataset with ``REAL/001`` black in both compressions, and a profile whose backend
    finds no face on black frames; returns the profile's path."""
    monkeypatch.setattr(PickyBackend, "see_in_the_dark", False)
    for compression in ("c23", "c40"):
        _write_clip(env.dataset / "originals" / compression / "001.avi", dark=True)
    _register("picky", PickyBackend)
    return _write_profile(env.tmp, "picky")


# ---------------------------------------------------------------------------------- scoping


def test_without_a_protocol_every_inventory_video_is_processed(env):
    raw_before = _tree(env.raw)
    summary = _run()

    store = _store_path(env.work)
    assert summary == RunSummary(counts_by_status={"ok": 7}, n_skipped=0, store=store)
    rows = _rows(store)
    assert set(_keys(rows)) == ALL_KEYS
    assert len(rows) == 7  # each video scheduled exactly once
    for row in rows:
        assert row.status == "ok"
        assert row.n_frames == 8
        directory = store / row.relpath
        assert (directory / "clip.json").is_file()
        assert len(list(directory.glob("frame_*.png"))) == 8

    recorded = json.loads((store / "profile.json").read_text("utf-8"))
    assert recorded["profile_id"] == load_profile(PROFILE).profile_id()
    assert recorded["backend"] == {"name": "center", "version": "1", "license": "MIT", "meta": {}}
    assert _tree(env.raw) == raw_before  # nothing is ever written under a datasets root


def test_a_protocol_split_restricts_the_run_to_its_videos_and_counts_the_missing(env, caplog):
    with caplog.at_level(logging.WARNING, logger="dfwb"):
        summary = _run(protocol="demo/official", split="test")
    assert summary.counts_by_status == {"ok": 3}
    assert set(_keys(_rows(summary.store))) == {
        ("REAL/001", "c23"),
        ("REAL/001", "c40"),
        ("FS_SWAP/000_001", "c23"),
    }
    # REAL/003 and FS_SWAP/002_000 are in the test split but not on this machine.
    assert "2 video(s)" in caplog.text
    assert "not in the inventory" in caplog.text


def test_a_protocol_without_a_split_takes_every_split_but_exclude(env):
    summary = _run(protocol="demo/official")
    assert set(_keys(_rows(summary.store))) == ALL_KEYS - {("REAL/002", "c23")}


def test_where_narrows_a_protocol_split_with_the_protocol_semantics(env):
    summary = _run(
        protocol="demo/official", split="test", where={"task": "REAL", "compression": "c40"}
    )
    assert _keys(_rows(summary.store)) == [("REAL/001", "c40")]


def test_where_without_a_protocol_filters_the_inventory_the_same_way(env):
    summary = _run(where={"task": "FS_SWAP"})
    assert _keys(_rows(summary.store)) == [("FS_SWAP/000_001", "c23"), ("FS_SWAP/001_000", "c23")]


def test_where_without_a_protocol_takes_a_list_as_membership(env):
    summary = _run(where={"identity": ["000", "002"], "compression": "c23"})
    # A fake's identity is its target, so FS_SWAP/000_001 is identity 000 too.
    assert _keys(_rows(summary.store)) == [
        ("FS_SWAP/000_001", "c23"),
        ("REAL/000", "c23"),
        ("REAL/002", "c23"),
    ]


def test_an_unknown_where_field_is_a_config_error_with_a_suggestion(env):
    with pytest.raises(ConfigError, match="unknown where field 'idenity'") as caught:
        _run(where={"idenity": "000"})
    assert "identity" in caught.value.message
    assert not (env.work / "demo" / "processed").exists()


def test_an_unknown_attribute_in_where_is_a_config_error(env):
    with pytest.raises(ConfigError, match="unknown attribute 'lang'"):
        _run(where={"attrs.lang": "en"})


def test_limit_keeps_the_first_videos_by_key_and_compression(env):
    summary = _run(limit=3)
    assert _keys(_rows(summary.store)) == [
        ("FS_SWAP/000_001", "c23"),
        ("FS_SWAP/001_000", "c23"),
        ("REAL/000", "c23"),
    ]


def test_limit_orders_a_protocol_split_by_key_and_compression_too(env):
    summary = _run(protocol="demo/official", split="test", limit=2)
    assert _keys(_rows(summary.store)) == [("FS_SWAP/000_001", "c23"), ("REAL/001", "c23")]


def _shard_of(key: Key, count: int) -> int:
    text = f"{key[0]}\t{key[1] or ''}"
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % count


def test_shards_partition_the_scope_by_a_stable_hash_of_key_and_compression(env):
    count = 3
    store = _store_path(env.work)
    seen: list[set[Key]] = []
    for index in range(count):
        before = len(_rows_if_any(store))
        summary = _run(shard=(index, count))
        added = set(_keys(_rows_if_any(store)[before:]))
        assert added == {key for key in ALL_KEYS if _shard_of(key, count) == index}
        assert summary.n_skipped == 0  # another shard's videos are out of scope, not skipped
        seen.append(added)
    assert sum(1 for keys in seen if keys) >= 2  # a real split, not everything in one shard
    assert set().union(*seen) == ALL_KEYS
    assert sum(len(keys) for keys in seen) == len(ALL_KEYS)  # pairwise disjoint


def test_the_shards_of_a_limited_run_add_up_to_the_limited_run(env):
    # The limit picks the videos first; sharding only divides them up.
    store = _store_path(env.work)
    for index in range(2):
        _run(limit=4, shard=(index, 2))
    assert _keys(_rows(store)) == [
        ("FS_SWAP/000_001", "c23"),
        ("FS_SWAP/001_000", "c23"),
        ("REAL/000", "c23"),
        ("REAL/000", "c40"),
    ]


@pytest.mark.parametrize("shard", [(3, 3), (-1, 2), (0, 0), (1, -2)])
def test_a_shard_outside_zero_to_n_is_a_config_error(env, shard):
    with pytest.raises(ConfigError, match="shard"):
        _run(shard=shard)


def test_a_split_without_a_protocol_is_a_config_error(env):
    with pytest.raises(ConfigError, match="protocol"):
        _run(split="test")


def test_a_split_the_scheme_does_not_have_is_a_config_error_with_a_suggestion(env):
    with pytest.raises(ConfigError, match="'tset'") as caught:
        _run(protocol="demo/official", split="tset")
    assert "test" in caught.value.message


def test_a_protocol_of_another_dataset_is_a_config_error(env):
    with pytest.raises(ConfigError, match="'other'"):
        _run(protocol="other/official")


@pytest.mark.parametrize(("option", "value"), [("workers", -1), ("limit", 0)])
def test_negative_workers_and_an_empty_limit_are_config_errors(env, option, value):
    with pytest.raises(ConfigError, match=option):
        _run(**{option: value})


def test_an_unknown_profile_is_refused(env):
    with pytest.raises(UnknownKeyError):
        _run(profile="no-such-profile")


# ------------------------------------------------------------------------------ resume, redo


def test_a_rerun_skips_every_video_already_in_the_index(env):
    first = _run()
    lines = _index_lines(first.store)
    second = _run()
    assert second == RunSummary(counts_by_status={}, n_skipped=7, store=first.store)
    assert _index_lines(first.store) == lines


def test_a_rerun_also_skips_failures_unless_asked_to_redo_them(env, picky):
    first = _run(profile=picky)
    assert first.counts_by_status == {"ok": 5, "no_face": 2}
    assert _run(profile=picky) == RunSummary({}, 7, first.store)


def test_redo_no_face_reprocesses_only_those_videos(env, picky, monkeypatch):
    first = _run(profile=picky)
    before = _index_lines(first.store)
    monkeypatch.setattr(PickyBackend, "see_in_the_dark", True)

    second = _run(profile=picky, redo={"no_face"})

    assert second == RunSummary(counts_by_status={"ok": 2}, n_skipped=5, store=first.store)
    after = _index_lines(first.store)
    assert after[: len(before)] == before  # the index is only ever appended to
    added = [ProcessedRecord(**json.loads(line)) for line in after[len(before) :]]
    assert _keys(added) == [("REAL/001", "c23"), ("REAL/001", "c40")]
    assert {row.status for row in added} == {"ok"}
    assert {row.status for row in Store(first.store, load_profile(picky)).records()} == {"ok"}


def test_redo_ok_reprocesses_videos_that_already_succeeded(env, picky):
    first = _run(profile=picky)
    second = _run(profile=picky, redo={"ok"})
    assert second == RunSummary(counts_by_status={"ok": 5}, n_skipped=2, store=first.store)


def test_an_unknown_redo_status_is_a_config_error_naming_the_statuses(env):
    with pytest.raises(ConfigError, match="'no-face'") as caught:
        _run(redo={"no-face"})
    assert "no_face" in caught.value.message
    for status in ("ok", "no_face", "decode_error", "too_short", "skipped"):
        assert status in caught.value.hint
    assert not (env.work / "demo" / "processed").exists()


def test_a_video_interrupted_mid_write_is_cleaned_up_and_redone(env):
    first = _run()
    store = first.store
    # A run killed while writing REAL/000 c23: its index row never landed, and a private
    # temporary directory is left next to where the video's output would go.
    kept = [
        line
        for line, row in zip(_index_lines(store), _rows(store), strict=True)
        if (row.key, row.compression) != ("REAL/000", "c23")
    ]
    (store / "index.jsonl").write_text("".join(f"{line}\n" for line in kept), "utf-8")
    shutil.rmtree(store / "REAL" / "000" / "c23")
    stray = store / "REAL" / "000" / "c23.tmp-4242"
    stray.mkdir()
    (stray / "frame_000000.png").write_bytes(b"half a frame")
    # The same kill also caught a redo of REAL/001 c23 mid-swap, with its finished output renamed
    # aside and a half-written replacement next to it, and a video that is not redone this time.
    done = store / "REAL" / "001" / "c23"
    done.rename(done.with_name("c23.old-4242"))
    (done.with_name("c23.tmp-4242") / "frame_000000.png").parent.mkdir()
    (done.with_name("c23.tmp-4242") / "frame_000000.png").write_bytes(b"half a frame")
    finished = sorted(path.name for path in done.with_name("c23.old-4242").iterdir())

    second = _run()

    assert second == RunSummary(counts_by_status={"ok": 1}, n_skipped=6, store=store)
    assert not stray.exists()
    assert len(list((store / "REAL" / "000" / "c23").glob("frame_*.png"))) == 8
    # The finished output is put back and the half-written one removed, though REAL/001 c23 is
    # skipped: nothing but whole video directories is left anywhere in the store.
    assert sorted(path.name for path in done.iterdir()) == finished
    assert sorted(path.name for path in done.parent.iterdir()) == ["c23", "c40"]


def test_a_missing_source_is_recorded_as_a_decode_error_and_the_run_goes_on(env, caplog):
    (env.dataset / "originals" / "c40" / "001.avi").unlink()
    with caplog.at_level(logging.WARNING, logger="dfwb"):
        summary = _run()
    assert "1 video(s)" in caplog.text
    assert "REAL/001 (c40)" in caplog.text  # the first one missing is named
    assert summary.counts_by_status == {"ok": 6, "decode_error": 1}
    (row,) = [row for row in _rows(summary.store) if row.status != "ok"]
    assert (row.key, row.compression) == ("REAL/001", "c40")
    assert row.reason == "source not found"
    assert row.n_frames == 0
    assert row.frame_indices == []
    assert row.relpath == "REAL/001/c40"
    assert not (summary.store / "REAL" / "001" / "c40").exists()


def test_a_record_in_a_sibling_folder_is_read_from_there(env):
    # A record whose file lives in another dataset's folder (``folder``) is found there, across
    # the datasets roots, not in the Demo folder.
    sibling = env.raw / "Sibling" / "clips"
    _write_clip(sibling / "x.avi")
    inventory = env.work / "demo" / "inventory.jsonl"
    records = read_jsonl(inventory, InventoryRecord)
    borrowed = dataclasses.replace(
        records[0], key="REAL/borrowed", compression=None, relpath="clips/x.avi", folder="Sibling"
    )
    write_jsonl(inventory, [*records, borrowed])

    summary = _run(where={"key": "REAL/borrowed"})

    assert summary.counts_by_status == {"ok": 1}
    assert (summary.store / "REAL" / "borrowed" / "_" / "clip.json").is_file()


# --------------------------------------------------------------------- workers, environment


@pytest.fixture
def process_wide_changes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every variable set or deleted in ``os.environ``, and every change of multiprocessing's
    start method, from here on. Recording the writes themselves catches a change even when an
    earlier run in this process already made it, which comparing snapshots alone would miss."""
    changes: list[str] = []
    real_set, real_delete = os._Environ.__setitem__, os._Environ.__delitem__

    def set_item(self: Any, key: str, value: str) -> None:
        changes.append(f"set {key}")
        real_set(self, key, value)

    def delete_item(self: Any, key: str) -> None:
        changes.append(f"delete {key}")
        real_delete(self, key)

    def set_start_method(*args: Any, **kwargs: Any) -> None:
        changes.append(f"set_start_method {args[-1]}")

    monkeypatch.setattr(os._Environ, "__setitem__", set_item)
    monkeypatch.setattr(os._Environ, "__delitem__", delete_item)
    monkeypatch.setattr(multiprocessing, "set_start_method", set_start_method)
    monkeypatch.setattr(
        multiprocessing.context.DefaultContext, "set_start_method", set_start_method
    )
    return changes


def test_two_workers_write_what_one_process_writes_and_leave_the_environment_alone(
    env, tmp_path, process_wide_changes
):
    environment = dict(os.environ)
    process_wide_changes.clear()  # pytest itself sets PYTEST_CURRENT_TEST before the test body
    serial = _run(workers=0)
    assert dict(os.environ) == environment
    assert process_wide_changes == []
    serial_lines = sorted(_index_lines(serial.store))
    serial_copy = tmp_path / "serial-store"
    shutil.copytree(serial.store, serial_copy)
    shutil.rmtree(serial.store)

    parallel = _run(workers=2)

    assert dict(os.environ) == environment
    assert process_wide_changes == []
    assert parallel == serial
    assert sorted(_index_lines(parallel.store)) == serial_lines
    # Every frame and clip.json is byte-identical too, not just the index.
    assert _tree(parallel.store) == _tree(serial_copy)
    for path in serial_copy.rglob("*"):
        if path.is_file() and path.name != "index.jsonl":
            twin = parallel.store / path.relative_to(serial_copy)
            assert twin.read_bytes() == path.read_bytes(), path


@pytest.mark.parametrize("workers", [0, 2])
def test_a_profile_the_backend_cannot_run_is_refused_before_any_work(env, workers):
    # identity-cluster needs embeddings, which the center backend does not provide: that is known
    # as soon as the backend is built, so no video is ever tried, in this process or a worker.
    profile = _write_profile(
        env.tmp,
        "center",
        track={"iou": 0.5, "strategy": "identity-cluster", "ema": None},
    )
    with pytest.raises(ConfigError, match="identity-cluster"):
        _run(profile=profile, workers=workers)
    assert not (env.work / "demo" / "processed").exists()


def test_a_worker_builds_its_backend_on_its_first_video_only_and_closes_it_at_exit(
    env, monkeypatch
):
    # What each spawned worker does, run here where it can be watched.
    monkeypatch.setattr(CountingBackend, "events", [])
    _register("counting", CountingBackend)
    at_exit: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []
    monkeypatch.setattr("atexit.register", lambda func, *args: at_exit.append((func, args)))
    monkeypatch.setattr(runner, "_worker", None)
    profile = load_profile(_write_profile(env.tmp, "counting"))

    runner._start_worker(runner._Settings(profile, "counting", {}, "cuda:0"))
    assert CountingBackend.events == []  # starting a worker builds nothing

    records = read_jsonl(env.work / "demo" / "inventory.jsonl", InventoryRecord)[:3]
    for record in records:
        out_dir = env.tmp / "out" / record.key / (record.compression or "_")
        job = runner._Job(env.dataset / record.relpath, record, out_dir)
        outcome = runner._work(job)
        assert (outcome.record.status, outcome.error) == ("ok", None)
    assert CountingBackend.events == ["build cuda:0"]  # built once, on the first video

    ((func, args),) = at_exit
    func(*args)
    assert CountingBackend.events == ["build cuda:0", "close"]


def test_a_worker_whose_backend_cannot_be_built_records_the_error_for_the_video(env, monkeypatch):
    monkeypatch.setattr(runner, "_worker", None)
    profile = load_profile(PROFILE)
    runner._start_worker(runner._Settings(profile, "no-such-backend", {}, "cpu"))
    (record, *_) = read_jsonl(env.work / "demo" / "inventory.jsonl", InventoryRecord)
    outcome = runner._work(runner._Job(env.dataset / record.relpath, record, env.tmp / "out"))
    assert outcome.record.status == "decode_error"
    assert outcome.error == outcome.record.reason
    assert outcome.error == "error: UnknownKeyError: face_backends: unknown key 'no-such-backend'"
    assert "Traceback" in outcome.detail


def test_a_process_the_pool_did_not_start_refuses_work(env, monkeypatch):
    monkeypatch.setattr(runner, "_worker", None)
    (record, *_) = read_jsonl(env.work / "demo" / "inventory.jsonl", InventoryRecord)
    with pytest.raises(RuntimeError, match="did not initialise"):
        runner._work(runner._Job(env.dataset / record.relpath, record, env.tmp / "out"))


# ------------------------------------------------------------------------- failing videos


def _failing_on(
    monkeypatch: pytest.MonkeyPatch, fails: Callable[[int, InventoryRecord], BaseException | None]
) -> None:
    """Make in-process processing raise ``fails(n, record)`` for the n-th video it is given
    (counting from 0), when that is not ``None``, and process the video as usual otherwise."""
    real = runner.process_video
    calls = iter(range(1_000))

    def process_video(source: Path, record: InventoryRecord, *args: Any) -> ProcessedRecord:
        error = fails(next(calls), record)
        if error is not None:
            raise error
        return real(source, record, *args)

    monkeypatch.setattr(runner, "process_video", process_video)


@pytest.fixture
def fifteen_videos(env: Env) -> list[Key]:
    """Eight more reals in c23, the inventory rebuilt: 15 videos, returned in processing order."""
    for index in range(3, 11):
        _write_clip(env.dataset / "originals" / "c23" / f"{index:03d}.avi")
    build_inventory("demo")
    records = read_jsonl(env.work / "demo" / "inventory.jsonl", InventoryRecord)
    return [(record.key, record.compression) for record in records]


def test_a_video_that_raises_is_recorded_as_a_decode_error_naming_it(env, monkeypatch, caplog):
    def fails(n: int, record: InventoryRecord) -> BaseException | None:
        if (record.key, record.compression) == ("REAL/001", "c40"):
            return RuntimeError("a frame the detector could not take\nsecond line of detail")
        return None

    real = runner.process_video
    _failing_on(monkeypatch, fails)
    with caplog.at_level(logging.WARNING, logger="dfwb"):
        summary = _run()

    assert summary.counts_by_status == {"ok": 6, "decode_error": 1}
    (row,) = [row for row in _rows(summary.store) if row.status != "ok"]
    assert (row.key, row.compression) == ("REAL/001", "c40")
    assert row.reason == "error: RuntimeError: a frame the detector could not take"
    assert (row.n_frames, row.frame_indices, row.track) == (0, [], None)
    assert "REAL/001 (c40)" in caplog.text
    assert "a frame the detector could not take" in caplog.text

    # Such a row is retried like any other failure.
    monkeypatch.setattr(runner, "process_video", real)
    retried = _run(redo={"decode_error"})
    assert retried == RunSummary({"ok": 1}, 6, summary.store)


def test_ten_errors_in_a_row_abort_the_run_naming_the_last_video(env, monkeypatch, fifteen_videos):
    def fails(n: int, record: InventoryRecord) -> BaseException | None:
        return None if n == 0 else RuntimeError("CUDA out of memory")

    _failing_on(monkeypatch, fails)
    with pytest.raises(DFWBError) as caught:
        _run()

    last_key, last_compression = fifteen_videos[10]
    assert f"{last_key} ({last_compression})" in caught.value.message
    assert "10 videos in a row" in caught.value.message
    assert "error: RuntimeError: CUDA out of memory" in caught.value.message
    assert "--redo decode_error" in caught.value.hint
    # Everything recorded before the abort stays: one ok, then the ten errors.
    rows = _rows(_store_path(env.work))
    assert [(row.key, row.compression) for row in rows] == fifteen_videos[:11]
    assert [row.status for row in rows] == ["ok"] + ["decode_error"] * 10


def test_errors_that_are_not_consecutive_never_abort(env, monkeypatch, fifteen_videos):
    def fails(n: int, record: InventoryRecord) -> BaseException | None:
        return None if n == 9 else RuntimeError("flaky")

    _failing_on(monkeypatch, fails)
    summary = _run()
    assert summary.counts_by_status == {"ok": 1, "decode_error": 14}


def test_an_interrupt_stops_the_run_keeping_every_row_already_written(env, monkeypatch):
    # Rows are appended as each video finishes, not at the end: stopping on the third video
    # leaves the first two in the index.
    def fails(n: int, record: InventoryRecord) -> BaseException | None:
        return KeyboardInterrupt() if n == 2 else None

    _failing_on(monkeypatch, fails)
    with pytest.raises(KeyboardInterrupt):
        _run()
    assert _keys(_rows(_store_path(env.work))) == [
        ("FS_SWAP/000_001", "c23"),
        ("FS_SWAP/001_000", "c23"),
    ]


@pytest.fixture
def failing_plugin(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The ``failing`` backend, installed as a plugin spawned workers find, and registered here;
    REAL/001 c40 is black (its worker dies) and FS_SWAP/001_000 white (its worker raises).
    Returns the profile's path."""
    site = tmp_path / "site"
    install_plugin(site)
    monkeypatch.syspath_prepend(str(site))
    _register("failing", FailingBackend)
    _write_clip(env.dataset / "originals" / "c40" / "001.avi", dark=True)
    _write_clip(env.dataset / "swapped" / "c23" / "001_000.avi", white=True)
    return _write_profile(env.tmp, "failing")


def test_a_worker_crash_is_pinned_on_its_video_and_the_run_completes(env, failing_plugin, caplog):
    with caplog.at_level(logging.WARNING, logger="dfwb"):
        summary = _run(profile=failing_plugin, workers=2)

    assert summary.counts_by_status == {"ok": 5, "decode_error": 2}
    rows = {(row.key, row.compression): row for row in _rows(summary.store)}
    assert set(rows) == ALL_KEYS  # every video recorded exactly once
    assert len(_rows(summary.store)) == 7
    assert rows[("REAL/001", "c40")].reason == "error: worker crashed"
    assert rows[("FS_SWAP/001_000", "c23")].reason == (
        "error: RuntimeError: the detector choked on a white frame"
    )
    assert all(
        row.status == "ok"
        for key, row in rows.items()
        if key not in {("REAL/001", "c40"), ("FS_SWAP/001_000", "c23")}
    )
    assert "REAL/001 (c40)" in caplog.text
    assert "FS_SWAP/001_000 (c23)" in caplog.text


class FakePool:
    """A stand-in process pool that runs each video here, at once, when it is submitted.

    ``breaks`` says how many more times handing out a video breaks the pool, as a worker crash
    does: that video's future fails with ``BrokenProcessPool``, and so does every later
    submission to the same pool. Once is an innocent video that happened to be in flight when
    another crashed; more is the video that crashes its worker every time. A video in ``hangs``
    never finishes.
    """

    made: ClassVar[list[FakePool]] = []
    breaks: ClassVar[dict[Key, int]] = {}
    hangs: ClassVar[set[Key]] = set()

    def __init__(self, *, max_workers: int, mp_context: Any, initializer: Any, initargs: Any):
        self.max_workers = max_workers
        self.shutdowns: list[tuple[bool, bool]] = []
        self.broken = False
        initializer(*initargs)
        type(self).made.append(self)

    def submit(self, fn: Callable[..., Any], job: Any) -> Future[Any]:
        if self.broken:
            raise BrokenProcessPool("a process in the pool was terminated abruptly")
        key = (job.record.key, job.record.compression)
        future: Future[Any] = Future()
        if type(self).breaks.get(key, 0) > 0:
            type(self).breaks[key] -= 1
            self.broken = True
            future.set_exception(BrokenProcessPool("a process in the pool was terminated"))
        elif key not in type(self).hangs:
            try:
                future.set_result(fn(job))
            except BaseException as exc:  # a real pool reports even an interrupt this way
                future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdowns.append((wait, cancel_futures))


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch) -> type[FakePool]:
    monkeypatch.setattr(FakePool, "made", [])
    monkeypatch.setattr(FakePool, "breaks", {})
    monkeypatch.setattr(FakePool, "hangs", set())
    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(runner, "_worker", None)
    monkeypatch.setattr("atexit.register", lambda *args: None)
    return FakePool


def test_after_a_crash_each_suspect_runs_alone_and_the_rest_in_a_new_pool(env, fake_pool):
    # FS_SWAP/001_000 is only caught up in the first crash; REAL/000 c23 crashes every time.
    fake_pool.breaks = {("FS_SWAP/001_000", "c23"): 1, ("REAL/000", "c23"): 2}
    summary = _run(workers=2)

    assert summary.counts_by_status == {"ok": 6, "decode_error": 1}
    rows = {(row.key, row.compression): row for row in _rows(summary.store)}
    assert len(rows) == len(_rows(summary.store)) == 7  # every video recorded exactly once
    assert rows[("FS_SWAP/001_000", "c23")].status == "ok"
    assert rows[("REAL/000", "c23")].reason == "error: worker crashed"
    # A pool broke on FS_SWAP/001_000, which then ran fine alone; a fresh pool broke on
    # REAL/000 c23, which broke its own pool of one too; a third pool took the rest.
    assert [pool.max_workers for pool in fake_pool.made] == [2, 1, 2, 1, 2]
    assert all(pool.shutdowns == [(True, False)] for pool in fake_pool.made)


def test_errors_in_a_row_stop_a_pooled_run_too(env, fake_pool, monkeypatch, fifteen_videos):
    _failing_on(monkeypatch, lambda n, record: RuntimeError("CUDA out of memory"))
    with pytest.raises(DFWBError, match="videos in a row"):
        _run(workers=2)
    rows = _rows(_store_path(env.work))
    # Results arrive four at a time here, so the stop comes at the first batch that reaches ten.
    assert 10 <= len(rows) < len(fifteen_videos)
    assert {row.reason for row in rows} == {"error: RuntimeError: CUDA out of memory"}


def test_crashes_count_towards_the_errors_in_a_row_that_stop_a_run(env, fake_pool, fifteen_videos):
    # A backend that kills every worker would otherwise pin down and record one video at a time,
    # indefinitely: its crashes stop the run just as repeated errors do.
    fake_pool.breaks = dict.fromkeys(fifteen_videos, 99)
    with pytest.raises(DFWBError, match="10 videos in a row") as caught:
        _run(workers=2)
    assert "error: worker crashed" in caught.value.message
    rows = _rows(_store_path(env.work))
    assert [row.reason for row in rows] == ["error: worker crashed"] * 10


def test_an_error_carried_back_from_a_worker_is_recorded_for_its_video(env, fake_pool, monkeypatch):
    real_work = runner._work

    def work(job: Any) -> Any:
        if (job.record.key, job.record.compression) == ("REAL/002", "c23"):
            raise OSError("the result could not be sent back")
        return real_work(job)

    monkeypatch.setattr(runner, "_work", work)
    summary = _run(workers=2)
    assert summary.counts_by_status == {"ok": 6, "decode_error": 1}
    (row,) = [row for row in _rows(summary.store) if row.status != "ok"]
    assert row.reason == "error: OSError: the result could not be sent back"


def _interrupted_in_worker(monkeypatch: pytest.MonkeyPatch, key: Key) -> None:
    """Make the worker handling video ``key`` be interrupted, as Ctrl-C interrupts every process."""
    real_work = runner._work

    def work(job: Any) -> Any:
        if (job.record.key, job.record.compression) == key:
            raise KeyboardInterrupt
        return real_work(job)

    monkeypatch.setattr(runner, "_work", work)


def test_an_interrupt_during_the_pool_keeps_finished_videos_and_cancels_the_rest(
    env, fake_pool, monkeypatch
):
    fake_pool.hangs = {("REAL/000", "c40")}
    fake_pool.breaks = {("REAL/002", "c23"): 1}
    _interrupted_in_worker(monkeypatch, ("REAL/001", "c40"))
    real_wait = runner.wait
    calls = iter(range(1_000))

    def wait(*args: Any, **kwargs: Any) -> Any:
        if next(calls) == 1:
            raise KeyboardInterrupt
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(runner, "wait", wait)
    with pytest.raises(KeyboardInterrupt):
        _run(workers=2)
    # Four videos were handed out first, one of which never finishes; the three that did were
    # recorded. Then REAL/001 c23 finished, REAL/001 c40 was interrupted in its worker and
    # REAL/002 c23 broke the pool, before this process was interrupted too: only the finished one
    # is recorded, and the pool is shut down with everything still queued cancelled.
    assert _keys(_rows(_store_path(env.work))) == [
        ("FS_SWAP/000_001", "c23"),
        ("FS_SWAP/001_000", "c23"),
        ("REAL/000", "c23"),
        ("REAL/001", "c23"),
    ]
    (pool,) = fake_pool.made
    assert pool.shutdowns == [(True, True)]


def test_an_interrupt_in_a_worker_stops_the_run_recording_nothing_for_its_video(
    env, fake_pool, monkeypatch
):
    _interrupted_in_worker(monkeypatch, ("REAL/000", "c23"))
    with pytest.raises(KeyboardInterrupt):
        _run(workers=2)
    # The other three videos handed out with it finished and are recorded; the interrupted one
    # has no row, so the next run does it again.
    assert _keys(_rows(_store_path(env.work))) == [
        ("FS_SWAP/000_001", "c23"),
        ("FS_SWAP/001_000", "c23"),
        ("REAL/000", "c40"),
    ]
    (pool,) = fake_pool.made
    assert pool.shutdowns == [(True, True)]


def test_the_parent_closes_its_backend_before_the_pool_starts(env, monkeypatch):
    monkeypatch.setattr(CountingBackend, "events", [])
    _register("counting", CountingBackend)
    seen: list[list[str]] = []
    monkeypatch.setattr(
        runner, "_run_pool", lambda *args, **kwargs: seen.append(list(CountingBackend.events))
    )
    _run(profile=_write_profile(env.tmp, "counting"), workers=2)
    assert seen == [["build cpu", "prepare", "close"]]
    assert CountingBackend.events == ["build cpu", "prepare", "close"]  # closed once only


# ---------------------------------------------------------------- backend lifecycle, licence


def test_the_backend_is_built_prepared_and_closed_once_in_the_parent(env, monkeypatch):
    monkeypatch.setattr(CountingBackend, "events", [])
    _register("counting", CountingBackend)
    summary = _run(profile=_write_profile(env.tmp, "counting"), device="cuda:1")
    assert summary.counts_by_status == {"ok": 7}
    assert CountingBackend.events == ["build cuda:1", "prepare", "close"]
    recorded = json.loads((summary.store / "profile.json").read_text("utf-8"))
    assert recorded["backend"] == {
        "name": "counting",
        "version": "1",
        "license": "MIT",
        "meta": {"picky": True},
    }


def test_the_backend_is_closed_even_when_the_run_fails(env, monkeypatch):
    monkeypatch.setattr(CountingBackend, "events", [])
    _register("counting", CountingBackend)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "process_video", fail)
    with pytest.raises(KeyboardInterrupt):
        _run(profile=_write_profile(env.tmp, "counting"))
    assert CountingBackend.events == ["build cpu", "prepare", "close"]


def test_a_gated_backend_refuses_before_any_work_without_the_acknowledgement(env):
    _register("gated", GatedBackend)
    profile = _write_profile(env.tmp, "gated")
    with pytest.raises(InstallationError) as caught:
        _run(profile=profile)
    assert caught.value.exit_code == 5
    assert "--accept-license" in caught.value.hint
    assert not licenses.is_accepted("gated-weights")
    assert not (env.work / "demo" / "processed").exists()


def test_accept_license_records_the_acknowledgement_before_building(env):
    _register("gated", GatedBackend)
    summary = _run(profile=_write_profile(env.tmp, "gated"), accept_license=True)
    assert summary.counts_by_status == {"ok": 7}
    acceptance = licenses.all_accepted()["gated-weights"]
    assert acceptance.license == "for testing only"
    # Once recorded, later runs need no flag.
    assert _run(profile=_write_profile(env.tmp, "gated"), redo={"ok"}).counts_by_status == {"ok": 7}


def test_accept_license_on_an_ungated_backend_records_nothing(env):
    _run(accept_license=True)
    assert licenses.all_accepted() == {}


# ---------------------------------------------------------------------------------- progress


def test_progress_is_shown_with_rich_on_a_terminal(env, monkeypatch, capsys):
    monkeypatch.setattr(runner, "_interactive", lambda: True)
    summary = _run()
    assert summary.counts_by_status == {"ok": 7}
    assert "7/7" in capsys.readouterr().err


def test_no_progress_is_shown_off_a_terminal(env, capsys):
    _run()
    assert capsys.readouterr().err == ""


def test_progress_needs_rich_and_a_terminal(monkeypatch):
    class Terminal:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stderr", Terminal())
    assert runner._interactive() is True
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert runner._interactive() is False

    def broken(name: str) -> None:
        raise ValueError(name)

    monkeypatch.setattr("importlib.util.find_spec", broken)
    assert runner._interactive() is False
    monkeypatch.setattr("sys.stderr", None)
    assert runner._interactive() is False
