"""Tests for the optional media probe (``--probe``): PyAV probing and frame directories.

``probe_file`` never raises for one bad file (a decode failure or a missing path both give an
empty :class:`Probe` and a warning); it raises :class:`InstallationError` only when PyAV itself
is not installed. ``av`` is imported lazily, so every test that needs it skips cleanly when it is
missing (``pytest.importorskip``), except the "av is missing" test, which blocks it on purpose.
"""

from __future__ import annotations

import importlib.abc
import logging
import sys
from pathlib import Path

import pytest
from tests.unit.preprocess.inventory._demo import install

from dfwb.core.errors import InstallationError
from dfwb.core.records import InventoryRecord, Probe, read_jsonl
from dfwb.preprocess.inventory.probe import probe_file
from dfwb.preprocess.inventory.runner import build_inventory

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")


def _write_ffv1(
    path: Path, *, frames: int, width: int = 32, height: int = 32, fps: int = 10
) -> None:
    """A tiny lossless FFV1 clip in a Matroska container: real metadata, near-zero encode cost."""
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            array = np.full((height, width, 3), i % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


# ------------------------------------------------------------------------------- probe_file


def test_probe_file_reads_a_video(tmp_path):
    video = tmp_path / "clip.mkv"
    _write_ffv1(video, frames=10, width=32, height=32, fps=10)

    result = probe_file(video)

    assert result.frames == 10
    assert result.width == 32
    assert result.height == 32
    assert result.has_audio is False
    assert result.fps == pytest.approx(10.0)
    assert result.codec == "ffv1"
    assert result.duration_s is not None
    assert result.duration_s > 0


def test_probe_file_counts_a_frame_directory_without_pyav(tmp_path, monkeypatch):
    frames_dir = tmp_path / "real_test_1_2"
    frames_dir.mkdir()
    for i in range(5):
        (frames_dir / f"{i:06d}.png").touch()
    (frames_dir / "005.JPG").touch()  # suffix matched ignoring case
    (frames_dir / ".hidden.png").touch()  # hidden files are skipped
    (frames_dir / "notes.txt").touch()  # not an image suffix

    # No PyAV needed at all for a frame directory: block it and confirm probing still works.
    monkeypatch.delitem(sys.modules, "av", raising=False)

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] == "av":
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        result = probe_file(frames_dir)
    finally:
        sys.meta_path.remove(blocker)

    assert result == Probe(frames=6)


def test_probe_file_returns_an_empty_probe_and_warns_on_a_garbage_file(tmp_path, caplog):
    garbage = tmp_path / "broken.mp4"
    garbage.write_bytes(b"this is not a video file")

    with caplog.at_level(logging.WARNING):
        result = probe_file(garbage)

    assert result == Probe()
    assert "broken.mp4" in caplog.text


def test_probe_file_returns_an_empty_probe_and_warns_on_a_missing_file(tmp_path, caplog):
    missing = tmp_path / "nope.mp4"

    with caplog.at_level(logging.WARNING):
        result = probe_file(missing)

    assert result == Probe()
    assert "nope.mp4" in caplog.text


def test_probe_file_raises_installation_error_when_pyav_is_missing(tmp_path, monkeypatch):
    video = tmp_path / "clip.mkv"
    video.touch()
    for name in [n for n in sys.modules if n == "av" or n.startswith("av.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] == "av":
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(InstallationError) as info:
            probe_file(video)
    finally:
        sys.meta_path.remove(blocker)

    assert "[preprocess]" in info.value.hint


# ------------------------------------------------------------------------------- build_inventory


@pytest.fixture
def env(monkeypatch, tmp_path):
    """An isolated environment: no config files, no overrides, a datasets root and a work root."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home" / ".config"))
    raw, work = tmp_path / "raw", tmp_path / "work"
    raw.mkdir()
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(raw))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work))
    return raw, work


def _write_demo_tree(root: Path) -> None:
    """A Demo release (see ``_demo.DemoBuilder``) with real, distinctly-sized clips per file."""
    plan = {
        ("originals", "c23", "000"): 4,
        ("originals", "c40", "000"): 6,
        ("swapped", "c23", "000_000"): 3,
        ("swapped", "c40", "000_000"): 2,
    }
    for (task_dir, compression, stem), frames in plan.items():
        folder = root / task_dir / compression
        folder.mkdir(parents=True)
        _write_ffv1(folder / f"{stem}.mkv", frames=frames)


def test_build_inventory_probe_is_deterministic_across_job_counts(env, monkeypatch):
    raw, work = env
    install(monkeypatch)
    _write_demo_tree(raw / "Demo")

    one = build_inventory("demo", probe=True, jobs=1)
    one_bytes = (one.path.read_bytes(), (work / "demo" / "inventory.meta.json").read_bytes())
    records_one = read_jsonl(one.path, InventoryRecord)

    four = build_inventory("demo", probe=True, jobs=4)
    four_bytes = (four.path.read_bytes(), (work / "demo" / "inventory.meta.json").read_bytes())
    records_four = read_jsonl(four.path, InventoryRecord)

    assert one_bytes == four_bytes
    assert records_one == records_four

    probes = {(r.key, r.compression): r.probe for r in records_one}
    real_c23 = probes[("REAL/000", "c23")]
    assert real_c23.frames == 4
    assert real_c23.fps == pytest.approx(10.0)
    assert (real_c23.width, real_c23.height) == (32, 32)
    assert real_c23.has_audio is False
    assert real_c23.codec == "ffv1"
    assert real_c23.duration_s == pytest.approx(0.4, abs=0.05)
    assert probes[("REAL/000", "c40")].frames == 6
    assert probes[("FS_SWAP/000_000", "c23")].frames == 3
    assert probes[("FS_SWAP/000_000", "c40")].frames == 2


def test_build_inventory_probe_never_raises_for_a_garbage_video(env, monkeypatch, caplog):
    raw, _work = env
    install(monkeypatch)
    from tests.unit.preprocess.inventory._demo import make_demo_tree

    make_demo_tree(raw / "Demo")  # empty stub .mp4 files: every one fails to decode

    with caplog.at_level(logging.WARNING):
        result = build_inventory("demo", probe=True, jobs=2)

    records = read_jsonl(result.path, InventoryRecord)
    assert len(records) == 8
    assert all(r.probe == Probe() for r in records)


def test_build_inventory_probe_raises_installation_error_when_pyav_is_missing(
    env, monkeypatch, caplog
):
    raw, _work = env
    install(monkeypatch)
    from tests.unit.preprocess.inventory._demo import make_demo_tree

    make_demo_tree(raw / "Demo")

    for name in [n for n in sys.modules if n == "av" or n.startswith("av.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] == "av":
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(InstallationError) as info:
            build_inventory("demo", probe=True)
    finally:
        sys.meta_path.remove(blocker)

    assert "[preprocess]" in info.value.hint
