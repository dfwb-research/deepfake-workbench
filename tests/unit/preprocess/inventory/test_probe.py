"""Tests for the optional media probe (``--probe``): PyAV probing and frame directories.

``probe_file`` never raises for one bad file (a decode failure or a missing path both give an
empty :class:`Probe` and a warning naming ``display``, never an absolute path); it raises
:class:`InstallationError` only when PyAV itself is not installed. ``require_pyav`` is the same
check, run once before a probe run starts. ``av`` is imported lazily, so every test that needs it
skips cleanly when it is missing (``pytest.importorskip``), except the "av is missing" tests,
which block it on purpose (``block_av``).
"""

from __future__ import annotations

import importlib.abc
import logging
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.preprocess.inventory._demo import install, make_demo_tree

from dfwb.core.errors import InstallationError
from dfwb.core.records import InventoryRecord, Probe, read_jsonl
from dfwb.preprocess.inventory.probe import _duration_s, _probe_open_video, probe_file, require_pyav
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


@pytest.fixture
def block_av(monkeypatch):
    """Block ``import av`` and ``importlib.util.find_spec("av")`` for the duration of a test.

    Purges any already-imported ``av`` (and submodules) from ``sys.modules`` first, so this
    works no matter what earlier tests in this process imported, then installs a meta-path
    finder that raises ``ModuleNotFoundError`` for ``av`` instead of the usual "return None".
    """
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
        yield
    finally:
        sys.meta_path.remove(blocker)


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


def test_probe_file_counts_a_frame_directory_without_pyav(tmp_path, block_av):
    frames_dir = tmp_path / "real_test_1_2"
    frames_dir.mkdir()
    for i in range(5):
        (frames_dir / f"{i:06d}.png").touch()
    (frames_dir / "005.JPG").touch()  # suffix matched ignoring case
    (frames_dir / ".hidden.png").touch()  # hidden files are skipped
    (frames_dir / "notes.txt").touch()  # not an image suffix

    # No PyAV needed at all for a frame directory: block_av confirms probing still works.
    result = probe_file(frames_dir)

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


def test_probe_file_warning_names_the_given_display_not_the_absolute_path(tmp_path, caplog):
    garbage = tmp_path / "broken.mp4"
    garbage.write_bytes(b"this is not a video file")

    with caplog.at_level(logging.WARNING):
        result = probe_file(garbage, display="originals/c23/broken.mp4")

    assert result == Probe()
    messages = [record.getMessage() for record in caplog.records]
    assert any("originals/c23/broken.mp4" in message for message in messages)
    assert not any(str(tmp_path) in message for message in messages)


def test_probe_file_raises_installation_error_when_pyav_is_missing(tmp_path, block_av):
    video = tmp_path / "clip.mkv"
    video.touch()

    with pytest.raises(InstallationError) as info:
        probe_file(video)

    assert "[preprocess]" in info.value.hint


# ------------------------------------------------------------------------------- require_pyav


def test_require_pyav_passes_when_av_is_installed():
    require_pyav()  # av is installed in this dev environment (the [preprocess] extra); no raise


def test_require_pyav_raises_installation_error_when_av_is_missing(block_av):
    with pytest.raises(InstallationError) as info:
        require_pyav()

    assert "[preprocess]" in info.value.hint


# --------------------------------------------------------------- _duration_s / _probe_open_video


def test_duration_s_falls_back_to_stream_duration_and_time_base_without_a_container_duration():
    container = SimpleNamespace(duration=None)
    stream = SimpleNamespace(duration=48000, time_base=Fraction(1, 48000))

    assert _duration_s(container, stream) == pytest.approx(1.0)


def test_duration_s_is_none_when_neither_container_nor_stream_report_one():
    container = SimpleNamespace(duration=None)
    stream = SimpleNamespace(duration=None, time_base=None)

    assert _duration_s(container, stream) is None


def test_probe_open_video_returns_an_audio_only_probe_without_a_video_stream(tmp_path):
    class _Streams:
        video: list[object] = []
        audio = [object()]

    class _Container:
        streams = _Streams()

        def close(self) -> None:
            pass

    class _AVModule:
        @staticmethod
        def open(path: str) -> _Container:
            return _Container()

    result = _probe_open_video(_AVModule(), tmp_path / "audio-only.mka")

    assert result == Probe(has_audio=True)


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
    make_demo_tree(raw / "Demo")  # empty stub .mp4 files: every one fails to decode

    with caplog.at_level(logging.WARNING):
        result = build_inventory("demo", probe=True, jobs=2)

    records = read_jsonl(result.path, InventoryRecord)
    assert len(records) == 8
    assert all(r.probe == Probe() for r in records)


def test_build_inventory_probe_warning_names_the_relpath_not_an_absolute_path(
    env, monkeypatch, caplog
):
    raw, _work = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo", reals=("000",), fakes=(), compressions=("c23",))

    with caplog.at_level(logging.WARNING):
        build_inventory("demo", probe=True, jobs=1)

    messages = [record.getMessage() for record in caplog.records]
    assert any("originals/c23/000.mp4" in message for message in messages)
    assert not any(str(raw) in message for message in messages)


def test_build_inventory_probe_raises_installation_error_when_pyav_is_missing(
    env, monkeypatch, caplog, block_av
):
    raw, _work = env
    install(monkeypatch)
    make_demo_tree(raw / "Demo")

    with pytest.raises(InstallationError) as info:
        build_inventory("demo", probe=True)

    assert "[preprocess]" in info.value.hint


def test_build_inventory_probe_fails_before_any_record_is_touched_when_pyav_is_missing(
    env, monkeypatch, block_av
):
    raw, _work = env
    install(monkeypatch)
    (raw / "Demo").mkdir()  # no originals/ or swapped/ at all: zero records to probe

    # Without the preflight, an empty record list would never call probe_file, so this would
    # succeed silently; the preflight makes a missing extra fail regardless of what is probed.
    with pytest.raises(InstallationError) as info:
        build_inventory("demo", probe=True)

    assert "[preprocess]" in info.value.hint
