"""The optional media probe (``--probe``): PyAV video probing and frame-directory counting.

:func:`probe_file` never raises for one bad file: a decode or open failure, or a path that does
not exist, gives an empty :class:`~dfwb.core.records.Probe` and logs a warning naming the file.
It raises :class:`~dfwb.core.errors.InstallationError` only when the file is a video and PyAV
itself is not installed.

PyAV is imported lazily, inside :func:`probe_file`, so this module -- and every module that
imports it -- stays import-safe without the ``preprocess`` extra: importing it costs nothing, and
only calling :func:`probe_file` on a video ever needs ``av``. A frame directory (a dataset that
ships pre-cropped frames instead of videos, e.g. WildDeepfake) is counted directly and never needs
PyAV at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dfwb.core.errors import InstallationError
from dfwb.core.records import Probe

__all__ = ["probe_file"]

_log = logging.getLogger(__name__)

# The same suffixes a frame-directory decoder reads: matched ignoring case, hidden files skipped.
_FRAME_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


def _is_frame(path: Path) -> bool:
    return (
        not path.name.startswith(".")
        and path.suffix.lower() in _FRAME_SUFFIXES
        and path.is_file()  # follows symlinks; a dangling link does not count
    )


def _probe_frame_dir(path: Path) -> Probe:
    """``Probe(frames=N)``: the number of image files directly in ``path``."""
    return Probe(frames=sum(1 for entry in path.iterdir() if _is_frame(entry)))


def _duration_s(container: Any, stream: Any) -> float | None:
    if container.duration is not None:
        return float(container.duration) / 1e6
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    return None


def _probe_open_video(av_module: Any, path: Path) -> Probe:
    container = av_module.open(str(path))
    try:
        has_audio = bool(container.streams.audio)
        video_streams = container.streams.video
        if not video_streams:
            return Probe(has_audio=has_audio)
        stream = video_streams[0]
        codec_context = stream.codec_context
        width = codec_context.width
        height = codec_context.height
        codec = codec_context.name
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        duration_s = _duration_s(container, stream)
        frames = stream.frames or sum(1 for _ in container.decode(stream))
        return Probe(
            frames=frames,
            fps=fps,
            width=width,
            height=height,
            duration_s=duration_s,
            has_audio=has_audio,
            codec=codec,
        )
    finally:
        container.close()


def probe_file(path: Path) -> Probe:
    """The media properties of ``path``: a video file, or a directory of frames.

    A directory is counted by its image files (:data:`_FRAME_SUFFIXES`, matched ignoring case,
    hidden files skipped), without importing PyAV. Anything else is opened as a video with PyAV;
    ``frames`` is the stream's declared frame count, or a decode count when that is 0 or absent;
    ``fps`` is the stream's average rate; ``width``/``height``/``codec`` come from the codec
    context; ``duration_s`` is the container's duration, or the stream's own duration and time
    base when the container does not report one; ``has_audio`` is whether the container has an
    audio stream.

    Raises:
        InstallationError: ``path`` is not a directory and PyAV is not installed.
    """
    if path.is_dir():
        return _probe_frame_dir(path)
    try:
        import av
    except ImportError as exc:
        raise InstallationError(
            "--probe needs PyAV", hint='pip install "deepfake-workbench[preprocess]"'
        ) from exc
    try:
        return _probe_open_video(av, path)
    except Exception:  # a broken or missing video must not abort the rest of the inventory
        _log.warning("%s: could not be probed (open or decode failed)", path)
        return Probe()
