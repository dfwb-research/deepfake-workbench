"""Opening a video (or a directory of frame images) and decoding chosen frames from it.

Both backends decode forward only, exactly once, with no seeking: whichever frames were asked for
are handed back as they are reached, in ascending order, and decoding simply stops the moment the
source runs out -- whether that is the real end of the clip or a truncated file that claims more
frames than it can actually deliver. That last case is common with hand-me-down datasets, whose
frame counts were sometimes written by a different, more permissive decoder than the one reading
them years later; treating it as "the clip ended a little early" rather than an error is what the
original per-video pipeline did, and this keeps that behaviour.

``cv2`` and ``av`` (PyAV) are optional extras and are only imported inside the functions that need
them, so this module itself always imports cleanly. Numpy is a hard dependency of the framework,
but it is likewise only named here in type annotations (inert under ``from __future__ import
annotations``); the decoded arrays themselves come from whichever library did the decoding.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from dfwb.core.errors import InstallationError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["DecodeError", "VideoSource", "open_source"]

_EXTRA_HINT = 'pip install "deepfake-workbench[preprocess]"'
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


class DecodeError(Exception):
    """A video or frame directory could not be decoded.

    Raised by :func:`open_source` for a source that cannot be read at all (a corrupt or truncated
    header, an unreadable image, ...). It is deliberately not one of the framework's own error
    types: it is caught by the code that processes one video at a time and turned into a per-video
    result rather than stopping a whole run.
    """


class VideoSource(Protocol):
    """A source of decoded RGB frames: an opened video, or a directory of frame images."""

    total_frames: int
    fps: float
    width: int
    height: int

    def read(self, indices: Sequence[int]) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
        """Decode forward once, yielding ``(index, rgb_frame)`` for each of ``indices`` reached.

        ``indices`` need not be sorted; results are always yielded in ascending order. Decoding
        stops as soon as the source is exhausted, so trailing indices that never decode are simply
        never yielded -- they are not an error.
        """
        ...


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError:
        raise InstallationError("decoding frames needs OpenCV", hint=_EXTRA_HINT) from None
    return cv2


def _require_av() -> Any:
    try:
        import av
    except ImportError:
        raise InstallationError("decoding frames needs PyAV", hint=_EXTRA_HINT) from None
    return av


class _OpenCVSource:
    """A video decoded with OpenCV's ``VideoCapture``, BGR frames converted to RGB."""

    def __init__(self, path: Path) -> None:
        cv2 = _require_cv2()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise DecodeError(f"{path}: OpenCV could not open this file")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                count += 1
            capture.release()
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise DecodeError(f"{path}: OpenCV could not reopen this file")
        self._cv2 = cv2
        self._capture = capture
        self.total_frames = count
        self.fps = float(capture.get(cv2.CAP_PROP_FPS))
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self, indices: Sequence[int]) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
        wanted = iter(sorted(set(indices)))
        target = next(wanted, None)
        index = 0
        while target is not None:
            ok, frame = self._capture.read()
            if not ok:
                return
            if index == target:
                yield index, self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                target = next(wanted, None)
            index += 1


class _PyAVSource:
    """A video decoded with PyAV, frames converted to ``rgb24``."""

    def __init__(self, path: Path) -> None:
        av = _require_av()
        try:
            container = av.open(str(path))
        except av.error.FFmpegError as exc:
            raise DecodeError(f"{path}: PyAV could not open this file ({exc})") from exc
        try:
            stream = container.streams.video[0]
        except IndexError:
            container.close()
            raise DecodeError(f"{path}: no video stream found") from None
        count = stream.frames
        if not count:
            count = sum(1 for _ in container.decode(stream))
            container.close()
            container = av.open(str(path))
            stream = container.streams.video[0]
        self._container = container
        self._stream = stream
        self.total_frames = count
        self.fps = float(stream.average_rate) if stream.average_rate else 0.0
        self.width = int(stream.codec_context.width)
        self.height = int(stream.codec_context.height)

    def read(self, indices: Sequence[int]) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
        wanted = iter(sorted(set(indices)))
        target = next(wanted, None)
        if target is None:
            return
        for index, frame in enumerate(self._container.decode(self._stream)):
            if index == target:
                yield index, frame.to_ndarray(format="rgb24")
                target = next(wanted, None)
                if target is None:
                    return


class _FrameDirectorySource:
    """A directory of already-extracted frame images, in file-name order."""

    def __init__(self, path: Path) -> None:
        files = sorted(
            (child for child in path.iterdir() if child.suffix.lower() in _IMAGE_SUFFIXES),
            key=lambda child: child.name,
        )
        if not files:
            raise DecodeError(f"{path}: no frame images found")
        cv2 = _require_cv2()
        first = cv2.imread(str(files[0]))
        if first is None:
            raise DecodeError(f"{files[0]}: OpenCV could not read this image")
        self._cv2 = cv2
        self._files = files
        self.total_frames = len(files)
        self.fps = 0.0
        self.height = int(first.shape[0])
        self.width = int(first.shape[1])

    def read(self, indices: Sequence[int]) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
        for index in sorted(set(indices)):
            if index >= len(self._files):
                return
            image = self._cv2.imread(str(self._files[index]))
            if image is None:
                return
            yield index, self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)


def open_source(path: Path, *, library: Literal["opencv", "pyav"]) -> VideoSource:
    """Open ``path`` for sequential decoding.

    ``path`` may be a video file, decoded with ``library``, or a directory of already-extracted
    frame images (``library`` is then irrelevant: there is no video to decode, just files to read
    in name order).

    Raises:
        InstallationError: the chosen library (or, for a frame directory, OpenCV) is not
            installed.
        DecodeError: the source cannot be opened or has no readable frames.
    """
    if path.is_dir():
        return _FrameDirectorySource(path)
    if library == "opencv":
        return _OpenCVSource(path)
    return _PyAVSource(path)
