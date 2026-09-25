"""Generate toyfake, dfwb's synthetic dataset, on this machine: ``dfwb datasets synth toyfake``.

toyfake lets anyone run the whole pipeline without first obtaining licensed data. :func:`synth`
writes ``<out>/toyfake/`` in the layout :mod:`dfwb.preprocess.inventory.builders.toyfake` reads:

* reals, ``original/<id>.mkv``: smooth, textured backgrounds with coloured blobs drifting across
  them. The identities are ``p000``, ``p001``, ...; about 40% of the videos are reals.
* fakes, ``blend-a/<target>_<source>.mkv`` and ``blend-b/<target>_<source>.mkv`` (about 30% each):
  real ``<target>`` with a 24x24 patch of real ``<source>`` blended in at a fixed alpha. The patch
  carries a high-frequency artefact -- a period-2 checkerboard for ``blend-a``, a period-3
  diagonal stripe for ``blend-b`` -- that the smooth reals never have, so a detector has
  something to learn. The patch sits at a seeded place inside the central 64x64 pixels, so a
  64-pixel centre crop keeps it; everywhere else a fake is its real, pixel for pixel.
* ``official_splits.json``: the identities of train, val and test (60/20/20, disjoint), and a
  ``README.txt``.

Everything is deterministic. The tree, the ids and the split depend only on ``seed`` and
``videos`` (drawn with :class:`random.Random`), and writing them needs neither numpy nor PyAV:
``write_media=False`` writes each video as an empty placeholder, which is all an inventory, a
protocol pack or ``dfwb protocols verify`` looks at. A video's frames depend only on ``seed`` and
the video's own id: its generator is seeded from a SHA-256 of the two, never from Python's
salted ``hash()``. They are written losslessly (FFV1 in Matroska), so the decoded frames are
exactly the generated ones and the artefact survives; the container is written bit-exact, so the
same seed gives the same bytes.

numpy and PyAV are imported only when media is written. Without PyAV (the ``preprocess`` extra),
asking for media raises :class:`~dfwb.core.errors.InstallationError` before anything is written.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from dfwb.core.errors import ConfigError, InstallationError
from dfwb.preprocess.inventory.builders.toyfake import (
    FOLDER,
    OFFICIAL_SPLITS_FILE,
    SPLIT_NAMES,
    TASK_DIRS,
    VIDEO_SUFFIX,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

__all__ = ["DEFAULT_SEED", "DEFAULT_VIDEOS", "MIN_VIDEOS", "SynthResult", "synth"]

# The built-in toyfake protocol pack lists the tree these defaults make.
DEFAULT_VIDEOS: Final = 200
DEFAULT_SEED: Final = 0
# The fewest videos that still give every task a video and every split an identity.
MIN_VIDEOS: Final = 5

# The blended patch: its side, the central window it stays inside, the blend weight of the donor
# patch, and the artefact's amplitude before blending (grey levels).
PATCH: Final = 24
CENTRE: Final = 64
_ALPHA: Final = 0.5
_ARTEFACT_AMPLITUDE: Final = 40.0

# The reals: base colour range, texture amplitude and feature size, and the blobs.
_BASE_RANGE: Final = (90.0, 165.0)
_TEXTURE_AMPLITUDE: Final = 20.0
_TEXTURE_SPACING: Final = 16
_BLOBS: Final = 3
_BLOB_AMPLITUDE: Final = 60.0
_BLOB_SPEED: Final = 1.5  # pixels per frame, at most, along each axis

_FAKE_TASKS: Final = ("BLEND_A", "BLEND_B")
_MISSING_PYAV_HINT: Final = (
    'pip install "deepfake-workbench[preprocess]", or pass --no-media to write the file tree only'
)


@dataclass(frozen=True)
class SynthResult:
    """What :func:`synth` wrote: the dataset folder, its video count, and videos per task."""

    root: Path
    n_videos: int
    by_task: dict[str, int]


@dataclass(frozen=True)
class _Video:
    """One video of the plan: its task, its file stem, and the reals it is made from."""

    task: str
    stem: str
    target: str
    source: str | None

    @property
    def relpath(self) -> str:
        return f"{TASK_DIRS[self.task]}/{self.stem}{VIDEO_SUFFIX}"


@dataclass(frozen=True)
class _Plan:
    """Every video to write, and the identities of each official split."""

    videos: tuple[_Video, ...]
    splits: dict[str, list[str]]


def _plan(videos: int, seed: int) -> _Plan:
    """The videos and the official split for ``videos`` and ``seed``; no numpy, no media.

    ``(3 * videos) // 10`` fakes per method, the rest reals. Each method draws its targets from
    the reals without repeats and gives each one a source drawn from the other reals. The
    identities are shuffled and a fifth of them (at least one) go to val, as many to test, and the
    rest to train.
    """
    rng = random.Random(seed)
    n_fake = (3 * videos) // 10
    n_real = videos - 2 * n_fake
    width = max(3, len(str(n_real - 1)))
    identities = [f"p{index:0{width}d}" for index in range(n_real)]
    planned = [_Video("REAL", identity, identity, None) for identity in identities]
    for task in _FAKE_TASKS:
        for target_index in sorted(rng.sample(range(n_real), n_fake)):
            other = rng.randrange(n_real - 1)
            source_index = other if other < target_index else other + 1
            target, source = identities[target_index], identities[source_index]
            planned.append(_Video(task, f"{target}_{source}", target, source))
    order = list(identities)
    rng.shuffle(order)
    held = max(1, n_real // 5)
    splits = {
        "train": sorted(order[2 * held :]),
        "val": sorted(order[:held]),
        "test": sorted(order[held : 2 * held]),
    }
    return _Plan(tuple(planned), splits)


def _check_arguments(videos: int, frames: int, size: int, fps: int) -> None:
    if videos < MIN_VIDEOS:
        raise ConfigError(
            f"toyfake needs at least {MIN_VIDEOS} videos, got {videos}",
            hint=f"pass --videos {MIN_VIDEOS} or more (the default is {DEFAULT_VIDEOS})",
        )
    if frames < 1:
        raise ConfigError(f"toyfake frames must be at least 1, got {frames}", hint="use 24")
    if size < PATCH:
        raise ConfigError(
            f"toyfake frame size must be at least {PATCH} pixels (the patch), got {size}",
            hint="use 64",
        )
    if fps < 1:
        raise ConfigError(f"toyfake fps must be at least 1, got {fps}", hint="use 8")


def _check_empty(root: Path) -> None:
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ConfigError(
            f"{root} exists and is not empty",
            hint="remove it, or pass another --out: toyfake never writes over existing files",
        )


def _import_pyav() -> Any:
    """The ``av`` module; raises :class:`InstallationError` when the extra is missing."""
    try:
        return importlib.import_module("av")
    except ImportError as exc:
        raise InstallationError("toyfake media needs PyAV", hint=_MISSING_PYAV_HINT) from exc


def _readme(videos: int, seed: int) -> str:
    return "\n".join(
        [
            "toyfake: a synthetic deepfake dataset generated by dfwb.",
            "",
            f"Made with: dfwb datasets synth toyfake --videos {videos} --seed {seed}",
            "",
            "original/<id>.mkv               real videos: smooth textured blobs; <id> is an "
            "identity",
            "blend-a/<target>_<source>.mkv   fakes: a patch of <source> blended into <target>, "
            "with a period-2 checkerboard artefact",
            "blend-b/<target>_<source>.mkv   fakes: the same, with a period-3 diagonal stripe "
            "artefact",
            f"{OFFICIAL_SPLITS_FILE:<31} the identities of train, val and test (disjoint)",
            "",
            "Videos are FFV1 in Matroska, lossless. A tree written with --no-media holds empty "
            "placeholder files instead.",
            "",
            "Every video is synthetic: no real person appears in any of them. The data is "
            "released under the MIT licence, as dfwb is.",
            "",
        ]
    )


# ---------------------------------------------------------------------------------------------
# Media (numpy and PyAV, imported only here)
# ---------------------------------------------------------------------------------------------


def _stream_seed(seed: int, name: str) -> int:
    """A 64-bit generator seed for ``name`` under ``seed``, stable across processes."""
    digest = hashlib.sha256(f"toyfake:{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _smooth_field(rng: np.random.Generator, height: int, width: int) -> NDArray[np.float64]:
    """Smooth colour noise: random values on a coarse grid, interpolated bilinearly."""
    import numpy as np

    spacing = _TEXTURE_SPACING
    coarse = rng.uniform(
        -_TEXTURE_AMPLITUDE,
        _TEXTURE_AMPLITUDE,
        size=(height // spacing + 2, width // spacing + 2, 3),
    )
    ys = np.arange(height) / spacing
    xs = np.arange(width) / spacing
    y0 = ys.astype(np.int64)
    x0 = xs.astype(np.int64)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x0 + 1] * wx
    bottom = coarse[y0 + 1][:, x0] * (1 - wx) + coarse[y0 + 1][:, x0 + 1] * wx
    field: NDArray[np.float64] = top * (1 - wy) + bottom * wy
    return field


def _bounce(position: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    """``position`` folded into ``[0, size]``, so a blob bounces off the edges."""
    import numpy as np

    folded = np.mod(position, 2 * size)
    result: NDArray[np.float64] = np.where(folded <= size, folded, 2 * size - folded)
    return result


def _real_frames(seed: int, identity: str, frames: int, size: int) -> NDArray[np.uint8]:
    """A real's frames, ``(frames, size, size, 3)`` RGB: drifting texture and moving blobs."""
    import numpy as np

    real = _Video("REAL", identity, identity, None)
    rng = np.random.default_rng(_stream_seed(seed, real.relpath))
    base = rng.uniform(*_BASE_RANGE, size=3)
    texture = _smooth_field(rng, size + frames, size + frames)
    ys = np.arange(size, dtype=np.float64)[None, :, None]
    xs = np.arange(size, dtype=np.float64)[None, None, :]
    times = np.arange(frames, dtype=np.float64)[:, None]

    image = np.empty((frames, size, size, 3), dtype=np.float64)
    for t in range(frames):  # the texture drifts one pixel down and right per frame
        image[t] = texture[t : t + size, t : t + size]
    image += base
    for _ in range(_BLOBS):
        colour = rng.uniform(-_BLOB_AMPLITUDE, _BLOB_AMPLITUDE, size=3)
        sigma = rng.uniform(size / 10, size / 5)
        start = rng.uniform(0, size, size=2)
        velocity = rng.uniform(-_BLOB_SPEED, _BLOB_SPEED, size=2)
        centres = _bounce(start + velocity * times, size)  # (frames, 2)
        cy = centres[:, 0][:, None, None]
        cx = centres[:, 1][:, None, None]
        weight = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma**2))
        image += weight[..., None] * colour
    pixels: NDArray[np.uint8] = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return pixels


def _artefact(task: str) -> NDArray[np.float64]:
    """The patch's artefact in ``[-1, 1]``: a checkerboard (``BLEND_A``) or a diagonal stripe."""
    import numpy as np

    y, x = np.indices((PATCH, PATCH))
    if task == "BLEND_A":
        checker: NDArray[np.float64] = np.where((x + y) % 2 == 0, 1.0, -1.0)
        return checker
    stripe: NDArray[np.float64] = np.array([1.0, 0.0, -1.0])[(x + y) % 3]
    return stripe


def _render(video: _Video, *, seed: int, frames: int, size: int) -> NDArray[np.uint8]:
    """The frames of ``video``, ``(frames, size, size, 3)`` RGB, from ``seed`` and its id alone."""
    import numpy as np

    target = _real_frames(seed, video.target, frames, size)
    if video.source is None:
        return target
    source = _real_frames(seed, video.source, frames, size)
    rng = np.random.default_rng(_stream_seed(seed, video.relpath))
    centre = min(size, CENTRE)
    low = (size - centre) // 2
    y0 = low + int(rng.integers(0, centre - PATCH + 1))
    x0 = low + int(rng.integers(0, centre - PATCH + 1))
    window = (slice(None), slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))

    donor = source[window].astype(np.float64)
    donor += _ARTEFACT_AMPLITUDE * _artefact(video.task)[None, :, :, None]
    fake = target.astype(np.float64)
    fake[window] = (1 - _ALPHA) * fake[window] + _ALPHA * donor
    pixels: NDArray[np.uint8] = np.clip(np.rint(fake), 0, 255).astype(np.uint8)
    return pixels


def _write_video(av: Any, path: Path, frames: NDArray[np.uint8], fps: int) -> None:
    """Write ``frames`` to ``path``: FFV1 (lossless, 8-bit RGB) in a bit-exact Matroska file."""
    container = av.open(str(path), mode="w", format="matroska", options={"fflags": "+bitexact"})
    try:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "bgr0"  # FFV1's 8-bit RGB layout; rgb24 converts to it losslessly
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


# ---------------------------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------------------------


def _write_tree(root: Path, plan: _Plan, readme: str) -> None:
    for folder in TASK_DIRS.values():
        (root / folder).mkdir(parents=True, exist_ok=True)
    splits = {name: plan.splits[name] for name in SPLIT_NAMES}
    (root / OFFICIAL_SPLITS_FILE).write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (root / "README.txt").write_text(readme, encoding="utf-8", newline="\n")


def _by_task(videos: Sequence[_Video]) -> dict[str, int]:
    counts = dict.fromkeys(TASK_DIRS, 0)
    for video in videos:
        counts[video.task] += 1
    return counts


def synth(
    out: Path,
    *,
    videos: int = DEFAULT_VIDEOS,
    seed: int = DEFAULT_SEED,
    frames: int = 24,
    size: int = 64,
    fps: int = 8,
    write_media: bool = True,
) -> SynthResult:
    """Write the toyfake dataset to ``out/toyfake/``.

    Args:
        out: The folder to write ``toyfake/`` into, e.g. a datasets root.
        videos: How many videos: about 40% reals and 30% of each fake method.
        seed: Decides the ids, the pairs, the official split and every frame.
        frames: Frames per video.
        size: Frame width and height, in pixels.
        fps: Frame rate recorded in each video.
        write_media: Encode the videos (needs PyAV); ``False`` writes empty placeholder files,
            the same tree, without numpy or PyAV.

    Raises:
        ConfigError: an argument is out of range, or ``out/toyfake`` exists and is not empty.
        InstallationError: media was asked for and PyAV (the ``preprocess`` extra) is missing;
            nothing has been written.
    """
    _check_arguments(videos, frames, size, fps)
    root = out / FOLDER
    _check_empty(root)
    av = _import_pyav() if write_media else None
    plan = _plan(videos, seed)
    _write_tree(root, plan, _readme(videos, seed))
    for video in plan.videos:
        path = root / video.relpath
        if av is None:
            path.touch()
        else:
            _write_video(av, path, _render(video, seed=seed, frames=frames, size=size), fps)
    return SynthResult(root, len(plan.videos), _by_task(plan.videos))
