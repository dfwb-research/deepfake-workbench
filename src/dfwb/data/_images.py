"""Image files, read and written with Pillow: stored frames, and the JPEG round trip.

torchvision's own image codecs (``torchvision.io.decode_png``, ``encode_jpeg``, ``decode_jpeg``)
are deprecated and due to be removed, so nothing here uses them. Pillow decodes a stored PNG to
exactly the same pixels they did, and its libjpeg encodes a frame to exactly the same JPEG bytes
at the same quality, so switching changes no number a run produces.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

__all__ = ["jpeg_round_trip", "read_frame"]

# Channel count -> Pillow image mode, for the frames a clip can hold.
_MODES = {1: "L", 3: "RGB"}


def read_frame(path: Path) -> Tensor:
    """One stored frame: ``[3, H, W]``, ``uint8``, RGB."""
    with Image.open(path) as image:
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        array = np.array(rgb, dtype=np.uint8)  # [H, W, 3]; a copy, so writable
    return torch.from_numpy(array).permute(2, 0, 1)


def _encode(frame: Tensor, quality: int) -> bytes:
    channels = int(frame.shape[0])
    if channels not in _MODES:
        raise ValueError(f"jpeg: a frame needs 1 or 3 channels, not {channels}")
    array = frame.permute(1, 2, 0).contiguous().numpy()
    # Pillow reads the mode off the array: [H, W] uint8 is "L", [H, W, 3] uint8 is "RGB"
    image = Image.fromarray(array[:, :, 0] if channels == 1 else array)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _decode(data: bytes, channels: int) -> Tensor:
    with Image.open(io.BytesIO(data)) as image:
        array = np.array(image.convert(_MODES[channels]), dtype=np.uint8)
    if channels == 1:
        array = array[:, :, np.newaxis]
    return torch.from_numpy(array).permute(2, 0, 1)


def jpeg_round_trip(frames: Sequence[Tensor], quality: int) -> list[Tensor]:
    """Every ``[C, H, W]`` ``uint8`` frame (``C`` is 1 or 3) encoded as a JPEG at ``quality``,
    then decoded back to the same shape."""
    return [_decode(_encode(frame, quality), int(frame.shape[0])) for frame in frames]
