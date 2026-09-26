"""``dfwb.data._images.read_frame``: decoding a stored frame, and the two ways it now refuses it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import cv2

from dfwb.core.errors import ContractError
from dfwb.data._images import CorruptFrameError, read_frame


def _write_png(path: Path, size: int = 8) -> None:
    cv2.imwrite(str(path), np.zeros((size, size, 3), dtype=np.uint8))


def test_read_frame_decodes_a_valid_png(tmp_path):
    path = tmp_path / "frame_000000.png"
    _write_png(path, size=8)

    frame = read_frame(path)

    assert frame.shape == (3, 8, 8)


def test_a_matching_expected_size_is_accepted(tmp_path):
    path = tmp_path / "frame_000000.png"
    _write_png(path, size=8)

    frame = read_frame(path, expected_size=8)

    assert frame.shape == (3, 8, 8)


def test_a_mismatched_expected_size_is_refused_with_a_clear_error(tmp_path):
    path = tmp_path / "frame_000000.png"
    _write_png(path, size=8)

    with pytest.raises(ContractError, match="stored frame is 8x8, not 16x16") as excinfo:
        read_frame(path, expected_size=16)

    assert "profile.crop.size" in excinfo.value.message
    assert not isinstance(excinfo.value, CorruptFrameError)  # a size mismatch, not corruption


def test_no_expected_size_means_no_size_check(tmp_path):
    path = tmp_path / "frame_000000.png"
    _write_png(path, size=8)

    read_frame(path)  # any size is accepted when nothing is asked for


def test_a_truncated_file_raises_corrupt_frame_error(tmp_path):
    path = tmp_path / "frame_000000.png"
    _write_png(path, size=64)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    with pytest.raises(CorruptFrameError, match="cannot decode this stored frame"):
        read_frame(path)


def test_garbage_bytes_raise_corrupt_frame_error(tmp_path):
    path = tmp_path / "frame_000000.png"
    path.write_bytes(b"not a real image" * 4)

    with pytest.raises(CorruptFrameError):
        read_frame(path)


def test_corrupt_frame_error_is_a_contract_error(tmp_path):
    path = tmp_path / "frame_000000.png"
    path.write_bytes(b"garbage")

    with pytest.raises(ContractError):
        read_frame(path)
