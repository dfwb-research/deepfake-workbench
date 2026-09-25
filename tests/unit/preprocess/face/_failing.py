"""A face backend that fails on purpose, for tests whose failures must happen in a worker process.

Spawned workers import everything afresh and find backends only through installed plugins, so a
test cannot hand them a stand-in the way it can register one in its own process. Instead,
:func:`install_plugin` makes this module an installed plugin, exactly as a wheel would: a
``.dist-info`` folder, on a directory the test puts on ``sys.path`` (which spawned processes
inherit), whose entry point registers :class:`FailingBackend` as ``failing``.

The backend finds the centred square of an ordinary frame, like ``center``. On a white frame it
raises, as a buggy backend or library would. On a black frame it ends its process on the spot,
without raising anything, as a crash inside native code (a decoder or an inference runtime)
does. It must therefore never detect in the test process itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from dfwb.preprocess.face.types import Face

MODULE = "tests.unit.preprocess.face._failing"


class FailingBackend:
    name = "failing"
    version = "1"
    license = "MIT"
    meta: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device

    def detect(self, frames: np.ndarray) -> list[list[Face]]:
        faces: list[list[Face]] = []
        for frame in frames:
            level = float(frame.mean())
            if level < 5:
                os._exit(1)
            if level > 250:
                raise RuntimeError("the detector choked on a white frame")
            height, width = frame.shape[:2]
            side = min(height, width)
            x1, y1 = (width - side) / 2, (height - side) / 2
            faces.append([Face(bbox=(x1, y1, x1 + side, y1 + side), score=1.0)])
        return faces


def register(api: Any) -> None:
    api.face_backends.add(
        "failing", target=f"{MODULE}:FailingBackend", summary="fails on white frames, dies on black"
    )


def install_plugin(site: Path) -> None:
    """Make this module an installed ``dfwb.plugins`` distribution under ``site``."""
    info = site / "dfwb_failing_backend-0.0.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: dfwb-failing-backend\nVersion: 0.0.0\n", "utf-8"
    )
    (info / "entry_points.txt").write_text(
        f"[dfwb.plugins]\nfailing = {MODULE}:register\n", "utf-8"
    )
