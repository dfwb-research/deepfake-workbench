"""``dfwb.preprocess.face.decode`` imports without OpenCV or PyAV, and fails cleanly without them.

Both are optional extras. The module itself must never import either at module scope, so a user
who has not installed the ``preprocess`` extra can still, say, load a profile that names a decode
library without the import failing; only actually opening a source needs the library.
"""

from __future__ import annotations

import sys

IMPORT_ONLY = "import dfwb.preprocess.face.decode"

OPEN_VIDEO = """
import sys
from pathlib import Path
from dfwb.core.errors import InstallationError
from dfwb.preprocess.face.decode import open_source
try:
    open_source(Path("missing.mkv"), library={library!r})
except InstallationError as exc:
    print(exc.message)
    print(exc.hint)
    sys.exit(0)
sys.exit(1)
"""

OPEN_FRAME_DIRECTORY = """
import sys
from pathlib import Path
from dfwb.core.errors import InstallationError
from dfwb.preprocess.face.decode import open_source
directory = Path("frames")
directory.mkdir()
(directory / "frame_000000.png").touch()
try:
    open_source(directory, library="pyav")
except InstallationError as exc:
    print(exc.message)
    print(exc.hint)
    sys.exit(0)
sys.exit(1)
"""


def test_decode_module_imports_with_both_libraries_blocked(blocked):
    done = blocked([sys.executable, "-c", IMPORT_ONLY], block=("cv2", "av"))
    assert done.returncode == 0, done.stderr


def test_opening_an_opencv_video_without_cv2_raises_installation_error(blocked, tmp_path):
    code = OPEN_VIDEO.format(library="opencv")
    done = blocked([sys.executable, "-c", code], block=("cv2",), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "OpenCV" in done.stdout
    assert 'pip install "deepfake-workbench[preprocess]"' in done.stdout


def test_opening_a_pyav_video_without_av_raises_installation_error(blocked, tmp_path):
    code = OPEN_VIDEO.format(library="pyav")
    done = blocked([sys.executable, "-c", code], block=("av",), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "PyAV" in done.stdout
    assert 'pip install "deepfake-workbench[preprocess]"' in done.stdout


def test_opening_a_frame_directory_without_cv2_raises_installation_error(blocked, tmp_path):
    done = blocked([sys.executable, "-c", OPEN_FRAME_DIRECTORY], block=("cv2",), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "OpenCV" in done.stdout
    assert 'pip install "deepfake-workbench[preprocess]"' in done.stdout
