"""``docs/gen_datasets.py`` documents each dataset's real inventory-builder facts.

Skipped unless the ``docs`` dependency group is installed (``mkdocs_gen_files``), the same way
``test_gen_cli.py`` skips: this script is never run by the standard, torch-free or
torch-installed test jobs, only where the docs group is present.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

mkdocs_gen_files = pytest.importorskip("mkdocs_gen_files")

REPO = Path(__file__).resolve().parents[3]
GEN_DATASETS = REPO / "docs" / "gen_datasets.py"


class _FakeHandle:
    def __init__(self, written: dict[str, str], path: str) -> None:
        self._written = written
        self._path = path
        self._buffer: list[str] = []

    def __enter__(self) -> _FakeHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self._written[self._path] = "".join(self._buffer)

    def write(self, text: str) -> None:
        self._buffer.append(text)


def _generated_pages(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Run the real generator with ``mkdocs_gen_files.open`` captured in memory, not on disk."""
    written: dict[str, str] = {}
    monkeypatch.setattr(
        mkdocs_gen_files, "open", lambda path, mode="r", *a, **k: _FakeHandle(written, path)
    )
    runpy.run_path(str(GEN_DATASETS), run_name="gen_datasets_under_test")
    assert "datasets/index.md" in written, "the generator did not write datasets/index.md"
    return written


def test_index_folder_column_shows_the_real_folder_name_not_the_dataset_id(monkeypatch):
    """The index's ``Folder`` column must show the folder dfwb expects under a datasets root
    (``builder.expected_folder``, the same name the builder's own error message gives), not the
    dataset id repeated a second time.

    ``ffpp``'s own page already gets this right (``| Folder | \\`FaceForensics++\\` |``, from
    ``builder.expected_folder`` directly); only the index page's own table had the bug.
    """
    written = _generated_pages(monkeypatch)
    index = written["datasets/index.md"]
    assert "FaceForensics++" in index, "the index must show ffpp's real folder name"
    assert "| [FaceForensics++](ffpp.md) | `FaceForensics++` |" in index
    # the old, wrong shape: the dataset id shown twice, once as the "folder"
    assert "| `ffpp` |" not in index, "the Folder column must not just repeat the dataset id"
