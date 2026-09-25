"""``uv build`` ships every non-Python data file dfwb needs at runtime, not just the source.

Slow: it really builds the package with `uv build`, so it is not part of the default test run
(``pytest -m slow`` selects it). It builds this repository as it stands (not a scaffolded
package), so unlike a cold, from-scratch build it needs no network access: `hatchling`, the only
build-system requirement, is already resolvable from the project's own lock and cache.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "dfwb"


def _wheel_names(tmp_path: Path) -> set[str]:
    out_dir = tmp_path / "dist"
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out_dir), str(REPO)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as archive:
        return set(archive.namelist())


def test_wheel_ships_the_built_in_pack_the_newpack_templates_and_the_catalogue(tmp_path):
    names = _wheel_names(tmp_path)

    assert "dfwb/_packs/toyfake/pack.yaml" in names

    pack_dir = SRC / "_packs" / "toyfake" / "toyfake"
    expected_pack_files = [p for p in pack_dir.rglob("*") if p.is_file()]
    assert expected_pack_files  # the fixture pack must not be empty, or this checks nothing
    for path in expected_pack_files:
        rel = path.relative_to(SRC).as_posix()
        assert f"dfwb/{rel}" in names, f"missing from wheel: dfwb/{rel}"

    templates = sorted((SRC / "protocols" / "_newpack").glob("*.tmpl"))
    assert templates  # the new-pack scaffold must ship at least one template
    for template in templates:
        assert f"dfwb/protocols/_newpack/{template.name}" in names

    assert "dfwb/core/catalogue.toml" in names
