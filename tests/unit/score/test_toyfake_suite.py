"""The framework's own ``toyfake`` eval suite, pinned against the real toyfake protocol pack: a
hand-built processed store over every real toyfake video, then ``dfwb score --suite toyfake``
followed by ``dfwb eval --suite toyfake``, through the CLI end to end."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tests._dfwb_cli import run_dfwb
from tests.unit.data.conftest import processed_record, write_store_frames, write_store_index
from tests.unit.score._toy import toy_profile

_REPO = Path(__file__).resolve().parents[3]
_TOYFAKE_PACK = _REPO / "src" / "dfwb" / "_packs" / "toyfake" / "toyfake"


@pytest.fixture
def cli(capsys, monkeypatch, tmp_path, score_roots):
    """``score_roots`` (from ``tests/unit/score/conftest.py``) installs the ``fake:`` detector
    source through the ``scoretoy_pack`` fixture, and points ``DFWB_WORK_ROOT``/
    ``DFWB_RUNS_ROOT`` at a throwaway tree; the real, built-in ``toyfake`` pack and ``toyfake``
    eval suite stay registered underneath it (only the ``dfwb.plugins`` entry-point group is
    intercepted to *add* the scoretoy fixtures, never to replace the framework's own builtins)."""
    monkeypatch.chdir(tmp_path)

    def _run(*args: str):
        return run_dfwb(capsys, *args)

    return _run


def _toyfake_video_keys() -> list[str]:
    with gzip.open(_TOYFAKE_PACK / "videos.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line)["key"] for line in handle]


def _write_real_toyfake_store(work_root: Path, profile) -> Path:
    """A hand-built processed store covering every video of the real, installed ``toyfake``
    dataset (200 videos x 4 tiny frames each, sized to ``profile`` -- about a second's worth of
    file I/O), so both the suite's entries (``official``/test: 41 videos, ``ident-72-14-14``/test:
    12 videos) score fully, with both classes present in each."""
    store_dir = work_root / "toyfake" / "processed" / profile.profile_id()
    store_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile.model_dump(mode="json"),
        "sha256": profile.sha256(),
        "profile_id": profile.profile_id(),
        "backend": {"name": profile.backend.name, "version": None, "license": None, "meta": {}},
    }
    (store_dir / "profile.json").write_text(json.dumps(payload), encoding="utf-8")
    records = [processed_record(key, n_frames=4) for key in _toyfake_video_keys()]
    for record in records:
        write_store_frames(store_dir, record, size=profile.crop.size)
    write_store_index(store_dir, records)
    return store_dir


def test_toyfake_suite_round_trips_through_score_and_eval(cli, score_roots):
    profile = toy_profile("toy-face")  # scale=1.3, size=32: matches fake:'s own defaults
    _write_real_toyfake_store(score_roots, profile)

    scored = cli("score", "--detector", "fake:", "--suite", "toyfake", "--json")
    assert scored.code == 0, scored.err
    results = json.loads(scored.out)["results"]
    assert {r["group"] for r in results} == {"in-domain", "cross"}
    for row in results:
        assert row["coverage"]["ok"] == row["coverage"]["expected"] > 0
        assert row["coverage"]["missing"] == row["coverage"]["error"] == 0

    files = [r["csv"] for r in results]
    evaluated = cli("eval", *files, "--suite", "toyfake", "--bootstrap", "20", "--json")
    assert evaluated.code == 0, evaluated.err
    data = json.loads(evaluated.out)
    suite_rows = {row["group"]: row for row in data["tables"]["suite"]}
    assert set(suite_rows) == {"in-domain", "cross"}
    for row in suite_rows.values():
        assert row["metric"] == "auc"
        assert row["n_entries"] == row["n_expected"] == 1
        assert 0.0 <= row["value"] <= 1.0
