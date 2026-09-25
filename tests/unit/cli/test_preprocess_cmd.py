"""``dfwb preprocess``: flag wiring, output shapes and exit codes.

The scoping, resume/redo, worker and licence-gate behaviour these commands wrap is already
covered at the library level (``tests/unit/preprocess/face/test_runner.py`` and
``test_shard.py``); these tests only check that the CLI parses its flags into the right calls,
formats their results, and reports the framework's exit codes.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from tests.unit.preprocess.inventory._demo import install

from dfwb.cli.preprocess import _crop_text, _sampling_text
from dfwb.core import licenses, plugins
from dfwb.core.errors import InstallationError
from dfwb.core.plugins import get_registry
from dfwb.core.records.local import CropSpec, SamplingSpec
from dfwb.preprocess.face.backends.center import CenterBackend
from dfwb.preprocess.inventory.runner import build_inventory

PROFILE = "toy-64-center-8f"


def _write_clip(path, *, frames: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10.0, (48, 32))
    assert writer.isOpened(), path
    for index in range(frames):
        frame = np.zeros((32, 48, 3), np.uint8)
        frame[:] = 40
        cv2.circle(frame, (4 + 4 * index, 16), 6, (60, 180, 220), -1)
        writer.write(frame)
    writer.release()


@pytest.fixture
def demo(run, monkeypatch, tmp_path):
    """The demo builder, three tiny clips on disk, and a built inventory (no protocol pack).

    ``install`` replaces every entry point with a fake plugin that only registers the demo
    inventory builder; the real ``face_backends`` (``center``, and so on) live under the builtins
    group, so that group's real entry points are kept, exactly as
    ``tests.unit.preprocess.face.test_runner``'s own ``env`` fixture does.
    """
    real_entry_points = plugins._entry_points
    install(monkeypatch)
    fake_entry_points = plugins._entry_points
    monkeypatch.setattr(
        plugins,
        "_entry_points",
        lambda group: (
            real_entry_points(group)
            if group == plugins.BUILTINS_GROUP
            else fake_entry_points(group)
        ),
    )
    raw, work = tmp_path / "raw", tmp_path / "work"
    for stem in ("000", "001"):
        _write_clip(raw / "Demo" / "originals" / "c23" / f"{stem}.avi")
    _write_clip(raw / "Demo" / "swapped" / "c23" / "000_001.avi")
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(raw))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work))
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("DFWB_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.delenv("DFWB_DATASET_DEMO", raising=False)
    build_inventory("demo")
    return raw, work


class GatedBackend(CenterBackend):
    """A backend whose weights need a licence acknowledgement, for the exit-5 test."""

    name = "cli-gated"
    license_gate = "cli-gated-weights"
    license_terms = "for testing only"

    def __init__(self, *, device: str = "cpu") -> None:
        if not licenses.is_accepted("cli-gated-weights"):
            raise InstallationError(
                "cli-gated-weights: needs a one-time licence acknowledgement",
                hint="re-run with --accept-license",
            )
        super().__init__(device=device)


def _gated_profile(tmp_path) -> str:
    from dfwb.preprocess.face.profiles import load_profile

    get_registry("face_backends").register("cli-gated", summary="gated stand-in")(GatedBackend)
    data = load_profile(PROFILE).model_dump(mode="json")
    data["id"] = "toy-cli-gated"
    data["backend"] = {"name": "cli-gated"}
    path = tmp_path / "gated.yaml"
    path.write_text(json.dumps(data))
    return str(path)


# ------------------------------------------------------------------------------------- help


def test_preprocess_help_lists_every_subcommand(run):
    result = run("preprocess", "--help")
    assert result.code == 0, result.err
    for name in ("run", "status", "merge", "profiles"):
        assert name in result.out


# ---------------------------------------------------------------------------------- profiles


def test_profiles_table_and_json(run):
    table = run("preprocess", "profiles")
    assert table.code == 0, table.err
    header, *rows = table.out.splitlines()
    assert header.split() == ["ID", "PROFILE_ID", "BACKEND", "SAMPLING", "CROP"]
    assert any(PROFILE in row for row in rows)

    as_json = run("preprocess", "profiles", "--json")
    assert as_json.code == 0, as_json.err
    data = json.loads(as_json.out)
    (toy,) = [row for row in data if row["id"] == PROFILE]
    assert toy["backend"] == "center"
    assert toy["sampling"] == "uniform 8f"
    assert toy["crop"] == "1.0x/64/square"
    assert toy["profile_id"].startswith(f"{PROFILE}-")


# --------------------------------------------------------------------------------------- run


def test_run_processes_the_dataset_and_prints_a_summary_table(run, demo):
    result = run("preprocess", "run", "demo", "--profile", PROFILE)
    assert result.code == 0, result.err
    header, *rest = result.out.splitlines()
    assert header.split() == ["STATUS", "COUNT"]
    assert any(line.startswith("ok") for line in rest)
    assert any(line.startswith("store:") for line in rest)


def test_the_summary_table_counts_videos_already_done_apart_from_the_skipped_status(run, demo):
    assert run("preprocess", "run", "demo", "--profile", PROFILE).code == 0
    result = run("preprocess", "run", "demo", "--profile", PROFILE)
    assert result.code == 0, result.err
    header, *rest = result.out.splitlines()
    assert header.split() == ["STATUS", "COUNT"]
    assert [line.split() for line in rest if not line.startswith("store:")] == [
        ["already", "done", "3"]
    ]


def test_run_json_prints_a_run_summary(run, demo):
    result = run("preprocess", "run", "demo", "--profile", PROFILE, "--json")
    assert result.code == 0, result.err
    data = json.loads(result.out)
    assert data["counts_by_status"] == {"ok": 3}
    assert data["n_skipped"] == 0
    assert "/demo/processed/" in data["store"]
    assert data["store"].rsplit("/", 1)[-1].startswith(f"{PROFILE}-")


def test_run_where_and_limit_narrow_the_scope(run, demo):
    result = run(
        "preprocess",
        "run",
        "demo",
        "--profile",
        PROFILE,
        "--where",
        "task=REAL",
        "--limit",
        "1",
        "--json",
    )
    assert result.code == 0, result.err
    assert json.loads(result.out)["counts_by_status"] == {"ok": 1}


def test_run_where_repeated_for_the_same_key_means_any_of_its_values(run, demo):
    result = run(
        "preprocess",
        "run",
        "demo",
        "--profile",
        PROFILE,
        "--where",
        "identity=000",
        "--where",
        "identity=001",
        "--json",
    )
    assert result.code == 0, result.err
    assert json.loads(result.out)["counts_by_status"] == {"ok": 3}


def test_run_redo_reprocesses_a_status(run, demo):
    first = run("preprocess", "run", "demo", "--profile", PROFILE, "--json")
    first_data = json.loads(first.out)
    assert first_data["counts_by_status"] == {"ok": 3}

    rerun = run("preprocess", "run", "demo", "--profile", PROFILE, "--json")
    assert json.loads(rerun.out) == {
        "counts_by_status": {},
        "n_skipped": 3,
        "store": first_data["store"],
    }

    redone = run("preprocess", "run", "demo", "--profile", PROFILE, "--redo", "ok", "--json")
    assert json.loads(redone.out)["counts_by_status"] == {"ok": 3}


def test_run_shard_then_merge_round_trips_through_status(run, demo):
    first = run("preprocess", "run", "demo", "--profile", PROFILE, "--shard", "0/2", "--json")
    second = run("preprocess", "run", "demo", "--profile", PROFILE, "--shard", "1/2", "--json")
    assert first.code == 0, first.err
    assert second.code == 0, second.err
    total = json.loads(first.out)["counts_by_status"].get("ok", 0) + json.loads(second.out)[
        "counts_by_status"
    ].get("ok", 0)
    assert total == 3

    merged = run("preprocess", "merge", "demo", "--profile", PROFILE)
    assert merged.code == 0, merged.err
    assert "merged 2 shard file(s)" in merged.out

    status = run("preprocess", "status", "demo", "--profile", PROFILE)
    assert status.code == 0, status.err
    header, *rows = status.out.splitlines()
    assert header.split() == ["TASK", "STATUS", "COUNT"]
    assert sum(int(row.split()[-1]) for row in rows if row.split()[1] == "ok") == 3


def test_merge_with_nothing_to_merge_says_so(run, demo):
    result = run("preprocess", "merge", "demo", "--profile", PROFILE)
    assert result.code == 0, result.err
    assert "nothing to merge" in result.out


# ------------------------------------------------------------------------------------ status


def test_status_table_and_json(run, demo):
    assert run("preprocess", "run", "demo", "--profile", PROFILE, "--where", "task=REAL").code == 0

    table = run("preprocess", "status", "demo", "--profile", PROFILE)
    assert table.code == 0, table.err
    header, *rows = table.out.splitlines()
    assert header.split() == ["TASK", "STATUS", "COUNT"]
    body = {tuple(row.split()) for row in rows}
    assert ("REAL", "ok", "2") in body
    assert ("FS_SWAP", "not-processed", "1") in body

    as_json = run("preprocess", "status", "demo", "--profile", PROFILE, "--json")
    assert as_json.code == 0, as_json.err
    rows_json = json.loads(as_json.out)
    assert {"task": "REAL", "split": None, "status": "ok", "count": 2} in rows_json


def test_status_with_a_protocol_adds_a_split_column(run, monkeypatch):
    # The scoping and counting a --protocol status draws on is already covered at the library
    # level (test_shard.py); this only checks that the CLI adds the SPLIT column for it.
    from dfwb.preprocess.face import shard as shard_module

    fake_table = shard_module.StatusTable(
        rows=(shard_module.StatusRow(task="REAL", split="train", status="ok", count=2),)
    )
    monkeypatch.setattr(shard_module, "status", lambda *args, **kwargs: fake_table)

    result = run(
        "preprocess", "status", "demo", "--profile", PROFILE, "--protocol", "demo/official"
    )
    assert result.code == 0, result.err
    header, row = result.out.splitlines()
    assert header.split() == ["TASK", "SPLIT", "STATUS", "COUNT"]
    assert row.split() == ["REAL", "train", "ok", "2"]


# --------------------------------------------------------------------------------- exit codes


def test_unknown_profile_exits_2_with_a_suggestion(run, demo):
    result = run("preprocess", "run", "demo", "--profile", "toy-64-center-8")
    assert result.code == 2, result.err
    assert "did you mean" in result.err
    assert "hint:" in result.err


def test_a_missing_licence_acknowledgement_exits_5_with_the_accept_license_hint(
    run, demo, tmp_path
):
    profile = _gated_profile(tmp_path)
    result = run("preprocess", "run", "demo", "--profile", profile)
    assert result.code == 5, result.err
    assert "--accept-license" in result.err
    assert "hint:" in result.err


def test_accept_license_records_the_acknowledgement_and_the_run_then_succeeds(run, demo, tmp_path):
    profile = _gated_profile(tmp_path)
    result = run("preprocess", "run", "demo", "--profile", profile, "--accept-license", "--json")
    assert result.code == 0, result.err
    assert json.loads(result.out)["counts_by_status"] == {"ok": 3}


def test_a_malformed_shard_flag_exits_2(run, demo):
    result = run("preprocess", "run", "demo", "--profile", PROFILE, "--shard", "nope")
    assert result.code == 2, result.err
    assert "hint:" in result.err


def test_a_malformed_where_flag_exits_2(run, demo):
    result = run("preprocess", "run", "demo", "--profile", PROFILE, "--where", "no-equals-sign")
    assert result.code == 2, result.err
    assert "hint:" in result.err


def test_split_without_protocol_exits_2(run, demo):
    result = run("preprocess", "status", "demo", "--profile", PROFILE, "--split", "test")
    assert result.code == 2, result.err
    assert "hint:" in result.err


# ------------------------------------------------------------------ profiles table formatting


def test_sampling_text_covers_every_shape():
    assert _sampling_text(SamplingSpec(mode="uniform", frames=64)) == "uniform 64f"
    assert _sampling_text(SamplingSpec(mode="all")) == "all"
    assert _sampling_text(SamplingSpec(mode="stride", frames=10, stride=3)) == "stride 10f/3"


def test_crop_text_gives_scale_size_and_shape():
    assert _crop_text(CropSpec(scale=1.3, size=256, square=True, align="none")) == "1.3x/256/square"
