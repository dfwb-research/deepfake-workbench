import gzip
import json
from pathlib import Path

import yaml
from tests.unit.protocols.conftest import (
    make_pack,
    register_packs,
    write_release_mismatch_dataset,
    write_toyone_dataset,
)

from dfwb.core.records import (
    BuilderRef,
    DatasetCard,
    InventoryRecord,
    LabelVocab,
    PackCard,
    PackProvenance,
    SplitRow,
    VideoRecord,
    read_jsonl,
    write_jsonl,
)
from dfwb.core.records.protocol import LabelMappingSpec, LicenseInfo
from dfwb.protocols.writer import scheme_card_for, write_dataset_files


def _install_toyone(monkeypatch, tmp_path):
    root = make_pack(
        tmp_path, "toyone-pack", {"toyone": {}}, builders={"toyone": write_toyone_dataset}
    )
    register_packs(monkeypatch, {"toyone-pack": root})
    return root


def _install_release(monkeypatch, tmp_path):
    root = make_pack(
        tmp_path,
        "release-pack",
        {"release": {}},
        builders={"release": write_release_mismatch_dataset},
    )
    register_packs(monkeypatch, {"release-pack": root})
    return root


def _inventory_record(video: VideoRecord, **overrides: object) -> InventoryRecord:
    fields: dict[str, object] = {
        "key": video.key,
        "compression": video.compression,
        "label_key": video.label_key,
        "method": video.method,
        "relpath": f"{video.key}.mp4",
        "builder": BuilderRef("fixture", "1"),
        "identity": video.identity,
        "source_id": video.source_id,
        "target_id": video.target_id,
        "pair_key": video.pair_key,
        "attrs": dict(video.attrs),
    }
    fields.update(overrides)
    return InventoryRecord(**fields)


def test_list_empty(run):
    result = run("protocols", "list")
    assert result.code == 0


def test_list_human_marks_default(run, monkeypatch, tmp_path):
    _install_toyone(monkeypatch, tmp_path)
    result = run("protocols", "list")
    assert result.code == 0
    assert "PROTOCOL" in result.out
    assert "toyone/official*" in result.out
    assert "toyone/all-test" in result.out
    assert "toyone/all-test*" not in result.out


def test_list_json(run, monkeypatch, tmp_path):
    _install_toyone(monkeypatch, tmp_path)
    data = json.loads(run("protocols", "list", "--json").out)
    rows = {(r["dataset_id"], r["scheme"]): r for r in data}
    assert rows[("toyone", "official")]["default"] is True
    assert rows[("toyone", "official")]["pack"] == "toyone-pack"
    assert rows[("toyone", "official")]["counts"] == {"train": 6, "val": 4, "test": 4}
    assert rows[("toyone", "all-test")]["default"] is False


def test_info_human_and_json(run, monkeypatch, tmp_path):
    _install_toyone(monkeypatch, tmp_path)
    result = run("protocols", "info", "toyone/official")
    assert result.code == 0
    assert "toyone/official" in result.out

    data = json.loads(run("protocols", "info", "toyone/official", "--json").out)
    assert data["ref"] == "toyone/official"
    assert data["pack"] == "toyone-pack"
    assert data["scheme_card"]["kind"] == "official"
    total = sum(row["count"] for row in data["counts"])
    assert total == 14  # 16 rows minus REAL/r4's two unassigned rows


def test_info_unknown_dataset_exits_2_with_hint(run):
    result = run("protocols", "info", "nope")
    assert result.code == 2
    assert "hint: " in result.err


def test_info_exits_4_on_a_malformed_video_row(run, monkeypatch, tmp_path):
    root = _install_toyone(monkeypatch, tmp_path)
    videos_path = root / "toyone" / "videos.jsonl.gz"
    with gzip.open(videos_path, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    del_data = json.loads(lines[0])
    del del_data["key"]
    lines[0] = json.dumps(del_data)
    with gzip.open(videos_path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    result = run("protocols", "info", "toyone/official")
    assert result.code == 4
    assert "videos.jsonl.gz:1: missing 'key'" in result.err
    assert "hint: " in result.err


def test_verify_cli_exit_codes_and_json(run, monkeypatch, tmp_path):
    root = _install_toyone(monkeypatch, tmp_path)
    work_root = tmp_path / "work"
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work_root))
    inventory_path = work_root / "toyone" / "inventory.jsonl"
    inventory_path.parent.mkdir(parents=True)
    videos = read_jsonl(root / "toyone" / "videos.jsonl.gz", VideoRecord)

    write_jsonl(inventory_path, [_inventory_record(v) for v in videos])
    result = run("protocols", "verify", "toyone")
    assert result.code == 0
    assert "have: 16" in result.out
    assert "report written to" in result.out
    report_path = work_root / "toyone" / "verify" / "official.json"
    assert report_path.is_file()

    # Drop a train-split video: exits 3 (partial coverage), still writes the report.
    write_jsonl(inventory_path, [_inventory_record(v) for v in videos if v.key != "FAKE_A/a1"])
    result = run("protocols", "verify", "toyone")
    assert result.code == 3
    assert "missing_requested: 1" in result.out

    # Relabel a video: exits 4 (contract mismatch).
    write_jsonl(
        inventory_path,
        [
            _inventory_record(v, method="Wrong") if v.key == "FAKE_A/a1" else _inventory_record(v)
            for v in videos
        ],
    )
    result = run("protocols", "verify", "toyone/official", "--json")
    assert result.code == 4
    data = json.loads(result.out)
    assert data["counts"]["label_mismatch"] == 1
    assert data["exit_code"] == 4
    assert data["report_path"] == str(report_path)


def test_verify_cli_prints_release_mismatch_warning(run, monkeypatch, tmp_path):
    root = _install_release(monkeypatch, tmp_path)
    work_root = tmp_path / "work"
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work_root))
    videos = read_jsonl(root / "release" / "videos.jsonl.gz", VideoRecord)
    kept = [v for v in videos if int(v.key.rsplit("v", 1)[1]) <= 20]
    records = [_inventory_record(v) for v in kept]
    for i in range(1, 11):
        records.append(
            InventoryRecord(
                key=f"FAKE_A/x{i:02d}",
                compression=None,
                label_key="RELEASE-FAKE_A",
                method="FakeA",
                relpath=f"FAKE_A/x{i:02d}.mp4",
                builder=BuilderRef("fixture", "1"),
            )
        )
    inventory_path = work_root / "release" / "inventory.jsonl"
    inventory_path.parent.mkdir(parents=True)
    write_jsonl(inventory_path, records)

    result = run("protocols", "verify", "release")

    assert "warning: FAKE_A: 10 missing and 10 extra" in result.out


def test_verify_without_inventory_hints_inventory_build(run, monkeypatch, tmp_path):
    _install_toyone(monkeypatch, tmp_path)
    monkeypatch.setenv("DFWB_WORK_ROOT", str(tmp_path / "work"))

    result = run("protocols", "verify", "toyone")

    assert result.code == 2
    assert "hint: run: dfwb inventory build toyone" in result.err


def test_verify_cli_rejects_unknown_split_with_suggestion(run, monkeypatch, tmp_path):
    root = _install_toyone(monkeypatch, tmp_path)
    work_root = tmp_path / "work"
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work_root))
    inventory_path = work_root / "toyone" / "inventory.jsonl"
    inventory_path.parent.mkdir(parents=True)
    videos = read_jsonl(root / "toyone" / "videos.jsonl.gz", VideoRecord)
    write_jsonl(inventory_path, [_inventory_record(v) for v in videos])

    result = run("protocols", "verify", "toyone", "--split", "tset")

    assert result.code == 2
    assert "split 'tset' is not in this scheme" in result.err
    assert "did you mean 'test'" in result.err
    assert "hint: splits in this scheme: test, train, val" in result.err


# -------------------------------------------------------------------------------------------
# `dfwb protocols lint`: pack integrity checks. ``lint``/``diff`` take a pack directory path
# directly, so no registry install is needed.
# -------------------------------------------------------------------------------------------


def _write_lint_pack(root: Path, *, distribution: str = "list") -> Path:
    dataset_dir = root / "toy"
    rows = [SplitRow("REAL/r1", None, "train"), SplitRow("REAL/r2", None, "test")]
    scheme = scheme_card_for(rows, kind="official", rule="official")
    card = DatasetCard(
        id="toy",
        name="toy",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only; never media",
        distribution=distribution,
        modalities=["video"],
        key_rule="<task>/<stem>",
        schemes={"official": scheme},
        default_scheme="official",
    )
    labels = LabelVocab(
        vocab={"TOY-REAL": {"binary": 0}}, mappings={"binary": LabelMappingSpec(from_="binary")}
    )
    provenance = PackProvenance(
        builder={"id": "toy", "version": "1"},
        dfwb="0.1.0",
        source_listing_sha256="0" * 64,
        rules={"official": {"rule": "official", "params": {}}},
    )
    write_dataset_files(
        dataset_dir,
        videos=[
            VideoRecord("REAL/r1", None, "TOY-REAL", "original", identity="r1"),
            VideoRecord("REAL/r2", None, "TOY-REAL", "original", identity="r2"),
        ],
        schemes={"official": rows},
        pairs=[],
        card=card,
        labels=labels,
        provenance=provenance,
        notice="# toy\n\nSynthetic fixture; never real media.\n",
    )
    pack_card = PackCard(schema_version=1, name="lint-pack", version="1.0.0", datasets=["toy"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "pack.yaml").write_text(
        yaml.safe_dump(pack_card.model_dump(mode="json", by_alias=True), sort_keys=True)
    )
    return root


def test_lint_cli_warns_on_undecided_distribution_and_exits_0(run, tmp_path):
    root = _write_lint_pack(tmp_path, distribution="undecided")

    result = run("protocols", "lint", str(root))

    assert result.code == 0
    assert "warning: toy/dataset.yaml: distribution is undecided" in result.out


def test_lint_cli_release_flag_turns_the_warning_into_an_error(run, tmp_path):
    root = _write_lint_pack(tmp_path, distribution="undecided")

    result = run("protocols", "lint", str(root), "--release")

    assert result.code == 4
    assert "error: toy/dataset.yaml: distribution is undecided" in result.out


def test_lint_cli_exits_4_when_any_error_is_found(run, tmp_path):
    root = _write_lint_pack(tmp_path)
    (root / "toy" / "NOTICE.md").unlink()

    result = run("protocols", "lint", str(root))

    assert result.code == 4
    assert "error: toy/NOTICE.md: file is missing" in result.out


def test_lint_cli_json_output(run, tmp_path):
    root = _write_lint_pack(tmp_path)

    data = json.loads(run("protocols", "lint", str(root), "--json").out)

    assert data == []


# -------------------------------------------------------------------------------------------
# `dfwb protocols diff`: the required SemVer bump between two pack directories.
# -------------------------------------------------------------------------------------------


def _diff_rows() -> list[SplitRow]:
    return [
        SplitRow("REAL/r1", None, "train"),
        SplitRow("REAL/r2", None, "test"),
        SplitRow("FAKE/f1", None, "train"),
        SplitRow("FAKE/f2", None, "test"),
    ]


def _write_diff_pack(root: Path, version: str, rows: list[SplitRow]) -> Path:
    dataset_dir = root / "diffcli"
    scheme = scheme_card_for(rows, kind="official", rule="official")
    card = DatasetCard(
        id="diffcli",
        name="diffcli",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only; never media",
        distribution="list",
        modalities=["video"],
        key_rule="<task>/<stem>",
        schemes={"official": scheme},
        default_scheme="official",
    )
    labels = LabelVocab(
        vocab={"DIFFCLI-REAL": {"binary": 0}, "DIFFCLI-FAKE": {"binary": 1}},
        mappings={"binary": LabelMappingSpec(from_="binary")},
    )
    provenance = PackProvenance(
        builder={"id": "diffcli", "version": "1"},
        dfwb="0.1.0",
        source_listing_sha256="0" * 64,
        rules={"official": {"rule": "official", "params": {}}},
    )
    write_dataset_files(
        dataset_dir,
        videos=[
            VideoRecord("REAL/r1", None, "DIFFCLI-REAL", "original", identity="r1"),
            VideoRecord("REAL/r2", None, "DIFFCLI-REAL", "original", identity="r2"),
            VideoRecord("FAKE/f1", None, "DIFFCLI-FAKE", "FakeA", identity="f1", target_id="r1"),
            VideoRecord("FAKE/f2", None, "DIFFCLI-FAKE", "FakeA", identity="f2", target_id="r2"),
        ],
        schemes={"official": rows},
        pairs=[],
        card=card,
        labels=labels,
        provenance=provenance,
        notice="# diffcli\n\nSynthetic fixture; never real media.\n",
    )
    pack_card = PackCard(schema_version=1, name="diff-pack", version=version, datasets=["diffcli"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "pack.yaml").write_text(
        yaml.safe_dump(pack_card.model_dump(mode="json", by_alias=True), sort_keys=True)
    )
    return root


def _moved_rows() -> list[SplitRow]:
    return [SplitRow("REAL/r1", None, "test") if r.key == "REAL/r1" else r for r in _diff_rows()]


def test_diff_cli_prints_the_table_and_required_bump(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.0.1", _diff_rows())

    result = run("protocols", "diff", str(old), str(new))

    assert result.code == 0
    assert "diffcli" in result.out
    assert "official" in result.out
    assert "required bump: patch" in result.out


def test_diff_cli_json_reports_a_moved_video_as_a_major_bump(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.1.0", _moved_rows())

    data = json.loads(run("protocols", "diff", str(old), str(new), "--json").out)

    assert data["required_bump"] == "major"
    assert data["schemes"] == [
        {
            "dataset": "diffcli",
            "scheme": "official",
            "status": "changed",
            "added": 0,
            "removed": 0,
            "moved": 1,
        }
    ]


def test_diff_cli_expect_bump_exits_4_when_the_actual_bump_is_smaller_than_required(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.1.0", _moved_rows())  # an actual 1.0.0 -> 1.1.0

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "minor")

    assert result.code == 4


def test_diff_cli_expect_bump_passes_when_the_version_was_bumped_enough(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "2.0.0", _moved_rows())

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "major")

    assert result.code == 0


def test_diff_cli_expect_bump_exits_4_on_a_version_downgrade(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.2.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.1.0", _diff_rows())

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "patch")

    assert result.code == 4
    assert "downgrade" in result.err


def test_diff_cli_expect_bump_exits_4_on_a_malformed_version(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.1.0", _diff_rows())

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "minor")

    assert result.code == 4
    assert "is not MAJOR.MINOR.PATCH" in result.err
