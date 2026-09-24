import gzip
import json

from tests.unit.protocols.conftest import (
    make_pack,
    register_packs,
    write_release_mismatch_dataset,
    write_toyone_dataset,
)

from dfwb.core.records import BuilderRef, InventoryRecord, VideoRecord, read_jsonl, write_jsonl


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
