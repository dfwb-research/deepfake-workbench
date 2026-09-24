import gzip
import json

from tests.unit.protocols.conftest import make_pack, register_packs, write_toyone_dataset


def _install_toyone(monkeypatch, tmp_path):
    root = make_pack(
        tmp_path, "toyone-pack", {"toyone": {}}, builders={"toyone": write_toyone_dataset}
    )
    register_packs(monkeypatch, {"toyone-pack": root})
    return root


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
