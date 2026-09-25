import json

from tests.unit.preprocess.inventory._demo import DEMO_TARGET, install, make_demo_tree
from tests.unit.protocols.conftest import make_pack


def test_list_json_works_with_no_roots_set(run, monkeypatch):
    install(monkeypatch)
    monkeypatch.delenv("DFWB_DATASET_DEMO", raising=False)
    result = run("datasets", "list", "--json")
    assert result.code == 0, result.err
    assert json.loads(result.out) == [
        {
            "id": "demo",
            "name": "Demo",
            "folder": "Demo",
            "path": None,
            "source": "not found",
            "also_found": [],
            "packs": [],
        }
    ]


def test_list_never_imports_a_builder(run, monkeypatch):
    # The target does not exist: listing must still work, from registry metadata alone.
    install(monkeypatch, {"ghost": ("tests.unit.preprocess.inventory.nowhere:Ghost", "Ghost", "G")})
    result = run("datasets", "list")
    assert result.code == 0, result.err
    assert result.out.splitlines()[0].split() == ["ID", "NAME", "FOLDER", "FOUND", "AT", "PACKS"]
    assert "ghost" in result.out
    assert "—" in result.out


def test_list_shows_the_folder_and_the_packs(run, monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    make_demo_tree(raw / "Demo", compressions=("c23",))
    pack = make_pack(tmp_path, "demo-pack", {"demo": {}, "packonly": {}})
    install(monkeypatch, {"demo": (DEMO_TARGET, "Demo", "Demo")}, packs={"demo-pack": pack})
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(raw))
    rows = {row["id"]: row for row in json.loads(run("datasets", "list", "--json").out)}
    assert rows["demo"]["path"] == str(raw / "Demo")
    assert rows["demo"]["source"] == "root 1"
    assert rows["demo"]["packs"] == ["demo-pack"]
    assert rows["packonly"] == {
        "id": "packonly",
        "name": None,
        "folder": None,
        "path": None,
        "source": "not found",
        "also_found": [],
        "packs": ["demo-pack"],
    }
    text = run("datasets", "list").out
    assert str(raw / "Demo") in text
    assert "demo-pack" in text


def test_info_shows_the_card_layout_folder_and_schemes(run, monkeypatch, tmp_path):
    raw, second = tmp_path / "raw", tmp_path / "second"
    make_demo_tree(raw / "Demo", compressions=("c23",))
    (second / "Demo").mkdir(parents=True)
    pack = make_pack(tmp_path, "demo-pack", {"demo": {}})
    install(monkeypatch, packs={"demo-pack": pack})
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    result = run("datasets", "info", "demo")
    assert result.code == 0, result.err
    assert "Demo" in result.out
    assert "originals/{cX}/" in result.out  # the layout text
    assert f"root 1: {raw / 'Demo'}" in result.out
    assert "videos: originals/c23, swapped/c23" in result.out
    assert f"root 2: {second / 'Demo'}" in result.out
    assert "videos: no videos found" in result.out
    assert "demo-pack" in result.out
    assert "official" in result.out

    data = json.loads(run("datasets", "info", "demo", "--json").out)
    assert data["id"] == "demo"
    assert data["builder"] == {"id": "demo", "version": "1"}
    assert data["card"]["name"] == "Demo"
    assert "swapped/{cX}/" in data["layout"]
    assert data["location"] == {
        "source": "root 1",
        "problem": None,
        "copies": [
            {
                "path": str(raw / "Demo"),
                "source": "root 1",
                "video_dirs": ["originals/c23", "swapped/c23"],
            },
            {"path": str(second / "Demo"), "source": "root 2", "video_dirs": []},
        ],
    }
    assert data["schemes"] == [
        {"pack": "demo-pack", "scheme": "official", "kind": "official", "default": True}
    ]


def test_info_lists_every_copy_and_its_populated_video_dirs(run, monkeypatch, tmp_path):
    # root 1 has the folder but no video (e.g. a processed copy); root 2 has the raw layout.
    raw, second = tmp_path / "raw", tmp_path / "second"
    (raw / "Demo").mkdir(parents=True)
    make_demo_tree(second / "Demo", compressions=("c23",))
    install(monkeypatch)
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")

    result = run("datasets", "info", "demo")
    assert result.code == 0, result.err
    assert f"root 1: {raw / 'Demo'}" in result.out
    assert f"root 2: {second / 'Demo'}" in result.out
    assert "videos: no videos found" in result.out
    assert "videos: originals/c23, swapped/c23" in result.out

    data = json.loads(run("datasets", "info", "demo", "--json").out)
    assert data["location"] == {
        "source": "root 1",
        "problem": None,
        "copies": [
            {"path": str(raw / "Demo"), "source": "root 1", "video_dirs": []},
            {
                "path": str(second / "Demo"),
                "source": "root 2",
                "video_dirs": ["originals/c23", "swapped/c23"],
            },
        ],
    }


def test_info_when_the_folder_is_missing_and_no_pack_publishes_it(run, monkeypatch):
    install(monkeypatch)
    result = run("datasets", "info", "demo")
    assert result.code == 0, result.err
    assert "not found" in result.out
    assert "no installed protocol pack publishes demo" in result.out


def test_info_of_an_unknown_dataset_suggests_close_matches(run, monkeypatch):
    install(monkeypatch)
    result = run("datasets", "info", "dmeo")
    assert result.code == 2
    assert "did you mean 'demo'" in result.err
