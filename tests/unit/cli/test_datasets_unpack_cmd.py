"""``dfwb datasets unpack wilddeepfake``: the public, safe archive unpacker.

Every archive is synthetic (``tests/unit/preprocess/_wdf_archives.py``); no test reads or writes
the machine's real, read-only copy of WildDeepfake.
"""

from __future__ import annotations

import json
import tarfile

from tests.unit.preprocess._wdf_archives import write_raw_shard, write_shard

_OG = "224w_224h_wild_precropped"
REAL_DIR = f"original_content/real/frames/{_OG}"
FAKE_DIR = f"manipulated_content/fake/frames/{_OG}"


def _write_sample_archives(archives):
    write_shard(
        archives / "real_train" / "6.tar.gz",
        "6",
        "real",
        [("54", "0.png", b"r0"), ("54", "1.png", b"r1")],
    )
    write_shard(archives / "fake_test" / "100.tar.gz", "100", "fake", [("0", "0.png", b"f0")])


def test_unpack_writes_the_tree_and_reports_counts(run, tmp_path, monkeypatch):
    archives = tmp_path / "archives"
    _write_sample_archives(archives)
    root = tmp_path / "out"
    to = root / "WildDeepfake"
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(root))

    result = run(
        "datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(root), "--json"
    )
    assert result.code == 0, result.err
    data = json.loads(result.out)
    assert data["dataset_id"] == "wilddeepfake"
    assert data["datasets_root"] == str(root)
    assert data["to"] == str(to)
    assert data["inventory_folder"] == str(to)
    assert sorted(data["categories"]) == ["fake_test", "real_train"]
    assert data["shards_found"] == 2
    assert data["sequences_written"] == 2
    assert data["sequences_skipped"] == 0
    assert data["frames_written"] == 3
    assert data["by_task"] == {"REAL": 1, "FAKE": 1}
    assert data["warnings"] == {"count": 0, "first": []}
    assert data["layout_warnings"] == []
    assert (to / REAL_DIR / "real_train_6_54" / "000000.png").read_bytes() == b"r0"
    assert (to / FAKE_DIR / "fake_test_100_0" / "000000.png").read_bytes() == b"f0"


def test_unpack_prints_a_plain_summary_without_json(run, tmp_path, monkeypatch):
    archives = tmp_path / "archives"
    _write_sample_archives(archives)
    root = tmp_path / "out"
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(root))

    result = run("datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(root))
    assert result.code == 0, result.err
    assert str(root / "WildDeepfake") in result.out
    assert "REAL 1" in result.out
    assert "FAKE 1" in result.out
    assert "warning" not in result.out + result.err


def test_after_unpack_to_a_root_inventory_build_with_that_root_finds_the_records(
    run, tmp_path, monkeypatch
):
    archives = tmp_path / "archives"
    _write_sample_archives(archives)
    root = tmp_path / "datasets"
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(root))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(tmp_path / "work"))

    unpacked = run(
        "datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(root), "--json"
    )
    assert unpacked.code == 0, unpacked.err
    built = run("inventory", "build", "wilddeepfake", "--json")
    assert built.code == 0, built.err

    found = json.loads(unpacked.out)["by_task"]
    inventory = json.loads(built.out)
    assert inventory["dataset_dir"] == str(root / "WildDeepfake")
    assert inventory["by_task"] == found == {"REAL": 1, "FAKE": 1}


def test_unpack_warns_when_inventory_build_would_not_look_there(run, tmp_path):
    archives = tmp_path / "archives"
    _write_sample_archives(archives)
    root = tmp_path / "datasets"  # no DFWB_DATASETS_ROOT names it

    result = run("datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(root))
    assert result.code == 0, result.err
    assert "warning: `dfwb inventory build wilddeepfake` will not find" in result.err
    assert f"DFWB_DATASETS_ROOT={root}" in result.err

    data = json.loads(
        run(
            "datasets",
            "unpack",
            "wilddeepfake",
            "--from",
            str(archives),
            "--to",
            str(root),
            "--json",
        ).out
    )
    assert data["inventory_folder"] is None
    assert data["by_task"] is None


def test_unpack_help_says_to_is_a_datasets_root(run):
    result = run("datasets", "unpack", "--help")
    assert result.code == 0
    text = " ".join(result.out.split())
    assert "--to DIRECTORY The datasets root to unpack into" in text


def test_unpack_is_resumable_across_two_invocations(run, tmp_path):
    archives = tmp_path / "archives"
    _write_sample_archives(archives)
    to = tmp_path / "out"

    first = run(
        "datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(to), "--json"
    )
    assert first.code == 0, first.err
    assert json.loads(first.out)["sequences_written"] == 2

    second = run(
        "datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(to), "--json"
    )
    assert second.code == 0, second.err
    data = json.loads(second.out)
    assert data["sequences_written"] == 0
    assert data["sequences_skipped"] == 2


def test_unpack_defaults_to_is_the_first_datasets_root(run, tmp_path, monkeypatch):
    archives = tmp_path / "archives"
    _write_sample_archives(archives)
    datasets_root = tmp_path / "data"
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(datasets_root))

    result = run("datasets", "unpack", "wilddeepfake", "--from", str(archives), "--json")
    assert result.code == 0, result.err
    data = json.loads(result.out)
    assert data["to"] == str(datasets_root / "WildDeepfake")
    assert (datasets_root / "WildDeepfake" / REAL_DIR / "real_train_6_54").is_dir()


def test_unpack_reports_a_mismatched_inner_label_as_a_warning_not_a_failure(run, tmp_path):
    archives = tmp_path / "archives"
    write_shard(archives / "real_train" / "6.tar.gz", "6", "fake", [("54", "0.png", b"x")])
    root = tmp_path / "out"
    to = root / "WildDeepfake"

    json_result = run(
        "datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(root), "--json"
    )
    assert json_result.code == 0, json_result.err
    data = json.loads(json_result.out)
    assert data["sequences_written"] == 1
    assert data["warnings"]["count"] == 1
    assert len(data["warnings"]["first"]) == 1
    assert "fake" in data["warnings"]["first"][0]
    assert (to / REAL_DIR / "real_train_6_54" / "000000.png").read_bytes() == b"x"

    plain_result = run(
        "datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(root)
    )
    assert plain_result.code == 0, plain_result.err
    assert "warning: 1 frame(s)" in plain_result.err
    assert "fake" in plain_result.err


def test_unpack_creates_nothing_when_there_are_no_shards(run, tmp_path):
    archives = tmp_path / "archives"
    (archives / "real_train").mkdir(parents=True)
    root = tmp_path / "out"

    result = run("datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(root))
    assert result.code == 2
    assert "hint: " in result.err
    assert not root.exists()


def test_unpack_from_a_missing_directory_fails_with_a_hint(run, tmp_path):
    result = run(
        "datasets",
        "unpack",
        "wilddeepfake",
        "--from",
        str(tmp_path / "nope"),
        "--to",
        str(tmp_path / "datasets"),
    )
    assert result.code == 2
    assert "not a directory" in result.err
    assert "hint: " in result.err


def test_unpack_refuses_an_unsafe_archive_member_with_a_hint(run, tmp_path):
    archives = tmp_path / "archives"
    info = tarfile.TarInfo(name="/etc/passwd")
    write_raw_shard(archives / "real_train" / "6.tar.gz", [(info, b"evil")])

    result = run(
        "datasets",
        "unpack",
        "wilddeepfake",
        "--from",
        str(archives),
        "--to",
        str(tmp_path / "WildDeepfake"),
    )
    assert result.code == 4
    assert "/etc/passwd" in result.err
    assert "hint: " in result.err


def test_unpack_only_knows_wilddeepfake(run, tmp_path):
    result = run("datasets", "unpack", "ffpp", "--from", str(tmp_path))
    assert result.code == 2
    assert "wilddeepfake" in result.err
    assert "hint: " in result.err


def test_unpack_requires_from(run, tmp_path):
    result = run("datasets", "unpack", "wilddeepfake")
    assert result.code == 2
    assert "hint: " in result.err
