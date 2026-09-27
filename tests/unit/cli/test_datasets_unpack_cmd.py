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


def test_unpack_writes_the_tree_and_reports_counts(run, tmp_path):
    archives = tmp_path / "archives"
    _write_sample_archives(archives)
    to = tmp_path / "out" / "WildDeepfake"

    result = run(
        "datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(to), "--json"
    )
    assert result.code == 0, result.err
    data = json.loads(result.out)
    assert data["dataset_id"] == "wilddeepfake"
    assert data["to"] == str(to)
    assert sorted(data["categories"]) == ["fake_test", "real_train"]
    assert data["shards_found"] == 2
    assert data["sequences_written"] == 2
    assert data["sequences_skipped"] == 0
    assert data["frames_written"] == 3
    assert data["by_task"] == {"REAL": 1, "FAKE": 1}
    assert (to / REAL_DIR / "real_train_6_54" / "000000.png").read_bytes() == b"r0"
    assert (to / FAKE_DIR / "fake_test_100_0" / "000000.png").read_bytes() == b"f0"


def test_unpack_prints_a_plain_summary_without_json(run, tmp_path):
    archives = tmp_path / "archives"
    _write_sample_archives(archives)
    to = tmp_path / "out" / "WildDeepfake"

    result = run("datasets", "unpack", "wilddeepfake", "--from", str(archives), "--to", str(to))
    assert result.code == 0, result.err
    assert str(to) in result.out
    assert "REAL 1" in result.out
    assert "FAKE 1" in result.out


def test_unpack_is_resumable_across_two_invocations(run, tmp_path):
    archives = tmp_path / "archives"
    _write_sample_archives(archives)
    to = tmp_path / "out" / "WildDeepfake"

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


def test_unpack_from_a_missing_directory_fails_with_a_hint(run, tmp_path):
    result = run(
        "datasets",
        "unpack",
        "wilddeepfake",
        "--from",
        str(tmp_path / "nope"),
        "--to",
        str(tmp_path / "WildDeepfake"),
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
