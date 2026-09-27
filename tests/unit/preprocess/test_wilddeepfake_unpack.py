"""``unpack_wilddeepfake``: safely turning the release's tar shards into the builder's tree.

Every archive here is synthetic, built by ``tests/unit/preprocess/_wdf_archives.py``: no test
reads or writes the machine's real, read-only copy of WildDeepfake.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest
from tests.unit.preprocess._wdf_archives import write_raw_shard, write_realistic_shard, write_shard

from dfwb.core.errors import ConfigError, ContractError
from dfwb.preprocess.inventory.builders.wilddeepfake import WildDeepfakeBuilder
from dfwb.preprocess.wilddeepfake_unpack import unpack_wilddeepfake

_OG = "224w_224h_wild_precropped"
REAL_DIR = f"original_content/real/frames/{_OG}"
FAKE_DIR = f"manipulated_content/fake/frames/{_OG}"


@pytest.fixture
def builder() -> WildDeepfakeBuilder:
    return WildDeepfakeBuilder()


def _frame_files(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


# ------------------------------------------------------------------------------ happy path


def test_unpacks_shards_into_the_builders_tree(tmp_path, builder):
    archives = tmp_path / "archives"
    write_shard(
        archives / "real_train" / "6.tar.gz",
        "6",
        "real",
        [
            ("54", "0.png", b"r0"),
            ("54", "1.PNG", b"r1"),
            ("54", "2.png", b"r2"),
            ("7", "000005.png", b"r5"),
        ],
    )
    write_shard(
        archives / "fake_test" / "100.tar.gz",
        "100",
        "fake",
        [("0", "0.png", b"f0"), ("0", "1.png", b"f1")],
    )
    to = tmp_path / "WildDeepfake"

    result = unpack_wilddeepfake(builder, archives, to)

    assert set(result.categories) == {"real_train", "fake_test"}
    assert result.shards_found == 2
    assert result.sequences_total == 3
    assert result.sequences_written == 3
    assert result.sequences_skipped == 0
    assert result.frames_written == 6
    assert result.by_task == {"REAL": 2, "FAKE": 1}

    real_54 = to / REAL_DIR / "real_train_6_54"
    assert _frame_files(real_54) == {
        "000000.png": b"r0",
        "000001.png": b"r1",
        "000002.png": b"r2",
    }
    real_7 = to / REAL_DIR / "real_train_6_7"
    assert _frame_files(real_7) == {"000005.png": b"r5"}
    fake_0 = to / FAKE_DIR / "fake_test_100_0"
    assert _frame_files(fake_0) == {"000000.png": b"f0", "000001.png": b"f1"}


def test_frame_names_are_normalised(tmp_path, builder):
    archives = tmp_path / "archives"
    write_shard(
        archives / "real_test" / "9.tar.gz",
        "9",
        "real",
        [("1", "1919.PNG", b"a"), ("1", "notes.png", b"b"), ("1", "7.png", b"c")],
    )
    to = tmp_path / "WildDeepfake"

    unpack_wilddeepfake(builder, archives, to)

    frame_dir = to / REAL_DIR / "real_test_9_1"
    assert _frame_files(frame_dir) == {
        "001919.png": b"a",
        "notes.png": b"b",
        "000007.png": b"c",
    }


def test_a_realistic_archive_with_dot_slash_paths_and_directory_entries(tmp_path, builder):
    """Built the way a plain ``tar -cf`` of a real directory tree actually shapes it: every
    member (files and directories alike) prefixed ``./``, with an explicit directory entry for
    the shard, the label and each sequence -- not just the frame files."""
    archives = tmp_path / "archives"
    write_realistic_shard(
        archives / "real_train" / "6.tar.gz",
        "6",
        "real",
        [("54", "0.png", b"r0"), ("54", "1.png", b"r1"), ("7", "000005.png", b"r5")],
    )
    to = tmp_path / "WildDeepfake"

    result = unpack_wilddeepfake(builder, archives, to)

    assert result.warnings == ()
    assert result.sequences_written == 2
    assert result.frames_written == 3
    assert _frame_files(to / REAL_DIR / "real_train_6_54") == {
        "000000.png": b"r0",
        "000001.png": b"r1",
    }
    assert _frame_files(to / REAL_DIR / "real_train_6_7") == {"000005.png": b"r5"}


def test_only_present_categories_are_scanned(tmp_path, builder):
    archives = tmp_path / "archives"
    write_shard(archives / "real_train" / "1.tar.gz", "1", "real", [("1", "0.png", b"x")])
    to = tmp_path / "WildDeepfake"

    result = unpack_wilddeepfake(builder, archives, to)

    assert result.categories == ("real_train",)
    assert result.by_task == {"REAL": 1, "FAKE": 0}
    assert not (to / FAKE_DIR).exists()


# ------------------------------------------------------------------------------ resumability


def test_already_complete_sequences_are_skipped_on_a_second_run(tmp_path, builder):
    archives = tmp_path / "archives"
    write_shard(
        archives / "real_train" / "6.tar.gz",
        "6",
        "real",
        [("54", "0.png", b"r0"), ("54", "1.png", b"r1")],
    )
    to = tmp_path / "WildDeepfake"
    unpack_wilddeepfake(builder, archives, to)
    frame_path = to / REAL_DIR / "real_train_6_54" / "000000.png"
    before = frame_path.stat().st_mtime_ns

    result = unpack_wilddeepfake(builder, archives, to)

    assert result.sequences_written == 0
    assert result.sequences_skipped == 1
    assert frame_path.stat().st_mtime_ns == before  # never rewritten


def test_a_partial_sequence_is_redone_without_leaving_stale_files(tmp_path, builder):
    archives = tmp_path / "archives"
    write_shard(
        archives / "real_train" / "6.tar.gz",
        "6",
        "real",
        [("54", "0.png", b"r0"), ("54", "1.png", b"r1"), ("54", "2.png", b"r2")],
    )
    to = tmp_path / "WildDeepfake"
    # Simulate an interrupted previous run: only one of the three frames landed, plus a stray
    # file that is not part of this archive at all.
    partial = to / REAL_DIR / "real_train_6_54"
    partial.mkdir(parents=True)
    (partial / "000000.png").write_bytes(b"stale")
    (partial / "999999.png").write_bytes(b"stray")

    result = unpack_wilddeepfake(builder, archives, to)

    assert result.sequences_written == 1
    assert result.sequences_skipped == 0
    assert _frame_files(partial) == {
        "000000.png": b"r0",
        "000001.png": b"r1",
        "000002.png": b"r2",
    }


# ------------------------------------------------------------------------------ safety


def _member(name: str, **kwargs) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    for key, value in kwargs.items():
        setattr(info, key, value)
    return info


def test_refuses_an_absolute_path_member_and_writes_nothing(tmp_path, builder):
    archives = tmp_path / "archives"
    write_raw_shard(
        archives / "real_train" / "6.tar.gz",
        [
            (_member("6/real/54/0.png"), b"ok"),
            (_member("/etc/passwd"), b"evil"),
        ],
    )
    to = tmp_path / "WildDeepfake"

    with pytest.raises(ContractError, match="/etc/passwd") as excinfo:
        unpack_wilddeepfake(builder, archives, to)
    assert excinfo.value.hint
    # The whole archive is refused: even the otherwise-safe sequence is never written.
    assert not (to / REAL_DIR / "real_train_6_54").exists()


def test_refuses_a_dotdot_path_segment(tmp_path, builder):
    archives = tmp_path / "archives"
    write_raw_shard(
        archives / "real_train" / "6.tar.gz",
        [(_member("6/real/../../../etc/passwd"), b"evil")],
    )
    to = tmp_path / "WildDeepfake"

    with pytest.raises(ContractError, match=r"\.\."):
        unpack_wilddeepfake(builder, archives, to)


def test_refuses_a_symlink_member(tmp_path, builder):
    archives = tmp_path / "archives"
    link = _member("6/real/54/0.png", type=tarfile.SYMTYPE, linkname="/etc/passwd")
    write_raw_shard(archives / "real_train" / "6.tar.gz", [(link, None)])
    to = tmp_path / "WildDeepfake"

    with pytest.raises(ContractError, match="symlink"):
        unpack_wilddeepfake(builder, archives, to)


def test_refuses_a_hardlink_member(tmp_path, builder):
    archives = tmp_path / "archives"
    hard = _member("6/real/54/1.png", type=tarfile.LNKTYPE, linkname="6/real/54/0.png")
    write_raw_shard(
        archives / "real_train" / "6.tar.gz",
        [(_member("6/real/54/0.png"), b"ok"), (hard, None)],
    )
    to = tmp_path / "WildDeepfake"

    with pytest.raises(ContractError, match="hard link"):
        unpack_wilddeepfake(builder, archives, to)


def test_refuses_a_corrupt_archive(tmp_path, builder):
    archive = tmp_path / "archives" / "real_train" / "6.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"not actually a tar file")
    to = tmp_path / "WildDeepfake"

    with pytest.raises(ContractError, match="not a readable tar archive") as excinfo:
        unpack_wilddeepfake(builder, tmp_path / "archives", to)
    assert excinfo.value.hint


def test_refuses_a_device_file_member(tmp_path, builder):
    archives = tmp_path / "archives"
    device = _member("6/real/54/0.png", type=tarfile.CHRTYPE, devmajor=1, devminor=5)
    write_raw_shard(archives / "real_train" / "6.tar.gz", [(device, None)])
    to = tmp_path / "WildDeepfake"

    with pytest.raises(ContractError, match="device"):
        unpack_wilddeepfake(builder, archives, to)


def test_refuses_colliding_normalised_frame_names(tmp_path, builder):
    archives = tmp_path / "archives"
    write_shard(
        archives / "real_train" / "6.tar.gz",
        "6",
        "real",
        [("54", "07.png", b"a"), ("54", "7.png", b"b")],
    )
    to = tmp_path / "WildDeepfake"

    with pytest.raises(ContractError, match=r"000007\.png"):
        unpack_wilddeepfake(builder, archives, to)


def test_a_mismatched_inner_label_is_a_warning_not_a_refusal(tmp_path, builder):
    """The maintainer's own build of the release keys every frame by its *category* folder's
    label, regardless of what the shard's internal <label> folder says; an unpacker that refused
    the archive here would refuse real, correctly-built releases."""
    archives = tmp_path / "archives"
    write_shard(archives / "real_train" / "6.tar.gz", "6", "fake", [("54", "0.png", b"x")])
    to = tmp_path / "WildDeepfake"

    result = unpack_wilddeepfake(builder, archives, to)

    # Unpacked under the category's label (real), not the shard's inner one (fake).
    assert result.sequences_written == 1
    assert _frame_files(to / REAL_DIR / "real_train_6_54") == {"000000.png": b"x"}
    assert not (to / FAKE_DIR).exists()
    assert len(result.warnings) == 1
    assert "fake" in result.warnings[0]
    assert "real" in result.warnings[0]
    assert "6.tar.gz" in result.warnings[0]


def test_no_warnings_on_a_consistent_archive(tmp_path, builder):
    archives = tmp_path / "archives"
    write_shard(archives / "real_train" / "6.tar.gz", "6", "real", [("54", "0.png", b"x")])
    to = tmp_path / "WildDeepfake"

    result = unpack_wilddeepfake(builder, archives, to)

    assert result.warnings == ()


def test_non_frame_members_are_ignored_not_refused(tmp_path, builder):
    archives = tmp_path / "archives"
    write_raw_shard(
        archives / "real_train" / "6.tar.gz",
        [
            (_member("6/real/54/0.png"), b"ok"),
            (_member("6/real/54/notes.txt"), b"metadata, not a frame"),
            (_member("README.txt"), b"top-level, not shaped like a frame"),
        ],
    )
    to = tmp_path / "WildDeepfake"

    result = unpack_wilddeepfake(builder, archives, to)

    assert result.frames_written == 1
    assert _frame_files(to / REAL_DIR / "real_train_6_54") == {"000000.png": b"ok"}


def test_directory_members_are_ignored_not_refused(tmp_path, builder):
    """A real ``tar`` archive typically has an explicit entry for each directory it holds; those
    are safe (checked like any other member) but carry no frame of their own, and must not be
    mistaken for one."""
    archives = tmp_path / "archives"
    write_raw_shard(
        archives / "real_train" / "6.tar.gz",
        [
            (_member("6/", type=tarfile.DIRTYPE), None),
            (_member("6/real/", type=tarfile.DIRTYPE), None),
            (_member("6/real/54/", type=tarfile.DIRTYPE), None),
            (_member("6/real/54/0.png"), b"ok"),
        ],
    )
    to = tmp_path / "WildDeepfake"

    result = unpack_wilddeepfake(builder, archives, to)

    assert result.frames_written == 1
    assert _frame_files(to / REAL_DIR / "real_train_6_54") == {"000000.png": b"ok"}


# ------------------------------------------------------------------------------ bad input


def test_from_dir_must_exist(tmp_path, builder):
    with pytest.raises(ConfigError, match="not a directory") as excinfo:
        unpack_wilddeepfake(builder, tmp_path / "nope", tmp_path / "WildDeepfake")
    assert excinfo.value.hint


def test_no_category_folders_found(tmp_path, builder):
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / "something-else").mkdir()
    to = tmp_path / "WildDeepfake"

    with pytest.raises(ConfigError, match="real_train") as excinfo:
        unpack_wilddeepfake(builder, archives, to)
    assert excinfo.value.hint
    assert not to.exists()


def test_no_shard_files_found(tmp_path, builder):
    archives = tmp_path / "archives"
    (archives / "real_train").mkdir(parents=True)
    to = tmp_path / "WildDeepfake"

    with pytest.raises(ConfigError, match=r"tar\.gz") as excinfo:
        unpack_wilddeepfake(builder, archives, to)
    assert excinfo.value.hint
    # Refused before writing anything: --to is never even created.
    assert not to.exists()


def test_to_must_not_be_an_existing_file(tmp_path, builder):
    archives = tmp_path / "archives"
    write_shard(archives / "real_train" / "1.tar.gz", "1", "real", [("1", "0.png", b"x")])
    to = tmp_path / "WildDeepfake"
    to.write_text("not a directory")

    with pytest.raises(ConfigError, match="not a directory") as excinfo:
        unpack_wilddeepfake(builder, archives, to)
    assert excinfo.value.hint
