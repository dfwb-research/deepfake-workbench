"""The LAV-DF inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``, and the metadata is a
few hand-written rows in the release's ``metadata.json`` format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.lavdf import LAVDFBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    resolve_pairs,
    task_of,
)

RVRA_DIR = "original_content/RealVideo-RealAudio/videos"
FVFA_DIR = "manipulated_content/FakeVideo-FakeAudio/videos"
FVRA_DIR = "manipulated_content/FakeVideo-RealAudio/videos"
RVFA_DIR = "manipulated_content/RealVideo-FakeAudio/videos"
RELEASE = ".official_files/LAV-DF"
ATTRS = {
    "task_name",
    "n_fakes",
    "fake_periods",
    "modify_video",
    "modify_audio",
    "duration",
    "video_frames",
    "audio_channels",
    "audio_frames",
}

# (stem, modify_video, modify_audio, original stem or None, task folder)
_SPEC = [
    ("000001", False, False, None, RVRA_DIR),
    ("000002", True, True, "000001", FVFA_DIR),
    ("000003", True, False, "000001", FVRA_DIR),
    ("000004", False, True, "000001", RVFA_DIR),
]


def _row(stem: str, video: bool, audio: bool, original: str | None, split: str = "test"):
    return {
        "file": f"{split}/{stem}.mp4",
        "n_fakes": 0 if original is None else 1,
        "fake_periods": [] if original is None else [[2.5, 3.224]],
        "timestamps": [["word", 0.0, 0.4]],
        "duration": 4.8,
        "transcript": "a few words",
        "original": f"{split}/{original}.mp4" if original else None,
        "modify_video": video,
        "modify_audio": audio,
        "split": split,
        "video_frames": 116,
        "audio_channels": 1,
        "audio_frames": 74752,
    }


def _write_metadata(root: Path, rows: Any, name: str = "metadata.json") -> None:
    folder = root / RELEASE
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(rows), encoding="utf-8")


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"\x00")


def _make_synthetic_lavdf_root(tmp_path: Path) -> Path:
    """A tiny LAV-DF tree: one real and three fakes of it, one per task folder, plus metadata."""
    root = tmp_path / "LAV-DF"
    for stem, _video, _audio, _original, folder in _SPEC:
        _touch(root / folder, f"{stem}.mp4")
    _write_metadata(root, [_row(*spec[:4]) for spec in _SPEC])
    return root


@pytest.fixture
def lavdf_root(tmp_path: Path) -> Path:
    return _make_synthetic_lavdf_root(tmp_path)


def _records(root, **kwargs):
    return {rec.key: rec for rec in collect_records(LAVDFBuilder(), root, **kwargs)}


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(lavdf_root):
    # 1 real + 3 fakes.
    assert len(collect_records(LAVDFBuilder(), lavdf_root)) == 4


def test_entry_schema(lavdf_root):
    for rec in collect_records(LAVDFBuilder(), lavdf_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"RVRA", "FVFA", "FVRA", "RVFA"}
        assert rec.builder == BuilderRef("lav-df", LAVDFBuilder.version)
        assert rec.compression is None  # LAV-DF has no compression levels
        assert rec.label_key == f"LAVDF-{task}"
        # The transcript and its word timestamps are large and are not copied.
        assert set(rec.attrs) == ATTRS
        assert not rec.relpath.startswith("/")
        assert rec.folder is None
        assert rec.probe is None
        assert not hasattr(rec, "media")
        assert not hasattr(rec, "label")


def _task_of_folder(folder: str) -> str:
    return {RVRA_DIR: "RVRA", FVFA_DIR: "FVFA", FVRA_DIR: "FVRA", RVFA_DIR: "RVFA"}[folder]


def test_fake_pair_key_is_the_original_stem(lavdf_root):
    records = _records(lavdf_root)
    for stem, video, audio, original, folder in _SPEC:
        if original is None:
            continue
        rec = records[f"{_task_of_folder(folder)}/{stem}"]
        assert rec.pair_key == original
        assert f"RVRA/{original}" in records
        assert rec.attrs["n_fakes"] == 1
        assert rec.attrs["fake_periods"] == [[2.5, 3.224]]
        assert (rec.attrs["modify_video"], rec.attrs["modify_audio"]) == (video, audio)
        assert rec.relpath == f"{folder}/{stem}.mp4"


def test_real_entry_attributes(lavdf_root):
    real = _records(lavdf_root)["RVRA/000001"]
    assert real.pair_key is None
    assert real.method == "original"
    assert real.attrs == {
        "task_name": "RealVideo-RealAudio",
        "n_fakes": 0,
        "fake_periods": [],
        "modify_video": False,
        "modify_audio": False,
        "duration": 4.8,
        "video_frames": 116,
        "audio_channels": 1,
        "audio_frames": 74752,
    }


def test_lavdf_has_no_identity_axis(lavdf_root):
    for rec in collect_records(LAVDFBuilder(), lavdf_root):
        assert (rec.identity, rec.target_id, rec.source_id) == (None, None, None)


def test_every_task_and_method(lavdf_root):
    records = _records(lavdf_root)
    assert {key: rec.method for key, rec in records.items()} == {
        "RVRA/000001": "original",
        "FVFA/000002": "FakeVideo-FakeAudio",
        "FVRA/000003": "FakeVideo-RealAudio",
        "RVFA/000004": "RealVideo-FakeAudio",
    }


def test_the_task_folders_are_searched_recursively(lavdf_root):
    _touch(lavdf_root / FVFA_DIR / "extra", "000009.mp4")
    rec = _records(lavdf_root)["FVFA/000009"]
    assert rec.relpath == f"{FVFA_DIR}/extra/000009.mp4"


def test_the_full_metadata_file_is_preferred_to_the_minified_one(lavdf_root):
    rows = [_row(*spec[:4]) for spec in _SPEC]
    for row in rows:
        row["duration"] = 9.9
        del row["transcript"], row["timestamps"]
    _write_metadata(lavdf_root, rows, name="metadata.min.json")
    assert _records(lavdf_root)["RVRA/000001"].attrs["duration"] == 4.8
    (lavdf_root / RELEASE / "metadata.json").unlink()
    assert _records(lavdf_root)["RVRA/000001"].attrs["duration"] == 9.9


def test_without_metadata_the_attributes_are_minimal_with_a_warning(lavdf_root, caplog):
    (lavdf_root / RELEASE / "metadata.json").unlink()
    with caplog.at_level(logging.WARNING):
        records = _records(lavdf_root)
    assert "metadata" in caplog.text
    assert len(records) == 4
    fake = records["FVFA/000002"]
    assert fake.pair_key is None
    assert fake.attrs == {
        "task_name": "FakeVideo-FakeAudio",
        "n_fakes": 0,
        "fake_periods": [],
        "modify_video": None,
        "modify_audio": None,
        "duration": None,
        "video_frames": None,
        "audio_channels": None,
        "audio_frames": None,
    }


def test_a_video_without_a_metadata_row_is_kept_with_minimal_attributes(lavdf_root):
    _touch(lavdf_root / FVRA_DIR, "999999.mp4")
    rec = _records(lavdf_root)["FVRA/999999"]
    assert rec.pair_key is None
    assert rec.attrs["n_fakes"] == 0
    assert rec.attrs["modify_video"] is None


def test_missing_row_fields_take_the_release_defaults(lavdf_root):
    _write_metadata(lavdf_root, [{"file": "train/000002.mp4"}])
    rec = _records(lavdf_root)["FVFA/000002"]
    assert rec.pair_key is None
    assert rec.attrs["n_fakes"] == 0
    assert rec.attrs["fake_periods"] == []
    assert rec.attrs["duration"] is None


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('{"file": "a.mp4"}', "list"),
        ('[{"n_fakes": 0}]', "file"),
        ('["train/000001.mp4"]', "file"),
        ("[oops", "metadata.json"),
    ],
)
def test_malformed_metadata_is_a_contract_error(lavdf_root, content, match):
    (lavdf_root / RELEASE / "metadata.json").write_text(content, encoding="utf-8")
    with pytest.raises(ContractError, match=match) as caught:
        collect_records(LAVDFBuilder(), lavdf_root)
    assert "metadata" in caught.value.hint


def test_no_compression_can_be_requested(lavdf_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        collect_records(LAVDFBuilder(), lavdf_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ release layout


def _make_release_root(tmp_path: Path) -> Path:
    """LAV-DF as the release unpacks: flat train/, dev/ and test/ folders beside its metadata."""
    root = tmp_path / "LAV-DF"
    _touch(root / RELEASE / "train", "000001.mp4", "000002.mp4")
    _touch(root / RELEASE / "dev", "000003.mp4", "000005.mp4")
    _touch(root / RELEASE / "test", "000004.mp4", "notes.txt")
    _write_metadata(
        root,
        [
            _row("000001", False, False, None, "train"),
            _row("000002", True, True, "000001", "train"),
            _row("000003", True, False, "000001", "dev"),
            _row("000004", False, True, "000001", "test"),
        ],
    )
    return root


def test_the_release_as_unpacked_takes_each_task_from_the_metadata(tmp_path):
    root = _make_release_root(tmp_path)
    builder = LAVDFBuilder()
    assert builder.layout_present(root)
    assert builder.videos_present(root) == (f"{RELEASE}/train", f"{RELEASE}/dev", f"{RELEASE}/test")
    records = _records(root)
    # 000005 has no metadata row, so its task is unknown and it is left out.
    assert {key: rec.relpath for key, rec in records.items()} == {
        "RVRA/000001": f"{RELEASE}/train/000001.mp4",
        "FVFA/000002": f"{RELEASE}/train/000002.mp4",
        "FVRA/000003": f"{RELEASE}/dev/000003.mp4",
        "RVFA/000004": f"{RELEASE}/test/000004.mp4",
    }
    fake = records["FVRA/000003"]
    assert fake.pair_key == "000001"
    assert fake.method == "FakeVideo-RealAudio"
    assert set(fake.attrs) == ATTRS


def test_a_release_row_without_both_modify_flags_gives_no_task(tmp_path, caplog):
    root = _make_release_root(tmp_path)
    rows = [
        _row("000001", False, False, None, "train"),
        _row("000002", True, True, "000001", "train"),
        _row("000003", True, False, "000001", "dev"),
        _row("000004", False, True, "000001", "test"),
    ]
    del rows[1]["modify_audio"]  # a flag missing altogether
    rows[2]["modify_video"] = None  # a flag that is not true or false
    _write_metadata(root, rows)
    with caplog.at_level(logging.WARNING):
        records = _records(root)
    # Neither is labelled real by default: without both flags the category is unknown, so they
    # are skipped like 000005, which has no row at all.
    assert set(records) == {"RVRA/000001", "RVFA/000004"}
    assert "3 video(s)" in caplog.text


def test_a_row_without_modify_flags_still_describes_a_video_in_a_task_folder(lavdf_root):
    # In a task folder the folder names the category, so the row only adds its attributes.
    rows = [_row(*spec[:4]) for spec in _SPEC]
    del rows[1]["modify_video"], rows[1]["modify_audio"]
    _write_metadata(lavdf_root, rows)
    rec = _records(lavdf_root)["FVFA/000002"]
    assert rec.pair_key == "000001"
    assert (rec.attrs["modify_video"], rec.attrs["modify_audio"]) == (None, None)


def test_the_layout_dirs_are_the_task_folders_then_the_release_folders():
    assert LAVDFBuilder().layout_dirs() == (
        RVRA_DIR,
        FVFA_DIR,
        FVRA_DIR,
        RVFA_DIR,
        f"{RELEASE}/train",
        f"{RELEASE}/dev",
        f"{RELEASE}/test",
    )


def test_without_metadata_the_release_as_unpacked_yields_nothing(tmp_path, caplog):
    root = _make_release_root(tmp_path)
    (root / RELEASE / "metadata.json").unlink()
    with caplog.at_level(logging.WARNING):
        assert collect_records(LAVDFBuilder(), root) == []
    assert "metadata" in caplog.text


def test_a_video_in_both_layouts_is_a_duplicate(lavdf_root):
    _touch(lavdf_root / RELEASE / "test", "000002.mp4")
    with pytest.raises(ContractError, match="FVFA/000002"):
        collect_records(LAVDFBuilder(), lavdf_root)


def test_each_folder_is_taken_from_the_first_copy_holding_it(tmp_path):
    first = _make_release_root(tmp_path / "a")
    second = tmp_path / "b" / "LAV-DF"
    _touch(second / RVRA_DIR, "000011.mp4")
    _touch(second / FVFA_DIR, "000012.mp4")
    _touch(second / RELEASE / "test", "000007.mp4")  # the first copy has a test folder
    builder = LAVDFBuilder()
    copies = (first, second)
    builder.bind_copies(copies, builder.choose_copies(copies))
    records = {rec.key: rec.relpath for rec in collect_records(builder, first)}
    # The release folders come from the first copy and the task folders from the second (their
    # videos have no metadata row in the first copy's metadata, so their attributes are minimal).
    assert records == {
        "RVRA/000001": f"{RELEASE}/train/000001.mp4",
        "FVFA/000002": f"{RELEASE}/train/000002.mp4",
        "FVRA/000003": f"{RELEASE}/dev/000003.mp4",
        "RVFA/000004": f"{RELEASE}/test/000004.mp4",
        "RVRA/000011": f"{RVRA_DIR}/000011.mp4",
        "FVFA/000012": f"{FVFA_DIR}/000012.mp4",
    }


# ------------------------------------------------------------------------------ official split


def test_the_official_split_reads_each_videos_split(lavdf_root):
    rows = [_row(*spec[:4]) for spec in _SPEC]
    rows[0]["split"] = "train"
    rows[1]["split"] = " Dev "
    rows[2]["split"] = "unknown"
    rows.append(_row("000099", True, True, "000001", "train"))  # not on disk: ignored
    _write_metadata(lavdf_root, rows)
    builder = LAVDFBuilder()
    official = builder.official_splits(lavdf_root, collect_records(builder, lavdf_root))
    assert official == {"RVRA/000001": "train", "FVFA/000002": "val", "RVFA/000004": "test"}


def test_a_stem_in_two_splits_goes_to_the_first_of_train_val_test(lavdf_root):
    rows = [_row("000002", True, True, "000001", "test"), _row("000002", True, True, "x", "dev")]
    _write_metadata(lavdf_root, rows)
    builder = LAVDFBuilder()
    official = builder.official_splits(lavdf_root, collect_records(builder, lavdf_root))
    assert official == {"FVFA/000002": "val"}


def test_the_official_split_falls_back_to_the_minified_metadata(lavdf_root):
    (lavdf_root / RELEASE / "metadata.json").unlink()
    _write_metadata(lavdf_root, [_row("000001", False, False, None, "dev")], "metadata.min.json")
    builder = LAVDFBuilder()
    official = builder.official_splits(lavdf_root, collect_records(builder, lavdf_root))
    assert official == {"RVRA/000001": "val"}


def test_the_official_scheme_assigns_the_published_splits(lavdf_root):
    builder = LAVDFBuilder()
    records = collect_records(builder, lavdf_root)
    assignment = assign_official(records, builder.official_splits(lavdf_root, records))
    assert set(assignment.values()) == {"test"}
    assert len(assignment) == 4


def test_the_official_split_needs_the_metadata(lavdf_root):
    builder = LAVDFBuilder()
    records = collect_records(builder, lavdf_root)
    (lavdf_root / RELEASE / "metadata.json").unlink()
    with pytest.raises(ConfigError, match=r"metadata\.json") as caught:
        builder.official_splits(lavdf_root, records)
    assert ".official_files/LAV-DF" in caught.value.hint


# ------------------------------------------------------------------------------ pairs, labels


def test_pairs_link_each_visual_fake_to_its_original(lavdf_root):
    _touch(lavdf_root / FVRA_DIR, "000008.mp4")
    builder = LAVDFBuilder()
    records = collect_records(builder, lavdf_root)
    fakes = {r.key: builder.pair_candidates(r) for r in records if not builder.is_real(r)}
    # Real video with fake audio is real under the visual label, so it is not a fake here.
    assert fakes == {"FVFA/000002": "000001", "FVRA/000003": "000001", "FVRA/000008": None}
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    assert pairs == [
        PairRecord("FVFA/000002", "RVRA/000001", "original-video"),
        PairRecord("FVRA/000003", "RVRA/000001", "original-video"),
    ]


def test_label_vocab_covers_every_task():
    builder = LAVDFBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"LAVDF-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {
        "LAVDF-RVRA": (0, 0, 1, "real"),
        "LAVDF-FVFA": (1, 1, 2, "video-and-audio"),
        "LAVDF-FVRA": (1, 1, 3, "video-only"),
        "LAVDF-RVFA": (0, 1, 1, "audio-only"),
    }


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = LAVDFBuilder()
    assert builder.default_scheme == "official"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=100, strata=("task",))
    assert builder.pairing_rule == "original-video"
    assert builder.pairing_fanout is None
    assert builder.metadata_files == (f"{RELEASE}/metadata.json",)


def test_the_benchmark_counts_real_video_fake_audio_as_real(lavdf_root):
    builder = LAVDFBuilder()
    records = collect_records(builder, lavdf_root)
    chosen = {
        key
        for key, _ in assign_benchmark(
            records,
            spec=builder.benchmark,
            is_real=builder.is_real,
            task_rank=builder.task_rank(),
            pool_keys=None,
        )
    }
    # Two fakes, and the two visual reals (RVRA and RVFA) balance them.
    assert chosen == {"FVFA/000002", "FVRA/000003", "RVRA/000001", "RVFA/000004"}
    assert {task_of(key) for key in chosen} == {"RVRA", "FVFA", "FVRA", "RVFA"}


def test_dataset_card():
    builder = LAVDFBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "lav-df"
    assert card.name == "LAV-DF"
    assert "LAVDF" in card.aliases
    assert card.compressions is None
    assert card.modalities == ["video", "audio"]
    assert card.default_scheme == "official"
    assert card.homepage == "https://github.com/ControlNet/LAV-DF"
    assert card.paper is not None
    assert card.paper.title.startswith("Do You Really Mean That?")
    assert card.paper.year == 2022
    assert card.license.spdx is None


def test_the_layout_names_the_metadata_and_the_release_folders():
    text = LAVDFBuilder().describe_layout()
    assert "'LAV-DF'" in text
    assert f"{RVRA_DIR}/**/<video>" in text
    assert f"{RELEASE}/metadata.json" in text
    assert "train/, dev/ and test/" in text
    assert "modify_video" in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("lav-df"), LAVDFBuilder)
    assert LAVDFBuilder.expected_folder == "LAV-DF"
    assert LAVDFBuilder.label_prefix == "LAVDF"
