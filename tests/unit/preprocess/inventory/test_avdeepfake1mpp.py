"""The AV-Deepfake1M++ inventory builder, run on synthetic trees and hand-written metadata.

No test here reads a real dataset: every tree is made in ``tmp_path``, and the metadata is a few
rows in the release's ``val_metadata.json`` format. Three properties of the release drive most of
the checks:

1. the task is the row's ``(modify_type, video_model)`` pair, never a folder, because the release
   keeps every task's videos side by side in one folder per utterance;
2. real video with fake audio is its own task, real to a visual detector;
3. a row's ``original`` means three different things: for a fake, the real it was made from; for
   a plain real, the upstream VoxCeleb2 path; for a ``_p1`` part-clip real, its parent real.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.avdeepfake1mpp import AVDeepfake1MPPBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_benchmark,
    assign_official,
    local_key,
    resolve_pairs,
    task_of,
)

METADATA = ".official_files/val_metadata.json"
SPEAKER = "id01358"
UTTERANCE = f"vox_celeb_2/{SPEAKER}/_1nATum8x78/00030"
UTTERANCE_KEY = f"vox_celeb_2__{SPEAKER}__1nATum8x78__00030"
SILENT = "silent_videos/subject_17_ddvyjnjfad_vid_2_6"
SILENT_KEY = "silent_videos__subject_17_ddvyjnjfad_vid_2_6"
ATTRS = {
    "task_name",
    "vox2_id",
    "source_corpus",
    "original_raw",
    "upstream_source",
    "parent_key",
    "modify_type",
    "video_model",
    "audio_model",
    "modify_video",
    "modify_audio",
    "fake_segments",
    "visual_fake_segments",
    "audio_fake_segments",
    "n_fakes",
    "n_visual_fakes",
    "n_audio_fakes",
    "video_frames",
    "audio_frames",
}


def _row(
    file: str,
    modify_type: str,
    video_model: str | None = None,
    *,
    original: str | None = None,
    audio_model: str | None = None,
    visual: list[list[float]] | None = None,
    audio: list[list[float]] | None = None,
    video_frames: int = 161,
    audio_frames: int = 102656,
    split: str = "val",
) -> dict[str, Any]:
    visual, audio = visual or [], audio or []
    return {
        "file": file,
        "original": original,
        "split": split,
        "modify_type": modify_type,
        "audio_model": audio_model,
        "video_model": video_model,
        "fake_segments": visual or audio,
        "visual_fake_segments": visual,
        "audio_fake_segments": audio,
        "video_frames": video_frames,
        "audio_frames": audio_frames,
    }


# A tiny release covering every task and all three source corpora.
ROWS = [
    _row(
        f"{UTTERANCE}/real.mp4",
        "real",
        original=f"VoxCeleb2/dev/mp4/{SPEAKER}/_1nATum8x78/00030.mp4",
    ),
    _row(
        f"{UTTERANCE}/fake_video_real_audio.mp4",
        "visual_modified",
        "Talklip",
        original=f"{UTTERANCE}/real.mp4",
        visual=[[0.48, 0.86]],
        video_frames=132,
        audio_frames=83968,
    ),
    _row(
        f"{UTTERANCE}/fake_video_fake_audio.mp4",
        "both_modified",
        "Talklip",
        original=f"{UTTERANCE}/real.mp4",
        audio_model="yourtts",
        visual=[[4.4, 4.64]],
        audio=[[4.4, 4.64]],
    ),
    _row(
        f"{UTTERANCE}/real_video_fake_audio.mp4",
        "audio_modified",
        original=f"{UTTERANCE}/real.mp4",
        audio_model="vits",
        audio=[[1.0, 1.5]],
    ),
    # A part clip of a real: its original is the in-dataset parent real.
    _row(
        f"{UTTERANCE}/real_p1.mp4",
        "real",
        original=f"{UTTERANCE}/real.mp4",
        video_frames=80,
        audio_frames=51200,
    ),
    # lrs3: the folder below the corpus is a YouTube clip id; original may be null.
    _row(
        "lrs3/ue2ZEmTJSXo/00013/real.mp4",
        "real",
        original="lrs3/ue2ZEmTJSXo/00013/real.mp4",
        video_frames=120,
        audio_frames=76800,
    ),
    _row(
        "lrs3/uhRhtFFhNzQ/00033/fake_video_fake_audio_p1.mp4",
        "both_modified",
        "Talklip",
        audio_model="yourtts",
        visual=[[0.1, 0.4]],
        audio=[[0.1, 0.4]],
        video_frames=60,
        audio_frames=38400,
    ),
    # silent_videos: every diff2lip fake, and none has an audio track.
    _row(
        f"{SILENT}/fake.mp4",
        "visual_modified",
        "diff2lip",
        original=f"{SILENT}/real.mp4",
        visual=[[0.0, 1.0]],
        video_frames=150,
        audio_frames=0,
    ),
    _row(
        f"{SILENT}/real.mp4",
        "real",
        original=f"{SILENT}/real.mp4",
        video_frames=150,
        audio_frames=480256,
    ),
]


def _write_metadata(root: Path, rows: Any) -> None:
    path = root / METADATA
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _make_root(tmp_path: Path, rows: Any = None) -> Path:
    root = tmp_path / "AV-Deepfake1M++"
    _write_metadata(root, ROWS if rows is None else rows)
    return root


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")


@pytest.fixture
def avdf_root(tmp_path: Path) -> Path:
    return _make_root(tmp_path)


def _records(root: Path, builder: AVDeepfake1MPPBuilder | None = None) -> dict[str, Any]:
    """The records by local key (every key of this release is unique across tasks)."""
    records = collect_records(builder or AVDeepfake1MPPBuilder(), root)
    by_local = {local_key(rec.key): rec for rec in records}
    assert len(by_local) == len(records)
    return by_local


def _one(tmp_path: Path, row: dict[str, Any]) -> Any:
    (rec,) = collect_records(AVDeepfake1MPPBuilder(), _make_root(tmp_path, [row]))
    return rec


# ------------------------------------------------------------------------------ keys


def test_the_key_is_the_sanitised_path_join(tmp_path):
    rec = _one(
        tmp_path,
        _row(f"{UTTERANCE}/fake_video_fake_audio.mp4", "both_modified", "Talklip"),
    )
    assert local_key(rec.key) == f"{UTTERANCE_KEY}__fake_video_fake_audio"
    assert rec.key == f"LS_TALKLIP_TTS/{UTTERANCE_KEY}__fake_video_fake_audio"


def test_the_key_does_not_depend_on_the_content_folder(tmp_path):
    root = _make_root(tmp_path, [_row("lrs3/ue2ZEmTJSXo/00013/real.mp4", "real")])
    (root / "val" / "vox_celeb_2").mkdir(parents=True)
    (rec,) = collect_records(AVDeepfake1MPPBuilder(), root)
    assert rec.key == "RVRA/lrs3__ue2ZEmTJSXo__00013__real"
    assert rec.relpath == "val/lrs3/ue2ZEmTJSXo/00013/real.mp4"


def test_repeated_file_names_get_distinct_keys(tmp_path):
    rows = [
        _row("vox_celeb_2/id01358/yt/00030/real.mp4", "real"),
        _row("vox_celeb_2/id01358/yt/00031/real.mp4", "real"),
    ]
    keys = {rec.key for rec in collect_records(AVDeepfake1MPPBuilder(), _make_root(tmp_path, rows))}
    assert keys == {
        "RVRA/vox_celeb_2__id01358__yt__00030__real",
        "RVRA/vox_celeb_2__id01358__yt__00031__real",
    }


# ------------------------------------------------------------------------------ identities


@pytest.mark.parametrize(
    ("file", "identity", "vox2_id", "corpus"),
    [
        ("vox_celeb_2/id01358/yt/00030/real.mp4", "vox2:id01358", "id01358", "vox_celeb_2"),
        ("lrs3/ue2ZEmTJSXo/00013/real.mp4", "lrs3:ue2ZEmTJSXo", None, "lrs3"),
        (f"{SILENT}/real.mp4", "silent:subject_17", None, "silent_videos"),
    ],
)
def test_the_identity_is_namespaced_by_its_source_corpus(tmp_path, file, identity, vox2_id, corpus):
    rec = _one(tmp_path, _row(file, "real"))
    assert rec.identity == identity
    assert rec.attrs["vox2_id"] == vox2_id
    assert rec.attrs["source_corpus"] == corpus
    # Lip sync changes the mouth only: the person on screen is the target, and there is no donor.
    assert rec.target_id == identity.split(":", 1)[1]
    assert rec.source_id is None


def test_a_voxceleb2_folder_that_is_not_a_speaker_id_claims_no_speaker(tmp_path):
    rec = _one(tmp_path, _row("vox_celeb_2/misc/yt/00001/real.mp4", "real"))
    assert rec.identity == "vox2:misc"
    assert rec.target_id == "misc"
    assert rec.attrs["vox2_id"] is None


def test_silent_videos_group_their_clips_by_subject(tmp_path):
    rows = [
        _row("silent_videos/subject_0_2msdhgqawh_vid_0_20/real.mp4", "real"),
        _row("silent_videos/subject_0_2msdhgqawh_vid_2_18/real.mp4", "real"),
    ]
    records = collect_records(AVDeepfake1MPPBuilder(), _make_root(tmp_path, rows))
    assert {rec.identity for rec in records} == {"silent:subject_0"}


def test_an_unknown_source_corpus_is_a_contract_error(tmp_path):
    root = _make_root(tmp_path, [_row("some_new_corpus/x/y/real.mp4", "real")])
    with pytest.raises(ContractError, match="some_new_corpus"):
        collect_records(AVDeepfake1MPPBuilder(), root)


def test_every_record_has_its_identity(avdf_root):
    for rec in _records(avdf_root).values():
        assert rec.identity is not None
        assert rec.source_id is None
        assert rec.target_id == rec.identity.split(":", 1)[1]


# ------------------------------------------------------------------------------ tasks


@pytest.mark.parametrize(
    ("modify_type", "video_model", "task", "method"),
    [
        ("real", None, "RVRA", "original"),
        ("audio_modified", None, "RVFA", "tts_audio_only"),
        ("visual_modified", "Talklip", "LS_TALKLIP", "Talklip"),
        ("visual_modified", "diff2lip", "LS_D2L", "diff2lip"),
        ("both_modified", "Talklip", "LS_TALKLIP_TTS", "Talklip"),
    ],
)
def test_the_task_comes_from_modify_type_and_video_model(
    tmp_path, modify_type, video_model, task, method
):
    rec = _one(tmp_path, _row(f"{UTTERANCE}/x.mp4", modify_type, video_model))
    assert task_of(rec.key) == task
    assert rec.method == method
    assert rec.label_key == f"AVDF1MPP-{task}"


@pytest.mark.parametrize(
    ("modify_type", "video_model"),
    [("visual_modified", "SomeNewLipSyncGAN"), ("geometry_modified", "Talklip"), ("real", "x")],
)
def test_an_unseen_task_combination_is_a_contract_error_naming_it(
    tmp_path, modify_type, video_model
):
    root = _make_root(tmp_path, [_row(f"{UTTERANCE}/x.mp4", modify_type, video_model)])
    with pytest.raises(ContractError) as caught:
        collect_records(AVDeepfake1MPPBuilder(), root)
    assert modify_type in caught.value.message
    assert video_model in caught.value.message


def test_every_task_and_method(avdf_root):
    records = _records(avdf_root)
    assert {rec.key: rec.method for rec in records.values()} == {
        f"RVRA/{UTTERANCE_KEY}__real": "original",
        f"LS_TALKLIP/{UTTERANCE_KEY}__fake_video_real_audio": "Talklip",
        f"LS_TALKLIP_TTS/{UTTERANCE_KEY}__fake_video_fake_audio": "Talklip",
        f"RVFA/{UTTERANCE_KEY}__real_video_fake_audio": "tts_audio_only",
        f"RVRA/{UTTERANCE_KEY}__real_p1": "original",
        "RVRA/lrs3__ue2ZEmTJSXo__00013__real": "original",
        "LS_TALKLIP_TTS/lrs3__uhRhtFFhNzQ__00033__fake_video_fake_audio_p1": "Talklip",
        f"LS_D2L/{SILENT_KEY}__fake": "diff2lip",
        f"RVRA/{SILENT_KEY}__real": "original",
    }


# ------------------------------------------------------------------------------ schema


def test_discover_yields_one_record_per_metadata_row(avdf_root):
    assert len(collect_records(AVDeepfake1MPPBuilder(), avdf_root)) == len(ROWS)


def test_entry_schema(avdf_root):
    for rec in collect_records(AVDeepfake1MPPBuilder(), avdf_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert rec.builder == BuilderRef("av-deepfake1m-pp", AVDeepfake1MPPBuilder.version)
        assert rec.compression is None  # a single version, no compression levels
        assert rec.label_key == f"AVDF1MPP-{task}"
        assert set(rec.attrs) - {"audio_relpath"} == ATTRS
        assert rec.folder is None
        assert rec.probe is None


def test_the_relpath_is_the_metadata_file_below_the_content_folder(avdf_root):
    rec = _records(avdf_root)["lrs3__ue2ZEmTJSXo__00013__real"]
    # No corpus folder is on disk yet, so the content folder is the dataset folder itself.
    assert rec.relpath == "lrs3/ue2ZEmTJSXo/00013/real.mp4"


# ------------------------------------------------------------------------------ original


def test_a_fake_pairs_with_the_in_dataset_real_it_names(avdf_root):
    records = _records(avdf_root)
    fake = records[f"{UTTERANCE_KEY}__fake_video_real_audio"]
    assert fake.pair_key == f"{UTTERANCE_KEY}__real"
    assert fake.pair_key in records
    assert fake.attrs["upstream_source"] is None
    assert fake.attrs["parent_key"] is None


def test_a_plain_real_records_its_upstream_path_not_a_pair(avdf_root):
    real = _records(avdf_root)[f"{UTTERANCE_KEY}__real"]
    assert real.pair_key is None
    assert real.attrs["upstream_source"] == f"VoxCeleb2/dev/mp4/{SPEAKER}/_1nATum8x78/00030.mp4"
    assert real.attrs["parent_key"] is None


def test_a_part_clip_real_records_its_in_dataset_parent(avdf_root):
    part = _records(avdf_root)[f"{UTTERANCE_KEY}__real_p1"]
    assert part.pair_key is None
    assert part.attrs["upstream_source"] is None
    assert part.attrs["parent_key"] == f"{UTTERANCE_KEY}__real"


def test_real_video_with_fake_audio_is_visually_real_so_its_original_is_a_parent(avdf_root):
    rvfa = _records(avdf_root)[f"{UTTERANCE_KEY}__real_video_fake_audio"]
    assert rvfa.pair_key is None
    assert rvfa.attrs["parent_key"] == f"{UTTERANCE_KEY}__real"


def test_a_null_original_is_tolerated(avdf_root):
    rec = _records(avdf_root)["lrs3__uhRhtFFhNzQ__00033__fake_video_fake_audio_p1"]
    assert rec.pair_key is None
    assert rec.attrs["original_raw"] is None
    assert rec.attrs["upstream_source"] is None
    assert rec.attrs["parent_key"] is None


# ------------------------------------------------------------------------------ attributes


def test_a_fakes_attributes(avdf_root):
    rec = _records(avdf_root)[f"{UTTERANCE_KEY}__fake_video_real_audio"]
    assert rec.attrs == {
        "task_name": "FakeVideo-RealAudio-Talklip",
        "vox2_id": SPEAKER,
        "source_corpus": "vox_celeb_2",
        "original_raw": f"{UTTERANCE}/real.mp4",
        "upstream_source": None,
        "parent_key": None,
        "modify_type": "visual_modified",
        "video_model": "Talklip",
        "audio_model": None,
        "modify_video": True,
        "modify_audio": False,
        "fake_segments": [[0.48, 0.86]],
        "visual_fake_segments": [[0.48, 0.86]],
        "audio_fake_segments": [],
        "n_fakes": 1,
        "n_visual_fakes": 1,
        "n_audio_fakes": 0,
        "video_frames": 132,
        "audio_frames": 83968,
        "audio_relpath": f"{UTTERANCE}/fake_video_real_audio.mp4",
    }


def test_audio_only_fakes_are_visually_unmodified(avdf_root):
    rec = _records(avdf_root)[f"{UTTERANCE_KEY}__real_video_fake_audio"]
    assert task_of(rec.key) == "RVFA"
    assert (rec.attrs["modify_video"], rec.attrs["modify_audio"]) == (False, True)
    assert rec.attrs["visual_fake_segments"] == []
    assert rec.attrs["n_visual_fakes"] == 0
    assert rec.attrs["n_audio_fakes"] == 1


def test_an_audio_model_alone_marks_the_audio_modified(tmp_path):
    rec = _one(tmp_path, _row(f"{UTTERANCE}/x.mp4", "both_modified", "Talklip", audio_model="vits"))
    assert rec.attrs["audio_fake_segments"] == []
    assert rec.attrs["modify_audio"] is True


def test_missing_segment_lists_are_empty(tmp_path):
    row = _row(f"{UTTERANCE}/x.mp4", "real")
    for name in ("fake_segments", "visual_fake_segments", "audio_fake_segments"):
        row[name] = None
    rec = _one(tmp_path, row)
    assert rec.attrs["fake_segments"] == rec.attrs["visual_fake_segments"] == []
    assert rec.attrs["n_fakes"] == 0


def test_the_generator_is_recorded(avdf_root):
    d2l = _records(avdf_root)[f"{SILENT_KEY}__fake"]
    assert task_of(d2l.key) == "LS_D2L"
    assert d2l.attrs["video_model"] == "diff2lip"
    assert d2l.method == "diff2lip"


def test_the_audio_is_inside_the_video_and_absent_without_audio_frames(avdf_root):
    records = _records(avdf_root)
    real = records[f"{UTTERANCE_KEY}__real"]
    assert real.attrs["audio_relpath"] == real.relpath
    # The diff2lip clips carry no audio track at all.
    d2l = records[f"{SILENT_KEY}__fake"]
    assert d2l.attrs["audio_frames"] == 0
    assert "audio_relpath" not in d2l.attrs


# ------------------------------------------------------------------------------ content folder


@pytest.mark.parametrize(
    ("on_disk", "prefix"),
    [
        (["raw_content/vox_celeb_2"], "raw_content/"),
        (["vox_celeb_2"], ""),
        (["val/vox_celeb_2"], "val/"),
        (["val/vox_celeb_2", "vox_celeb_2"], ""),
        (["val/vox_celeb_2", "vox_celeb_2", "raw_content/vox_celeb_2"], "raw_content/"),
        (["raw_content/lrs3"], ""),  # only a vox_celeb_2 folder marks the content folder
    ],
)
def test_the_content_folder_is_detected(avdf_root, on_disk, prefix):
    for folder in on_disk:
        (avdf_root / folder).mkdir(parents=True)
    rec = _records(avdf_root)["lrs3__ue2ZEmTJSXo__00013__real"]
    assert rec.relpath == f"{prefix}lrs3/ue2ZEmTJSXo/00013/real.mp4"
    assert local_key(rec.key) == "lrs3__ue2ZEmTJSXo__00013__real"


def test_without_a_content_folder_the_dataset_folder_is_assumed_with_a_warning(avdf_root, caplog):
    with caplog.at_level(logging.WARNING, logger="dfwb.preprocess.inventory.builders"):
        collect_records(AVDeepfake1MPPBuilder(), avdf_root)
    assert "vox_celeb_2" in caplog.text


def test_the_content_folder_is_detected_across_copies(tmp_path):
    first = _make_root(tmp_path / "a")
    second = tmp_path / "b" / "AV-Deepfake1M++"
    (first / "val" / "vox_celeb_2").mkdir(parents=True)
    (second / "raw_content" / "vox_celeb_2").mkdir(parents=True)
    builder = AVDeepfake1MPPBuilder()
    builder.bind_copies([first, second])
    rec = _records(first, builder)["lrs3__ue2ZEmTJSXo__00013__real"]
    # raw_content comes first in the order, whichever copy holds it.
    assert rec.relpath == "raw_content/lrs3/ue2ZEmTJSXo/00013/real.mp4"


# ------------------------------------------------------------------------------ metadata


def test_without_metadata_nothing_is_yielded_with_a_warning(tmp_path, caplog):
    root = tmp_path / "AV-Deepfake1M++"
    _touch(root / "vox_celeb_2" / SPEAKER / "yt" / "00001" / "real.mp4")
    with caplog.at_level(logging.WARNING, logger="dfwb.preprocess.inventory.builders"):
        assert collect_records(AVDeepfake1MPPBuilder(), root) == []
    assert "val_metadata.json" in caplog.text


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{not json", "cannot read"),
        ('{"file": "x"}', "not a list"),
        ('["x"]', "entry 0"),
        ('[{"original": null}]', "entry 0"),
        ('[{"file": ""}]', "entry 0"),
    ],
)
def test_malformed_metadata_is_a_contract_error(tmp_path, content, match):
    root = tmp_path / "AV-Deepfake1M++"
    (root / ".official_files").mkdir(parents=True)
    (root / METADATA).write_text(content, encoding="utf-8")
    with pytest.raises(ContractError, match=match):
        collect_records(AVDeepfake1MPPBuilder(), root)


def test_no_compression_can_be_requested(avdf_root):
    with pytest.raises(ConfigError, match="unknown compression"):
        collect_records(AVDeepfake1MPPBuilder(), avdf_root, compressions=["c23"])
    with pytest.raises(ConfigError, match="unknown compression"):
        list(AVDeepfake1MPPBuilder().discover(avdf_root, compressions=["c23"]))


# ------------------------------------------------------------------------------ layout


def test_the_layout_dirs_are_the_corpus_folders_in_each_content_folder():
    assert AVDeepfake1MPPBuilder().layout_dirs() == (
        "raw_content/vox_celeb_2",
        "raw_content/lrs3",
        "raw_content/silent_videos",
        "vox_celeb_2",
        "lrs3",
        "silent_videos",
        "val/vox_celeb_2",
        "val/lrs3",
        "val/silent_videos",
    )


def test_the_layout_is_present_only_with_a_video_in_a_corpus_folder(avdf_root):
    builder = AVDeepfake1MPPBuilder()
    (avdf_root / "raw_content" / "vox_celeb_2").mkdir(parents=True)
    assert not builder.layout_present(avdf_root)  # metadata and an empty folder are not videos
    _touch(avdf_root / "val" / "lrs3" / "ue2ZEmTJSXo" / "00013" / "real.mp4")
    assert builder.videos_present(avdf_root) == ("val/lrs3",)
    assert builder.layout_present(avdf_root)


def test_every_task_is_backed_by_the_first_copy_holding_a_video(tmp_path):
    empty = _make_root(tmp_path / "a")
    full = tmp_path / "b" / "AV-Deepfake1M++"
    later = tmp_path / "c" / "AV-Deepfake1M++"
    _touch(full / "raw_content" / "lrs3" / "x" / "00001" / "real.mp4")
    _touch(later / "vox_celeb_2" / SPEAKER / "yt" / "00001" / "real.mp4")
    builder = AVDeepfake1MPPBuilder()
    copies = [empty, full, later]
    chosen = builder.choose_copies(copies)
    assert chosen == {(task.abbr, None): full for task in builder.tasks}
    assert builder.copies_after(copies, full, builder.tasks[0], None) == [later]


# ------------------------------------------------------------------------------ official split


def test_the_official_split_is_the_labelled_val_split(avdf_root):
    builder = AVDeepfake1MPPBuilder()
    records = collect_records(builder, avdf_root)
    official = builder.official_splits(avdf_root, records)
    assert official == {rec.key: "val" for rec in records}
    assignment = assign_official(records, official)
    assert set(assignment.values()) == {"val"}
    assert len(assignment) == len(ROWS)


def test_only_videos_first_listed_as_val_are_published(avdf_root):
    rows = [dict(row) for row in ROWS[:4]]
    rows[0]["split"] = "train"
    rows[1]["split"] = " Dev "
    rows[2]["split"] = "unknown"
    rows[3]["split"] = "TEST"
    rows.append(_row(f"{UTTERANCE}/fake_video_real_audio.mp4", "real", split="train"))
    rows.append(_row("lrs3/zzz/00001/real.mp4", "real"))  # not on disk: ignored
    builder = AVDeepfake1MPPBuilder()
    records = collect_records(builder, avdf_root)
    _write_metadata(avdf_root, rows)
    official = builder.official_splits(avdf_root, records)
    # A key also listed as train goes to train first, and only val is published.
    assert official == {}
    del rows[4]  # now listed once, as " Dev ": dev is val, whatever its case and spacing
    _write_metadata(avdf_root, rows)
    official = builder.official_splits(avdf_root, records)
    assert official == {f"LS_TALKLIP/{UTTERANCE_KEY}__fake_video_real_audio": "val"}


def test_the_official_split_needs_the_metadata(avdf_root):
    builder = AVDeepfake1MPPBuilder()
    records = collect_records(builder, avdf_root)
    (avdf_root / METADATA).unlink()
    with pytest.raises(ConfigError, match=r"val_metadata\.json") as caught:
        builder.official_splits(avdf_root, records)
    assert ".official_files" in caught.value.hint


# ------------------------------------------------------------------------------ pairs, labels


def test_pairs_link_each_visual_fake_to_the_real_it_names(avdf_root):
    rows = [
        *ROWS,
        _row(
            "lrs3/abc/00001/fake_video_real_audio.mp4",
            "visual_modified",
            "Talklip",
            original="lrs3/abc/00001/real.mp4",
        ),
    ]
    _write_metadata(avdf_root, rows)
    builder = AVDeepfake1MPPBuilder()
    records = collect_records(builder, avdf_root)
    fakes = {r.key: builder.pair_candidates(r) for r in records if not builder.is_real(r)}
    # Real video with fake audio is real under the visual label, so it is not a fake here.
    assert fakes == {
        f"LS_TALKLIP/{UTTERANCE_KEY}__fake_video_real_audio": f"{UTTERANCE_KEY}__real",
        f"LS_TALKLIP_TTS/{UTTERANCE_KEY}__fake_video_fake_audio": f"{UTTERANCE_KEY}__real",
        "LS_TALKLIP_TTS/lrs3__uhRhtFFhNzQ__00033__fake_video_fake_audio_p1": None,
        f"LS_D2L/{SILENT_KEY}__fake": f"{SILENT_KEY}__real",
        # Its real is not part of the release, so it pairs with nothing.
        "LS_TALKLIP/lrs3__abc__00001__fake_video_real_audio": "lrs3__abc__00001__real",
    }
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    assert pairs == [
        PairRecord(f"LS_D2L/{SILENT_KEY}__fake", f"RVRA/{SILENT_KEY}__real", "original-video"),
        PairRecord(
            f"LS_TALKLIP/{UTTERANCE_KEY}__fake_video_real_audio",
            f"RVRA/{UTTERANCE_KEY}__real",
            "original-video",
        ),
        PairRecord(
            f"LS_TALKLIP_TTS/{UTTERANCE_KEY}__fake_video_fake_audio",
            f"RVRA/{UTTERANCE_KEY}__real",
            "original-video",
        ),
    ]


def test_label_vocab_covers_every_task():
    builder = AVDeepfake1MPPBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"AVDF1MPP-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {
        "AVDF1MPP-RVRA": (0, 0, 1, "real"),
        "AVDF1MPP-LS_TALKLIP": (1, 1, 2, "lip-sync"),
        "AVDF1MPP-LS_TALKLIP_TTS": (1, 1, 3, "lip-sync"),
        "AVDF1MPP-LS_D2L": (1, 1, 4, "lip-sync"),
        "AVDF1MPP-RVFA": (0, 1, 1, "audio-only"),
    }


def test_the_task_table_order():
    assert [task.abbr for task in AVDeepfake1MPPBuilder.tasks] == [
        "RVRA",
        "LS_TALKLIP",
        "LS_TALKLIP_TTS",
        "LS_D2L",
        "RVFA",
    ]


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = AVDeepfake1MPPBuilder()
    assert builder.default_scheme == "official"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "official": "official",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=100, strata=("task",))
    assert builder.pairing_rule == "original-video"
    assert builder.pairing_fanout is None
    assert builder.metadata_files == (METADATA,)


def test_the_benchmark_counts_real_video_fake_audio_as_real(tmp_path):
    rows = [ROWS[0], ROWS[1], ROWS[3], ROWS[7]]
    builder = AVDeepfake1MPPBuilder()
    records = collect_records(builder, _make_root(tmp_path, rows))
    assert builder.benchmark is not None
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
    assert {task_of(key) for key in chosen} == {"RVRA", "RVFA", "LS_TALKLIP", "LS_D2L"}
    assert len(chosen) == 4


def test_dataset_card():
    builder = AVDeepfake1MPPBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "av-deepfake1m-pp"
    assert card.name == "AV-Deepfake1M++"
    assert "AVDF1MPP" in card.aliases
    assert card.compressions is None
    assert card.modalities == ["video", "audio"]
    assert card.default_scheme == "official"
    assert card.homepage == "https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus"
    assert card.paper is not None
    assert (card.paper.title, card.paper.venue, card.paper.year, card.paper.doi) == (
        "AV-Deepfake1M++: A Large-Scale Audio-Visual Deepfake Benchmark with Real-World "
        "Perturbations",
        "ACM MM",
        2025,
        "10.1145/3746027.3761979",
    )
    assert card.license.spdx is None
    assert "77,326" in card.release


def test_the_layout_names_the_metadata_the_content_folders_and_the_tasks():
    text = AVDeepfake1MPPBuilder().describe_layout()
    assert "'AV-Deepfake1M++'" in text
    assert METADATA in text
    for word in ("raw_content", "val", "vox_celeb_2", "lrs3", "silent_videos"):
        assert word in text
    for word in ("RVRA", "LS_TALKLIP_TTS", "both_modified", "diff2lip", "audio_modified"):
        assert word in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("av-deepfake1m-pp"), AVDeepfake1MPPBuilder)
    assert AVDeepfake1MPPBuilder.expected_folder == "AV-Deepfake1M++"
    assert AVDeepfake1MPPBuilder.label_prefix == "AVDF1MPP"
