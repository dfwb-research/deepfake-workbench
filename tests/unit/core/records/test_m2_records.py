import pytest

from dfwb.core.errors import ContractError
from dfwb.core.records import (
    BuilderRef,
    InventoryRecord,
    PackProvenance,
    PairRecord,
    SchemeCard,
    VideoRecord,
    read_jsonl,
    write_jsonl,
)
from dfwb.core.records.local import to_video_record

B = BuilderRef("ffpp", "1")


def inv(**kw):
    base = dict(
        key="FS_DF/000_003",
        compression="c23",
        label_key="FF-FS_DF",
        method="Deepfakes",
        relpath="manipulated_content/Deepfakes/c23/videos/000_003.mp4",
        builder=B,
    )
    return InventoryRecord(**{**base, **kw})


def test_scheme_card_counts_and_params_are_optional():
    card = SchemeCard(
        kind="derived",
        rule="md5-carve",
        sha256="a" * 64,
        counts={"train": 7, "val": 2, "test": 1},
        params={"val_hi": 14, "test_hi": 28},
    )
    assert card.counts["train"] == 7
    assert SchemeCard(kind="official", sha256="b" * 64).params == {}


def test_pair_records_round_trip(tmp_path):
    rows = [PairRecord("FS_DF/000_003", "REAL/000", "target-id")]
    write_jsonl(tmp_path / "pairs.jsonl.gz", rows)
    assert read_jsonl(tmp_path / "pairs.jsonl.gz", PairRecord, strict=True) == rows


def test_provenance_model():
    prov = PackProvenance(
        builder={"id": "ffpp", "version": "1"},
        dfwb="0.1.0a2",
        source_listing_sha256="c" * 64,
        rules={"official": {"rule": "official"}},
    )
    assert prov.builder["id"] == "ffpp"


@pytest.mark.parametrize(
    "bad", ["/abs/x.mp4", "../FaceForensics++/x.mp4", "a/../b.mp4", "a\\b.mp4", ""]
)
def test_inventory_relpath_guard(bad):
    with pytest.raises(ContractError, match="relpath"):
        inv(relpath=bad)


def test_inventory_folder_for_records_in_a_sibling_dataset():
    rec = inv(folder="FaceForensics++", relpath="original_content/YouTube/c23/videos/000.mp4")
    assert rec.folder == "FaceForensics++"


def test_to_video_record_keeps_only_portable_columns():
    rec = inv(
        identity="000",
        source_id="003",
        target_id="000",
        pair_key="000_003",
        attrs={"task_name": "Deepfakes"},
        folder=None,
    )
    assert to_video_record(rec) == VideoRecord(
        key="FS_DF/000_003",
        compression="c23",
        label_key="FF-FS_DF",
        method="Deepfakes",
        identity="000",
        source_id="003",
        target_id="000",
        pair_key="000_003",
        attrs={"task_name": "Deepfakes"},
    )


def test_to_video_record_leaves_out_the_attrs_that_describe_the_local_copy():
    rec = inv(attrs={"task_name": "Deepfakes", "audio_relpath": "a/b.wav"})

    published = to_video_record(rec, local_attrs=frozenset({"audio_relpath", "absent"}))

    assert published.attrs == {"task_name": "Deepfakes"}
    assert rec.attrs == {"task_name": "Deepfakes", "audio_relpath": "a/b.wav"}  # kept locally
    assert to_video_record(rec).attrs == rec.attrs
