"""The DeepFakeDetection inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfwb.core.errors import ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.dfd import DeepFakeDetectionBuilder
from dfwb.preprocess.inventory.builders.ffpp import FaceForensicsBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_72_14_14,
    local_key,
    resolve_pairs,
    task_of,
)

REAL_DIR = ("original_content", "Actors", "c23", "videos")
FAKE_DIR = ("manipulated_content", "DeepFakeDetection", "c23", "videos")


def _make_synthetic_dfd_root(tmp_path: Path) -> Path:
    """A tiny DeepFakeDetection tree (inside a FaceForensics++ folder): two reals, two fakes."""
    root = tmp_path / "FaceForensics++"
    real_dir = root.joinpath(*REAL_DIR)
    fake_dir = root.joinpath(*FAKE_DIR)
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    # Reals: <actor>__<scene>.
    (real_dir / "01__exit_phone_room.mp4").write_bytes(b"\x00")
    (real_dir / "02__hugging_happy.mp4").write_bytes(b"\x00")
    # Fakes: <target>_<source>__<scene>__<code>; the first pairs with the first real.
    (fake_dir / "01_02__exit_phone_room__ABCDEF.mp4").write_bytes(b"\x00")
    (fake_dir / "03_04__kitchen_pan__123456.mp4").write_bytes(b"\x00")
    return root


def _touch(folder: Path, *stems: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (folder / f"{stem}.mp4").touch()


@pytest.fixture
def dfd_root(tmp_path: Path) -> Path:
    return _make_synthetic_dfd_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(DeepFakeDetectionBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_entries(dfd_root):
    # 2 real + 2 fake stubs -> 4 records.
    assert len(_discover(dfd_root)) == 4


def test_entry_schema(dfd_root):
    for rec in _discover(dfd_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"AR", "FS_DFD"}
        assert rec.builder == BuilderRef("dfd", DeepFakeDetectionBuilder.version)
        assert rec.attrs["task_name"] in {"Actors-real", "DeepFakeDetection"}
        assert not rec.relpath.startswith("/")
        assert rec.folder is None


def test_records_carry_a_label_key_but_no_label_or_media(dfd_root):
    for rec in _discover(dfd_root):
        assert rec.label_key in {"DFD-AR", "DFD-FS_DFD"}
        assert rec.probe is None
        assert not hasattr(rec, "label")
        assert not hasattr(rec, "media")


def test_pairing_attrs_present(dfd_root):
    records = _discover(dfd_root)
    fakes = [r for r in records if task_of(r.key) == "FS_DFD"]
    reals = [r for r in records if task_of(r.key) == "AR"]
    assert fakes
    assert reals
    paired = [r for r in fakes if r.pair_key]
    assert paired
    sample = paired[0]
    assert sample.target_id
    assert sample.source_id
    # pair_key is "<target_id>__<scene>"
    assert sample.pair_key.startswith(sample.target_id + "__")
    # at least one fake's pair_key is a real's local key (the seeded match)
    real_keys = {local_key(r.key) for r in reals}
    assert [r for r in paired if r.pair_key in real_keys]


def test_relpath_is_the_concrete_path_of_its_compression(dfd_root):
    records = {rec.key: rec for rec in _discover(dfd_root)}
    fake = records["FS_DFD/01_02__exit_phone_room__ABCDEF"]
    assert fake.relpath == (
        "manipulated_content/DeepFakeDetection/c23/videos/01_02__exit_phone_room__ABCDEF.mp4"
    )
    assert fake.compression == "c23"
    assert records["AR/01__exit_phone_room"].relpath == (
        "original_content/Actors/c23/videos/01__exit_phone_room.mp4"
    )
    assert all("{cX}" not in rec.relpath for rec in records.values())


def test_real_entries_have_null_pairing(dfd_root):
    reals = [r for r in _discover(dfd_root) if task_of(r.key) == "AR"]
    assert reals
    for rec in reals:
        assert rec.source_id is None
        assert rec.pair_key is None
        assert rec.identity
        assert rec.target_id == rec.identity
        assert rec.method == "original"


def test_fields_parsed_from_the_names(dfd_root):
    records = {rec.key: rec for rec in _discover(dfd_root)}
    fake = records["FS_DFD/01_02__exit_phone_room__ABCDEF"]
    assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
        "01",
        "01",
        "02",
        "01__exit_phone_room",
    )
    assert fake.method == "deepfakedetection"
    assert fake.attrs == {"task_name": "DeepFakeDetection"}
    real = records["AR/02__hugging_happy"]
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "02",
        "02",
        None,
        None,
    )
    assert real.attrs == {"task_name": "Actors-real"}


def test_names_that_do_not_parse_are_kept_without_fields(dfd_root):
    _touch(dfd_root.joinpath(*FAKE_DIR), "01_02__scene_only", "0102__scene__CODE")
    _touch(dfd_root.joinpath(*REAL_DIR), "__no_actor", "07")
    records = {rec.key: rec for rec in _discover(dfd_root)}
    for key in ("FS_DFD/01_02__scene_only", "FS_DFD/0102__scene__CODE", "AR/__no_actor"):
        rec = records[key]
        assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (None,) * 4, key
    # a real without "__" is an actor id on its own
    assert (records["AR/07"].identity, records["AR/07"].target_id) == ("07", "07")


def test_ffpp_and_dfd_share_a_folder_without_clashing(dfd_root):
    _touch(dfd_root / "original_content" / "YouTube" / "c23" / "videos", "000", "001")
    _touch(dfd_root / "manipulated_content" / "Deepfakes" / "c23" / "videos", "000_001")
    _touch(dfd_root / "manipulated_content" / "NeuralTextures" / "c23" / "videos", "001_000")

    dfd = collect_records(DeepFakeDetectionBuilder(), dfd_root)
    ffpp = collect_records(FaceForensicsBuilder(), dfd_root)

    assert {task_of(r.key) for r in dfd} == {"AR", "FS_DFD"}
    assert len(dfd) == 4
    assert {task_of(r.key) for r in ffpp} == {"REAL", "FS_DF", "FR_NT"}
    assert len(ffpp) == 4
    assert not {r.relpath for r in dfd} & {r.relpath for r in ffpp}
    assert {r.builder.id for r in dfd} == {"dfd"}
    assert {r.builder.id for r in ffpp} == {"ffpp"}
    assert DeepFakeDetectionBuilder.expected_folder == FaceForensicsBuilder.expected_folder


# ------------------------------------------------------------------------------ splits, pairs


def test_there_is_no_official_split(dfd_root):
    builder = DeepFakeDetectionBuilder()
    with pytest.raises(ContractError, match="no official split"):
        builder.official_splits(dfd_root, collect_records(builder, dfd_root))


def test_the_carve_keeps_an_actor_on_one_side(dfd_root):
    _touch(dfd_root.joinpath(*REAL_DIR), "01__walking_outside")
    _touch(dfd_root.joinpath(*FAKE_DIR), "01_05__walking_outside__XYZ", "01_09__kitchen_pan__Q1")
    records = _discover(dfd_root)
    assignment = assign_72_14_14(records)
    actor = [r for r in records if r.identity == "01"]
    assert {task_of(r.key) for r in actor} == {"AR", "FS_DFD"}
    assert len(actor) == 5
    assert len({assignment[(r.key, r.compression)] for r in actor}) == 1


def test_pair_candidates_name_the_actor_scene(dfd_root):
    builder = DeepFakeDetectionBuilder()
    _touch(dfd_root.joinpath(*FAKE_DIR), "01_02__scene_only")
    records = collect_records(builder, dfd_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "FS_DFD/01_02__exit_phone_room__ABCDEF": "01__exit_phone_room",
        "FS_DFD/03_04__kitchen_pan__123456": "03__kitchen_pan",
        "FS_DFD/01_02__scene_only": None,
    }
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    # 03__kitchen_pan has no real, so that fake has no pair
    assert pairs == [
        PairRecord(
            "FS_DFD/01_02__exit_phone_room__ABCDEF", "AR/01__exit_phone_room", "target-scene"
        )
    ]


def test_label_vocab_covers_every_task():
    builder = DeepFakeDetectionBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"DFD-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"]) for k, v in vocab.items()
    }
    assert table == {"DFD-AR": (0, 0, 1, "real"), "DFD-FS_DFD": (1, 1, 2, "face-swap")}


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = DeepFakeDetectionBuilder()
    assert builder.default_scheme == "ident-72-14-14"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "ident-72-14-14": "ident-72-14-14",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    # The benchmark is defined at c23, whatever other compressions are on disk.
    assert builder.benchmark == BenchmarkSpec(k_fake=100, compressions=("c23",))
    assert builder.pairing_rule == "target-scene"
    assert builder.pairing_fanout is None


def test_dataset_card():
    builder = DeepFakeDetectionBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "dfd"
    assert card.name == "DeepFakeDetection"
    assert card.compressions == list(builder.known_compressions) == ["raw", "c23", "c40"]
    assert card.default_scheme == "ident-72-14-14"
    assert card.paper is None


def test_the_layout_says_the_folder_is_shared():
    text = DeepFakeDetectionBuilder().describe_layout()
    assert "'FaceForensics++'" in text
    assert "original_content/Actors/{cX}/videos" in text
    assert "ffpp" in text


def test_it_is_registered():
    assert isinstance(get_builder("dfd"), DeepFakeDetectionBuilder)
