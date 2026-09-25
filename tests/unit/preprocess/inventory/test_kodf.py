"""The KoDF inventory builder, run on synthetic trees of empty files.

No test here reads a real dataset: every tree is touched into ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.kodf import KoDFBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.protocols.rules import (
    BenchmarkSpec,
    assign_72_14_14,
    assign_benchmark,
    local_key,
    resolve_pairs,
    task_of,
)

REAL_DIR = ("original_content", "actors", "videos")
DFL_DIR = ("manipulated_content", "dfl", "videos")
HEX_ACTOR = "0123456789abcdef0123"


def _make_synthetic_kodf_root(tmp_path: Path) -> Path:
    """A tiny KoDF tree: three reals nested one folder per actor, two flat DeepFaceLab fakes.

    Layout::

        original_content/actors/videos/
            0123456789abcdef0123/0123456789abcdef0123_001.mp4   (a 20-character hex actor id)
            0123456789abcdef0123/0123456789abcdef0123_002.mp4
            100001/100001_001.mp4                               (a six-digit actor id)
        manipulated_content/dfl/videos/
            100002_100001_1_0090.mp4
            0123456789abcdef0123_100001_1_0091.mp4
    """
    root = tmp_path / "KoDF"
    real_dir = root.joinpath(*REAL_DIR)
    fake_dir = root.joinpath(*DFL_DIR)
    (real_dir / HEX_ACTOR).mkdir(parents=True)
    (real_dir / "100001").mkdir(parents=True)
    (real_dir / HEX_ACTOR / f"{HEX_ACTOR}_001.mp4").write_bytes(b"\x00")
    (real_dir / HEX_ACTOR / f"{HEX_ACTOR}_002.mp4").write_bytes(b"\x00")
    (real_dir / "100001" / "100001_001.mp4").write_bytes(b"\x00")
    fake_dir.mkdir(parents=True)
    (fake_dir / "100002_100001_1_0090.mp4").write_bytes(b"\x00")
    (fake_dir / f"{HEX_ACTOR}_100001_1_0091.mp4").write_bytes(b"\x00")
    return root


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()


@pytest.fixture
def kodf_root(tmp_path: Path) -> Path:
    return _make_synthetic_kodf_root(tmp_path)


def _discover(root, **kwargs):
    return collect_records(KoDFBuilder(), root, **kwargs)


# ------------------------------------------------------------------------------ discovery


def test_discover_yields_expected_count(kodf_root):
    # 3 reals + 2 fakes.
    assert len(_discover(kodf_root)) == 5


def test_entry_schema(kodf_root):
    for rec in _discover(kodf_root):
        task, _, legacy = rec.key.partition("/")
        assert rec.key == f"{task}/{legacy}"
        assert task in {"REAL", "FS_DFL"}
        assert rec.builder == BuilderRef("kodf", KoDFBuilder.version)
        assert rec.attrs["task_name"] in {"Actors", "dfl"}
        assert not rec.relpath.startswith("/")
        # KoDF has a single version: no compression level.
        assert rec.compression is None
        assert rec.folder is None


def test_no_media_or_label_fields(kodf_root):
    for rec in _discover(kodf_root):
        assert rec.label_key in {"KoDF-REAL", "KoDF-FS_DFL"}
        assert rec.probe is None
        assert not hasattr(rec, "label")
        assert not hasattr(rec, "media")


def test_pairing_attrs_present_on_fakes(kodf_root):
    fakes = [r for r in _discover(kodf_root) if task_of(r.key) != "REAL"]
    assert len(fakes) == 2
    for rec in fakes:
        assert rec.target_id
        assert rec.source_id
        # The on-screen identity is the target.
        assert rec.identity == rec.target_id
        # KoDF has no one-to-one pair table: pair_key stays None.
        assert rec.pair_key is None


def test_nested_real_videos_discovered(kodf_root):
    reals = [r for r in _discover(kodf_root) if task_of(r.key) == "REAL"]
    assert len(reals) == 3
    by_key = {local_key(r.key): r for r in reals}
    assert set(by_key) == {f"{HEX_ACTOR}_001", f"{HEX_ACTOR}_002", "100001_001"}
    # The identity is the stem up to its last "_".
    assert by_key[f"{HEX_ACTOR}_001"].identity == HEX_ACTOR
    assert by_key["100001_001"].identity == "100001"
    assert by_key["100001_001"].relpath == "original_content/actors/videos/100001/100001_001.mp4"


def test_fields_parsed_from_the_names(kodf_root):
    records = {rec.key: rec for rec in _discover(kodf_root)}
    fake = records["FS_DFL/100002_100001_1_0090"]
    assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
        "100002",
        "100002",
        "100001",
        None,
    )
    assert fake.method == "dfl"
    assert fake.attrs == {"task_name": "dfl"}
    assert fake.relpath == "manipulated_content/dfl/videos/100002_100001_1_0090.mp4"
    real = records["REAL/100001_001"]
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "100001",
        "100001",
        None,
        None,
    )
    assert real.method == "original"
    assert real.attrs == {"task_name": "Actors"}


def test_fakes_nest_by_date_and_target(kodf_root):
    # The release nests fakes as <date>/<target>/<video>: every task is searched recursively.
    _touch(
        kodf_root.joinpath("manipulated_content", "fsgan", "videos", "20200101", "100003"),
        "100003_100004_3_0001.mp4",
    )
    records = {rec.key: rec for rec in _discover(kodf_root)}
    fake = records["FS_FSG/100003_100004_3_0001"]
    assert fake.relpath == (
        "manipulated_content/fsgan/videos/20200101/100003/100003_100004_3_0001.mp4"
    )
    assert (fake.identity, fake.source_id, fake.method) == ("100003", "100004", "fsgan")


def test_the_method_comes_from_the_code_in_the_name(kodf_root):
    # Code 1 dfl, 2 dffs, 3 fsgan, 4 fo, 5 audio-driven, whatever folder the file is in; an
    # unknown code, or a name that is not four parts, keeps the task's method.
    fo_dir = kodf_root.joinpath("manipulated_content", "fo", "videos")
    _touch(fo_dir, "1_2_4_0001.mp4", "1_2_5_0002.mp4", "1_2_9_0003.mp4", "1_2_0004.mp4")
    _touch(kodf_root.joinpath("manipulated_content", "dffs", "videos"), "1_2_2_0005.mp4")
    _touch(kodf_root.joinpath("manipulated_content", "audio-driven", "videos"), "1_2_3_0006.mp4")
    records = {rec.key: rec for rec in _discover(kodf_root)}
    assert records["FR_FO/1_2_4_0001"].method == "fo"
    assert records["FR_FO/1_2_5_0002"].method == "audio-driven"
    assert records["FR_FO/1_2_9_0003"].method == "fo"
    assert records["FR_FO/1_2_0004"].method == "fo"
    assert records["FS_DFFS/1_2_2_0005"].method == "dffs"
    assert records["LS_AD/1_2_3_0006"].method == "fsgan"


def test_names_that_do_not_parse_are_kept_without_fields(kodf_root):
    _touch(kodf_root.joinpath(*DFL_DIR), "1_2_3.mp4", "1_2_3_4_5.mp4")
    _touch(kodf_root.joinpath(*REAL_DIR, "loose"), "noactor.mp4", "_007.mp4")
    records = {rec.key: rec for rec in _discover(kodf_root)}
    for key in ("FS_DFL/1_2_3", "FS_DFL/1_2_3_4_5", "REAL/noactor"):
        rec = records[key]
        assert (rec.identity, rec.target_id, rec.source_id, rec.pair_key) == (None,) * 4, key
    assert records["FS_DFL/1_2_3_4_5"].method == "dfl"
    # A real stem split at its last "_" keeps whatever is before it, even nothing.
    assert (records["REAL/_007"].identity, records["REAL/_007"].target_id) == ("", "")


def test_every_task_is_discovered(kodf_root):
    for folder, stem in (
        ("audio-driven", "1_2_5_0001"),
        ("dffs", "1_2_2_0001"),
        ("fo", "1_2_4_0001"),
        ("fsgan", "1_2_3_0001"),
    ):
        _touch(kodf_root.joinpath("manipulated_content", folder, "videos"), f"{stem}.mp4")
    assert {task_of(r.key) for r in _discover(kodf_root)} == {
        "REAL",
        "LS_AD",
        "FS_DFFS",
        "FS_DFL",
        "FR_FO",
        "FS_FSG",
    }


def test_no_compression_can_be_requested(kodf_root):
    with pytest.raises(ConfigError, match="c23") as caught:
        _discover(kodf_root, compressions=["c23"])
    assert "single version" in caught.value.hint


# ------------------------------------------------------------------------------ splits, pairs


def test_there_is_no_official_split(kodf_root):
    builder = KoDFBuilder()
    with pytest.raises(ContractError, match="no official split"):
        builder.official_splits(kodf_root, collect_records(builder, kodf_root))


def test_the_carve_keeps_an_actor_on_one_side(kodf_root):
    _touch(kodf_root.joinpath(*DFL_DIR), "100001_100002_1_0001.mp4")
    records = _discover(kodf_root)
    assignment = assign_72_14_14(records)
    actor = [r for r in records if r.identity == "100001"]
    assert {task_of(r.key) for r in actor} == {"REAL", "FS_DFL"}
    assert len({assignment[(r.key, r.compression)] for r in actor}) == 1


def test_pair_candidates_name_the_target_actor(kodf_root):
    builder = KoDFBuilder()
    _touch(kodf_root.joinpath(*DFL_DIR), "1_2_3.mp4")
    records = collect_records(builder, kodf_root)
    fakes = [r for r in records if not builder.is_real(r)]
    assert {r.key: builder.pair_candidates(r) for r in fakes} == {
        "FS_DFL/100002_100001_1_0090": "100002",
        f"FS_DFL/{HEX_ACTOR}_100001_1_0091": HEX_ACTOR,
        "FS_DFL/1_2_3": None,
    }


def test_each_fake_pairs_with_the_first_real_of_its_target(kodf_root):
    builder = KoDFBuilder()
    records = collect_records(builder, kodf_root)
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    # 100002 has no real, so that fake has no pair; the hex actor has two reals and the fan-out
    # keeps only the first by key.
    assert pairs == [
        PairRecord(f"FS_DFL/{HEX_ACTOR}_100001_1_0091", f"REAL/{HEX_ACTOR}_001", "identity-fanout")
    ]


def test_label_vocab_covers_every_task():
    builder = KoDFBuilder()
    vocab = builder.label_vocab().vocab
    assert set(vocab) == {f"KoDF-{task.abbr}" for task in builder.tasks}
    table = {
        k: (v["binary"], v["binary_av"], v["multiclass"], v["family"], v["method"])
        for k, v in vocab.items()
    }
    assert table == {
        "KoDF-REAL": (0, 0, 1, "real", "original"),
        "KoDF-LS_AD": (1, 1, 2, "lip-sync", "audio-driven"),
        "KoDF-FS_DFFS": (1, 1, 3, "face-swap", "dffs"),
        "KoDF-FS_DFL": (1, 1, 4, "face-swap", "dfl"),
        "KoDF-FR_FO": (1, 1, 5, "face-reenactment", "fo"),
        "KoDF-FS_FSG": (1, 1, 6, "face-swap", "fsgan"),
    }


# ------------------------------------------------------------------------------ schemes, card


def test_schemes_and_benchmark():
    builder = KoDFBuilder()
    assert builder.default_scheme == "ident-72-14-14"
    assert {name: s.rule for name, s in builder.schemes.items()} == {
        "ident-72-14-14": "ident-72-14-14",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=2, strata=("target_id", "task"))
    assert builder.pairing_rule == "identity-fanout"
    assert builder.pairing_fanout == 1


def test_the_benchmark_draws_per_target_and_task(kodf_root):
    _touch(kodf_root.joinpath(*DFL_DIR), "100002_1_1_0001.mp4", "100002_2_1_0002.mp4")
    builder = KoDFBuilder()
    records = collect_records(builder, kodf_root)
    chosen = assign_benchmark(
        records,
        spec=builder.benchmark,
        is_real=builder.is_real,
        task_rank=builder.task_rank(),
        pool_keys=None,
    )
    fakes = [key for key, _ in chosen if task_of(key) != "REAL"]
    # Target 100002 has three fakes and keeps two; the hex actor's one fake is kept.
    assert len([k for k in fakes if local_key(k).startswith("100002_")]) == 2
    assert f"FS_DFL/{HEX_ACTOR}_100001_1_0091" in fakes
    # As many reals as fakes, capped by the three there are.
    assert len(chosen) - len(fakes) == 3


def test_dataset_card():
    builder = KoDFBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "kodf"
    assert card.name == "KoDF"
    assert card.compressions is None
    assert builder.known_compressions == ()
    assert card.default_scheme == "ident-72-14-14"
    assert card.homepage == "https://github.com/deepbrainai-research/kodf"
    assert card.paper is not None
    assert (card.paper.title, card.paper.venue, card.paper.year, card.paper.doi) == (
        "KoDF: A Large-Scale Korean DeepFake Detection Dataset",
        "ICCV",
        2021,
        None,
    )
    assert card.license.spdx is None


def test_the_layout_says_every_task_nests():
    text = KoDFBuilder().describe_layout()
    assert "'KoDF'" in text
    assert "original_content/actors/videos/**/<video>" in text
    assert "manipulated_content/audio-driven/videos/**/<video>" in text
    assert "<date>/<target>" in text
    assert "{cX}" not in text


def test_it_is_registered():
    assert isinstance(get_builder("kodf"), KoDFBuilder)
    assert KoDFBuilder.expected_folder == "KoDF"
    assert KoDFBuilder.label_prefix == "KoDF"
