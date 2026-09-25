"""The toyfake synthetic dataset: its generator, its inventory builder and its built-in pack.

The file tree, the ids and the official split depend only on the seed and the number of videos,
and need neither numpy nor PyAV (``write_media=False``). The media tests need PyAV and skip
cleanly without it; they decode what was written and compare it with the generated frames.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import dfwb
from dfwb import __version__
from dfwb.core.errors import ConfigError, ContractError, InstallationError
from dfwb.core.records import BuilderRef, PairRecord, SchemeCard
from dfwb.preprocess.inventory.builders.toyfake import ToyfakeBuilder
from dfwb.preprocess.inventory.runner import collect_records, get_builder
from dfwb.preprocess.toyfake import SynthResult, synth
from dfwb.protocols.lint import lint_pack
from dfwb.protocols.packs import installed_packs
from dfwb.protocols.protocol import load
from dfwb.protocols.rules import BenchmarkSpec, local_key, resolve_pairs, task_of

REPO = Path(__file__).resolve().parents[3]
PACK = Path(dfwb.__file__).parent / "_packs" / "toyfake"
SCRIPT = REPO / "scripts" / "regenerate_toyfake_pack.py"
TASK_DIRS = {"REAL": "original", "BLEND_A": "blend-a", "BLEND_B": "blend-b"}


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root``, by POSIX relative path, with its bytes."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _stems(root: Path, folder: str) -> list[str]:
    return sorted(path.stem for path in (root / folder).iterdir())


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No roots, dataset overrides or config files leak in from the machine running the tests."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    for name in list(os.environ):
        if name.startswith("DFWB_"):
            monkeypatch.delenv(name)
    return tmp_path


@pytest.fixture
def block_av(monkeypatch):
    """Block ``import av`` (and ``find_spec("av")``) for the duration of a test."""
    for name in [n for n in sys.modules if n == "av" or n.startswith("av.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] == "av":
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


# ------------------------------------------------------------------------------ the file tree


def test_synth_tree_is_seed_deterministic_without_media(tmp_path):
    first = synth(tmp_path / "one", videos=40, seed=7, write_media=False)
    second = synth(tmp_path / "two", videos=40, seed=7, write_media=False)
    other = synth(tmp_path / "three", videos=40, seed=8, write_media=False)

    assert first == SynthResult(
        tmp_path / "one" / "toyfake", 40, {"REAL": 16, "BLEND_A": 12, "BLEND_B": 12}
    )
    assert _tree(first.root) == _tree(second.root)
    assert _tree(first.root) != _tree(other.root)
    tree = _tree(first.root)
    assert {"official_splits.json", "README.txt"} <= set(tree)
    videos = {name: data for name, data in tree.items() if name.endswith(".mkv")}
    assert len(videos) == 40
    assert set(videos.values()) == {b""}  # write_media=False: empty placeholders only
    assert {name.split("/")[0] for name in videos} == set(TASK_DIRS.values())


def test_media_parameters_never_change_the_tree(tmp_path):
    plain = synth(tmp_path / "a", videos=20, seed=1, write_media=False)
    resized = synth(tmp_path / "b", videos=20, seed=1, frames=3, size=32, fps=4, write_media=False)
    assert _tree(plain.root) == _tree(resized.root)


@pytest.mark.parametrize(("videos", "seed"), [(5, 0), (10, 3), (40, 11), (200, 0), (333, 5)])
def test_synth_pairs_and_splits_are_identity_disjoint(tmp_path, videos, seed):
    root = synth(tmp_path, videos=videos, seed=seed, write_media=False).root
    reals = _stems(root, "original")
    fakes = {method: _stems(root, method) for method in ("blend-a", "blend-b")}

    n_fake = (3 * videos) // 10
    assert len(fakes["blend-a"]) == len(fakes["blend-b"]) == n_fake
    assert len(reals) == videos - 2 * n_fake
    assert reals[0] == "p000"
    for stems in fakes.values():
        for stem in stems:
            target, source = stem.split("_")
            assert target in reals  # every fake has a real pair
            assert source in reals
            assert target != source

    splits = json.loads((root / "official_splits.json").read_text("utf-8"))
    assert list(splits) == ["train", "val", "test"]
    assert all(splits[name] for name in splits), "every split holds at least one identity"
    listed = [identity for name in splits for identity in splits[name]]
    assert len(listed) == len(set(listed)), "an identity is in one split only"
    assert sorted(listed) == reals


def test_synth_splits_depend_on_the_seed(tmp_path):
    def splits(seed):
        root = synth(tmp_path / str(seed), videos=200, seed=seed, write_media=False).root
        return json.loads((root / "official_splits.json").read_text("utf-8"))

    zero, one = splits(0), splits(1)
    assert zero != one
    assert {name: len(ids) for name, ids in zero.items()} == {
        "train": 48,
        "val": 16,
        "test": 16,
    }


def test_synth_readme_says_how_it_was_made(tmp_path):
    root = synth(tmp_path, videos=20, seed=4, write_media=False).root
    text = (root / "README.txt").read_text("utf-8")
    assert "dfwb datasets synth toyfake --videos 20 --seed 4" in text
    assert "MIT" in text
    assert "synthetic" in text.lower()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"videos": 4}, "at least 5 videos"),
        ({"frames": 0}, "frames"),
        ({"size": 16}, "size"),
        ({"fps": 0}, "fps"),
    ],
)
def test_synth_rejects_bad_arguments(tmp_path, kwargs, match):
    with pytest.raises(ConfigError, match=match):
        synth(tmp_path, write_media=False, **kwargs)
    assert not (tmp_path / "toyfake").exists()


def test_synth_refuses_a_folder_that_is_not_empty(tmp_path):
    (tmp_path / "toyfake").mkdir()
    (tmp_path / "toyfake" / "keep.txt").write_text("mine")
    with pytest.raises(ConfigError, match="not empty") as caught:
        synth(tmp_path, videos=10, write_media=False)
    assert "--out" in caught.value.hint
    assert _tree(tmp_path / "toyfake") == {"keep.txt": b"mine"}


def test_synth_accepts_an_empty_existing_folder(tmp_path):
    (tmp_path / "toyfake").mkdir()
    assert synth(tmp_path, videos=10, write_media=False).n_videos == 10


_NO_NUMPY_NO_AV = """
import importlib.abc, json, sys
from pathlib import Path

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in ("numpy", "av", "torch"):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None

sys.meta_path.insert(0, _Blocker())
from dfwb.preprocess.toyfake import synth
result = synth(Path(sys.argv[1]), videos=20, seed=2, write_media=False)
print(json.dumps({"n": result.n_videos, "loaded": sorted(
    m for m in ("numpy", "av", "torch") if m in sys.modules)}))
"""


def test_synth_without_media_needs_neither_numpy_nor_pyav(tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", _NO_NUMPY_NO_AV, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"n": 20, "loaded": []}


def test_synth_media_without_pyav_raises_before_writing_anything(tmp_path, block_av):
    with pytest.raises(InstallationError) as caught:
        synth(tmp_path, videos=10)
    assert "[preprocess]" in caught.value.hint
    assert "--no-media" in caught.value.hint
    assert not (tmp_path / "toyfake").exists()


# ------------------------------------------------------------------------------ the media


def _decode(path: Path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "ffv1"
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(stream)])


def _energy(frames) -> float:
    """Mean absolute difference between horizontally and vertically neighbouring pixels."""
    np = pytest.importorskip("numpy")
    pixels = frames.astype(np.int16)
    across = np.abs(np.diff(pixels, axis=2)).mean()
    down = np.abs(np.diff(pixels, axis=1)).mean()
    return float((across + down) / 2)


def _patch_box(fake, real) -> tuple[slice, slice]:
    """The bounding box of every pixel where a fake differs from its real pair."""
    np = pytest.importorskip("numpy")
    rows, cols = np.nonzero((fake != real).any(axis=(0, 3)))
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


@pytest.mark.parametrize("method", ["blend-a", "blend-b"])
def test_synth_media_is_lossless_and_has_the_artefact(tmp_path, method):
    pytest.importorskip("av")
    from dfwb.preprocess.toyfake import _plan, _render

    result = synth(tmp_path, videos=10, seed=3)
    plan = {video.relpath: video for video in _plan(10, 3).videos}
    fake_path = sorted((result.root / method).iterdir())[0]
    target = fake_path.stem.split("_")[0]
    real_path = result.root / "original" / f"{target}.mkv"

    fake = _decode(fake_path)
    real = _decode(real_path)
    assert fake.shape == real.shape == (24, 64, 64, 3)
    # FFV1 is lossless: what is decoded is exactly what was generated.
    expected_fake = _render(plan[f"{method}/{fake_path.name}"], seed=3, frames=24, size=64)
    expected_real = _render(plan[f"original/{real_path.name}"], seed=3, frames=24, size=64)
    assert (fake == expected_fake).all()
    assert (real == expected_real).all()

    # Outside one 24x24 window a fake is its real pair, pixel for pixel.
    rows, cols = _patch_box(fake, real)
    assert (rows.stop - rows.start, cols.stop - cols.start) == (24, 24)
    # The window carries a high-frequency artefact that the smooth real does not have.
    ratio = _energy(fake[:, rows, cols]) / _energy(real[:, rows, cols])
    assert ratio >= 5, ratio


def _lag(steps, k: int) -> float:
    """Mean product of first differences ``k`` pixels apart along a row (``steps``: T, H, W, C)."""
    np = pytest.importorskip("numpy")
    return float(np.mean(steps[:, :, :-k] * steps[:, :, k:]))


def test_the_two_methods_carry_different_artefacts():
    np = pytest.importorskip("numpy")
    from dfwb.preprocess.toyfake import _plan, _render

    videos = _plan(10, 0).videos
    reals = {video.stem: video for video in videos if video.task == "REAL"}
    steps = {}
    for task in ("BLEND_A", "BLEND_B"):
        video = next(v for v in videos if v.task == task)
        fake = _render(video, seed=0, frames=2, size=64)
        real = _render(reals[video.target], seed=0, frames=2, size=64)
        rows, cols = _patch_box(fake, real)
        # Along a row, the step from one pixel to the next: tiny on smooth content, large and
        # periodic on the artefact.
        steps[task] = np.diff(fake[:, rows, cols].astype(np.float64), axis=2)
    # A period-2 checkerboard flips sign at every pixel: its steps alternate in sign.
    assert _lag(steps["BLEND_A"], 1) < 0 < _lag(steps["BLEND_A"], 2)
    # A period-3 diagonal stripe repeats every third pixel: its steps do too.
    assert _lag(steps["BLEND_B"], 2) < 0 < _lag(steps["BLEND_B"], 3)
    assert _lag(steps["BLEND_A"], 3) < 0


def test_synth_media_is_seed_deterministic(tmp_path):
    pytest.importorskip("av")
    first = synth(tmp_path / "one", videos=5, seed=9, frames=4)
    second = synth(tmp_path / "two", videos=5, seed=9, frames=4)
    other = synth(tmp_path / "three", videos=5, seed=10, frames=4)

    assert _tree(first.root) == _tree(second.root)  # byte for byte, container included
    same = _decode(first.root / "original" / "p000.mkv")
    changed = _decode(other.root / "original" / "p000.mkv")
    assert same.shape == changed.shape == (4, 64, 64, 3)
    assert (same != changed).any()


def test_synth_media_honours_frames_size_and_fps(tmp_path):
    av = pytest.importorskip("av")
    root = synth(tmp_path, videos=5, seed=0, frames=5, size=96, fps=4).root
    path = root / "original" / "p000.mkv"
    assert _decode(path).shape == (5, 96, 96, 3)
    with av.open(str(path)) as container:
        assert float(container.streams.video[0].average_rate) == 4


def test_the_artefact_stays_inside_the_central_64_pixels(tmp_path):
    pytest.importorskip("av")
    from dfwb.preprocess.toyfake import _plan, _render

    videos = _plan(20, 5).videos
    reals = {video.stem: video for video in videos if video.task == "REAL"}
    for video in videos:
        if video.task == "REAL":
            continue
        fake = _render(video, seed=5, frames=1, size=128)
        real = _render(reals[video.target], seed=5, frames=1, size=128)
        rows, cols = _patch_box(fake, real)
        for span in (rows, cols):
            assert span.start >= 32, video.relpath
            assert span.stop <= 96, video.relpath


# ------------------------------------------------------------------------------ the builder


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory) -> Path:
    return synth(tmp_path_factory.mktemp("toy"), videos=40, seed=0, write_media=False).root


def test_discover_yields_one_record_per_video(toy_root):
    records = collect_records(ToyfakeBuilder(), toy_root)
    assert len(records) == 40
    by_task: dict[str, int] = {}
    for record in records:
        by_task[task_of(record.key)] = by_task.get(task_of(record.key), 0) + 1
    assert by_task == {"REAL": 16, "BLEND_A": 12, "BLEND_B": 12}
    for record in records:
        assert record.builder == BuilderRef("toyfake", ToyfakeBuilder.version)
        assert record.compression is None
        assert record.relpath == f"{TASK_DIRS[task_of(record.key)]}/{local_key(record.key)}.mkv"
        assert record.label_key == f"TOY-{task_of(record.key)}"


def test_records_carry_identity_target_source_and_pair_key(toy_root):
    records = {r.key: r for r in collect_records(ToyfakeBuilder(), toy_root)}
    real = records["REAL/p000"]
    assert (real.identity, real.target_id, real.source_id, real.pair_key) == (
        "p000",
        "p000",
        None,
        None,
    )
    assert real.method == "original"
    fakes = [r for r in records.values() if task_of(r.key) != "REAL"]
    for fake in fakes:
        target, source = local_key(fake.key).split("_")
        assert (fake.identity, fake.target_id, fake.source_id, fake.pair_key) == (
            target,
            target,
            source,
            target,
        )
        assert fake.method == {"BLEND_A": "blend-a", "BLEND_B": "blend-b"}[task_of(fake.key)]


def test_a_fake_whose_name_does_not_parse_is_kept_without_fields(tmp_path):
    root = synth(tmp_path, videos=10, write_media=False).root
    (root / "blend-a" / "stray.mkv").touch()
    builder = ToyfakeBuilder()
    record = {r.key: r for r in collect_records(builder, root)}["BLEND_A/stray"]
    assert (record.identity, record.target_id, record.source_id, record.pair_key) == (None,) * 4
    assert builder.pair_candidates(record) is None


def test_official_splits_follow_the_target_identity(toy_root):
    builder = ToyfakeBuilder()
    records = collect_records(builder, toy_root)
    assignment = builder.official_splits(toy_root, records)
    listed = json.loads((toy_root / "official_splits.json").read_text("utf-8"))
    split_of = {identity: split for split, ids in listed.items() for identity in ids}
    assert assignment == {record.key: split_of[record.identity] for record in records}
    assert set(assignment.values()) == {"train", "val", "test"}


@pytest.mark.parametrize(
    ("content", "error", "match"),
    [
        (None, ConfigError, "official_splits.json is missing"),
        ("not json", ContractError, "cannot read"),
        ("[]", ContractError, "not an object"),
        ('{"train": ["p000"], "dev": ["p001"]}', ContractError, "'dev'"),
        ('{"train": "p000"}', ContractError, "list of identities"),
        ('{"train": ["p000"], "test": ["p000"]}', ContractError, "p000"),
    ],
)
def test_official_splits_reject_a_bad_file(tmp_path, content, error, match):
    root = synth(tmp_path, videos=10, write_media=False).root
    splits = root / "official_splits.json"
    if content is None:
        splits.unlink()
    else:
        splits.write_text(content, encoding="utf-8")
    builder = ToyfakeBuilder()
    with pytest.raises(error, match=match):
        builder.official_splits(root, collect_records(builder, root))


def test_every_fake_pairs_with_its_target_real(toy_root):
    builder = ToyfakeBuilder()
    records = collect_records(builder, toy_root)
    pairs = resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=str(builder.pairing_rule),
        task_rank=builder.task_rank(),
    )
    fakes = [r for r in records if not builder.is_real(r)]
    assert len(pairs) == len(fakes) == 24
    expected = {
        PairRecord(r.key, f"REAL/{local_key(r.key).split('_')[0]}", "target-id") for r in fakes
    }
    assert set(pairs) == expected


def test_label_vocab():
    vocab = ToyfakeBuilder().label_vocab()
    table = {
        key: (v["binary"], v["binary_av"], v["multiclass"], v["family"], v["method"])
        for key, v in vocab.vocab.items()
    }
    assert table == {
        "TOY-REAL": (0, 0, 0, "real", "original"),
        "TOY-BLEND_A": (1, 1, 1, "blend", "blend-a"),
        "TOY-BLEND_B": (1, 1, 2, "blend", "blend-b"),
    }


def test_schemes_benchmark_and_pairing():
    builder = ToyfakeBuilder()
    assert builder.default_scheme == "official"
    assert {name: spec.rule for name, spec in builder.schemes.items()} == {
        "official": "official",
        "ident-72-14-14": "ident-72-14-14",
        "all-test": "all-test",
        "benchmark": "benchmark",
    }
    assert builder.benchmark == BenchmarkSpec(k_fake=20)
    assert builder.pairing_rule == "target-id"
    assert builder.pairing_fanout is None
    assert builder.metadata_files == ("official_splits.json",)
    assert (builder.dataset_id, builder.expected_folder, builder.label_prefix) == (
        "toyfake",
        "toyfake",
        "TOY",
    )


def test_dataset_card_is_synthetic_mit_and_listed():
    builder = ToyfakeBuilder()
    cards = {
        name: SchemeCard(kind=spec.kind, sha256="a" * 64) for name, spec in builder.schemes.items()
    }
    card = builder.dataset_card(cards)
    assert card.id == "toyfake"
    assert card.license.spdx == "MIT"
    assert "synthetic" in card.release.lower()
    assert "dfwb datasets synth toyfake" in card.access
    # Synthetic data made by dfwb itself: nothing stops the lists being redistributed.
    assert card.distribution == "list"
    assert card.terms.notes
    assert card.compressions is None


def test_the_layout_names_every_folder_and_the_split_file():
    text = ToyfakeBuilder().describe_layout()
    for folder in ("original", "blend-a", "blend-b", "official_splits.json"):
        assert folder in text
    assert "dfwb datasets synth toyfake" in text


def test_it_is_registered():
    assert isinstance(get_builder("toyfake"), ToyfakeBuilder)


# ------------------------------------------------------------------------------ the built-in pack


def _regeneration_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("regenerate_toyfake_pack", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_pack_matches_regeneration(isolated):
    fresh = isolated / "pack"
    _regeneration_script().regenerate(fresh)
    committed = _tree(PACK)
    regenerated = _tree(fresh)
    assert sorted(regenerated) == sorted(committed)
    for name, data in committed.items():
        assert regenerated[name] == data, f"{name} differs: run scripts/regenerate_toyfake_pack.py"


def test_regeneration_refuses_a_folder_that_is_not_empty(isolated):
    out = isolated / "pack"
    out.mkdir()
    (out / "stray").touch()
    with pytest.raises(ConfigError, match="not empty"):
        _regeneration_script().regenerate(out)


def test_regeneration_refuses_a_toyfake_folder_override(isolated, monkeypatch):
    monkeypatch.setenv("DFWB_DATASET_TOYFAKE", str(isolated / "elsewhere"))
    with pytest.raises(ConfigError, match="DFWB_DATASET_TOYFAKE"):
        _regeneration_script().regenerate(isolated / "pack")


def test_toyfake_pack_is_lint_clean():
    assert lint_pack(PACK, release=True) == []


def test_the_pack_card_carries_the_dfwb_version():
    text = (PACK / "pack.yaml").read_text("utf-8")
    expected = f"schema_version: 1\nname: toyfake\nversion: {__version__}\ndatasets:\n- toyfake\n"
    assert text == expected


def test_the_notice_records_the_terms_instead_of_a_pending_review():
    notice = (PACK / "toyfake" / "NOTICE.md").read_text("utf-8")
    assert "pending" not in notice.lower()
    assert "distribution: list" in notice
    assert "no real person" in notice.lower()


def test_the_built_in_pack_is_installed():
    (pack,) = [p for p in installed_packs() if p.name == "toyfake"]
    assert pack.provider == "dfwb"
    assert pack.error is None
    assert pack.root == PACK
    assert pack.version == __version__
    protocol = load("toyfake")
    assert protocol.scheme == "official"
    counts = protocol.scheme_card.counts
    assert counts is not None
    assert sum(counts.values()) == 200
    assert len(protocol.pairs()) == 120
