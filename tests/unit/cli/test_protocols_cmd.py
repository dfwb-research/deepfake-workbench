import gzip
import json
from pathlib import Path

import yaml
from tests.unit.preprocess.test_packbuild import setup_packdemo
from tests.unit.protocols.conftest import (
    make_pack,
    register_packs,
    write_release_mismatch_dataset,
    write_toyone_dataset,
)

from dfwb.core.records import (
    BuilderRef,
    DatasetCard,
    InventoryRecord,
    LabelVocab,
    PackCard,
    PackProvenance,
    SplitRow,
    VideoRecord,
    read_jsonl,
    write_jsonl,
)
from dfwb.core.records.protocol import LabelMappingSpec, LicenseInfo
from dfwb.protocols._yaml import read_card
from dfwb.protocols.writer import scheme_card_for, write_dataset_files


def _install_toyone(monkeypatch, tmp_path):
    root = make_pack(
        tmp_path, "toyone-pack", {"toyone": {}}, builders={"toyone": write_toyone_dataset}
    )
    register_packs(monkeypatch, {"toyone-pack": root})
    return root


def _install_release(monkeypatch, tmp_path):
    root = make_pack(
        tmp_path,
        "release-pack",
        {"release": {}},
        builders={"release": write_release_mismatch_dataset},
    )
    register_packs(monkeypatch, {"release-pack": root})
    return root


def _inventory_record(video: VideoRecord, **overrides: object) -> InventoryRecord:
    fields: dict[str, object] = {
        "key": video.key,
        "compression": video.compression,
        "label_key": video.label_key,
        "method": video.method,
        "relpath": f"{video.key}.mp4",
        "builder": BuilderRef("fixture", "1"),
        "identity": video.identity,
        "source_id": video.source_id,
        "target_id": video.target_id,
        "pair_key": video.pair_key,
        "attrs": dict(video.attrs),
    }
    fields.update(overrides)
    return InventoryRecord(**fields)


def test_list_empty(run):
    result = run("protocols", "list")
    assert result.code == 0


def test_list_and_info_work_while_other_packs_are_broken(run, monkeypatch, tmp_path):
    toyone = make_pack(
        tmp_path, "toyone-pack", {"toyone": {}}, builders={"toyone": write_toyone_dataset}
    )
    mixed = make_pack(tmp_path, "mixed", {"good": {}, "bad": {}})
    (mixed / "bad" / "dataset.yaml").write_text("id: [\n")
    ghost = tmp_path / "dfwb_no_such_pack_package" / "pack"
    register_packs(monkeypatch, {"toyone-pack": toyone, "mixed": mixed, "ghost": ghost})

    listed = run("protocols", "list")
    info = run("protocols", "info", "toyone/official")
    good = run("protocols", "info", "good")

    assert listed.code == 0
    assert "toyone/official*" in listed.out
    assert "good/official*" in listed.out
    assert "warning: dataset 'bad' of pack 'mixed' is broken:" in listed.err
    assert "warning: pack 'ghost' is broken:" in listed.err
    assert (info.code, good.code) == (0, 0)
    assert "toyone/official" in info.out


def test_list_human_marks_default(run, monkeypatch, tmp_path):
    _install_toyone(monkeypatch, tmp_path)
    result = run("protocols", "list")
    assert result.code == 0
    assert "PROTOCOL" in result.out
    assert "toyone/official*" in result.out
    assert "toyone/all-test" in result.out
    assert "toyone/all-test*" not in result.out


def test_list_json(run, monkeypatch, tmp_path):
    _install_toyone(monkeypatch, tmp_path)
    data = json.loads(run("protocols", "list", "--json").out)
    rows = {(r["dataset_id"], r["scheme"]): r for r in data}
    assert rows[("toyone", "official")]["default"] is True
    assert rows[("toyone", "official")]["pack"] == "toyone-pack"
    assert rows[("toyone", "official")]["counts"] == {"train": 6, "val": 4, "test": 4}
    assert rows[("toyone", "all-test")]["default"] is False


def test_info_human_and_json(run, monkeypatch, tmp_path):
    _install_toyone(monkeypatch, tmp_path)
    result = run("protocols", "info", "toyone/official")
    assert result.code == 0
    assert "toyone/official" in result.out

    data = json.loads(run("protocols", "info", "toyone/official", "--json").out)
    assert data["ref"] == "toyone/official"
    assert data["pack"] == "toyone-pack"
    assert data["scheme_card"]["kind"] == "official"
    total = sum(row["count"] for row in data["counts"])
    assert total == 14  # 16 rows minus REAL/r4's two unassigned rows


def test_info_unknown_dataset_exits_2_with_hint(run):
    result = run("protocols", "info", "nope")
    assert result.code == 2
    assert "hint: " in result.err


def test_info_exits_4_on_a_malformed_video_row(run, monkeypatch, tmp_path):
    root = _install_toyone(monkeypatch, tmp_path)
    videos_path = root / "toyone" / "videos.jsonl.gz"
    with gzip.open(videos_path, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    del_data = json.loads(lines[0])
    del del_data["key"]
    lines[0] = json.dumps(del_data)
    with gzip.open(videos_path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    result = run("protocols", "info", "toyone/official")
    assert result.code == 4
    assert "videos.jsonl.gz:1: missing 'key'" in result.err
    assert "hint: " in result.err


def test_verify_cli_exit_codes_and_json(run, monkeypatch, tmp_path):
    root = _install_toyone(monkeypatch, tmp_path)
    work_root = tmp_path / "work"
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work_root))
    inventory_path = work_root / "toyone" / "inventory.jsonl"
    inventory_path.parent.mkdir(parents=True)
    videos = read_jsonl(root / "toyone" / "videos.jsonl.gz", VideoRecord)

    write_jsonl(inventory_path, [_inventory_record(v) for v in videos])
    result = run("protocols", "verify", "toyone")
    assert result.code == 0
    assert "have: 16" in result.out
    assert "report written to" in result.out
    report_path = work_root / "toyone" / "verify" / "official.json"
    assert report_path.is_file()

    # Drop a train-split video: exits 3 (partial coverage), still writes the report.
    write_jsonl(inventory_path, [_inventory_record(v) for v in videos if v.key != "FAKE_A/a1"])
    result = run("protocols", "verify", "toyone")
    assert result.code == 3
    assert "missing_requested: 1" in result.out

    # Relabel a video: exits 4 (contract mismatch).
    write_jsonl(
        inventory_path,
        [
            _inventory_record(v, method="Wrong") if v.key == "FAKE_A/a1" else _inventory_record(v)
            for v in videos
        ],
    )
    result = run("protocols", "verify", "toyone/official", "--json")
    assert result.code == 4
    data = json.loads(result.out)
    assert data["counts"]["label_mismatch"] == 1
    assert data["exit_code"] == 4
    assert data["report_path"] == str(report_path)


def test_verify_cli_refuses_a_work_root_inside_a_datasets_root(run, monkeypatch, tmp_path):
    root = _install_toyone(monkeypatch, tmp_path)
    datasets = tmp_path / "datasets"
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(datasets))
    monkeypatch.setenv("DFWB_WORK_ROOT", str(datasets / "dfwb-work"))
    inventory = tmp_path / "inventory.jsonl"
    videos = read_jsonl(root / "toyone" / "videos.jsonl.gz", VideoRecord)
    write_jsonl(inventory, [_inventory_record(v) for v in videos])

    result = run("protocols", "verify", "toyone", "--inventory", str(inventory))

    assert result.code == 2
    assert "DFWB_WORK_ROOT" in result.err
    assert not (datasets / "dfwb-work").exists()


def test_verify_cli_prints_release_mismatch_warning(run, monkeypatch, tmp_path):
    root = _install_release(monkeypatch, tmp_path)
    work_root = tmp_path / "work"
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work_root))
    videos = read_jsonl(root / "release" / "videos.jsonl.gz", VideoRecord)
    kept = [v for v in videos if int(v.key.rsplit("v", 1)[1]) <= 20]
    records = [_inventory_record(v) for v in kept]
    for i in range(1, 11):
        records.append(
            InventoryRecord(
                key=f"FAKE_A/x{i:02d}",
                compression=None,
                label_key="RELEASE-FAKE_A",
                method="FakeA",
                relpath=f"FAKE_A/x{i:02d}.mp4",
                builder=BuilderRef("fixture", "1"),
            )
        )
    inventory_path = work_root / "release" / "inventory.jsonl"
    inventory_path.parent.mkdir(parents=True)
    write_jsonl(inventory_path, records)

    result = run("protocols", "verify", "release")

    assert "warning: FAKE_A: 10 missing and 10 extra" in result.out


def test_verify_without_inventory_hints_inventory_build(run, monkeypatch, tmp_path):
    _install_toyone(monkeypatch, tmp_path)
    monkeypatch.setenv("DFWB_WORK_ROOT", str(tmp_path / "work"))

    result = run("protocols", "verify", "toyone")

    assert result.code == 2
    assert "hint: run: dfwb inventory build toyone" in result.err


def test_verify_cli_rejects_unknown_split_with_suggestion(run, monkeypatch, tmp_path):
    root = _install_toyone(monkeypatch, tmp_path)
    work_root = tmp_path / "work"
    monkeypatch.setenv("DFWB_WORK_ROOT", str(work_root))
    inventory_path = work_root / "toyone" / "inventory.jsonl"
    inventory_path.parent.mkdir(parents=True)
    videos = read_jsonl(root / "toyone" / "videos.jsonl.gz", VideoRecord)
    write_jsonl(inventory_path, [_inventory_record(v) for v in videos])

    result = run("protocols", "verify", "toyone", "--split", "tset")

    assert result.code == 2
    assert "split 'tset' is not in this scheme" in result.err
    assert "did you mean 'test'" in result.err
    assert "hint: splits in this scheme: test, train, val" in result.err


# -------------------------------------------------------------------------------------------
# `dfwb protocols lint`: pack integrity checks. ``lint``/``diff`` take a pack directory path
# directly, so no registry install is needed.
# -------------------------------------------------------------------------------------------


def _write_lint_pack(root: Path, *, distribution: str = "list") -> Path:
    dataset_dir = root / "toy"
    rows = [SplitRow("REAL/r1", None, "train"), SplitRow("REAL/r2", None, "test")]
    scheme = scheme_card_for(rows, kind="official", rule="official")
    card = DatasetCard(
        id="toy",
        name="toy",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only; never media",
        distribution=distribution,
        modalities=["video"],
        key_rule="<task>/<stem>",
        schemes={"official": scheme},
        default_scheme="official",
    )
    labels = LabelVocab(
        vocab={"TOY-REAL": {"binary": 0}}, mappings={"binary": LabelMappingSpec(from_="binary")}
    )
    provenance = PackProvenance(
        builder={"id": "toy", "version": "1"},
        dfwb="0.1.0",
        source_listing_sha256="0" * 64,
        rules={"official": {"rule": "official", "params": {}}},
    )
    write_dataset_files(
        dataset_dir,
        videos=[
            VideoRecord("REAL/r1", None, "TOY-REAL", "original", identity="r1"),
            VideoRecord("REAL/r2", None, "TOY-REAL", "original", identity="r2"),
        ],
        schemes={"official": rows},
        pairs=[],
        card=card,
        labels=labels,
        provenance=provenance,
        notice="# toy\n\nSynthetic fixture; never real media.\n",
    )
    pack_card = PackCard(schema_version=1, name="lint-pack", version="1.0.0", datasets=["toy"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "pack.yaml").write_text(
        yaml.safe_dump(pack_card.model_dump(mode="json", by_alias=True), sort_keys=True)
    )
    return root


def test_lint_cli_warns_on_undecided_distribution_and_exits_0(run, tmp_path):
    root = _write_lint_pack(tmp_path, distribution="undecided")

    result = run("protocols", "lint", str(root))

    assert result.code == 0
    assert "warning: toy/dataset.yaml: distribution is undecided" in result.out


def test_lint_cli_release_flag_turns_the_warning_into_an_error(run, tmp_path):
    root = _write_lint_pack(tmp_path, distribution="undecided")

    result = run("protocols", "lint", str(root), "--release")

    assert result.code == 4
    assert "error: toy/dataset.yaml: distribution is undecided" in result.out


def test_lint_cli_exits_4_when_any_error_is_found(run, tmp_path):
    root = _write_lint_pack(tmp_path)
    (root / "toy" / "NOTICE.md").unlink()

    result = run("protocols", "lint", str(root))

    assert result.code == 4
    assert "error: toy/NOTICE.md: file is missing" in result.out


def test_lint_cli_json_output(run, tmp_path):
    root = _write_lint_pack(tmp_path)

    data = json.loads(run("protocols", "lint", str(root), "--json").out)

    assert data == []


# -------------------------------------------------------------------------------------------
# `dfwb protocols diff`: the required SemVer bump between two pack directories.
# -------------------------------------------------------------------------------------------


def _diff_rows() -> list[SplitRow]:
    return [
        SplitRow("REAL/r1", None, "train"),
        SplitRow("REAL/r2", None, "test"),
        SplitRow("FAKE/f1", None, "train"),
        SplitRow("FAKE/f2", None, "test"),
    ]


def _write_diff_pack(root: Path, version: str, rows: list[SplitRow]) -> Path:
    dataset_dir = root / "diffcli"
    scheme = scheme_card_for(rows, kind="official", rule="official")
    card = DatasetCard(
        id="diffcli",
        name="diffcli",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only; never media",
        distribution="list",
        modalities=["video"],
        key_rule="<task>/<stem>",
        schemes={"official": scheme},
        default_scheme="official",
    )
    labels = LabelVocab(
        vocab={"DIFFCLI-REAL": {"binary": 0}, "DIFFCLI-FAKE": {"binary": 1}},
        mappings={"binary": LabelMappingSpec(from_="binary")},
    )
    provenance = PackProvenance(
        builder={"id": "diffcli", "version": "1"},
        dfwb="0.1.0",
        source_listing_sha256="0" * 64,
        rules={"official": {"rule": "official", "params": {}}},
    )
    write_dataset_files(
        dataset_dir,
        videos=[
            VideoRecord("REAL/r1", None, "DIFFCLI-REAL", "original", identity="r1"),
            VideoRecord("REAL/r2", None, "DIFFCLI-REAL", "original", identity="r2"),
            VideoRecord("FAKE/f1", None, "DIFFCLI-FAKE", "FakeA", identity="f1", target_id="r1"),
            VideoRecord("FAKE/f2", None, "DIFFCLI-FAKE", "FakeA", identity="f2", target_id="r2"),
        ],
        schemes={"official": rows},
        pairs=[],
        card=card,
        labels=labels,
        provenance=provenance,
        notice="# diffcli\n\nSynthetic fixture; never real media.\n",
    )
    pack_card = PackCard(schema_version=1, name="diff-pack", version=version, datasets=["diffcli"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "pack.yaml").write_text(
        yaml.safe_dump(pack_card.model_dump(mode="json", by_alias=True), sort_keys=True)
    )
    return root


def _moved_rows() -> list[SplitRow]:
    return [SplitRow("REAL/r1", None, "test") if r.key == "REAL/r1" else r for r in _diff_rows()]


def test_diff_cli_prints_the_table_and_required_bump(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.0.1", _diff_rows())

    result = run("protocols", "diff", str(old), str(new))

    assert result.code == 0
    assert "diffcli" in result.out
    assert "official" in result.out
    assert "required bump: none" in result.out


def test_diff_cli_json_reports_a_moved_video_as_a_major_bump(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.1.0", _moved_rows())

    data = json.loads(run("protocols", "diff", str(old), str(new), "--json").out)

    assert data["required_bump"] == "major"
    assert data["schemes"] == [
        {
            "dataset": "diffcli",
            "scheme": "official",
            "status": "changed",
            "added": 0,
            "removed": 0,
            "moved": 1,
        }
    ]


def test_diff_cli_expect_bump_exits_4_when_the_claimed_bump_is_smaller_than_required(run, tmp_path):
    # The claim is what is checked: a moved video needs a major bump, and "minor" is not enough,
    # whatever the pack.yaml versions say.
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "2.0.0", _moved_rows())

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "minor")

    assert result.code == 4
    assert "'minor'" in result.err
    assert "'major'" in result.err


def test_diff_cli_expect_bump_passes_when_the_claim_covers_the_changes(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.1.0", _moved_rows())

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "major")

    assert result.code == 0


def test_diff_cli_expect_bump_accepts_none_or_more_for_an_unchanged_pack(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.0.0", _diff_rows())

    for claim in ("none", "patch", "major"):
        result = run("protocols", "diff", str(old), str(new), "--expect-bump", claim, "--json")
        assert result.code == 0, claim
        data = json.loads(result.out)
        assert (data["required_bump"], data["expected_bump"]) == ("none", claim)


def test_diff_cli_expect_bump_accepts_pre_release_versions(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "0.1.0a1", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "0.1.0a2", _diff_rows())

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "none", "--json")

    assert result.code == 0, result.err
    data = json.loads(result.out)
    assert (data["required_bump"], data["actual_bump"]) == ("none", "none")


def test_diff_cli_expect_bump_none_fails_on_a_text_change(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.0.0", _diff_rows())
    (new / "diffcli" / "NOTICE.md").write_text("# diffcli\n\nReworded.\n")

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "none")

    assert result.code == 4
    assert "'none'" in result.err
    assert "'patch'" in result.err


def test_diff_cli_lists_relabelled_videos(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.0.1", _diff_rows())
    videos = [
        VideoRecord(v.key, v.compression, "DIFFCLI-REAL", v.method) if v.key == "FAKE/f1" else v
        for v in read_jsonl(new / "diffcli" / "videos.jsonl.gz", VideoRecord)
    ]
    write_jsonl(new / "diffcli" / "videos.jsonl.gz", videos)

    text = run("protocols", "diff", str(old), str(new))
    data = json.loads(run("protocols", "diff", str(old), str(new), "--json").out)

    assert "videos relabelled: 1" in text.out
    assert "diffcli/FAKE/f1|" in text.out
    assert "required bump: major" in text.out
    assert data["relabelled"] == ["diffcli/FAKE/f1|"]
    assert data["relabelled_count"] == 1
    assert data["required_bump"] == "major"


def test_diff_cli_expect_bump_exits_4_on_a_version_downgrade(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.2.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.1.0", _diff_rows())

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "patch")

    assert result.code == 4
    assert "downgrade" in result.err


def test_diff_cli_expect_bump_exits_4_on_a_malformed_version(run, tmp_path):
    old = _write_diff_pack(tmp_path / "old", "1.0", _diff_rows())
    new = _write_diff_pack(tmp_path / "new", "1.1.0", _diff_rows())

    result = run("protocols", "diff", str(old), str(new), "--expect-bump", "minor")

    assert result.code == 4
    assert "is not MAJOR.MINOR.PATCH" in result.err


# -------------------------------------------------------------------------------------------
# `dfwb protocols new-pack`: scaffolding a new protocol pack distribution.
# -------------------------------------------------------------------------------------------


def test_new_pack_cli_writes_files_and_prints_them(run, tmp_path):
    result = run("protocols", "new-pack", "my-pack", "--name", "my-pack")

    assert result.code == 0
    written = tmp_path / "my-pack"
    assert (written / "pyproject.toml").is_file()
    assert (written / "src" / "my_pack" / "packs" / "pack.yaml").is_file()
    assert "wrote my-pack/pyproject.toml" in result.out


def test_new_pack_cli_json_reports_the_written_files(run, tmp_path):
    data = json.loads(run("protocols", "new-pack", "my-pack", "--name", "my-pack", "--json").out)

    assert data["directory"] == "my-pack"
    assert "my-pack/pyproject.toml" in data["written"]
    assert len(data["written"]) == 7


def test_new_pack_cli_accepts_an_author(run, tmp_path):
    result = run("protocols", "new-pack", "my-pack", "--name", "my-pack", "--author", "Ada")

    assert result.code == 0
    pyproject = (tmp_path / "my-pack" / "src" / "my_pack" / "__init__.py").read_text("utf-8")
    assert "my-pack" in pyproject
    notice = (tmp_path / "my-pack" / "LICENSE-DATA").read_text("utf-8")
    assert "Ada" in notice


def test_new_pack_cli_rejects_a_bad_name_with_exit_2(run, tmp_path):
    result = run("protocols", "new-pack", "my-pack", "--name", "Not_Kebab")

    assert result.code == 2
    assert "hint: " in result.err


def test_new_pack_cli_rejects_a_non_empty_directory_with_exit_2(run, tmp_path):
    (tmp_path / "my-pack").mkdir()
    (tmp_path / "my-pack" / "keep.txt").write_text("hi")

    result = run("protocols", "new-pack", "my-pack", "--name", "my-pack")

    assert result.code == 2
    assert "is not empty" in result.err


def test_new_pack_cli_rejects_a_bad_author_with_exit_2(run, tmp_path):
    result = run("protocols", "new-pack", "my-pack", "--name", "my-pack", "--author", 'bad"author')

    assert result.code == 2
    assert "hint: " in result.err
    assert not (tmp_path / "my-pack").exists()


# -------------------------------------------------------------------------------------------
# `dfwb protocols build` and `dfwb protocols materialize`, on the synthetic packdemo dataset.
# -------------------------------------------------------------------------------------------


def test_build_cli_json_updates_pack_yaml_and_lints_clean(run, monkeypatch, tmp_path):
    paths = setup_packdemo(monkeypatch, tmp_path)
    out = paths["pack"] / "packdemo"

    result = run(
        "protocols", "build", "packdemo", "--out", str(out), "--update-pack-yaml", "--json"
    )

    assert result.code == 0, result.err
    data = json.loads(result.out)
    assert data["dataset_id"] == "packdemo"
    assert data["out"] == str(out)
    assert (data["n_videos"], data["n_pairs"]) == (30, 20)
    assert sorted(data["schemes"]) == ["all-test", "benchmark", "ident-72-14-14", "official"]
    assert data["schemes"]["official"]["counts"] == {"test": 6, "train": 18, "val": 6}
    assert data["schemes"]["official"]["sha256"] == read_card(out).schemes["official"].sha256
    assert data["pack_yaml"] == {"path": str(paths["pack"] / "pack.yaml"), "added": True}
    assert yaml.safe_load((paths["pack"] / "pack.yaml").read_text("utf-8"))["datasets"] == [
        "packdemo"
    ]
    lint = run("protocols", "lint", str(paths["pack"]))
    assert lint.code == 0
    assert "error:" not in lint.out

    again = run("protocols", "build", "packdemo", "--out", str(out), "--update-pack-yaml")
    assert again.code == 0
    assert "already lists packdemo" in again.out


def test_build_cli_human_output_and_scheme_selection(run, monkeypatch, tmp_path):
    setup_packdemo(monkeypatch, tmp_path)
    out = tmp_path / "somewhere" / "packdemo"

    result = run(
        "protocols",
        "build",
        "packdemo",
        "--out",
        str(out),
        "--scheme",
        "official",
        "--scheme",
        "all-test",
    )

    assert result.code == 0, result.err
    assert f"wrote packdemo to {out}: 30 videos, 20 pairs" in result.out
    assert "official" in result.out
    assert "all-test" in result.out
    assert "benchmark" not in result.out
    assert sorted(p.name for p in (out / "splits").iterdir()) == [
        "all-test.tsv.gz",
        "official.tsv.gz",
    ]


def test_build_cli_update_pack_yaml_needs_an_out_named_after_the_dataset(
    run, monkeypatch, tmp_path
):
    paths = setup_packdemo(monkeypatch, tmp_path)

    result = run(
        "protocols", "build", "packdemo", "--out", str(paths["pack"] / "demo"), "--update-pack-yaml"
    )

    assert result.code == 2
    assert "hint: " in result.err
    assert not (paths["pack"] / "demo").exists()


def test_build_cli_update_pack_yaml_with_out_dot_inside_the_dataset_folder(
    run, monkeypatch, tmp_path
):
    paths = setup_packdemo(monkeypatch, tmp_path)
    out = paths["pack"] / "packdemo"
    out.mkdir()
    monkeypatch.chdir(out)

    result = run("protocols", "build", "packdemo", "--out", ".", "--update-pack-yaml", "--json")

    assert result.code == 0, result.err
    data = json.loads(result.out)
    assert data["out"] == str(out)
    assert data["pack_yaml"] == {"path": str(paths["pack"] / "pack.yaml"), "added": True}
    assert (out / "dataset.yaml").is_file()


def test_build_cli_without_an_inventory_hints_inventory_build(run, monkeypatch, tmp_path):
    paths = setup_packdemo(monkeypatch, tmp_path)
    paths["inventory"].unlink()

    result = run("protocols", "build", "packdemo", "--out", str(paths["pack"] / "packdemo"))

    assert result.code == 2
    assert "hint: run: dfwb inventory build packdemo" in result.err


def test_build_cli_with_an_explicit_inventory(run, monkeypatch, tmp_path):
    paths = setup_packdemo(monkeypatch, tmp_path)
    moved = tmp_path / "moved.jsonl"
    paths["inventory"].rename(moved)

    result = run(
        "protocols",
        "build",
        "packdemo",
        "--out",
        str(paths["pack"] / "packdemo"),
        "--inventory",
        str(moved),
        "--json",
    )

    assert result.code == 0, result.err
    assert json.loads(result.out)["pack_yaml"] is None


def test_materialize_cli_exit_codes(run, monkeypatch, tmp_path):
    paths = setup_packdemo(monkeypatch, tmp_path)
    out = paths["pack"] / "packdemo"
    assert run("protocols", "build", "packdemo", "--out", str(out), "--update-pack-yaml").code == 0
    for scheme in ("official", "ident-72-14-14"):
        (out / "splits" / f"{scheme}.tsv.gz").unlink()

    # The official rule needs the publisher's split: the command reads it through the builder.
    result = run("protocols", "materialize", "packdemo/official", "--json")
    assert result.code == 0, result.err
    data = json.loads(result.out)
    assert data == {
        "ref": "packdemo/official",
        "path": str(paths["work"] / "packdemo" / "materialized" / "splits" / "official.tsv.gz"),
        "sha256": read_card(out).schemes["official"].sha256,
        "matched": True,
    }

    result = run("protocols", "materialize", "packdemo/ident-72-14-14")
    assert result.code == 0, result.err
    assert "matches the published hash" in result.out
    assert run("protocols", "info", "packdemo/ident-72-14-14").code == 0

    # A local copy that differs from the release the pack describes: exit 4.
    short = tmp_path / "short.jsonl"
    write_jsonl(short, read_jsonl(paths["inventory"], InventoryRecord)[1:])
    result = run("protocols", "materialize", "packdemo/ident-72-14-14", "--inventory", str(short))
    assert result.code == 4
    assert "materialized hash" in result.err
    assert (
        "hint: your local copy differs from the release the pack describes; "
        "run dfwb protocols verify"
    ) in result.err

    # An unknown dataset is a usage error.
    assert run("protocols", "materialize", "nope/official").code == 2


def test_materialize_cli_without_the_dataset_folder_hints_location(run, monkeypatch, tmp_path):
    paths = setup_packdemo(monkeypatch, tmp_path)
    out = paths["pack"] / "packdemo"
    assert run("protocols", "build", "packdemo", "--out", str(out), "--update-pack-yaml").code == 0
    (out / "splits" / "official.tsv.gz").unlink()
    (paths["raw"] / "PackDemo").rename(paths["raw"] / "Elsewhere")

    result = run("protocols", "materialize", "packdemo/official")

    assert result.code == 2
    assert "hint: locate the dataset folder (see dfwb datasets info packdemo)" in result.err


def test_materialize_cli_without_an_inventory_hints_inventory_build(run, monkeypatch, tmp_path):
    paths = setup_packdemo(monkeypatch, tmp_path)
    out = paths["pack"] / "packdemo"
    assert run("protocols", "build", "packdemo", "--out", str(out), "--update-pack-yaml").code == 0
    paths["inventory"].unlink()

    result = run("protocols", "materialize", "packdemo/all-test")

    assert result.code == 2
    assert "hint: run: dfwb inventory build packdemo" in result.err
