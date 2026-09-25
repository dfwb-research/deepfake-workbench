"""``run_experiment``: checks before any data loads, the self-describing run directory, and one
run per seed."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

pytest.importorskip("lightning")

import torch
from safetensors.torch import load_file
from tests.unit.train._toy import (
    PROFILE,
    PROTOCOL,
    toy_config,
    toy_profile,
    toy_source,
    write_toy_config,
    write_toy_store,
)

from dfwb.core.config import load_config
from dfwb.core.config.loader import LoadedConfig, dump_yaml
from dfwb.core.config.schema import TrainConfig
from dfwb.core.errors import ConfigError
from dfwb.models.source import load_run
from dfwb.protocols.protocol import load as load_protocol
from dfwb.train import run as run_module
from dfwb.train.datamodule import ProtocolDataModule
from dfwb.train.run import RunResult, run_experiment

SOURCE = "toytrain-official"


def load(directory: Path, config: TrainConfig) -> LoadedConfig:
    return load_config(write_toy_config(directory, config))


def train(tmp_path: Path, work_root: Path, config: TrainConfig | None = None) -> list[RunResult]:
    return run_experiment(
        load(tmp_path, config or toy_config()),
        work_root=work_root,
        runs_root=tmp_path / "runs",
        progress=False,
    )


def _files(directory: Path) -> list[str]:
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())


def _no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(self, stage=None):
        raise AssertionError("data was loaded")

    monkeypatch.setattr(ProtocolDataModule, "setup", _refuse)


# ------------------------------------------------------------------------- run directory


def test_run_directory_layout_after_one_epoch(tmp_path, toy_work_root):
    (result,) = train(tmp_path, toy_work_root)
    run_dir = result.run_dir

    assert run_dir.parent == tmp_path / "runs" / "toy"
    assert re.fullmatch(r"\d{8}-\d{6}-s0", run_dir.name)
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "checkpoints",
        "config.resolved.yaml",
        "data.json",
        "env.json",
        "fingerprint.txt",
        "logs",
        "metrics.json",
        "report.md",
        "scores",
    ]
    assert _files(run_dir / "checkpoints") == [
        "best/detector.json",
        "best/model.safetensors",
        "last/detector.json",
        "last/model.safetensors",
    ]
    assert _files(run_dir / "scores") == [
        f"val/{SOURCE}.scores.csv",
        f"val/{SOURCE}.scores.meta.json",
    ]
    assert "metrics.csv" in {p.name for p in (run_dir / "logs").iterdir()}
    # weights and JSON only: nothing pickled anywhere in the run
    assert not [p for p in run_dir.rglob("*") if p.suffix in {".ckpt", ".pt", ".pth", ".pkl"}]


def test_the_run_records_how_it_was_produced(tmp_path, toy_work_root):
    loaded = load(tmp_path, toy_config())
    (result,) = run_experiment(
        loaded,
        work_root=toy_work_root,
        runs_root=tmp_path / "runs",
        progress=False,
        argv=["dfwb", "train", "-c", str(tmp_path / "exp.yaml")],
    )
    run_dir = result.run_dir

    assert (run_dir / "config.resolved.yaml").read_text("utf-8") == dump_yaml(loaded.data)
    assert (run_dir / "fingerprint.txt").read_text("utf-8") == f"{loaded.fingerprint}\n"

    env = json.loads((run_dir / "env.json").read_text("utf-8"))
    assert env["seed"] == 0
    assert env["env"]["dfwb"]
    assert env["env"]["device"] == "cpu"
    assert env["command"] == "dfwb train -c '<abs>/exp.yaml'"  # no machine-specific path
    assert env["created"].endswith("Z")
    assert "plugins" in env
    assert "git" in env

    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics == result.metrics
    assert (metrics["seed"], metrics["epochs"], metrics["global_step"]) == (0, 1, 4)
    assert metrics["fingerprint"] == loaded.fingerprint
    assert metrics["monitor"]["key"] == "val/video_auc"
    assert metrics["monitor"]["best_epoch"] == 0
    assert metrics["val"]["val/video_auc"] == pytest.approx(metrics["monitor"]["best"])
    assert f"val/{SOURCE}/eer" in metrics["val"]

    report = (run_dir / "report.md").read_text("utf-8")
    assert loaded.fingerprint in report
    assert "val/video_auc" in report
    assert SOURCE in report


def test_data_json_records_every_source(tmp_path, toy_pack, deterministic_torch):
    work_root = tmp_path / "work"
    write_toy_store(work_root, skip=["REAL/r00", "FAKE/f11"])  # one train, one val video
    (result,) = train(tmp_path, work_root)

    data = json.loads((result.run_dir / "data.json").read_text("utf-8"))
    assert (data["processing"], data["labels"]) == (PROFILE, "binary")
    assert (data["pairs"], data["allow_input_mismatch"]) == (False, False)
    protocol = load_protocol(PROTOCOL, work_root=work_root)
    profile = toy_profile()
    for role, split, excluded in [("train", "train", 1), ("val", "val", 1)]:
        (source,) = data[role]
        assert source["name"] == SOURCE
        assert source["protocol"] == {
            "ref": PROTOCOL,
            "id": protocol.ref,
            "split": split,
            "where": {},
            "pack": "toytrain-pack",
            "pack_version": "1.0.0",
            "scheme_sha256": protocol.sha256,
        }
        assert source["profile"] == {"id": profile.profile_id(), "sha256": profile.sha256()}
        assert source["input_adaptation"]["mismatch"] is False
        (summary,) = source["index"]["sources"]
        assert summary["counts"]["excluded"] == {"not-processed": excluded}
        assert summary["counts"]["in_split"] == 16 if split == "train" else 8


def test_an_allowed_input_mismatch_is_recorded(tmp_path, toy_pack, deterministic_torch):
    # a store cropped tighter (1.0) than the detector asks for (1.3): refused unless allowed
    work_root = tmp_path / "work"
    write_toy_store(work_root, profile=toy_profile(scale=1.0))
    config = toy_config(data={"allow_input_mismatch": True})
    (result,) = train(tmp_path, work_root, config)

    data = json.loads((result.run_dir / "data.json").read_text("utf-8"))
    assert data["allow_input_mismatch"] is True
    adaptation = data["train"][0]["input_adaptation"]
    assert adaptation["mismatch"] is True
    assert adaptation["reason"]


def test_input_overrides_reach_the_detector(tmp_path, toy_work_root):
    config = toy_config(model={"input": {"size": [32, 32]}})
    (result,) = train(tmp_path, toy_work_root, config)
    detector = load_run(str(result.run_dir))
    assert detector.meta.input.size == (32, 32)
    assert detector.meta.source == f"run:{load(tmp_path, config).fingerprint}"


def test_latest_is_a_relative_symlink_to_the_newest_run(tmp_path, toy_work_root):
    (first,) = train(tmp_path, toy_work_root)
    (second,) = train(tmp_path, toy_work_root, toy_config(run={"name": "toy", "seeds": [1]}))

    latest = tmp_path / "runs" / "toy" / "latest"
    assert latest.is_symlink()
    assert latest.readlink() == Path(second.run_dir.name)  # relative: the root can move
    assert latest.resolve() == second.run_dir.resolve()
    assert first.run_dir != second.run_dir
    load_run(str(latest))  # the default `#best` checkpoint resolves through it


def test_runs_started_in_the_same_second_get_distinct_directories(
    tmp_path, toy_work_root, monkeypatch
):
    monkeypatch.setattr(run_module, "_timestamp", lambda: "20260102-030405")
    (first,) = train(tmp_path, toy_work_root)
    (second,) = train(tmp_path, toy_work_root)
    assert first.run_dir.name == "20260102-030405-s0"
    assert second.run_dir.name == "20260102-030405-s0-2"


def test_several_seeds_give_one_run_each(tmp_path, toy_work_root):
    config = toy_config(run={"name": "toy", "seeds": [3, 4]})
    results = train(tmp_path, toy_work_root, config)

    assert [r.seed for r in results] == [3, 4]
    assert [r.run_dir.name.rsplit("-s", 1)[1] for r in results] == ["3", "4"]
    fingerprints = {(r.run_dir / "fingerprint.txt").read_text("utf-8") for r in results}
    assert len(fingerprints) == 1  # the same experiment, run with two seeds
    for result, seed in zip(results, [3, 4], strict=True):
        assert json.loads((result.run_dir / "env.json").read_text("utf-8"))["seed"] == seed
        assert result.metrics["seed"] == seed
    weights = [load_file(r.run_dir / "checkpoints/last/model.safetensors") for r in results]
    assert not torch.equal(weights[0]["head.linear.weight"], weights[1]["head.linear.weight"])
    assert (tmp_path / "runs" / "toy" / "latest").resolve() == results[1].run_dir.resolve()


def test_repeated_seeds_are_refused(tmp_path, toy_work_root, monkeypatch):
    _no_data(monkeypatch)
    config = toy_config(run={"name": "toy", "seeds": [3, 3]})
    with pytest.raises(ConfigError, match=r"run\.seeds: 3 appears more than once"):
        train(tmp_path, toy_work_root, config)


def test_without_val_sources_best_is_a_copy_of_last(tmp_path, toy_work_root, caplog):
    with caplog.at_level(logging.WARNING, logger="dfwb"):
        (result,) = train(tmp_path, toy_work_root, toy_config(data={"val": []}))

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("no validation sources" in m and "monitoring is off" in m for m in warnings)
    checkpoints = result.run_dir / "checkpoints"
    best = load_file(checkpoints / "best" / "model.safetensors")
    last = load_file(checkpoints / "last" / "model.safetensors")
    assert best.keys() == last.keys()
    assert all(torch.equal(best[k], last[k]) for k in best)
    load_run(str(result.run_dir))  # run:<dir> defaults to #best, which resolves
    assert result.metrics["monitor"] is None
    assert result.metrics["val"] == {}
    assert not (result.run_dir / "scores").exists()


def test_training_on_pairs_is_wired_from_the_config(tmp_path, toy_work_root, monkeypatch):
    built: list[ProtocolDataModule] = []
    original = ProtocolDataModule.setup

    def _spy(self, stage=None):
        built.append(self)
        original(self, stage)

    monkeypatch.setattr(ProtocolDataModule, "setup", _spy)
    config = toy_config(data={"pairs": True, "loader": {"batch_size": 4, "num_workers": 0}})
    (result,) = train(tmp_path, toy_work_root, config)

    assert built
    assert all(datamodule.pairs for datamodule in built)
    data = json.loads((result.run_dir / "data.json").read_text("utf-8"))
    assert data["pairs"] is True


def test_early_stop_and_guards_come_from_the_config(tmp_path, toy_work_root):
    config = toy_config(
        train={
            "max_epochs": 5,
            "early_stop": {"patience": 1, "min_delta": 10.0},
            "heartbeat_steps": 2,
            "lightning": {"log_every_n_steps": 1},
        }
    )
    (result,) = train(tmp_path, toy_work_root, config)
    assert result.metrics["epochs"] == 2  # stopped after one epoch without improvement
    assert (result.run_dir / "logs" / "heartbeat.json").is_file()


# ------------------------------------------------------------------------- early checks


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {
                "model": {
                    "backbone": {"name": "tiny-cnn", "freeze": {"mode": "partail"}},
                    "temporal_pool": {"name": "mean"},
                    "head": {"name": "linear"},
                }
            },
            "model.backbone.freeze.mode: 'partail' is not one of "
            "['none', 'full', 'partial', 'norm-only', 'lora'] (did you mean 'partial'?)",
        ),
        (
            {"data": {"transforms": {"train": [{"name": "normalize", "mean": [0.5], "std": [1]}]}}},
            "normalize does not belong in transforms.train",
        ),
        (
            {"optim": {"name": "adamw", "lr": 0.01, "groups": {"blocks.9": {"lr_scale": 0.1}}}},
            "optim.groups.blocks.9: unknown group",
        ),
        (
            {"schedule": {"name": "cosine", "warmup_epoch": 1}},
            "schedule.warmup_epoch: unknown key (did you mean 'warmup_epochs'?)",
        ),
        (
            {"train": {"lightning": {"determinstic": True}}},
            "train.lightning: 'determinstic' is not a Lightning Trainer argument "
            "(did you mean 'deterministic'?)",
        ),
        (
            {"train": {"monitor": "loss"}},
            "train.monitor: 'loss' is not a validation value",
        ),
        (
            {"train": {"precision": "64-bogus"}},
            "train: Lightning refused these trainer options",
        ),
    ],
    ids=["component", "transform", "optim-group", "schedule", "lightning", "monitor", "precision"],
)
def test_config_errors_are_raised_before_any_data_loads(
    tmp_path, toy_work_root, monkeypatch, changes, expected
):
    _no_data(monkeypatch)
    with pytest.raises(ConfigError) as caught:
        train(tmp_path, toy_work_root, toy_config(**changes))
    assert expected in caught.value.message
    assert not (tmp_path / "runs").exists()  # nothing written either


def test_the_monitor_is_checked_against_the_sources_before_training(tmp_path, toy_work_root):
    config = toy_config(train={"monitor": f"val/{SOURCE[:-1]}/auc"})
    with pytest.raises(ConfigError) as caught:
        train(tmp_path, toy_work_root, config)
    assert f"did you mean 'val/{SOURCE}/auc'" in caught.value.message
    assert not (tmp_path / "runs").exists()


def test_a_missing_work_root_is_named(tmp_path, monkeypatch):
    monkeypatch.delenv("DFWB_WORK_ROOT", raising=False)
    monkeypatch.delenv("DFWB_DATASETS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    with pytest.raises(ConfigError, match="DFWB_WORK_ROOT is not set"):
        run_experiment(load(tmp_path, toy_config()), runs_root=tmp_path / "runs", progress=False)


def test_runs_go_under_the_configured_output_root(tmp_path, toy_work_root):
    config = toy_config(run={"name": "toy", "seeds": [0], "output_root": str(tmp_path / "out")})
    (result,) = train(tmp_path, toy_work_root, config)
    assert result.run_dir.parent == tmp_path / "out" / "toy"


def test_toy_sources_are_named_as_the_score_files(tmp_path, toy_work_root):
    config = toy_config(data={"val": [toy_source("val", **{"attrs.group": "a"})]})
    (result,) = train(tmp_path, toy_work_root, config)
    assert f"val/{SOURCE}-a/auc" in result.metrics["val"]


# ----------------------------------------------------------------------------- devices


@pytest.mark.parametrize(
    ("device", "options"),
    [
        ("cpu", {"accelerator": "cpu", "devices": 1}),
        ("cuda", {"accelerator": "gpu", "devices": 1}),
        ("gpu", {"accelerator": "gpu", "devices": 1}),
        ("cuda:1", {"accelerator": "gpu", "devices": [1]}),
    ],
)
def test_device_names_map_to_one_trainer_device(device, options):
    assert run_module.device_options(device) == options


def test_an_unknown_device_is_a_config_error():
    with pytest.raises(ConfigError, match=r"--device: 'cdua' is not a device") as caught:
        run_module.device_options("cdua")
    assert "did you mean 'cuda'" in caught.value.message
    with pytest.raises(ConfigError):
        run_module.device_options("cuda:x")
