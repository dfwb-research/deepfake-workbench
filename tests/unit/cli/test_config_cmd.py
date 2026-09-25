import json
from importlib import resources

import yaml


def _experiment(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "schema: dfwb.train/1\n"
        "extends: [dfwb://templates/binary-frame.yaml]\n"
        "train: {max_epochs: 3}\n"
    )
    (tmp_path / "team.yaml").write_text("extends: [base.yaml]\neval: {+metrics: [nll]}\n")
    (tmp_path / "exp.yaml").write_text(
        "extends: [team.yaml]\nrun: {name: r, seeds: [1]}\n"
        "data: {processing: p, train: [{protocol: toyfake/official, split: train}]}\n"
        "model: {backbone: {name: tiny-cnn}}\n"
    )
    return tmp_path / "exp.yaml"


def test_templates(run):
    result = run("config", "templates")
    assert "binary-frame  Binary real/fake detector trained on face clips." in result.out
    assert "toy-cpu       A tiny CPU training run on the synthetic toyfake dataset." in result.out
    names = [t["name"] for t in json.loads(run("config", "templates", "--json").out)]
    assert names == ["binary-frame", "toy-cpu"]


def test_validate_accepts_the_toy_cpu_template(run, requires_torch):
    # Components are checked against the installed plugins; the toyfake protocol pack and its
    # processed store are only needed once training starts.
    path = resources.files("dfwb").joinpath("templates", "toy-cpu.yaml")
    result = run("config", "validate", "-c", str(path))
    assert result.code == 0, result.err
    shown = json.loads(run("config", "show", "-c", str(path), "--json").out)["config"]
    assert shown["data"]["processing"] == "toy-64-center-8f"
    assert shown["data"]["train"] == [
        {"protocol": "toyfake/official", "split": "train", "where": {}}
    ]
    assert shown["data"]["clip"]["frames"] == 1
    assert shown["data"]["loader"]["batch_size"] == 16
    assert shown["data"]["loader"]["num_workers"] == 0
    assert shown["model"]["backbone"] == {"name": "tiny-cnn"}
    assert shown["model"]["input"] == {"crop": "full-frame", "crop_scale": None}
    assert shown["train"]["max_epochs"] == 2
    assert shown["train"]["precision"] == "32-true"
    assert shown["train"]["lightning"]["accelerator"] == "cpu"


def test_binary_frame_holds_the_contract_skeleton_defaults():
    text = resources.files("dfwb").joinpath("templates", "binary-frame.yaml").read_text("utf-8")
    template = yaml.safe_load(text)
    assert template["run"] == {"seeds": [42]}
    assert template["data"]["processing"] == "face-256-1.3x-32f"
    assert template["data"]["clip"] == {
        "frames": 8,
        "sampling": "uniform",
        "clips_per_video": {"train": 1, "eval": 4},
    }
    assert template["data"]["loader"] == {
        "batch_size": 32,
        "num_workers": 8,
        "balance": "video-label",
    }
    assert template["model"]["temporal_pool"] == {"name": "mean"}
    assert template["train"]["monitor"] == "val/video_auc"
    assert template["train"]["mode"] == "max"
    assert template["eval"]["metrics"] == ["auc", "eer", "tpr@fpr=0.01", "ece", "brier"]


def test_show_resolves_three_level_chain(run, tmp_path):
    exp = _experiment(tmp_path)
    result = run("config", "show", "-c", str(exp), "train.max_epochs=4")
    assert result.code == 0
    header, extends, *body = result.out.splitlines()
    assert header.startswith("# fingerprint: ")
    assert (
        extends
        == f"# extends: dfwb://templates/binary-frame.yaml -> base.yaml -> team.yaml -> {exp}"
    )
    data = yaml.safe_load("\n".join(body))
    assert data["train"]["max_epochs"] == 4
    assert data["eval"]["metrics"][-1] == "nll"
    as_json = json.loads(run("config", "show", "-c", str(exp), "--json").out)
    assert as_json["schema"] == "dfwb.train/1"
    assert len(as_json["sources"]) == 4


def test_show_reports_validation_errors(run, tmp_path):
    exp = _experiment(tmp_path)
    result = run("config", "show", "-c", str(exp), "train.max_epoch=4")
    assert result.code == 2
    assert "train.max_epoch: unknown key (did you mean 'max_epochs'?)" in result.err
    assert "hint: see `dfwb schema export c2`" in result.err


def test_validate_checks_components(run, tmp_path):
    exp = _experiment(tmp_path)
    # "tiny-cnn" is a real, registered backbone, so a bogus name keeps this test's own concern
    # (an unknown component key is reported with a hint) independent of which registries the
    # framework happens to have real built-ins for.
    exp.write_text(exp.read_text().replace("tiny-cnn", "nonexistent-backbone"))
    result = run("config", "validate", "-c", str(exp))
    assert result.code == 2
    assert "invalid component value(s)" in result.err
    assert "backbones: unknown key 'nonexistent-backbone'" in result.err


def test_validate_ok(run, tmp_path, monkeypatch):
    from dfwb.core import plugins

    def register(api):
        t = "tests.unit.core._targets:open_kwargs"
        for reg, key in [
            ("backbones", "tiny-cnn"),
            ("temporal_pools", "mean"),
            ("heads", "linear"),
            ("losses", "bce"),
            ("transforms", "hflip"),
            ("transforms", "color-jitter"),
        ]:
            getattr(api, reg).add(key, target=t, summary="test")

    class EP:
        name, value, group, dist = "t", "t:register", "dfwb.plugins", None

        def load(self):
            return register

    monkeypatch.setattr(
        plugins, "_entry_points", lambda group: [EP()] if group == "dfwb.plugins" else []
    )
    exp = _experiment(tmp_path)
    result = run("config", "validate", "-c", str(exp))
    assert result.code == 0, result.err
    assert result.out.startswith(f"ok: {exp} (dfwb.train/1, fingerprint ")
    assert json.loads(run("config", "validate", "-c", str(exp), "--json").out)["valid"] is True


def test_init_writes_a_valid_starter(run, tmp_path):
    result = run("config", "init")
    assert result.code == 0
    assert (tmp_path / "experiment.yaml").exists()
    assert run("config", "show", "-c", "experiment.yaml").code == 0
    again = run("config", "init")
    assert again.code == 2
    assert "hint: pass --force" in again.err
    assert run("config", "init", "--force", "--json").code == 0
    bad = run("config", "init", "--template", "binary-fram", "--out", "x.yaml")
    assert "did you mean 'binary-frame'" in bad.err


def test_missing_file(run):
    result = run("config", "show", "-c", "nope.yaml")
    assert result.code == 2
    assert "config file not found: nope.yaml" in result.err


def test_group_without_a_subcommand_shows_its_help(run):
    result = run("config")
    assert result.code == 0
    assert "Commands:" in result.out
