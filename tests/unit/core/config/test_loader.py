import pytest

from dfwb.core.config import compose, dump_yaml, load_config
from dfwb.core.errors import ConfigError


def test_three_level_extends_chain_with_template(write):
    write(
        "base.yaml",
        "schema: dfwb.train/1\n"
        "extends: [dfwb://templates/binary-frame.yaml]\n"
        "train: {max_epochs: 20}\n",
    )
    write("mid/team.yaml", "extends: ../base.yaml\noptim: {lr: 3.0e-4}\neval: {+metrics: [nll]}\n")
    exp = write(
        "exp.yaml",
        "extends: [mid/team.yaml]\n"
        "run: {name: r, seeds: [1, 2]}\n"
        "data: {processing: p, train: [{protocol: toyfake/official, split: train}]}\n"
        "model: {backbone: {name: tiny-cnn}}\n",
    )
    loaded = load_config(exp, env={})
    assert loaded.sources == (
        "dfwb://templates/binary-frame.yaml",
        "../base.yaml",
        "mid/team.yaml",
        str(exp),
    )
    assert loaded.data["train"]["max_epochs"] == 20
    assert loaded.data["train"]["precision"] == "bf16-mixed"
    assert loaded.data["optim"] == {
        "name": "adamw",
        "lr": 3.0e-4,
        "weight_decay": 0.05,
        "groups": {"backbone": {"lr_scale": 0.1}},
    }
    assert loaded.data["eval"]["metrics"] == ["auc", "eer", "tpr@fpr=0.01", "ece", "brier", "nll"]
    assert loaded.data["run"]["output_root"] is None  # default filled in
    assert loaded.schema == "dfwb.train/1"


def test_diamond_extends_applies_the_shared_base_once(write):
    write("d.yaml", "schema: dfwb.train/1\nx: {a: 1, b: 1}\nlst: [base]\n")
    write("b.yaml", "extends: d.yaml\nx: {a: 2}\n+lst: [from-b]\n")
    write("c.yaml", "extends: d.yaml\nx: {b: 3}\n")
    top = write("a.yaml", "extends: [b.yaml, c.yaml]\n")
    data, sources = compose(top)
    assert data["x"] == {"a": 2, "b": 3}
    assert data["lst"] == ["base", "from-b"]
    assert sources == ("d.yaml", "b.yaml", "c.yaml", str(top))


def test_a_mixin_appends_to_what_earlier_files_set(write):
    write("addon.yaml", "eval: {+metrics: [nll]}\n")
    exp = write(
        "exp.yaml",
        "schema: dfwb.train/1\nextends: [dfwb://templates/binary-frame.yaml, addon.yaml]\n",
    )
    data, _ = compose(exp)
    assert data["eval"]["metrics"] == ["auc", "eer", "tpr@fpr=0.01", "ece", "brier", "nll"]


def test_same_file_name_in_different_directories(write):
    write("a/base.yaml", "schema: dfwb.train/1\nx: {a: 1}\n")
    write("b/base.yaml", "x: {b: 2}\n")
    top = write("top.yaml", "extends: [a/base.yaml, b/base.yaml]\n")
    data, sources = compose(top)
    assert data["x"] == {"a": 1, "b": 2}
    assert sources == ("a/base.yaml", "b/base.yaml", str(top))


def test_cycle_is_detected(write):
    write("a.yaml", "extends: b.yaml\n")
    b = write("b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="extends cycle"):
        compose(b)


def test_mixed_schemas_are_rejected(write):
    write("p.yaml", "schema: dfwb.score/1\n")
    child = write("c.yaml", "schema: dfwb.train/1\nextends: p.yaml\n")
    with pytest.raises(ConfigError, match="mixes schemas"):
        compose(child)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("extends: dfwb://templates/nope.yaml\n", "does not exist"),
        ("extends: nosuchpkg_xyz://a.yaml\n", "is not installed"),
        ("extends: dfwb://../../etc/passwd\n", "outside package"),
        ("extends: [1]\n", "must be a list"),
        ("- a\n- b\n", "must be a mapping"),
        ("a: [\n", "invalid YAML"),
        ("extends: missing.yaml\n", "config file not found"),
    ],
)
def test_bad_files_and_references(write, text, message):
    with pytest.raises(ConfigError, match=message):
        compose(write("x.yaml", text))


def test_overrides_then_interpolation_then_validation(experiment):
    loaded = load_config(
        experiment,
        [
            "optim.lr=3e-4",
            "train.max_epochs=5",
            "data.train[0].split=val",
            "run.name=${ref:data.processing}",
        ],
        env={"DFWB_RUNS_ROOT": "/r"},
    )
    assert loaded.data["optim"]["lr"] == 3e-4  # a float, not the YAML 1.1 string "3e-4"
    assert loaded.data["train"]["max_epochs"] == 5
    assert loaded.data["data"]["train"][0]["split"] == "val"
    assert loaded.data["run"]["name"] == "face-256-1.3x-32f"
    assert loaded.data["run"]["output_root"] == "/r"


def test_fingerprint_ignores_run_only_fields_and_key_order(experiment, write):
    a = load_config(experiment, env={})
    b = load_config(experiment, ["run.name=other", "run.output_root=/elsewhere"], env={})
    c = load_config(experiment, ["train.max_epochs=11"], env={})
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    reordered = write(
        "r.yaml",
        experiment.read_text().replace("schema: dfwb.train/1\n", "") + "schema: dfwb.train/1\n",
    )
    assert load_config(reordered, env={}).fingerprint == a.fingerprint


@pytest.mark.parametrize("spelling", ["1e-4", "1.0e-4", "0.0001", "1E-4"])
def test_float_spellings_give_one_fingerprint(experiment, spelling):
    reference = load_config(experiment, ["optim.lr=0.0001"], env={})
    via_override = load_config(experiment, [f"optim.lr={spelling}"], env={})
    text = experiment.read_text().replace("\nmodel:", f"\noptim: {{lr: {spelling}}}\nmodel:", 1)
    assert via_override.data["optim"]["lr"] == 0.0001
    assert via_override.fingerprint == reference.fingerprint
    in_file = experiment.parent / f"lr-{spelling}.yaml"
    in_file.write_text(text)
    assert load_config(in_file, env={}).fingerprint == reference.fingerprint


def test_check_registries_uses_installed_plugins(experiment):
    with pytest.raises(ConfigError, match="invalid component"):
        load_config(experiment, env={}, check_registries=True)


def test_dump_yaml_round_trips(experiment):
    import yaml

    loaded = load_config(experiment, env={})
    assert yaml.safe_load(dump_yaml(loaded.data)) == loaded.data
