from importlib import resources

import pytest
import yaml

from dfwb.core.config.merge import deep_merge
from dfwb.core.config.schema import ComponentSpec, TrainConfig, check_components, validate_config
from dfwb.core.errors import ConfigError, InstallationError
from dfwb.core.registry import Registry


def _valid(experiment):
    """The template merged with the experiment, as `extends` would do (tested in test_loader)."""
    template = resources.files("dfwb").joinpath("templates", "binary-frame.yaml").read_text()
    child = yaml.safe_load(experiment.read_text())
    child.pop("extends")
    data = deep_merge(yaml.safe_load(template), child)
    data["run"]["output_root"] = "./runs"
    return data


def test_skeleton_from_contract_validates(experiment):
    model = validate_config(_valid(experiment), source="exp.yaml")
    assert isinstance(model, TrainConfig)
    assert model.model.backbone.name == "timm"
    assert model.model.backbone.params == {
        "model": "vit_base_patch16_224.augreg_in21k",
        "pretrained": True,
        "freeze": {"mode": "partial", "trainable_blocks": 2},
    }
    assert model.data.test.suite == "cross-dataset-v1"
    assert model.data.transforms.train[0] == ComponentSpec(name="hflip", p=0.5)


def test_numbers_are_accepted_for_strings(experiment):
    data = _valid(experiment)
    data["run"]["name"] = 123
    assert validate_config(data, source="x").run.name == "123"


def test_missing_and_unknown_schema():
    with pytest.raises(ConfigError, match="missing top-level 'schema:'"):
        validate_config({}, source="x.yaml")
    with pytest.raises(ConfigError) as info:
        validate_config({"schema": "dfwb.trian/1"}, source="x.yaml")
    assert "did you mean 'dfwb.train/1'" in info.value.message


def test_errors_name_exact_paths_with_suggestions(experiment):
    data = _valid(experiment)
    data["train"]["max_epoch"] = 3
    data["data"]["clip"]["sampling"] = "unifrom"
    data["data"]["train"][0]["split"] = "trian"
    del data["train"]["precision"]
    with pytest.raises(ConfigError) as info:
        validate_config(data, source="exp.yaml")
    lines = info.value.message.splitlines()
    assert lines[0] == "exp.yaml: 4 invalid value(s)"
    assert (
        "  data.clip.sampling: 'unifrom' is not one of "
        "['uniform', 'consecutive', 'random-window'] (did you mean 'uniform'?)" in lines
    )
    assert (
        "  data.train[0].split: 'trian' is not one of ['train', 'val', 'test'] "
        "(did you mean 'train'?)" in lines
    )
    assert "  train.precision: required key is missing" in lines
    assert "  train.max_epoch: unknown key (did you mean 'max_epochs'?)" in lines
    assert info.value.hint == "see `dfwb schema export c2` for every key and type"


T = "tests.unit.core._targets"
COMPONENTS = [  # every component the template and the experiment name, all valid
    ("backbones", "timm", "open_kwargs"),
    ("temporal_pools", "mean", "open_kwargs"),
    ("heads", "linear", "open_kwargs"),
    ("losses", "bce", "open_kwargs"),
    ("transforms", "hflip", "open_kwargs"),
    ("transforms", "color-jitter", "open_kwargs"),
]


def _registries(skip=()):
    registries = {
        name: Registry(name)
        for name in ("backbones", "temporal_pools", "heads", "losses", "transforms")
    }
    for registry, key, target in COMPONENTS:
        if (registry, key) not in skip:
            registries[registry].add(key, f"{T}:{target}", summary="x")
    return registries


def test_check_components_passes_when_all_valid(experiment):
    check_components(validate_config(_valid(experiment), source="x"), _registries())


def test_check_components_reports_every_problem_at_its_path(experiment):
    model = validate_config(_valid(experiment), source="x")
    registries = _registries(
        skip={("heads", "linear"), ("losses", "bce"), ("transforms", "color-jitter")}
    )
    registries["heads"].add("linear", f"{T}:make_head", summary="x")
    registries["losses"].add("bce", f"{T}:loss", summary="x")
    with pytest.raises(ConfigError) as info:
        check_components(model, registries)
    lines = info.value.message.splitlines()
    assert lines[0] == "2 invalid component value(s)"
    # a head's dim comes from the backbone when the detector is built: never missing here.
    assert not [line for line in lines if "model.head.dim" in line]
    assert "  loss.weight: required key is missing" in lines
    assert "  data.transforms.train[1]: transforms: unknown key 'color-jitter'" in lines


def test_component_param_errors_match_the_c2_example(experiment):
    data = _valid(experiment)
    data["model"]["backbone"] = {
        "name": "timm",
        "model": "vit_base_patch16_224",
        "freeze": {"mode": "partail", "trainable_block": 2},
    }
    registries = _registries(skip={("backbones", "timm")})
    registries["backbones"].add("timm", f"{T}:open_kwargs", summary="x", params=f"{T}:TimmParams")
    with pytest.raises(ConfigError) as info:
        check_components(validate_config(data, source="x"), registries)
    lines = info.value.message.splitlines()
    assert (
        "  model.backbone.freeze.mode: 'partail' is not one of "
        "['none', 'full', 'partial', 'lora'] (did you mean 'partial'?)"
    ) in lines
    assert (
        "  model.backbone.freeze.trainable_block: unknown key (did you mean 'trainable_blocks'?)"
        in lines
    )


def test_unknown_component_param_is_named_with_its_path(experiment):
    data = _valid(experiment)
    data["model"]["head"] = {"name": "linear", "dim": 8, "dropot": 0.1}
    registries = _registries(skip={("heads", "linear")})
    registries["heads"].add("linear", f"{T}:make_head", summary="x")
    with pytest.raises(ConfigError) as info:
        check_components(validate_config(data, source="x"), registries)
    assert (
        "  model.head.dropot: unknown parameter (did you mean 'dropout'?)"
        in info.value.message.splitlines()
    )


def test_only_missing_installs_give_an_installation_error(experiment):
    registries = _registries(skip={("backbones", "timm")})
    registries["backbones"].add(
        "timm", f"{T}:open_kwargs", summary="x", requires=("a_module_that_is_not_installed",)
    )
    with pytest.raises(InstallationError, match="needs 'a_module_that_is_not_installed'"):
        check_components(validate_config(_valid(experiment), source="x"), registries)


def test_errors_under_data_test_name_one_real_path(experiment):
    data = _valid(experiment)
    data["data"]["test"] = {"suite": "cross-dataset-v1", "spit": 1}
    with pytest.raises(ConfigError) as info:
        validate_config(data, source="x")
    lines = info.value.message.splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("  data.test.spit: unknown key (did you mean ")
    data["data"]["test"] = [{"protocol": "ffpp/official", "split": "tets"}]
    with pytest.raises(ConfigError) as info:
        validate_config(data, source="x")
    assert info.value.message.splitlines()[1:] == [
        "  data.test[0].split: 'tets' is not one of ['train', 'val', 'test'] (did you mean 'test'?)"
    ]


def test_each_problem_keeps_its_own_hint_when_hints_differ(experiment):
    # A key that only a failed plugin provided, next to an unrelated bad component.
    failed_hint = "heads 'half' comes from plugin 'broken' (broken-dist), which failed to load"
    registries = _registries(skip={("heads", "linear"), ("losses", "bce")})
    registries["heads"] = Registry("heads", unknown_hint=lambda registry, key: failed_hint)
    data = _valid(experiment)
    data["model"]["head"] = {"name": "half"}
    with pytest.raises(ConfigError) as info:
        check_components(validate_config(data, source="x"), registries)
    lines = info.value.message.splitlines()
    assert "  model.head: heads: unknown key 'half'" in lines
    assert f"    hint: {failed_hint}" in lines
    assert "  loss: losses: unknown key 'bce'" in lines
    assert "--all" in info.value.hint
