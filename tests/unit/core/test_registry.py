import sys

import pytest

from dfwb.core import registry as registry_module
from dfwb.core.errors import (
    AmbiguousKeyError,
    ConfigError,
    InstallationError,
    PluginError,
    UnknownKeyError,
)
from dfwb.core.registry import Entry, Registry, providing

T = "tests.unit.core._targets"


@pytest.fixture
def reg():
    return Registry("layers")


def test_add_and_resolve_case_insensitive_and_alias(reg):
    reg.add("srm-stem", f"{T}:Stem", summary="SRM stem", aliases=("SRM", "srm_v1"))
    for key in ("srm-stem", "SRM-Stem", "srm", "SRM_V1", "local:srm-stem"):
        assert reg.entry(key).key == "srm-stem"
    entry = reg.entry("srm")
    assert entry == Entry(
        registry="layers",
        key="srm-stem",
        target=f"{T}:Stem",
        provider="local",
        summary="SRM stem",
        aliases=("srm", "srm_v1"),
    )
    assert entry.qualified_key == "local:srm-stem"
    assert "SRM" in reg
    assert "nope" not in reg
    assert 3 not in reg
    assert reg.keys() == ["srm-stem"]


@pytest.mark.parametrize("key", ["SRM", "srm_stem", "srm stem", "", "-srm"])
def test_keys_must_be_lower_kebab(reg, key):
    with pytest.raises(PluginError, match="invalid key"):
        reg.add(key, f"{T}:Stem", summary="x")


@pytest.mark.parametrize("target", ["tests.unit.core._targets", "a:b:c", "a..b:C", "1a:b", ""])
def test_code_targets_must_be_import_paths(reg, target):
    with pytest.raises(PluginError, match="invalid target"):
        reg.add("x", target, summary="x")


def test_summary_and_alias_validation(reg):
    with pytest.raises(PluginError, match="summary"):
        reg.add("x", f"{T}:Stem", summary="")
    with pytest.raises(PluginError, match="summary"):
        reg.add("x", f"{T}:Stem", summary="two\nlines")
    with pytest.raises(PluginError, match="alias"):
        reg.add("x", f"{T}:Stem", summary="x", aliases=("a:b",))


def test_same_provider_cannot_register_twice(reg):
    reg.add("x", f"{T}:Stem", summary="x", aliases=("y",))
    with pytest.raises(PluginError, match="twice"):
        reg.add("x", f"{T}:Stem", summary="again")
    with pytest.raises(PluginError, match="twice"):
        reg.add("y", f"{T}:Stem", summary="alias clash")


def test_collision_between_providers_is_ambiguous_not_shadowed(reg):
    with providing("dfwb-torch-srm"):
        reg.add("srm", f"{T}:Stem", summary="from utility")
    with providing("dfwb"):
        reg.add("srm", f"{T}:make_head", summary="from framework")
    with pytest.raises(AmbiguousKeyError) as info:
        reg.entry("srm")
    assert "dfwb-torch-srm:srm" in info.value.message
    assert "dfwb:srm" in info.value.message
    assert "qualified" in info.value.hint
    assert reg.entry("dfwb-torch-srm:srm").provider == "dfwb-torch-srm"
    assert reg.entry("Dfwb_Torch_SRM:srm").provider == "dfwb-torch-srm"
    assert reg.keys() == ["srm"]
    assert [e.provider for e in reg.entries()] == ["dfwb", "dfwb-torch-srm"]


def test_unknown_key_suggests_close_matches_and_uses_hint_callback():
    reg = Registry("layers", unknown_hint=lambda r, k: f"{r}:{k} hint")
    reg.add("srm", f"{T}:Stem", summary="x")
    with pytest.raises(UnknownKeyError) as info:
        reg.entry("srn")
    assert info.value.message == "layers: unknown key 'srn' (did you mean 'srm'?)"
    assert info.value.hint == "layers:srn hint"


def test_unknown_key_default_hint(reg):
    with pytest.raises(UnknownKeyError) as info:
        reg.entry("anything")
    assert info.value.hint == "run `dfwb plugins list` to see the registered layers"


def test_on_access_runs_before_reads_not_before_add():
    calls = []
    reg = Registry("layers", on_access=lambda: calls.append(1))
    reg.add("x", f"{T}:Stem", summary="x")
    assert calls == []
    reg.keys()
    reg.entries()
    _ = "x" in reg
    reg.entry("x")
    assert len(calls) == 4


def test_load_is_lazy_and_cached(reg):
    sys.modules.pop("tests.unit.core._counting_target", None)
    reg.add("counted", "tests.unit.core._counting_target:Counted", summary="x")
    assert "tests.unit.core._counting_target" not in sys.modules
    first = reg.load("counted")
    assert "tests.unit.core._counting_target" in sys.modules
    assert reg.load("COUNTED") is first


def test_register_decorator_refuses_local_objects(reg):
    with pytest.raises(PluginError, match="module-level"):

        @reg.register("local-thing", summary="decorated")
        class Thing:
            pass


def test_register_decorator_on_module_level_object(reg):
    from tests.unit.core import _targets

    decorated = reg.register("stem", summary="stem")(_targets.Stem)
    assert decorated is _targets.Stem
    assert reg.entry("stem").target == f"{T}:Stem"
    assert reg.load("stem") is _targets.Stem


def test_requires_missing_raises_installation_error(reg, monkeypatch):
    monkeypatch.setattr(registry_module, "_catalogue", lambda: {"imports": {"torch": "train"}})
    reg.add("needs-torch", f"{T}:Stem", summary="x", requires=("torch_is_not_installed_here",))
    with pytest.raises(InstallationError) as info:
        reg.load("needs-torch")
    assert "needs 'torch_is_not_installed_here'" in info.value.message
    assert (
        info.value.hint
        == "install the package that provides the 'torch_is_not_installed_here' module"
    )
    reg.add("needs-torch-2", f"{T}:Stem", summary="x", requires=("torch.nn",))
    monkeypatch.setattr(registry_module, "_is_installed", lambda name: False)
    with pytest.raises(InstallationError) as info:
        reg.load("needs-torch-2")
    assert info.value.hint == 'pip install "deepfake-workbench[train]"'


def test_missing_target_module_or_attribute_is_a_plugin_error(reg):
    reg.add("gone", "tests.unit.core._no_such_module:X", summary="x")
    reg.add("typo", f"{T}:Stme", summary="x")
    with pytest.raises(PluginError, match="not found"):
        reg.load("gone")
    with pytest.raises(PluginError, match="does not exist"):
        reg.load("typo")


def test_missing_dependency_inside_target_is_an_installation_error(reg, tmp_path, monkeypatch):
    (tmp_path / "needs_dep.py").write_text("import a_dependency_that_is_missing\n")
    monkeypatch.syspath_prepend(tmp_path)
    reg.add("dep", "needs_dep:X", summary="x")
    with pytest.raises(InstallationError, match="a_dependency_that_is_missing"):
        reg.load("dep")


def test_build_validates_against_signature(reg):
    reg.add("stem", f"{T}:Stem", summary="x")
    stem = reg.build("stem", channels="6", mode="learned")
    assert (stem.channels, stem.mode) == (6, "learned")
    with pytest.raises(ConfigError) as info:
        reg.build("stem", chanels=6)
    assert (
        info.value.message == "layers/stem: chanels: unknown parameter (did you mean 'channels'?)"
    )
    assert info.value.problems == ((("chanels",), "unknown parameter (did you mean 'channels'?)"),)
    assert info.value.hint == "accepted parameters: channels, mode"
    with pytest.raises(ConfigError) as info:
        reg.build("stem", mode="lerned")
    assert (
        "mode: 'lerned' is not one of ['fixed', 'learned'] (did you mean 'learned'?)"
        in info.value.message
    )


def test_build_missing_required_param_and_kwargs_targets(reg):
    reg.add("head", f"{T}:make_head", summary="x")
    with pytest.raises(ConfigError, match="dim: required key is missing"):
        reg.build("head")
    assert reg.build("head", dim=4) == {"dim": 4, "dropout": 0.0}
    reg.add("open", f"{T}:open_kwargs", summary="x")
    assert reg.build("open", anything=1, other="x") == {"anything": 1, "other": "x"}


@pytest.mark.parametrize(
    ("key", "target", "params_model", "good", "expected", "bad_key"),
    [
        ("freeze", "backbone", "FreezeParams", {"mode": "partial"}, ("partial", 0), "moed"),
        ("pool", "pool", "PoolParams", {"kind": "max"}, ("max", 1.0), "knd"),
        ("loss", "loss", "LossParams", {"weight": "0.5"}, 0.5, "wieght"),
    ],
)
def test_build_validates_against_declared_params_model(
    reg, key, target, params_model, good, expected, bad_key
):
    reg.add(key, f"{T}:{target}", summary="x", params=f"{T}:{params_model}")
    assert reg.build(key, **good) == expected
    with pytest.raises(ConfigError, match="unknown parameter"):
        reg.build(key, **{bad_key: 1})


def test_validate_does_not_call_the_target(reg):
    reg.add("stem", f"{T}:Stem", summary="x")
    assert reg.validate("stem", channels="2") == {"channels": 2}


def test_params_must_be_a_model(reg):
    reg.add("bad", f"{T}:Stem", summary="x", params=f"{T}:make_head")
    with pytest.raises(PluginError, match="not a pydantic model"):
        reg.validate("bad")


def test_not_callable_target(reg):
    reg.add("const", f"{T}:NOT_CALLABLE", summary="x")
    with pytest.raises(PluginError, match="not callable"):
        reg.build("const")


def test_data_registry_locates_resources_without_importing_code(tmp_path, monkeypatch):
    package = tmp_path / "packpkg"
    (package / "packs").mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('pack code must not run')\n")
    (package / "packs" / "pack.yaml").write_text("schema_version: 1\n")
    monkeypatch.syspath_prepend(tmp_path)
    reg = Registry("protocol_packs", kind="data")
    reg.add("packs", "packpkg:packs", summary="a pack directory")
    resource = reg.load("packs")
    assert (resource / "pack.yaml").is_file()
    assert "packpkg" not in sys.modules
    with pytest.raises(PluginError, match="takes no parameters"):
        reg.validate("packs")
    reg.add("missing", "packpkg:no/such/dir", summary="x")
    with pytest.raises(PluginError, match="does not exist"):
        reg.load("missing")
    reg.add("gone", "no_such_package_xyz:packs", summary="x")
    with pytest.raises(PluginError, match="package 'no_such_package_xyz' not found"):
        reg.load("gone")
    with pytest.raises(PluginError, match="invalid target"):
        reg.add("escape", "packpkg:../etc", summary="x")


def test_params_model_is_used_without_importing_the_target(reg):
    # The params model validates even though the target (and its heavy requirement) is absent.
    reg.add(
        "stem",
        "a_module_that_is_not_installed:Stem",
        summary="x",
        requires=("a_module_that_is_not_installed",),
        params=f"{T}:FreezeParams",
    )
    assert reg.validate("stem", mode="partial") == {"mode": "partial"}
    with pytest.raises(InstallationError):
        reg.build("stem", mode="partial")


def test_params_models_with_validation_aliases_are_refused(reg):
    reg.add("aliased", f"{T}:open_kwargs", summary="x", params=f"{T}:AliasedParams")
    with pytest.raises(PluginError, match="validation aliases"):
        reg.validate("aliased", lr=0.1)


def test_plain_aliases_are_accepted(reg):
    reg.add("kw", f"{T}:open_kwargs", summary="x", params=f"{T}:KeywordParams")
    assert reg.validate("kw", **{"lambda": "0.5"}) == {"lambda": 0.5}
    assert reg.build("kw", **{"lambda": 2}) == {"lambda": 2.0}


def test_typing_extensions_typeddict_params(reg):
    reg.add("te", f"{T}:loss", summary="x", params=f"{T}:ExtensionsLossParams")
    assert reg.build("te", weight="2") == 2.0
    with pytest.raises(ConfigError, match="unknown parameter"):
        reg.validate("te", wieght=1)


def test_nested_param_errors_carry_paths_and_suggestions(reg):
    reg.add("timm", f"{T}:open_kwargs", summary="x", params=f"{T}:TimmParams")
    with pytest.raises(ConfigError) as info:
        reg.validate("timm", model="vit", freeze={"trainable_block": 2})
    assert info.value.problems == (
        (("freeze", "trainable_block"), "unknown key (did you mean 'trainable_blocks'?)"),
    )
