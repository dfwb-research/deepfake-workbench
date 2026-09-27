import pytest
from tests.unit.protocols.conftest import make_pack, register_packs, register_provider_packs

from dfwb.core.errors import AmbiguousKeyError, ContractError, UnknownKeyError
from dfwb.core.records import PackCard
from dfwb.core.registry import catalogue_requirement
from dfwb.protocols._yaml import read_card, read_labels, read_model
from dfwb.protocols.packs import Pack, find_dataset, installed_packs


def test_installed_packs_lists_registered_packs(fixture_packs):
    fixture_packs({"alpha": {"toyone": {}}, "beta": {"toytwo": {}}})
    names = [p.name for p in installed_packs()]
    assert names == ["alpha", "beta"]


def test_find_dataset(fixture_packs):
    fixture_packs({"alpha": {"toyone": {}}})
    pack = find_dataset("toyone")
    assert pack.name == "alpha"
    assert (pack.dataset_dir("toyone") / "dataset.yaml").is_file()


def test_same_dataset_in_two_packs_is_ambiguous(fixture_packs):
    fixture_packs({"alpha": {"toyone": {}}, "beta": {"toyone": {}}})
    with pytest.raises(AmbiguousKeyError) as info:
        find_dataset("toyone")
    assert "alpha:toyone" in info.value.message
    assert "beta:toyone" in info.value.message
    assert find_dataset("toyone", pack="beta").name == "beta"


def test_unknown_dataset_suggests_close_matches(fixture_packs):
    fixture_packs({"alpha": {"toyone": {}}})
    with pytest.raises(UnknownKeyError, match="did you mean 'toyone'"):
        find_dataset("toyon")


def test_a_broken_pack_does_not_hide_the_others(fixture_packs, tmp_path):
    root = fixture_packs({"alpha": {"toyone": {}}, "broken": {"toytwo": {}}})
    (root["broken"] / "pack.yaml").write_text("schema_version: 99\n")
    packs = {p.name: p for p in installed_packs()}
    assert packs["broken"].error is not None
    assert packs["broken"].card is None
    assert find_dataset("toyone").name == "alpha"
    with pytest.raises(ContractError, match="broken"):
        find_dataset("toytwo", pack="broken")


def test_a_pack_whose_folder_cannot_be_located_is_broken_not_fatal(tmp_path, monkeypatch):
    alpha = make_pack(tmp_path, "alpha", {"toyone": {}})
    ghost = tmp_path / "dfwb_no_such_pack_package" / "pack"  # never written: nothing to locate
    register_packs(monkeypatch, {"alpha": alpha, "ghost": ghost})

    packs = {p.name: p for p in installed_packs()}

    assert packs["alpha"].error is None
    assert packs["ghost"].card is None
    assert packs["ghost"].root is None
    assert "dfwb_no_such_pack_package" in (packs["ghost"].error or "")
    assert find_dataset("toyone").name == "alpha"
    with pytest.raises(ContractError, match="broken"):
        find_dataset("toyone", pack="ghost")
    with pytest.raises(ContractError, match="broken"):
        packs["ghost"].dataset_dir("toyone")


def test_an_unknown_dataset_hint_names_the_broken_packs(fixture_packs):
    root = fixture_packs({"alpha": {"toyone": {}}, "broken": {"toytwo": {}}})
    (root["broken"] / "pack.yaml").write_text("schema_version: 99\n")
    with pytest.raises(UnknownKeyError) as info:
        find_dataset("toytwo")
    assert "'broken'" in info.value.hint


# The tests above cover the documented interface's primary example flow, verbatim. The rest pin
# the remaining documented interface (Pack.version, read_card/read_labels, and find_dataset's
# other error branches) that is specified elsewhere but not exercised by that flow.


def test_pack_version(fixture_packs):
    fixture_packs({"alpha": {"toyone": {}}})
    assert find_dataset("toyone").version == "1.0.0"
    broken = Pack("x", "local", find_dataset("toyone").root, None, "boom")
    with pytest.raises(ContractError, match="boom"):
        _ = broken.version


def test_find_dataset_rejects_unknown_pack_name(fixture_packs):
    fixture_packs({"alpha": {"toyone": {}}})
    with pytest.raises(UnknownKeyError, match="unknown protocol pack 'nope'"):
        find_dataset("toyone", pack="nope")


def test_find_dataset_named_pack_without_the_dataset(fixture_packs):
    fixture_packs({"alpha": {"toyone": {}}, "beta": {"toytwo": {}}})
    with pytest.raises(UnknownKeyError, match="'beta' does not publish dataset 'toyone'") as info:
        find_dataset("toyone", pack="beta")
    # The hint names a command that exists: `dfwb protocols list` has no --pack option.
    assert "--pack" not in info.value.hint
    assert "dfwb protocols list" in info.value.hint


def test_read_card_and_labels(fixture_packs):
    root = fixture_packs({"alpha": {"toyone": {}}})
    dataset_dir = root["alpha"] / "toyone"
    assert read_card(dataset_dir).id == "toyone"
    assert "TOYONE-REAL" in read_labels(dataset_dir).vocab


def test_read_model_errors(tmp_path):
    with pytest.raises(ContractError, match="cannot read"):
        read_model(tmp_path / "nope.yaml", PackCard)
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [\n")
    with pytest.raises(ContractError, match="invalid YAML"):
        read_model(bad_yaml, PackCard)
    bad_schema = tmp_path / "bad_schema.yaml"
    bad_schema.write_text("schema_version: 99\n")
    with pytest.raises(ContractError, match="schema_version") as info:
        read_model(bad_schema, PackCard)
    assert info.value.hint == "not a valid PackCard"


# --- fix round 1 -------------------------------------------------------------------------------
# Finding 1: read_model errors named only the bare filename, so a broken dataset.yaml could not be
# told apart from any other pack's. Finding 2: two providers registering the same protocol_packs
# key were resolved silently (first one found), instead of raising like every other registry
# collision (C1).


def test_read_model_error_names_the_full_path_not_just_the_filename(fixture_packs):
    root = fixture_packs({"alpha": {"toyone": {}}})
    dataset_dir = root["alpha"] / "toyone"
    (dataset_dir / "dataset.yaml").write_text("id: not-a-real-card\n")
    with pytest.raises(ContractError) as info:
        read_card(dataset_dir)
    assert "toyone" in info.value.message
    assert str(dataset_dir / "dataset.yaml") in info.value.message


def test_two_providers_registering_the_same_pack_name_is_ambiguous(tmp_path, monkeypatch):
    root = make_pack(tmp_path, "gamma", {"toyone": {}})
    register_provider_packs(
        monkeypatch, {"provider-a": {"gamma": root}, "provider-b": {"gamma": root}}
    )
    assert [p.name for p in installed_packs()] == ["gamma", "gamma"]
    with pytest.raises(AmbiguousKeyError) as info:
        find_dataset("toyone", pack="gamma")
    assert "provider-a" in info.value.message
    assert "provider-b" in info.value.message
    assert info.value.hint == "uninstall one of the distributions that provide pack 'gamma'"
    with pytest.raises(AmbiguousKeyError):
        find_dataset("toyone")


# The dataset ids the public protocol pack publishes; each should point a user at that pack.
PUBLIC_DATASETS = (
    "ffpp",
    "dfd",
    "celebdf-v1",
    "celebdf-v2",
    "celebdf-v3",
    "dfdc",
    "dfdc-p",
    "deeperforensics",
    "wilddeepfake",
    "ffiw10k",
    "kodf",
    "dfdm",
    "fakeavceleb",
    "polyglotfake",
    "deepspeak-v1",
    "deepspeak-v2",
    "idforge-v1",
    "lav-df",
    "av-deepfake1m-pp",
    "talkingheadbench",
    "uadfv",
)


def test_a_public_dataset_that_is_not_installed_suggests_the_protocols_pack(fixture_packs):
    assert len(set(PUBLIC_DATASETS)) == 21
    for dataset_id in PUBLIC_DATASETS:
        assert catalogue_requirement("protocol_packs", dataset_id) == "dfwb-protocols", dataset_id
    fixture_packs({"alpha": {"toyone": {}}})
    with pytest.raises(UnknownKeyError) as info:
        find_dataset("celebdf-v2")
    assert info.value.hint == "pip install dfwb-protocols"
    with pytest.raises(UnknownKeyError) as info:
        find_dataset("my-own-dataset")
    assert "pip install" not in info.value.hint
