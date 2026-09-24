import pytest

from dfwb.core.errors import AmbiguousKeyError, ContractError, UnknownKeyError
from dfwb.core.records import PackCard
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


# The tests above are the brief's Step 1, verbatim. The rest pin the remaining documented
# interface (Pack.version, read_card/read_labels, and find_dataset's other error branches) that
# the brief's Interfaces section specifies but its literal test code does not exercise.


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
    with pytest.raises(UnknownKeyError, match="'beta' does not publish dataset 'toyone'"):
        find_dataset("toyone", pack="beta")


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
