"""The framework registers its built-in components through the ordinary plugin API."""

from types import SimpleNamespace

from dfwb import _builtins
from dfwb.core.plugins import DATA_REGISTRIES, REGISTRY_NAMES
from dfwb.core.registry import Registry


def _api():
    return SimpleNamespace(
        **{
            name: Registry(name, kind="data" if name in DATA_REGISTRIES else "code")
            for name in REGISTRY_NAMES
        }
    )


def test_inventory_builders_register_by_import_path_with_their_folder(monkeypatch):
    row = ("toy-set", "toyset:ToySetBuilder", "Toy Set", "Toy Set Folder")
    monkeypatch.setattr(_builtins, "INVENTORY_BUILDERS", (row,))
    api = _api()
    _builtins.register(api)
    (entry,) = api.inventory_builders.entries()
    assert entry.key == "toy-set"
    assert entry.target == "dfwb.preprocess.inventory.builders.toyset:ToySetBuilder"
    assert entry.summary == "Toy Set"
    assert dict(entry.meta) == {"folder": "Toy Set Folder"}


def test_every_built_in_builder_row_is_well_formed():
    ids = [row[0] for row in _builtins.INVENTORY_BUILDERS]
    assert len(ids) == len(set(ids))
    for dataset_id, target, name, folder in _builtins.INVENTORY_BUILDERS:
        assert ":" in target, dataset_id
        assert name.strip(), dataset_id
        assert folder.strip(), dataset_id


def test_built_in_protocol_packs_register_as_data(monkeypatch):
    monkeypatch.setattr(_builtins, "INVENTORY_BUILDERS", ())
    api = _api()
    _builtins.register(api)
    (entry,) = api.protocol_packs.entries()
    assert (entry.key, entry.target) == ("toyfake", "dfwb:_packs/toyfake")
    assert entry.summary == "Built-in synthetic toyfake protocol pack"
