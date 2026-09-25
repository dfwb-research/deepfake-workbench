import json

import pytest

from dfwb.core import licenses
from dfwb.core.errors import ContractError


def test_unaccepted_licence_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path))
    assert licenses.is_accepted("insightface-buffalo_l") is False


def test_accept_then_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path))
    licenses.accept("insightface-buffalo_l", license="non-commercial (insightface buffalo_l)")
    assert licenses.is_accepted("insightface-buffalo_l") is True
    assert licenses.is_accepted("some-other-model") is False


def test_acceptance_persists_across_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path))
    licenses.accept("insightface-buffalo_l", license="non-commercial")
    # A fresh read from disk (no in-process cache) must still see it.
    data = json.loads((tmp_path / "licenses.json").read_text())
    assert "insightface-buffalo_l" in data
    assert licenses.is_accepted("insightface-buffalo_l") is True


def test_dfwb_state_dir_isolates_separate_locations(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setenv("DFWB_STATE_DIR", str(first))
    licenses.accept("insightface-buffalo_l", license="non-commercial")
    monkeypatch.setenv("DFWB_STATE_DIR", str(second))
    assert licenses.is_accepted("insightface-buffalo_l") is False


def test_accept_records_the_licence_text_and_a_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path))
    licenses.accept("insightface-buffalo_l", license="non-commercial")
    (entry,) = [a for name, a in licenses.all_accepted().items() if name == "insightface-buffalo_l"]
    assert entry.license == "non-commercial"
    assert entry.accepted_at  # non-empty ISO timestamp


def test_accept_is_atomic_and_leaves_no_temp_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path))
    licenses.accept("a", license="x")
    licenses.accept("b", license="y")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["licenses.json"]


def _corrupt(tmp_path, monkeypatch, text):
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path))
    (tmp_path / "licenses.json").write_text(text)


def test_malformed_json_raises_contract_error(tmp_path, monkeypatch):
    _corrupt(tmp_path, monkeypatch, "{not json")
    with pytest.raises(ContractError, match="licence store is corrupt"):
        licenses.is_accepted("x")


@pytest.mark.parametrize("text", ["[]", '"just a string"', "42", "null"])
def test_non_object_top_level_raises_contract_error(tmp_path, monkeypatch, text):
    _corrupt(tmp_path, monkeypatch, text)
    with pytest.raises(ContractError, match="licence store is corrupt"):
        licenses.is_accepted("x")


def test_entry_missing_license_field_raises_contract_error(tmp_path, monkeypatch):
    _corrupt(tmp_path, monkeypatch, json.dumps({"insightface-buffalo_l": {"accepted_at": "x"}}))
    with pytest.raises(ContractError, match="licence store is corrupt"):
        licenses.is_accepted("x")


def test_entry_that_is_not_an_object_raises_contract_error(tmp_path, monkeypatch):
    _corrupt(tmp_path, monkeypatch, json.dumps({"insightface-buffalo_l": "non-commercial"}))
    with pytest.raises(ContractError, match="licence store is corrupt"):
        licenses.is_accepted("x")


def test_corrupt_store_error_names_the_file_and_hints_the_fix(tmp_path, monkeypatch):
    _corrupt(tmp_path, monkeypatch, "{not json")
    with pytest.raises(ContractError) as excinfo:
        licenses.is_accepted("x")
    assert str(tmp_path / "licenses.json") in excinfo.value.message
    assert excinfo.value.hint == (
        "fix or delete the file; accept the licence again with --accept-license"
    )
