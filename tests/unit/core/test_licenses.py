import json

from dfwb.core import licenses


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
