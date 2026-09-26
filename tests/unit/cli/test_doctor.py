import json

from tests.unit.preprocess.inventory._demo import install, make_demo_tree

import dfwb


def test_doctor_human(run):
    result = run("doctor")
    assert result.code == 0
    assert result.out.startswith(f"dfwb      {dfwb.__version__}\n")
    assert "datasets  (unset)" in result.out
    assert "note: DFWB_DATASETS_ROOT is unset" in result.err
    assert "PLUGIN" in result.out


def test_doctor_json_reports_roots_and_sources(run, monkeypatch, tmp_path):
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(tmp_path / "raw"))
    data = json.loads(run("doctor", "--json").out)
    assert data["roots"]["datasets"] == {
        "path": str(tmp_path / "raw"),
        "source": "env",
        "from": "DFWB_DATASETS_ROOT",
        "warning": None,
        "paths": [str(tmp_path / "raw")],
    }
    assert data["env_file"] is None
    assert data["roots"]["work"]["source"] == "default"
    assert "DFWB_WORK_ROOT is not set" in data["roots"]["work"]["warning"]
    assert data["torch"]["installed"] in (True, False)
    assert "rich" in data["extras"]
    assert "preprocess" in data["extras"]
    assert {"name": "dfwb", "provider": "dfwb"}.items() <= data["plugins"][0].items()


def test_doctor_warns_about_defaulted_work_root(run, monkeypatch, tmp_path):
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(tmp_path / "raw"))
    assert "warning: DFWB_WORK_ROOT is not set" in run("doctor").err


def test_doctor_json_lists_every_registered_dataset(run, monkeypatch, tmp_path):
    install(monkeypatch)
    raw, second = tmp_path / "raw", tmp_path / "second"
    make_demo_tree(raw / "Demo", compressions=("c23",))
    (second / "Demo").mkdir(parents=True)
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")
    monkeypatch.delenv("DFWB_DATASET_DEMO", raising=False)
    data = json.loads(run("doctor", "--json").out)
    (entry,) = data["datasets"]
    assert entry == {
        "id": "demo",
        "folder": "Demo",
        "path": str(raw / "Demo"),
        "source": "root 1",
        "also_found": [str(second / "Demo")],
        "warning": entry["warning"],
    }
    assert "several datasets roots" in entry["warning"]


def test_doctor_human_datasets_section(run, monkeypatch, tmp_path):
    install(monkeypatch)
    raw, second = tmp_path / "raw", tmp_path / "second"
    (raw / "Demo").mkdir(parents=True)
    (second / "Demo").mkdir(parents=True)
    monkeypatch.setenv("DFWB_DATASETS_ROOT", f"{raw}:{second}")
    result = run("doctor")
    assert result.code == 0
    assert "DATASET" in result.out
    assert str(raw / "Demo") in result.out
    assert "warning: demo is in several datasets roots" in result.err

    monkeypatch.setenv("DFWB_DATASET_DEMO", str(tmp_path / "nowhere"))
    result = run("doctor")
    assert result.code == 0
    assert "not found" in result.out
    assert "is not a directory" in result.err


def test_doctor_datasets_is_empty_without_builders(run, monkeypatch):
    install(monkeypatch, {})
    data = json.loads(run("doctor", "--json").out)
    assert data["datasets"] == []
    assert "no inventory builders are registered" in run("doctor").out


def test_doctor_reports_a_broken_project_config_and_carries_on(run, tmp_path):
    (tmp_path / "dfwb.toml").write_bytes(b'[roots]\ndatasets = "/x"  # caf\xe9\n')

    result = run("doctor")
    assert result.code == 0, result.err
    assert "not valid UTF-8" in result.out
    assert "hint: save the file as UTF-8" in result.out
    assert "PLUGIN" in result.out  # everything else is still reported

    data = json.loads(run("doctor", "--json").out)
    assert data["roots"] == {}
    assert "not valid UTF-8" in data["roots_error"]["message"]
    assert data["roots_error"]["hint"] == "save the file as UTF-8"
    assert data["datasets"] == []


def test_doctor_licences_section_is_empty_by_default(run):
    data = json.loads(run("doctor", "--json").out)
    assert data["licenses"] == {}
    assert "no licences acknowledged yet" in run("doctor").out


def test_doctor_reports_a_corrupt_licence_store_and_carries_on(run, monkeypatch, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("DFWB_STATE_DIR", str(state))
    (state / "licenses.json").write_text("{not json")
    hint = "fix or delete the file; accept the licence again with --accept-license"

    result = run("doctor")
    assert result.code == 0, result.err
    licences = result.out.split("LICENCES\n", 1)[1]
    assert "licence store is corrupt" in licences
    assert f"hint: {hint}" in licences
    assert "PLUGIN" in result.out  # everything else is still reported

    data = json.loads(run("doctor", "--json").out)
    assert data["licenses"] == {}
    assert "licence store is corrupt" in data["licenses_error"]["message"]
    assert data["licenses_error"]["hint"] == hint


def test_doctor_json_has_no_licence_error_when_the_store_reads_cleanly(run):
    assert json.loads(run("doctor", "--json").out)["licenses_error"] is None


def test_doctor_licences_section_lists_accepted_licences(run, monkeypatch, tmp_path):
    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path / "state"))
    from dfwb.core import licenses

    licenses.accept("insightface-buffalo_l", license="non-commercial")
    data = json.loads(run("doctor", "--json").out)
    assert data["licenses"]["insightface-buffalo_l"]["license"] == "non-commercial"
    assert data["licenses"]["insightface-buffalo_l"]["accepted_at"]

    result = run("doctor")
    assert "insightface-buffalo_l" in result.out
    assert "non-commercial" in result.out
