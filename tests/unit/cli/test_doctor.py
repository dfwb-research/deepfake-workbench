import json


def test_doctor_human(run):
    result = run("doctor")
    assert result.code == 0
    assert result.out.startswith("dfwb      0.1.0a1\n")
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
    }
    assert data["roots"]["work"]["source"] == "default"
    assert "DFWB_WORK_ROOT is not set" in data["roots"]["work"]["warning"]
    assert data["torch"]["installed"] in (True, False)
    assert "rich" in data["extras"]
    assert {"name": "dfwb", "provider": "dfwb"}.items() <= data["plugins"][0].items()


def test_doctor_warns_about_defaulted_work_root(run, monkeypatch, tmp_path):
    monkeypatch.setenv("DFWB_DATASETS_ROOT", str(tmp_path / "raw"))
    assert "warning: DFWB_WORK_ROOT is not set" in run("doctor").err
