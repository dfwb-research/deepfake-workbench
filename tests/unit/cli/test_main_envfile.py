import json

from dfwb.cli.main import main


def test_cli_applies_dotenv_before_the_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DFWB_DATASETS_ROOT", raising=False)
    monkeypatch.delenv("DFWB_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text(f"DFWB_DATASETS_ROOT={tmp_path}/a:{tmp_path}/b\n")
    assert main(["doctor", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["roots"]["datasets"]["paths"] == [f"{tmp_path}/a", f"{tmp_path}/b"]
    assert data["env_file"]["applied"] == ["DFWB_DATASETS_ROOT"]


def test_no_env_file_flag_skips_it(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DFWB_DATASETS_ROOT", raising=False)
    monkeypatch.delenv("DFWB_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("DFWB_DATASETS_ROOT=/nowhere\n")
    assert main(["--no-env-file", "doctor", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["roots"]["datasets"]["paths"] == []
    assert data["env_file"] is None


def test_a_broken_env_file_is_a_config_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DFWB_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("garbage line\n")
    assert main(["doctor"]) == 2
    err = capsys.readouterr().err
    assert "error: " in err
    assert ".env:1" in err
    assert "hint: " in err
