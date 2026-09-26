import json

from tests._dfwb_cli import run_dfwb


def test_cli_applies_dotenv_before_the_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DFWB_DATASETS_ROOT", raising=False)
    monkeypatch.delenv("DFWB_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text(f"DFWB_DATASETS_ROOT={tmp_path}/a:{tmp_path}/b\n")
    result = run_dfwb(capsys, "doctor", "--json")
    assert result.code == 0
    data = json.loads(result.out)
    assert data["roots"]["datasets"]["paths"] == [f"{tmp_path}/a", f"{tmp_path}/b"]
    assert data["env_file"]["applied"] == ["DFWB_DATASETS_ROOT"]


def test_no_env_file_flag_skips_it(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DFWB_DATASETS_ROOT", raising=False)
    monkeypatch.delenv("DFWB_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("DFWB_DATASETS_ROOT=/nowhere\n")
    result = run_dfwb(capsys, "--no-env-file", "doctor", "--json")
    assert result.code == 0
    data = json.loads(result.out)
    assert data["roots"]["datasets"]["paths"] == []
    assert data["env_file"] is None


def test_a_broken_env_file_is_a_config_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DFWB_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("garbage line\n")
    result = run_dfwb(capsys, "doctor")
    assert result.code == 2
    err = result.err
    assert "error: " in err
    assert ".env:1" in err
    assert "hint: " in err


def test_a_foreign_env_file_names_the_ways_around_it(tmp_path, monkeypatch, capsys):
    # A docker-compose style .env (a bare KEY line) is not dfwb's: the hint says how to skip it.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DFWB_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("COMPOSE_PROJECT_NAME=x\nDEBUG\n")
    result = run_dfwb(capsys, "doctor")
    assert result.code == 2
    err = result.err
    assert ".env:2: expected KEY=VALUE" in err
    hint = next(line for line in err.splitlines() if line.startswith("hint: "))
    assert "--no-env-file" in hint
    assert "DFWB_ENV_FILE" in hint
    assert run_dfwb(capsys, "--no-env-file", "doctor", "--json").code == 0
