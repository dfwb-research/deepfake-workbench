from pathlib import Path

import pytest

from dfwb.core.envfile import AppliedEnv
from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.paths import RelPath, relativize, require_root, resolve_roots


@pytest.fixture
def places(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir()
    user = tmp_path / "home" / ".config" / "dfwb" / "config.toml"
    user.parent.mkdir(parents=True)
    return cwd, user


def test_defaults_when_nothing_is_set(places, monkeypatch, tmp_path):
    cwd, user = places
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-home"))
    roots = resolve_roots(env={}, cwd=cwd, user_config=user)
    assert roots["datasets"].path is None
    assert roots["datasets"].source == "unset"
    assert roots["work"].source == "unset"
    assert roots["cache"].path == tmp_path / "cache-home" / "dfwb"
    assert roots["runs"].path == cwd / "runs"
    assert roots["runs"].source == "default"


def test_work_defaults_next_to_datasets_with_warning(places):
    cwd, user = places
    roots = resolve_roots(env={"DFWB_DATASETS_ROOT": "/data/raw"}, cwd=cwd, user_config=user)
    assert roots["work"].path == Path("/data/dfwb-work")
    assert roots["work"].source == "default"
    assert "DFWB_WORK_ROOT is not set" in roots["work"].warning


def test_precedence_flag_env_project_user(places):
    cwd, user = places
    user.write_text(
        '[roots]\ndatasets = "/u/ds"\nwork = "/u/work"\ncache = "/u/cache"\nruns = "rel-runs"\n'
    )
    (cwd / "dfwb.toml").write_text(
        '[roots]\ndatasets = "/p/ds"\nwork = "/p/work"\ncache = "/p/cache"\n'
    )
    env = {"DFWB_DATASETS_ROOT": "/e/ds", "DFWB_WORK_ROOT": "/e/work", "DFWB_CACHE_ROOT": ""}
    roots = resolve_roots(flags={"datasets": "/f/ds"}, env=env, cwd=cwd, user_config=user)
    assert (roots["datasets"].path, roots["datasets"].source) == (Path("/f/ds"), "flag")
    assert (roots["work"].path, roots["work"].source) == (Path("/e/work"), "env")
    assert (roots["cache"].path, roots["cache"].source) == (Path("/p/cache"), "project")
    assert (roots["runs"].path, roots["runs"].source) == (user.parent / "rel-runs", "user")


def test_relative_values_resolve_against_cwd_or_file(places):
    cwd, user = places
    (cwd / "dfwb.toml").write_text('[roots]\nwork = "work"\n')
    roots = resolve_roots(env={"DFWB_DATASETS_ROOT": "data"}, cwd=cwd, user_config=user)
    assert roots["datasets"].path == cwd / "data"
    assert roots["work"].path == cwd / "work"


def test_a_relative_env_value_set_by_dotenv_resolves_against_the_dotenv_directory(places):
    # DFWB_DATASETS_ROOT came from a .env file that lives outside cwd, with a relative value
    # (e.g. ./data/datasets): it must resolve against that file's own directory, not cwd.
    cwd, user = places
    env_dir = cwd.parent / "elsewhere"
    env_dir.mkdir()
    env_file = AppliedEnv(env_dir / ".env", ("DFWB_DATASETS_ROOT",), ())
    env = {"DFWB_DATASETS_ROOT": "./data/datasets", "DFWB_WORK_ROOT": "./data/work"}
    roots = resolve_roots(env=env, cwd=cwd, user_config=user, env_file=env_file)
    assert roots["datasets"].path == env_dir / "data" / "datasets"
    # DFWB_WORK_ROOT is a real environment variable here (not one the .env file set): it keeps
    # resolving against cwd, exactly as before.
    assert roots["work"].path == cwd / "data" / "work"


def test_a_relative_datasets_list_from_dotenv_resolves_every_entry_against_it(places):
    cwd, user = places
    env_dir = cwd.parent / "elsewhere"
    env_dir.mkdir()
    env_file = AppliedEnv(env_dir / ".env", ("DFWB_DATASETS_ROOT",), ())
    env = {"DFWB_DATASETS_ROOT": "./a:./b"}
    roots = resolve_roots(env=env, cwd=cwd, user_config=user, env_file=env_file)
    assert roots["datasets"].paths == (env_dir / "a", env_dir / "b")


def test_bad_config_files(places):
    cwd, user = places
    (cwd / "dfwb.toml").write_text("[roots\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        resolve_roots(env={}, cwd=cwd, user_config=user)
    (cwd / "dfwb.toml").write_text('[roots]\ndataset = "/x"\n')
    with pytest.raises(ConfigError, match="did you mean 'datasets'"):
        resolve_roots(env={}, cwd=cwd, user_config=user)


def test_a_non_utf8_project_config_is_a_config_error(places):
    cwd, user = places
    # Latin-1 bytes that are not valid UTF-8 (e.g. "café" saved with the wrong encoding).
    (cwd / "dfwb.toml").write_bytes('[roots]\ndatasets = "/x"  # caf\xe9\n'.encode("latin-1"))
    with pytest.raises(ConfigError, match="not valid UTF-8") as info:
        resolve_roots(env={}, cwd=cwd, user_config=user)
    assert str(cwd / "dfwb.toml") in info.value.message
    assert info.value.hint


def test_require_root_explains_how_to_set_it(places):
    cwd, user = places
    roots = resolve_roots(env={}, cwd=cwd, user_config=user)
    with pytest.raises(ConfigError) as info:
        require_root("datasets", roots)
    assert info.value.message == "DFWB_DATASETS_ROOT is not set"
    assert "export DFWB_DATASETS_ROOT=" in info.value.hint


def test_relpath_validation_and_resolution(places):
    cwd, user = places
    roots = resolve_roots(env={"DFWB_RUNS_ROOT": "/r"}, cwd=cwd, user_config=user)
    rel = RelPath("runs", "a/./b")
    assert str(rel) == "runs:a/b"
    assert rel.resolve(roots) == Path("/r/a/b")
    for bad in ("/abs", "../up", "a/../b", "", "a\\b"):
        with pytest.raises(ContractError):
            RelPath("runs", bad)
    with pytest.raises(ContractError):
        RelPath("elsewhere", "x")  # type: ignore[arg-type]


def test_relativize_picks_deepest_root(places):
    cwd, user = places
    env = {"DFWB_DATASETS_ROOT": "/d", "DFWB_WORK_ROOT": "/d/work"}
    roots = resolve_roots(env=env, cwd=cwd, user_config=user)
    assert relativize(Path("/d/work/ffpp/inventory.jsonl"), roots) == RelPath(
        "work", "ffpp/inventory.jsonl"
    )
    assert relativize(Path("/d/ffpp/x.mp4"), roots) == RelPath("datasets", "ffpp/x.mp4")
    with pytest.raises(ContractError, match="not inside any DFWB root"):
        relativize(Path("/elsewhere/x"), roots)
