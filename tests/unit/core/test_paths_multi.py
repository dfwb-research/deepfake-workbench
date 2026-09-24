from pathlib import Path

import pytest

from dfwb.core.errors import ConfigError
from dfwb.core.paths import current_host, dataset_overrides, locate_dataset, resolve_roots


@pytest.fixture
def places(tmp_path):
    cwd = tmp_path / "project"
    cwd.mkdir()
    user = tmp_path / "home" / "config.toml"
    user.parent.mkdir(parents=True)
    return cwd, user


def test_env_datasets_root_is_an_ordered_list(places, tmp_path):
    cwd, user = places
    value = f"{tmp_path}/a:{tmp_path}/b::"
    roots = resolve_roots(env={"DFWB_DATASETS_ROOT": value}, cwd=cwd, user_config=user)
    assert roots["datasets"].paths == (tmp_path / "a", tmp_path / "b")
    assert roots["datasets"].path == tmp_path / "a"


def test_toml_datasets_root_may_be_a_list(places):
    cwd, user = places
    (cwd / "dfwb.toml").write_text('[roots]\ndatasets = ["one", "/abs/two"]\n')
    roots = resolve_roots(env={}, cwd=cwd, user_config=user)
    assert roots["datasets"].paths == (cwd / "one", Path("/abs/two"))


def test_other_roots_must_stay_single_paths(places):
    cwd, user = places
    (cwd / "dfwb.toml").write_text('[roots]\nwork = ["a", "b"]\n')
    with pytest.raises(ConfigError, match=r"roots\.work must be a non-empty string"):
        resolve_roots(env={}, cwd=cwd, user_config=user)


def test_host_table_overrides_the_same_file_for_this_host_only(places):
    cwd, user = places
    (cwd / "dfwb.toml").write_text(
        '[roots]\ndatasets = "/shared/ds"\nwork = "/shared/work"\n'
        '[hosts.hades.roots]\ndatasets = ["/fast/ds", "/nfs/ds"]\n'
        '[hosts.other.roots]\nwork = "/never"\n'
    )
    roots = resolve_roots(env={"DFWB_HOST": "hades"}, cwd=cwd, user_config=user)
    assert roots["datasets"].paths == (Path("/fast/ds"), Path("/nfs/ds"))
    assert "[hosts.hades.roots]" in roots["datasets"].detail
    assert roots["work"].path == Path("/shared/work")
    assert roots["work"].source == "project"


def test_user_host_table_ranks_below_project(places):
    cwd, user = places
    user.write_text('[hosts.hades.roots]\nruns = "/u/runs"\ncache = "/u/cache"\n')
    (cwd / "dfwb.toml").write_text('[roots]\nruns = "/p/runs"\n')
    roots = resolve_roots(env={"DFWB_HOST": "hades"}, cwd=cwd, user_config=user)
    assert roots["runs"].path == Path("/p/runs")
    assert roots["cache"].path == Path("/u/cache")


def test_unknown_key_in_host_table_is_an_error(places):
    cwd, user = places
    (cwd / "dfwb.toml").write_text('[hosts.hades]\nroot = {datasets = "/x"}\n')
    with pytest.raises(ConfigError, match="did you mean 'roots'"):
        resolve_roots(env={"DFWB_HOST": "hades"}, cwd=cwd, user_config=user)


def test_current_host_is_short_and_lower_case(monkeypatch):
    assert current_host({"DFWB_HOST": "Lab-Box"}) == "lab-box"
    monkeypatch.setattr("socket.gethostname", lambda: "Hades.cluster.local")
    assert current_host({}) == "hades"


def test_work_default_warns_when_datasets_span_locations(places):
    cwd, user = places
    roots = resolve_roots(env={"DFWB_DATASETS_ROOT": "/a/ds:/b/ds"}, cwd=cwd, user_config=user)
    assert roots["work"].path == Path("/a/dfwb-work")
    assert "several locations" in roots["work"].warning


def _roots(cwd, user, value):
    return resolve_roots(env={"DFWB_DATASETS_ROOT": value}, cwd=cwd, user_config=user)


def test_locate_dataset_searches_roots_in_order_and_reports_duplicates(places, tmp_path):
    cwd, user = places
    for root in ("fast", "nfs"):
        (tmp_path / root / "FaceForensics++").mkdir(parents=True)
    roots = _roots(cwd, user, f"{tmp_path}/fast:{tmp_path}/nfs")
    where = locate_dataset("ffpp", "FaceForensics++", roots)
    assert where.path == tmp_path / "fast" / "FaceForensics++"
    assert where.root == tmp_path / "fast"
    assert where.also_found == (tmp_path / "nfs" / "FaceForensics++",)


def test_locate_dataset_falls_through_to_a_later_root(places, tmp_path):
    cwd, user = places
    (tmp_path / "nfs" / "UADFV").mkdir(parents=True)
    (tmp_path / "fast").mkdir()
    roots = _roots(cwd, user, f"{tmp_path}/fast:{tmp_path}/nfs")
    assert locate_dataset("uadfv", "UADFV", roots).path == tmp_path / "nfs" / "UADFV"


def test_locate_dataset_error_lists_the_searched_roots(places, tmp_path):
    cwd, user = places
    roots = _roots(cwd, user, f"{tmp_path}/fast:{tmp_path}/nfs")
    with pytest.raises(ConfigError) as info:
        locate_dataset("kodf", "KoDF", roots)
    assert "KoDF" in info.value.message
    assert str(tmp_path / "nfs") in info.value.message
    assert "DFWB_DATASET_KODF" in info.value.hint


def test_dataset_overrides_from_env_and_host_table(places, tmp_path):
    cwd, user = places
    (cwd / "dfwb.toml").write_text(
        '[datasets]\nkodf = "/shared/KoDF"\n[hosts.hades.datasets]\ncelebdf-v2 = "/fast/CDF2"\n'
    )
    env = {"DFWB_HOST": "hades", "DFWB_DATASET_KODF": str(tmp_path / "mine")}
    found = dataset_overrides(env=env, cwd=cwd, user_config=user)
    assert found["kodf"] == (tmp_path / "mine", "env: DFWB_DATASET_KODF")
    assert found["celebdf-v2"][0] == Path("/fast/CDF2")
    assert "[hosts.hades.datasets]" in found["celebdf-v2"][1]


def test_an_override_wins_over_root_search(places, tmp_path):
    cwd, user = places
    (tmp_path / "fast" / "KoDF").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    roots = _roots(cwd, user, f"{tmp_path}/fast")
    overrides = {"kodf": (tmp_path / "elsewhere", "env: DFWB_DATASET_KODF")}
    where = locate_dataset("kodf", "KoDF", roots, overrides=overrides)
    assert where.path == tmp_path / "elsewhere"
    assert where.root is None
    assert where.source == "env: DFWB_DATASET_KODF"


def test_an_override_that_does_not_exist_is_an_error(places, tmp_path):
    cwd, user = places
    roots = _roots(cwd, user, str(tmp_path))
    overrides = {"kodf": (tmp_path / "missing", "env: DFWB_DATASET_KODF")}
    with pytest.raises(ConfigError, match="is not a directory"):
        locate_dataset("kodf", "KoDF", roots, overrides=overrides)
