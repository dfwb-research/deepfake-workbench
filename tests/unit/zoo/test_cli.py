"""``dfwb zoo``: list, info, fetch, verify, parity and licenses over the dummy adapters and a
fixture adapter with weights served by a local HTTP server (never the real network). This
directory's autouse ``isolated``/guard fixtures (``conftest.py``) keep every test off the user's
real cache and state dirs."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from tests.unit.zoo._fixtures import WeightedTestAdapter

from dfwb.cli.main import main
from dfwb.core import licenses
from dfwb.core.plugins import get_registry
from dfwb.zoo.card import AdapterCard, parse_card

CONTENT = b"pretend these are model weights\n" * 20
SHA256 = hashlib.sha256(CONTENT).hexdigest()


@pytest.fixture
def run(capsys, monkeypatch, tmp_path):
    """Run ``dfwb ARGS...`` in-process, in the cache/state this directory's autouse ``isolated``
    fixture already redirected."""
    monkeypatch.chdir(tmp_path)

    def _run(*args: str) -> SimpleNamespace:
        code = main(list(args))
        captured = capsys.readouterr()
        return SimpleNamespace(code=code, out=captured.out, err=captured.err)

    return _run


def _register_weighted(name: str) -> None:
    get_registry("detectors").add(
        name, target="tests.unit.zoo._fixtures:WeightedTestAdapter", summary="cli test fixture"
    )


def _card_yaml(name: str, weights_block: str = "", *, requires_ack: bool = False) -> str:
    return f"""
name: {name}
display_name: Weighted Test
contract_version: [1, 0]
license: {{code: MIT, requires_ack: {"true" if requires_ack else "false"}}}
code_strategy: pip
input: {{crop: face, crop_scale: 1.3, size: [64, 64], frames: 1}}
{weights_block}
"""


def _init_repo(repo) -> str:
    """A tiny local git repo (never the real network), returning its one commit's sha."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.org"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True, capture_output=True)
    (repo / "entry.py").write_text("VALUE = 7\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True, capture_output=True
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _weighted_card(server_url: str, name: str, *, requires_ack: bool = False) -> AdapterCard:
    return parse_card(
        _card_yaml(
            name,
            f"""weights:
  - {{id: default, url: "{server_url}/w.safetensors", sha256: "{SHA256}",
     bytes: {len(CONTENT)}, format: safetensors}}""",
            requires_ack=requires_ack,
        )
    )


# =============================================================================================
# list / info / licenses -- no weights, no network, torch-free
# =============================================================================================


def test_list_shows_the_builtin_dummy_adapters(run):
    result = run("zoo", "list")

    assert result.code == 0
    assert "chance" in result.out
    assert "random" in result.out
    assert "MIT" in result.out


def test_list_json_reports_licence_input_and_reported_parity_counts(run):
    result = run("zoo", "list", "--json")

    assert result.code == 0
    rows = {row["name"]: row for row in json.loads(result.out)}
    assert rows["chance"]["license"]["code"] == "MIT"
    assert rows["chance"]["input"]["crop"] == "face"
    assert rows["chance"]["reported"] == 0
    assert rows["chance"]["parity"] == 0
    assert rows["chance"]["code_strategy"] == "pip"


def test_info_shows_the_full_card(run):
    result = run("zoo", "info", "chance")

    assert result.code == 0
    assert "chance" in result.out
    assert "MIT" in result.out
    assert "pip" in result.out


def test_info_json_dumps_the_validated_card(run):
    result = run("zoo", "info", "random", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data["card"]["name"] == "random"
    assert data["card"]["license"]["code"] == "MIT"
    assert data["license_accepted"] is None  # requires_ack is false: not applicable


def test_info_unknown_name_suggests_the_close_match(run):
    result = run("zoo", "info", "chnace")

    assert result.code == 2
    assert "chance" in result.err or "chance" in result.out
    assert "hint:" in result.err


def test_licenses_lists_nothing_acknowledged_yet(run):
    result = run("zoo", "licenses")

    assert result.code == 0
    assert "no licences" in result.out


def test_licenses_json_is_empty_list_when_nothing_accepted(run):
    result = run("zoo", "licenses", "--json")

    assert result.code == 0
    assert json.loads(result.out) == []


def test_licenses_plain_text_lists_an_acknowledged_licence(run):
    licenses.accept("some-adapter", license="MIT")

    result = run("zoo", "licenses")

    assert result.code == 0
    assert "some-adapter" in result.out
    assert "MIT" in result.out


_FULL_CARD_YAML = """
name: full-card
display_name: Full Card (fixture)
contract_version: [1, 0]
upstream:
  repo: "https://example.org/full-card"
  commit: "0123456789abcdef0123456789abcdef01234567"
license: {code: Apache-2.0, weights: LicenseRef-CC-BY-NC-4.0, requires_ack: true}
code_strategy: pip
input: {crop: face, crop_scale: 1.3, size: [64, 64], frames: 1}
weights:
  - {id: default, url: "https://example.org/w.safetensors",
     sha256: "170ead8b6650ce0eb473f2377ba63a8055e07f00a4fcde75574111d371e1cc6d",
     bytes: 12, format: safetensors}
reported:
  - {protocol: "some-pack:some/official", split: test, metric: auc, value: 0.9, source: paper}
parity:
  - {protocol: "some-pack:some/official", split: test, metric: auc, value: 0.89,
     tolerance: 0.02, dfwb_version: "0.1.0", date: "2026-01-01"}
"""


def test_list_and_info_plain_text_show_a_full_card(run, monkeypatch):
    get_registry("detectors").add(
        "full-card", target="tests.unit.zoo._fixtures:WeightedTestAdapter", summary="x"
    )
    card = parse_card(_FULL_CARD_YAML)
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    list_result = run("zoo", "list")
    assert list_result.code == 0
    assert "Apache-2.0/LicenseRef-CC-BY-NC-4.0 (ack)" in list_result.out

    licenses.accept("full-card", license="LicenseRef-CC-BY-NC-4.0")
    info_result = run("zoo", "info", "full-card")
    assert info_result.code == 0
    assert "upstream: https://example.org/full-card" in info_result.out
    assert "weights=LicenseRef-CC-BY-NC-4.0" in info_result.out
    assert "licence acknowledged: True" in info_result.out
    assert "weights:" in info_result.out
    assert "  default:" in info_result.out
    assert "reported:" in info_result.out
    assert "some-pack:some/official" in info_result.out
    assert "parity:" in info_result.out
    assert "tolerance=0.02" in info_result.out


# =============================================================================================
# fetch
# =============================================================================================


def test_fetch_an_unweighted_adapter_has_nothing_to_do(run):
    result = run("zoo", "fetch", "chance", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data == {"name": "chance", "code": None, "weights": None}


def test_fetch_downloads_and_caches_weights(run, server, monkeypatch):
    _register_weighted("fetch-weighted")
    server.routes["/w.safetensors"] = CONTENT
    card = _weighted_card(server.url, "fetch-weighted")
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    result = run("zoo", "fetch", "fetch-weighted", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data["weights"]["id"] == "default"
    assert data["weights"]["sha256"] == SHA256
    from pathlib import Path

    assert Path(data["weights"]["path"]).read_bytes() == CONTENT
    assert server.requests == ["/w.safetensors"]


def test_fetch_a_second_time_is_a_cache_hit(run, server, monkeypatch):
    _register_weighted("fetch-weighted-2")
    server.routes["/w.safetensors"] = CONTENT
    card = _weighted_card(server.url, "fetch-weighted-2")
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)
    run("zoo", "fetch", "fetch-weighted-2", "--json")
    server.requests.clear()

    result = run("zoo", "fetch", "fetch-weighted-2", "--json")

    assert result.code == 0
    assert server.requests == []


def test_fetch_unknown_name_exits_2_with_a_did_you_mean_hint(run):
    result = run("zoo", "fetch", "chnace")

    assert result.code == 2
    assert "hint:" in result.err
    assert "chance" in (result.out + result.err)


def test_fetch_a_gated_adapter_is_blocked_until_accepted(run, server, monkeypatch):
    _register_weighted("fetch-gated")
    server.routes["/w.safetensors"] = CONTENT
    card = _weighted_card(server.url, "fetch-gated", requires_ack=True)
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    blocked = run("zoo", "fetch", "fetch-gated")

    assert blocked.code == 5
    assert "--accept-license" in blocked.err
    assert server.requests == []

    accepted = run("zoo", "fetch", "fetch-gated", "--accept-license", "--json")

    assert accepted.code == 0
    assert json.loads(accepted.out)["weights"]["sha256"] == SHA256
    assert licenses.is_accepted("fetch-gated")
    assert server.requests == ["/w.safetensors"]


def test_fetch_records_the_acceptance_visible_via_licenses(run, server, monkeypatch):
    _register_weighted("fetch-gated-2")
    server.routes["/w.safetensors"] = CONTENT
    card = _weighted_card(server.url, "fetch-gated-2", requires_ack=True)
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    run("zoo", "fetch", "fetch-gated-2", "--accept-license")
    result = run("zoo", "licenses", "--json")

    assert result.code == 0
    rows = {row["name"]: row for row in json.loads(result.out)}
    assert "fetch-gated-2" in rows
    assert rows["fetch-gated-2"]["license"] == "MIT"


def test_fetch_a_tampered_cache_hit_is_silently_refetched(run, server, monkeypatch):
    from dfwb.zoo.weights import cache_dir

    _register_weighted("fetch-tampered")
    server.routes["/w.safetensors"] = CONTENT
    card = _weighted_card(server.url, "fetch-tampered")
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)
    dest = cache_dir("fetch-tampered", SHA256) / "weights.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"not the right bytes at all")

    result = run("zoo", "fetch", "fetch-tampered", "--json")

    assert result.code == 0
    assert dest.read_bytes() == CONTENT
    assert server.requests == ["/w.safetensors"]


def test_fetch_offline_refuses_an_uncached_download(run, server, monkeypatch):
    _register_weighted("fetch-offline")
    server.routes["/w.safetensors"] = CONTENT
    card = _weighted_card(server.url, "fetch-offline")
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)
    monkeypatch.setenv("DFWB_OFFLINE", "1")

    result = run("zoo", "fetch", "fetch-offline")

    assert result.code == 5
    assert "DFWB_OFFLINE" in result.err
    assert server.requests == []


def test_fetch_with_weights_flag_on_an_unweighted_adapter_is_a_config_error(run):
    result = run("zoo", "fetch", "chance", "--weights", "nope")

    assert result.code == 2
    assert "hint:" in result.err


def test_fetch_plain_text_reports_nothing_to_fetch(run):
    result = run("zoo", "fetch", "chance")

    assert result.code == 0
    assert "nothing to fetch" in result.out


def test_fetch_plain_text_reports_the_weights_path(run, server, monkeypatch):
    _register_weighted("fetch-plain")
    server.routes["/w.safetensors"] = CONTENT
    card = _weighted_card(server.url, "fetch-plain")
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    result = run("zoo", "fetch", "fetch-plain")

    assert result.code == 0
    assert "weights default:" in result.out


def test_fetch_clones_a_pinned_clone_adapter(run, tmp_path, monkeypatch):
    repo = tmp_path / "upstream"
    repo.mkdir()
    commit = _init_repo(repo)
    get_registry("detectors").add(
        "fetch-pinned-clone", target="tests.unit.zoo._fixtures:WeightedTestAdapter", summary="x"
    )
    card = parse_card(
        f"""
name: fetch-pinned-clone
display_name: Pinned Clone Fetch
contract_version: [1, 0]
upstream: {{repo: "{repo}", commit: "{commit}"}}
license: {{code: MIT}}
code_strategy: pinned-clone
input: {{}}
"""
    )
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    result = run("zoo", "fetch", "fetch-pinned-clone", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data["weights"] is None
    from pathlib import Path

    code_root = Path(data["code"])
    assert (code_root / "entry.py").is_file()

    plain = run("zoo", "fetch", "fetch-pinned-clone")
    assert plain.code == 0
    assert f"code: {code_root}" in plain.out


# =============================================================================================
# verify
# =============================================================================================


def test_verify_an_unweighted_adapter_is_trivially_ok(run):
    result = run("zoo", "verify", "chance", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data == {"name": "chance", "ok": True, "weights": [], "code": None}


def test_verify_an_unweighted_adapter_plain_text_says_nothing_to_verify(run):
    result = run("zoo", "verify", "chance")

    assert result.code == 0
    assert "chance: ok" in result.out
    assert "nothing to verify" in result.out


def test_verify_reports_not_cached_before_any_fetch(run, monkeypatch):
    _register_weighted("verify-uncached")
    card = _weighted_card("http://example.invalid", "verify-uncached")
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    result = run("zoo", "verify", "verify-uncached", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data["ok"] is True
    assert data["weights"][0]["status"] == "not-cached"


def test_verify_reports_ok_after_a_fetch(run, server, monkeypatch):
    _register_weighted("verify-ok")
    server.routes["/w.safetensors"] = CONTENT
    card = _weighted_card(server.url, "verify-ok")
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)
    run("zoo", "fetch", "verify-ok")

    result = run("zoo", "verify", "verify-ok", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data["ok"] is True
    assert data["weights"][0]["status"] == "ok"


def test_verify_detects_a_sha_mismatch_without_touching_the_network(run, server, monkeypatch):
    from dfwb.zoo.weights import cache_dir

    _register_weighted("verify-mismatch")
    card = _weighted_card(server.url, "verify-mismatch")
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)
    dest = cache_dir("verify-mismatch", SHA256) / "weights.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"corrupted")

    result = run("zoo", "verify", "verify-mismatch", "--json")

    assert result.code == 4
    data = json.loads(result.out)
    assert data["ok"] is False
    assert data["weights"][0]["status"] == "mismatch"
    assert server.requests == []  # verify never touches the network


def test_verify_plain_text_output(run, server, monkeypatch):
    _register_weighted("verify-plain")
    server.routes["/w.safetensors"] = CONTENT
    card = _weighted_card(server.url, "verify-plain")
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)
    run("zoo", "fetch", "verify-plain")

    result = run("zoo", "verify", "verify-plain")

    assert result.code == 0
    assert "verify-plain: ok" in result.out
    assert "weights default: ok" in result.out


def _pinned_card(name: str, repo, commit: str) -> AdapterCard:
    return parse_card(
        f"""
name: {name}
display_name: Pinned Clone Verify
contract_version: [1, 0]
upstream: {{repo: "{repo}", commit: "{commit}"}}
license: {{code: MIT}}
code_strategy: pinned-clone
input: {{}}
"""
    )


def test_verify_reports_not_cloned_before_any_fetch(run, tmp_path, monkeypatch):
    repo = tmp_path / "upstream"
    repo.mkdir()
    commit = _init_repo(repo)
    get_registry("detectors").add(
        "verify-not-cloned", target="tests.unit.zoo._fixtures:WeightedTestAdapter", summary="x"
    )
    card = _pinned_card("verify-not-cloned", repo, commit)
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    result = run("zoo", "verify", "verify-not-cloned", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data["code"]["status"] == "not-cloned"


def test_verify_reports_ok_for_a_clean_pinned_clone(run, tmp_path, monkeypatch):
    repo = tmp_path / "upstream"
    repo.mkdir()
    commit = _init_repo(repo)
    get_registry("detectors").add(
        "verify-clone-ok", target="tests.unit.zoo._fixtures:WeightedTestAdapter", summary="x"
    )
    card = _pinned_card("verify-clone-ok", repo, commit)
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)
    run("zoo", "fetch", "verify-clone-ok")

    result = run("zoo", "verify", "verify-clone-ok", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data["code"]["status"] == "ok"


def test_verify_reports_a_dirty_pinned_clone(run, tmp_path, monkeypatch):
    from dfwb.zoo.strategies import clone_cache_dir

    repo = tmp_path / "upstream"
    repo.mkdir()
    commit = _init_repo(repo)
    get_registry("detectors").add(
        "verify-clone-dirty", target="tests.unit.zoo._fixtures:WeightedTestAdapter", summary="x"
    )
    card = _pinned_card("verify-clone-dirty", repo, commit)
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)
    run("zoo", "fetch", "verify-clone-dirty")
    (clone_cache_dir("verify-clone-dirty", commit) / "entry.py").write_text("VALUE = 999\n")

    result = run("zoo", "verify", "verify-clone-dirty", "--json")

    assert result.code == 4
    data = json.loads(result.out)
    assert data["ok"] is False
    assert data["code"]["status"] == "dirty"


def test_verify_reports_a_pinned_clone_at_the_wrong_commit(run, tmp_path, monkeypatch):
    import subprocess

    from dfwb.zoo.strategies import clone_cache_dir

    repo = tmp_path / "upstream"
    repo.mkdir()
    first = _init_repo(repo)
    (repo / "entry.py").write_text("VALUE = 8\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "second"], cwd=repo, check=True, capture_output=True
    )
    second = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    get_registry("detectors").add(
        "verify-clone-wrong", target="tests.unit.zoo._fixtures:WeightedTestAdapter", summary="x"
    )
    card = _pinned_card("verify-clone-wrong", repo, second)
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)
    run("zoo", "fetch", "verify-clone-wrong")
    clone_dir = clone_cache_dir("verify-clone-wrong", second)
    subprocess.run(["git", "checkout", "-q", first], cwd=clone_dir, check=True, capture_output=True)

    result = run("zoo", "verify", "verify-clone-wrong")

    assert result.code == 4
    assert "wrong-commit" in result.out


# =============================================================================================
# parity
# =============================================================================================


def test_parity_with_no_reported_metrics_is_a_no_op(run):
    result = run("zoo", "parity", "chance", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data == {"name": "chance", "ok": True, "checks": [], "overlay": None}


def test_parity_with_no_reported_metrics_plain_text(run):
    result = run("zoo", "parity", "random")

    assert result.code == 0
    assert "no reported metrics" in result.out


def test_parity_unknown_protocol_filter_is_a_config_error(run):
    result = run("zoo", "parity", "chance", "--protocol", "nope/official")

    assert result.code == 2
    assert "hint:" in result.err


class ReportedChanceAdapter:
    """A ``chance``-like fixture adapter (constant 0.5): its card declares a ``reported`` AUC of
    0.5, which a fully tied constant score always measures exactly, so ``dfwb zoo parity`` has a
    real (if trivial) protocol to score, measure and compare without a trained model."""

    card: ClassVar[AdapterCard]

    def load(self, weights: Any, device: str, *, seed: int | None = None) -> Any:
        from dfwb.zoo.adapter import meta_from_card
        from dfwb.zoo.adapters.chance import ChanceDetector

        return ChanceDetector(meta_from_card(self.card, source=f"zoo:{self.card.name}"))


def _reported_card(name: str, protocol: str, *, value: float) -> AdapterCard:
    return parse_card(
        f"""
name: {name}
display_name: Reported Chance (fixture)
contract_version: [1, 0]
license: {{code: MIT, requires_ack: false}}
code_strategy: pip
input: {{crop: face, crop_scale: 1.3, size: [64, 64], frames: 1}}
reported:
  - {{protocol: "{protocol}", split: test, metric: auc, value: {value}, source: fixture}}
"""
    )


@pytest.fixture
def scoretoy(tmp_path, monkeypatch):
    """Installs the ``scoretoy`` fixture pack, points WORK/RUNS roots at ``tmp_path``, and skips
    cleanly if torch is not installed (actually scoring needs it)."""
    pytest.importorskip("torch")
    from tests.unit.score._toy import install_scoretoy_pack

    install_scoretoy_pack(tmp_path, monkeypatch)
    monkeypatch.setenv("DFWB_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("DFWB_RUNS_ROOT", str(tmp_path / "runs"))
    return tmp_path / "work"


def test_parity_scores_and_compares_against_a_real_protocol(run, scoretoy, monkeypatch):
    from tests.unit.score._toy import PROTOCOL, toy_run_profile, write_toy_store

    write_toy_store(scoretoy, toy_run_profile())
    get_registry("detectors").add(
        "reported-chance-pass", target="tests.unit.zoo.test_cli:ReportedChanceAdapter", summary="x"
    )
    card = _reported_card("reported-chance-pass", PROTOCOL, value=0.5)
    monkeypatch.setattr(ReportedChanceAdapter, "card", card, raising=False)

    result = run("zoo", "parity", "reported-chance-pass", "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert data["ok"] is True
    assert len(data["checks"]) == 1
    check = data["checks"][0]
    assert check["metric"] == "auc"
    assert check["reported"] == 0.5
    assert check["measured"] == pytest.approx(0.5)
    assert check["passed"] is True
    assert data["overlay"] is not None

    from dfwb.zoo.parity import read_parity_overlay

    overlay = read_parity_overlay("reported-chance-pass")
    assert len(overlay) == 1
    assert overlay[0].value == pytest.approx(0.5)


def test_parity_reports_a_mismatch_but_still_writes_the_overlay(run, scoretoy, monkeypatch):
    from tests.unit.score._toy import PROTOCOL, toy_run_profile, write_toy_store

    write_toy_store(scoretoy, toy_run_profile())
    get_registry("detectors").add(
        "reported-chance-fail", target="tests.unit.zoo.test_cli:ReportedChanceAdapter", summary="x"
    )
    card = _reported_card("reported-chance-fail", PROTOCOL, value=0.9)
    monkeypatch.setattr(ReportedChanceAdapter, "card", card, raising=False)

    result = run("zoo", "parity", "reported-chance-fail", "--tolerance", "0.01", "--json")

    assert result.code == 4
    data = json.loads(result.out)
    assert data["ok"] is False
    assert data["checks"][0]["passed"] is False

    from dfwb.zoo.parity import read_parity_overlay

    overlay = read_parity_overlay("reported-chance-fail")
    assert len(overlay) == 1  # still written, for the author to inspect


def test_parity_protocol_filter_only_scores_the_matching_reported_metrics(
    run, scoretoy, monkeypatch
):
    from tests.unit.score._toy import PROTOCOL, toy_run_profile, write_toy_store

    write_toy_store(scoretoy, toy_run_profile())
    get_registry("detectors").add(
        "reported-chance-filtered",
        target="tests.unit.zoo.test_cli:ReportedChanceAdapter",
        summary="x",
    )
    card = parse_card(
        f"""
name: reported-chance-filtered
display_name: Reported Chance (fixture)
contract_version: [1, 0]
license: {{code: MIT, requires_ack: false}}
code_strategy: pip
input: {{crop: face, crop_scale: 1.3, size: [64, 64], frames: 1}}
reported:
  - {{protocol: "{PROTOCOL}", split: test, metric: auc, value: 0.5, source: fixture}}
  - {{protocol: "some-other-pack:other/official", split: test, metric: auc, value: 0.5,
     source: fixture}}
"""
    )
    monkeypatch.setattr(ReportedChanceAdapter, "card", card, raising=False)

    result = run("zoo", "parity", "reported-chance-filtered", "--protocol", PROTOCOL, "--json")

    assert result.code == 0
    data = json.loads(result.out)
    assert len(data["checks"]) == 1
    assert data["checks"][0]["protocol"] == PROTOCOL


def test_parity_plain_text_passing(run, scoretoy, monkeypatch):
    from tests.unit.score._toy import PROTOCOL, toy_run_profile, write_toy_store

    write_toy_store(scoretoy, toy_run_profile())
    get_registry("detectors").add(
        "reported-chance-plain-pass",
        target="tests.unit.zoo.test_cli:ReportedChanceAdapter",
        summary="x",
    )
    card = _reported_card("reported-chance-plain-pass", PROTOCOL, value=0.5)
    monkeypatch.setattr(ReportedChanceAdapter, "card", card, raising=False)

    result = run("zoo", "parity", "reported-chance-plain-pass")

    assert result.code == 0
    assert "pass" in result.out
    assert "wrote " in result.out


def test_parity_plain_text_failing(run, scoretoy, monkeypatch):
    from tests.unit.score._toy import PROTOCOL, toy_run_profile, write_toy_store

    write_toy_store(scoretoy, toy_run_profile())
    get_registry("detectors").add(
        "reported-chance-plain-fail",
        target="tests.unit.zoo.test_cli:ReportedChanceAdapter",
        summary="x",
    )
    card = _reported_card("reported-chance-plain-fail", PROTOCOL, value=0.9)
    monkeypatch.setattr(ReportedChanceAdapter, "card", card, raising=False)

    result = run("zoo", "parity", "reported-chance-plain-fail")

    assert result.code == 4
    assert "FAIL" in result.out
    assert "wrote " in result.out
