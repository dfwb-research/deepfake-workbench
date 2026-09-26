"""The ``zoo:`` detector source: ``zoo:<name>[@<weights id>]`` resolves through the ``detectors``
registry to a C4 ``Detector`` with ``DetectorMeta``, verifying and caching weights (through a
local HTTP server, never the real network) and gating on any required licence first."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.zoo._fixtures import BadDetectorAdapter, NoLoadAdapter, WeightedTestAdapter

from dfwb.core import licenses
from dfwb.core.detector import DetectorMeta
from dfwb.core.errors import ConfigError, ContractError, InstallationError, UnknownKeyError
from dfwb.core.plugins import get_registry
from dfwb.score.sources import resolve_detector
from dfwb.zoo.card import parse_card

CONTENT = b"pretend weights\n" * 10
SHA256 = hashlib.sha256(CONTENT).hexdigest()


def _register_weighted(name: str = "weighted-test") -> None:
    get_registry("detectors").add(
        name, target="tests.unit.zoo._fixtures:WeightedTestAdapter", summary="test fixture"
    )


def _card_yaml(name: str, weights_block: str, *, requires_ack: bool = False) -> str:
    return f"""
name: {name}
display_name: Weighted Test
contract_version: [1, 0]
license: {{code: MIT, requires_ack: {"true" if requires_ack else "false"}}}
code_strategy: pip
input: {{crop: face, crop_scale: 1.3, size: [64, 64], frames: 1}}
{weights_block}
"""


# ---------------------------------------------------------------------------------- the dummies


def test_zoo_chance_resolves_to_a_detector_with_detector_meta(isolated):
    detector = resolve_detector("zoo:chance")
    assert isinstance(detector.meta, DetectorMeta)
    assert detector.meta.name == "chance"
    assert detector.meta.source == "zoo:chance"
    assert detector.checkpoint_sha256 is None


def test_zoo_random_resolves_to_a_detector_with_detector_meta(isolated):
    detector = resolve_detector("zoo:random")
    assert isinstance(detector.meta, DetectorMeta)
    assert detector.meta.name == "random"
    assert detector.meta.source == "zoo:random"
    assert detector.checkpoint_sha256 is None


# --------------------------------------------------------------------------------------- errors


def test_unknown_adapter_name_lists_the_known_ones(isolated):
    with pytest.raises(UnknownKeyError) as info:
        resolve_detector("zoo:chnace")
    assert "chance" in info.value.message or "chance" in (info.value.hint or "")


def test_an_empty_adapter_name_is_a_config_error(isolated):
    with pytest.raises(ConfigError):
        resolve_detector("zoo:@default")


def test_a_no_weights_adapter_with_an_explicit_id_is_a_config_error(isolated):
    with pytest.raises(ConfigError, match="no weights"):
        resolve_detector("zoo:chance@default")


def test_an_adapter_missing_load_is_a_contract_error(isolated, monkeypatch):
    get_registry("detectors").add(
        "no-load-adapter", target="tests.unit.zoo._fixtures:NoLoadAdapter", summary="test fixture"
    )
    card = parse_card(_card_yaml("no-load-adapter", ""))
    monkeypatch.setattr(NoLoadAdapter, "card", card, raising=False)

    with pytest.raises(ContractError, match="load"):
        resolve_detector("zoo:no-load-adapter")


def test_an_adapter_missing_card_is_a_contract_error(isolated):
    get_registry("detectors").add(
        "no-card-adapter", target="tests.unit.zoo._fixtures:NoCardAdapter", summary="test fixture"
    )

    with pytest.raises(ContractError, match="card"):
        resolve_detector("zoo:no-card-adapter")


def test_an_adapter_whose_card_name_does_not_match_the_registry_key_is_a_contract_error(
    isolated, monkeypatch
):
    # A well-formed adapter (has both `card` and `load`) so this actually exercises the name
    # check itself, not the earlier "missing load" check.
    get_registry("detectors").add(
        "registered-as-this", target="tests.unit.zoo._fixtures:WeightedTestAdapter", summary="x"
    )
    card = parse_card(_card_yaml("declared-as-something-else", ""))
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    with pytest.raises(ContractError, match="registered-as-this") as info:
        resolve_detector("zoo:registered-as-this")
    assert "declared-as-something-else" in info.value.message


def test_a_detector_missing_predict_and_to_is_a_contract_error(isolated, monkeypatch):
    get_registry("detectors").add(
        "bad-detector-adapter",
        target="tests.unit.zoo._fixtures:BadDetectorAdapter",
        summary="test fixture",
    )
    card = parse_card(_card_yaml("bad-detector-adapter", ""))
    monkeypatch.setattr(BadDetectorAdapter, "card", card, raising=False)

    with pytest.raises(ContractError) as info:
        resolve_detector("zoo:bad-detector-adapter")
    assert "predict" in info.value.message
    assert "to" in info.value.message


# --------------------------------------------------------------------------------- with weights


def test_resolving_a_single_weight_variant_needs_no_explicit_id(server, isolated, monkeypatch):
    _register_weighted()
    server.routes["/w.safetensors"] = CONTENT
    card = parse_card(
        _card_yaml(
            "weighted-test",
            f"""weights:
  - {{id: default, url: "{server.url}/w.safetensors", sha256: "{SHA256}", bytes: {len(CONTENT)},
     format: safetensors}}""",
        )
    )
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    detector = resolve_detector("zoo:weighted-test")

    assert detector.meta.source == "zoo:weighted-test@default"
    assert detector.checkpoint_sha256 == SHA256
    assert detector.weights_path.read_bytes() == CONTENT


def test_checkpoint_sha256_is_the_measured_hash_not_the_cards_own_string(
    server, isolated, monkeypatch, tmp_path
):
    # A defensive check: even if the weight manager ever returned a path whose content did not
    # actually match the card (a bug in `ensure_weights`, say), `load_zoo` must record what the
    # file on disk really hashes to, never just echo the card's own declared string back out.
    _register_weighted()
    card = parse_card(
        _card_yaml(
            "weighted-test",
            f"""weights:
  - {{id: default, url: "{server.url}/w.safetensors", sha256: "{SHA256}", bytes: {len(CONTENT)},
     format: safetensors}}""",
        )
    )
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    wrong_content = b"not what the card declares at all"
    decoy = tmp_path / "decoy.safetensors"
    decoy.write_bytes(wrong_content)
    import dfwb.zoo.source as source_module

    monkeypatch.setattr(source_module, "ensure_weights", lambda name, spec: decoy)

    detector = resolve_detector("zoo:weighted-test")

    assert detector.checkpoint_sha256 == hashlib.sha256(wrong_content).hexdigest()
    assert detector.checkpoint_sha256 != SHA256


def test_resolving_with_an_explicit_weights_id(server, isolated, monkeypatch):
    _register_weighted()
    server.routes["/a.safetensors"] = CONTENT
    other = b"other content\n"
    other_sha = hashlib.sha256(other).hexdigest()
    server.routes["/b.safetensors"] = other
    card = parse_card(
        _card_yaml(
            "weighted-test",
            f"""weights:
  - {{id: a, url: "{server.url}/a.safetensors", sha256: "{SHA256}", bytes: {len(CONTENT)},
     format: safetensors}}
  - {{id: b, url: "{server.url}/b.safetensors", sha256: "{other_sha}", bytes: {len(other)},
     format: safetensors}}""",
        )
    )
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    detector = resolve_detector("zoo:weighted-test@b")

    assert detector.meta.source == "zoo:weighted-test@b"
    assert detector.checkpoint_sha256 == other_sha
    assert detector.weights_path.read_bytes() == other


def test_ambiguous_weights_with_no_id_is_a_config_error(server, isolated, monkeypatch):
    _register_weighted()
    card = parse_card(
        _card_yaml(
            "weighted-test",
            f"""weights:
  - {{id: a, url: "{server.url}/a", sha256: "{SHA256}", bytes: 1, format: safetensors}}
  - {{id: b, url: "{server.url}/b", sha256: "{SHA256}", bytes: 1, format: safetensors}}""",
        )
    )
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    with pytest.raises(ConfigError) as info:
        resolve_detector("zoo:weighted-test")
    assert "a" in info.value.message
    assert "b" in info.value.message


def test_unknown_weights_id_is_reported(server, isolated, monkeypatch):
    _register_weighted()
    card = parse_card(
        _card_yaml(
            "weighted-test",
            f"""weights:
  - {{id: default, url: "{server.url}/w", sha256: "{SHA256}", bytes: 1, format: safetensors}}""",
        )
    )
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    with pytest.raises(UnknownKeyError, match="nope"):
        resolve_detector("zoo:weighted-test@nope")


def test_a_gated_adapter_is_blocked_until_its_licence_is_accepted(server, isolated, monkeypatch):
    _register_weighted()
    server.routes["/w.safetensors"] = CONTENT
    card = parse_card(
        _card_yaml(
            "weighted-test",
            f"""weights:
  - {{id: default, url: "{server.url}/w.safetensors", sha256: "{SHA256}", bytes: {len(CONTENT)},
     format: safetensors}}""",
            requires_ack=True,
        )
    )
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    with pytest.raises(InstallationError) as info:
        resolve_detector("zoo:weighted-test")
    assert info.value.exit_code == 5
    # Gated before anything is touched: no request made, nothing cached yet.
    assert server.requests == []
    from dfwb.zoo.weights import cache_dir

    assert not cache_dir("weighted-test", SHA256).exists()

    licenses.accept(card.name, license=card.license.code)
    detector = resolve_detector("zoo:weighted-test")
    assert detector.checkpoint_sha256 == SHA256
    assert server.requests == ["/w.safetensors"]


# ---------------------------------------------------------------------------- pinned-clone strategy


def _init_repo(repo: Path) -> str:
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


def test_load_zoo_clones_a_pinned_clone_adapter_before_loading_it(isolated, tmp_path, monkeypatch):
    repo = tmp_path / "upstream"
    repo.mkdir()
    commit = _init_repo(repo)
    _register_weighted("pinned-clone-test")
    card = parse_card(
        f"""
name: pinned-clone-test
display_name: Pinned Clone Test
contract_version: [1, 0]
upstream: {{repo: "{repo}", commit: "{commit}"}}
license: {{code: MIT}}
code_strategy: pinned-clone
input: {{}}
"""
    )
    monkeypatch.setattr(WeightedTestAdapter, "card", card, raising=False)

    detector = resolve_detector("zoo:pinned-clone-test")

    from dfwb.zoo.strategies import clone_cache_dir

    assert detector.code_root == clone_cache_dir(card.name, commit)
    assert (detector.code_root / "entry.py").is_file()


# ------------------------------------------------------------------------------- seed threading


def _fake_batch() -> SimpleNamespace:
    return SimpleNamespace(
        keys=["real/00", "fake/00"], dataset_ids=["toy", "toy"], compressions=[None, None]
    )


def test_the_same_seed_reproduces_random_scores_exactly(isolated):
    pytest.importorskip("torch")
    batch = _fake_batch()
    first = resolve_detector("zoo:random", seed=5).predict(batch).score
    second = resolve_detector("zoo:random", seed=5).predict(batch).score
    assert (first == second).all()


def test_two_different_seeds_give_different_random_scores(isolated):
    pytest.importorskip("torch")
    batch = _fake_batch()
    a = resolve_detector("zoo:random", seed=0).predict(batch).score
    b = resolve_detector("zoo:random", seed=1).predict(batch).score
    assert (a != b).any()


def test_resolve_detector_with_no_seed_uses_the_documented_default(isolated):
    pytest.importorskip("torch")
    batch = _fake_batch()
    default = resolve_detector("zoo:random").predict(batch).score
    explicit_zero = resolve_detector("zoo:random", seed=0).predict(batch).score
    assert (default == explicit_zero).all()


def test_random_scores_do_not_depend_on_batch_order(isolated):
    pytest.importorskip("torch")
    forward = (
        resolve_detector("zoo:random", seed=3)
        .predict(
            SimpleNamespace(keys=["a", "b"], dataset_ids=["toy", "toy"], compressions=[None, None])
        )
        .score
    )
    backward = (
        resolve_detector("zoo:random", seed=3)
        .predict(
            SimpleNamespace(keys=["b", "a"], dataset_ids=["toy", "toy"], compressions=[None, None])
        )
        .score
    )
    assert forward[0].item() == backward[1].item()
    assert forward[1].item() == backward[0].item()
