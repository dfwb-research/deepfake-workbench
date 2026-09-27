"""The adapter card schema: valid and invalid fixtures, and the two shipped cards."""

from __future__ import annotations

import pytest

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ContractError
from dfwb.zoo.adapter import read_builtin_card
from dfwb.zoo.card import AdapterCard, parse_card

_FULL_CARD = """
name: gend
display_name: GenD
contract_version: [1, 0]
upstream:
  repo: https://example.org/org/gend
  commit: "0123456789abcdef0123456789abcdef01234567"
  paper: {title: "A Paper", venue: CVPR, year: 2025, doi: "10.1/x"}
  bibtex: "@inproceedings{gend}"
license:
  code: Apache-2.0
  weights: LicenseRef-CC-BY-NC-4.0
  requires_ack: true
code_strategy: pinned-clone
install: {extra: zoo-gend, pip: ["open_clip_torch>=2.24,<3"]}
weights:
  - {id: default, url: "https://example.org/w.safetensors",
     sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", bytes: 123,
     format: safetensors, trained_on: [ffpp/official], redistribution: undecided}
input: {crop: face, crop_scale: 1.3, size: [224, 224], frames: 1,
        mean: [0.481, 0.458, 0.408], std: [0.269, 0.261, 0.276]}
polarity: fake-high
reported:
  - {protocol: celebdf-v2/official, split: test, metric: video_auc, value: 0.91, source: "Table 2"}
parity:
  - {protocol: celebdf-v2/official, split: test, metric: video_auc, value: 0.905, tolerance: 0.01,
     dfwb_version: "0.1.0", date: "2026-01-01"}
"""

_MINIMAL_CARD = """
name: chance
display_name: Chance
contract_version: [1, 0]
license: {code: MIT, requires_ack: false}
code_strategy: pip
input: {crop: face, crop_scale: 1.3, size: [64, 64], frames: 1}
"""


def test_parses_the_full_c4_example():
    card = parse_card(_FULL_CARD)
    assert isinstance(card, AdapterCard)
    assert card.name == "gend"
    assert card.upstream is not None
    assert card.upstream.commit == "0123456789abcdef0123456789abcdef01234567"
    assert card.upstream.paper is not None
    assert card.upstream.paper.year == 2025
    assert card.license.requires_ack is True
    assert card.code_strategy == "pinned-clone"
    assert card.install is not None
    assert card.install.extra == "zoo-gend"
    assert len(card.weights) == 1
    assert card.weights[0].sha256 == "a" * 64
    assert card.input.size == (224, 224)
    assert card.reported[0].value == 0.91
    assert card.parity[0].tolerance == 0.01


def test_parses_a_minimal_dummy_style_card():
    card = parse_card(_MINIMAL_CARD)
    assert card.upstream is None
    assert card.install is None
    assert card.weights == ()
    assert card.license.weights is None
    assert card.polarity == "fake-high"
    assert card.reported == ()
    assert card.parity == ()


def test_to_input_spec_completes_the_remaining_c4_defaults():
    card = parse_card(_MINIMAL_CARD)
    spec = card.input.to_input_spec()
    assert spec == InputSpec(crop="face", crop_scale=1.3, size=(64, 64), frames=1)


@pytest.mark.parametrize(
    "broken",
    [
        "name: x\ndisplay_name: X\ncontract_version: [1, 0]\n"
        "license: {code: MIT}\ncode_strategy: pip\ninput: {}\nunknown_top_level_key: true\n",
        "name: x\ndisplay_name: X\ncontract_version: [1, 0]\ncode_strategy: pip\ninput: {}\n",
        "name: x\ndisplay_name: X\ncontract_version: [1, 0]\n"
        "license: {code: MIT}\ncode_strategy: not-a-real-strategy\ninput: {}\n",
        "name: x\ndisplay_name: X\ncontract_version: [1, 0]\n"
        "license: {code: MIT}\ncode_strategy: pip\ninput: {}\npolarity: sideways\n",
        "name: x\ndisplay_name: X\ncontract_version: [1, 0]\n"
        "license: {code: MIT, unknown_field: 1}\ncode_strategy: pip\ninput: {}\n",
    ],
    ids=[
        "unknown-top-level-key",
        "missing-license",
        "bad-code-strategy",
        "bad-polarity",
        "unknown-nested-key",
    ],
)
def test_invalid_cards_raise_contract_error(broken):
    with pytest.raises(ContractError):
        parse_card(broken)


def test_invalid_yaml_raises_contract_error():
    with pytest.raises(ContractError, match="invalid YAML"):
        parse_card("name: [unterminated")


def test_read_card_names_a_missing_file(tmp_path):
    from dfwb.zoo.card import read_card

    with pytest.raises(ContractError, match="cannot read"):
        read_card(tmp_path / "does-not-exist.yaml")


@pytest.mark.parametrize("name", ["not/a-slug", "..", "Upper", "has space", "trailing-"])
def test_a_name_that_is_not_a_safe_slug_is_rejected(name):
    text = (
        f"name: {name!r}\ndisplay_name: X\ncontract_version: [1, 0]\n"
        "license: {code: MIT}\ncode_strategy: pip\ninput: {}\n"
    )
    with pytest.raises(ContractError):
        parse_card(text)


@pytest.mark.parametrize(
    "sha256",
    ["aa", "A" * 64, ("a" * 63) + "g", "a" * 65],
    ids=["too-short", "uppercase", "non-hex", "too-long"],
)
def test_a_weight_sha256_that_is_not_64_lowercase_hex_is_rejected(sha256):
    text = f"""
name: x
display_name: X
contract_version: [1, 0]
license: {{code: MIT}}
code_strategy: pip
input: {{}}
weights:
  - {{id: default, url: "https://example.org/w", sha256: "{sha256}", bytes: 1, format: safetensors}}
"""
    with pytest.raises(ContractError):
        parse_card(text)


def test_weight_bytes_is_required():
    # Matches C4's own adapter-card schema, which always gives it: dropping it is a card mistake,
    # not something to default away silently.
    text = """
name: x
display_name: X
contract_version: [1, 0]
license: {code: MIT}
code_strategy: pip
input: {}
weights:
  - {id: default, url: "https://example.org/w",
     sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
     format: safetensors}
"""
    with pytest.raises(ContractError, match="bytes"):
        parse_card(text)


@pytest.mark.parametrize(
    "commit", ["main", "v1.0", "0123456789abcdef", "0123456789ABCDEF0123456789abcdef01234567"]
)
def test_an_upstream_commit_that_is_not_a_full_lowercase_sha_is_rejected(commit):
    text = f"""
name: x
display_name: X
contract_version: [1, 0]
upstream: {{repo: "https://example.org/x", commit: "{commit}"}}
license: {{code: MIT}}
code_strategy: pinned-clone
input: {{}}
"""
    with pytest.raises(ContractError):
        parse_card(text)


@pytest.mark.parametrize("name", ["chance", "random"])
def test_shipped_dummy_cards_have_no_weights_and_mit_licence(name):
    card = read_builtin_card(name)
    assert card.name == name
    assert card.weights == ()
    assert card.license.code == "MIT"
    assert card.license.requires_ack is False
    assert card.input.crop == "face"
    assert card.input.crop_scale == 1.3
    assert card.input.size == (64, 64)
    assert card.input.frames == 1


# ------------------------------------------------------------------------- reported metric names


def _card_with_metric(metric: str, *, section: str = "reported") -> str:
    if section == "reported":
        entry = f'{{protocol: x/official, split: test, metric: "{metric}", value: 0.9, source: t}}'
    else:
        entry = (
            f'{{protocol: x/official, split: test, metric: "{metric}", value: 0.9, '
            'tolerance: 0.01, dfwb_version: "0.1.0", date: "2026-01-01"}'
        )
    return _MINIMAL_CARD + f"{section}:\n  - {entry}\n"


@pytest.mark.parametrize("section", ["reported", "parity"])
def test_an_unknown_metric_is_refused_with_a_did_you_mean(section):
    with pytest.raises(ContractError) as info:
        parse_card(_card_with_metric("aucc", section=section))

    assert f"{section}[0].metric" in info.value.message
    assert "did you mean 'auc'" in info.value.message


def test_a_metric_with_a_bad_parameter_is_refused():
    with pytest.raises(ContractError, match="fprr"):
        parse_card(_card_with_metric("tpr@fprr=0.01"))


def test_a_metric_spec_with_parameters_is_accepted():
    card = parse_card(_card_with_metric("tpr@fpr=0.01"))

    assert card.reported[0].metric == "tpr@fpr=0.01"


def test_the_c4_examples_video_auc_is_read_as_auc():
    """Every row of a score file is one video, so a video-level AUC is ``auc`` over it: the
    spelling the adapter-card example uses is accepted, and read as the metric it names."""
    card = parse_card(_FULL_CARD)

    assert card.reported[0].metric == "auc"
    assert card.parity[0].metric == "auc"
