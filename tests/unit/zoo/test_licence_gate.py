"""The licence gate: a card whose licence needs an explicit acknowledgement blocks use until
``dfwb.core.licenses`` records it; a card that needs none never asks. Zoo cards go through the
same gate (``dfwb.core.licenses.require_accepted``) the face backends use, so the shape of the
error is identical: ``InstallationError``, exit code 5, a hint naming ``--accept-license``."""

from __future__ import annotations

import pytest

from dfwb.core import licenses
from dfwb.core.errors import InstallationError
from dfwb.zoo.adapter import require_license_accepted
from dfwb.zoo.card import parse_card

_GATED_CARD = """
name: gend
display_name: GenD
contract_version: [1, 0]
license: {code: Apache-2.0, weights: LicenseRef-CC-BY-NC-4.0, requires_ack: true}
code_strategy: pip
input: {}
"""

_OPEN_CARD = """
name: chance
display_name: Chance
contract_version: [1, 0]
license: {code: MIT, requires_ack: false}
code_strategy: pip
input: {}
"""

_GATED_PINNED_CLONE_CARD = """
name: fsfm
display_name: FSFM
contract_version: [1, 0]
upstream: {repo: "https://example.org/fsfm", commit: "0123456789abcdef0123456789abcdef01234567"}
license: {code: LicenseRef-CC-BY-NC-4.0, requires_ack: true}
code_strategy: pinned-clone
input: {}
"""


def test_a_card_needing_no_acknowledgement_never_raises(isolated):
    card = parse_card(_OPEN_CARD)
    require_license_accepted(card)  # must not raise


def test_a_gated_card_raises_until_accepted(isolated):
    card = parse_card(_GATED_CARD)

    with pytest.raises(InstallationError) as info:
        require_license_accepted(card)
    assert info.value.exit_code == 5
    assert "gend" in info.value.message
    assert "--accept-license" in info.value.hint

    licenses.accept(card.name, license=card.license.weights or card.license.code)
    require_license_accepted(card)  # now accepted: must not raise


def test_the_gate_names_the_weights_licence_first_when_there_is_one(isolated):
    card = parse_card(_GATED_CARD)
    with pytest.raises(InstallationError) as info:
        require_license_accepted(card)
    assert "LicenseRef-CC-BY-NC-4.0" in info.value.hint


def test_the_hint_names_the_actionable_command(isolated):
    card = parse_card(_GATED_CARD)
    with pytest.raises(InstallationError) as info:
        require_license_accepted(card)
    assert "dfwb zoo fetch gend --accept-license" in info.value.hint


def test_a_pinned_clones_gate_names_the_code_licence_not_the_weights(isolated):
    card = parse_card(_GATED_PINNED_CLONE_CARD)
    with pytest.raises(InstallationError) as info:
        require_license_accepted(card)
    assert "code licence" in info.value.message
    assert "model weights" not in info.value.message
    assert "LicenseRef-CC-BY-NC-4.0" in info.value.hint
    assert "dfwb zoo fetch fsfm --accept-license" in info.value.hint


def test_acceptance_is_isolated_by_dfwb_state_dir(isolated, tmp_path, monkeypatch):
    card = parse_card(_GATED_CARD)
    licenses.accept(card.name, license=card.license.weights or card.license.code)
    require_license_accepted(card)

    monkeypatch.setenv("DFWB_STATE_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(InstallationError):
        require_license_accepted(card)
