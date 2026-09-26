"""The licence gate: a card whose licence needs an explicit acknowledgement blocks use until
``dfwb.core.licenses`` records it; a card that needs none never asks. Zoo cards go through the
same gate (``dfwb.core.licenses.require_accepted``) the face backends use, so the shape of the
error is identical: ``InstallationError``, exit code 5, a hint naming ``--accept-license``."""

from __future__ import annotations

import pytest

from dfwb.core import licenses
from dfwb.core.errors import InstallationError
from dfwb.zoo.adapter import accept_license, license_text_for, require_license_accepted
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

# Unlike _GATED_PINNED_CLONE_CARD above (whose weights licence is simply absent, so the fallback
# to the code licence would hide a bug that only shows up with a genuinely *distinct* one), this
# card's weights licence differs from its code licence -- the shape that exposed the original bug
# (recording the weights licence for a pinned-clone card the gate itself asks about by its code
# licence).
_GATED_PINNED_CLONE_CARD_WITH_DISTINCT_WEIGHTS_LICENCE = """
name: fsfm-weights
display_name: FSFM (distinct weights licence)
contract_version: [1, 0]
upstream: {repo: "https://example.org/fsfm", commit: "0123456789abcdef0123456789abcdef01234567"}
license: {code: Apache-2.0, weights: LicenseRef-CC-BY-NC-4.0, requires_ack: true}
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


# ------------------------------------------------------------------------ license_text_for


def test_license_text_for_prefers_the_weights_licence_when_there_is_one(isolated):
    card = parse_card(_GATED_CARD)
    assert license_text_for(card) == "LicenseRef-CC-BY-NC-4.0"


def test_license_text_for_falls_back_to_the_code_licence_with_no_weights_licence(isolated):
    card = parse_card(_OPEN_CARD)
    assert license_text_for(card) == "MIT"


def test_license_text_for_a_pinned_clone_is_always_the_code_licence(isolated):
    # Even though this card declares a distinct weights licence, a pinned-clone card is gated (and
    # its acceptance recorded) on its *code* licence -- the usual reason it uses that strategy.
    card = parse_card(_GATED_PINNED_CLONE_CARD_WITH_DISTINCT_WEIGHTS_LICENCE)
    assert license_text_for(card) == "Apache-2.0"


# --------------------------------------------------------------------------- accept_license


def test_accept_license_records_exactly_what_the_gate_checks(isolated):
    card = parse_card(_GATED_PINNED_CLONE_CARD_WITH_DISTINCT_WEIGHTS_LICENCE)

    accept_license(card)

    require_license_accepted(card)  # now accepted: must not raise
    assert licenses.all_accepted()[card.name].license == "Apache-2.0"


def test_accept_license_is_a_no_op_when_no_acknowledgement_is_needed(isolated):
    card = parse_card(_OPEN_CARD)

    accept_license(card)

    assert not licenses.is_accepted(card.name)


def test_accept_license_does_not_record_a_second_time_once_already_accepted(isolated, monkeypatch):
    import dfwb.zoo.adapter as adapter_module

    real_accept = licenses.accept
    calls: list[tuple[str, str]] = []

    def _tracking_accept(name: str, *, license: str) -> None:
        calls.append((name, license))
        real_accept(name, license=license)

    monkeypatch.setattr(adapter_module, "_accept_licence", _tracking_accept)
    card = parse_card(_GATED_CARD)

    accept_license(card)
    accept_license(card)

    assert calls == [(card.name, "LicenseRef-CC-BY-NC-4.0")]
