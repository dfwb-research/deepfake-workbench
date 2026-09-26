"""The parity harness: compares reproduced numbers against a card's ``reported`` claims, and
writes them to the local overlay file, shaped so they can be pasted straight into the card."""

from __future__ import annotations

import json

import pytest

from dfwb.core.errors import ContractError
from dfwb.zoo.card import parse_card
from dfwb.zoo.parity import compare_parity, parity_path, read_parity_overlay, write_parity_overlay

_CARD = """
name: gend
display_name: GenD
contract_version: [1, 0]
license: {code: Apache-2.0}
code_strategy: pip
input: {}
reported:
  - {protocol: celebdf-v2/official, split: test, metric: video_auc, value: 0.90, source: "Table 2"}
  - {protocol: ffpp/official, split: test, metric: video_auc, value: 0.95, source: "Table 3"}
"""


def _card():
    return parse_card(_CARD)


def test_compare_parity_matches_each_reported_metric_to_its_measured_value(isolated):
    checks = compare_parity(
        _card(),
        {
            ("celebdf-v2/official", "test", "video_auc"): 0.905,
            ("ffpp/official", "test", "video_auc"): 0.80,
        },
        tolerance=0.01,
    )
    by_protocol = {c.protocol: c for c in checks}
    assert by_protocol["celebdf-v2/official"].passed is True
    assert by_protocol["ffpp/official"].passed is False
    assert by_protocol["ffpp/official"].reported == 0.95
    assert by_protocol["ffpp/official"].measured == 0.80


def test_compare_parity_raises_when_a_reported_metric_has_no_measured_value(isolated):
    with pytest.raises(ContractError, match="ffpp/official"):
        compare_parity(_card(), {("celebdf-v2/official", "test", "video_auc"): 0.905})


def test_write_parity_overlay_writes_a_file_shaped_like_the_cards_own_parity_list(isolated):
    card = _card()
    checks = compare_parity(
        card,
        {
            ("celebdf-v2/official", "test", "video_auc"): 0.905,
            ("ffpp/official", "test", "video_auc"): 0.95,
        },
    )
    metrics = [c.as_metric(dfwb_version="0.1.0", date="2026-01-01") for c in checks]

    path = write_parity_overlay(card.name, metrics)

    assert path == parity_path(card.name)
    payload = json.loads(path.read_text("utf-8"))
    assert len(payload["parity"]) == 2
    entry = next(e for e in payload["parity"] if e["protocol"] == "celebdf-v2/official")
    assert entry["value"] == 0.905
    assert entry["metric"] == "video_auc"
    assert entry["dfwb_version"] == "0.1.0"
    assert entry["date"] == "2026-01-01"

    # Pasteable straight into the card: re-validating with this list as `parity:` succeeds, and
    # keeps exactly what was measured.
    from dfwb.zoo.card import AdapterCard

    updated = AdapterCard.model_validate(
        {**card.model_dump(mode="json"), "parity": payload["parity"]}
    )
    assert {c.value for c in updated.parity} == {0.905, 0.95}


def test_read_parity_overlay_round_trips_what_was_written(isolated):
    card = _card()
    checks = compare_parity(
        card,
        {
            ("celebdf-v2/official", "test", "video_auc"): 0.905,
            ("ffpp/official", "test", "video_auc"): 0.95,
        },
    )
    metrics = [c.as_metric(dfwb_version="0.1.0", date="2026-01-01") for c in checks]
    write_parity_overlay(card.name, metrics)

    read_back = read_parity_overlay(card.name)

    assert read_back == metrics


def test_read_parity_overlay_is_empty_before_any_run(isolated):
    assert read_parity_overlay("never-run") == []


def test_write_parity_overlay_is_under_the_cache_root(isolated):
    path = write_parity_overlay("x", [])
    assert path.parent == isolated.cache / "zoo" / "x"
