"""Suites: YAML loading, validation, and rolling per-entry results into aggregate rows."""

from __future__ import annotations

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.eval.suites import Suite, aggregate_suite, read_suite

SUITE_YAML = """
name: demo
entries:
  - {protocol: celebdf-v2/official, split: test, group: in-domain}
  - {protocol: ffpp/official, split: test, where: {compression: c23}, group: cross-dataset}
  - {protocol: dfdc/official, split: test, group: cross-dataset}
aggregates:
  - {group: cross-dataset, metric: auc, how: mean}
"""


def test_read_suite(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text(SUITE_YAML)
    suite = read_suite(path)
    assert suite.name == "demo"
    assert len(suite.entries) == 3
    assert suite.entries[1].where == {"compression": "c23"}
    assert suite.aggregates[0].how == "mean"


def test_read_suite_rejects_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: demo\nentries: not-a-list\n")
    with pytest.raises(ContractError):
        read_suite(path)


def test_read_suite_missing_file(tmp_path):
    with pytest.raises(ContractError, match="cannot read"):
        read_suite(tmp_path / "nope.yaml")


def test_aggregate_suite_means_the_group():
    suite = Suite.model_validate(
        {
            "name": "demo",
            "entries": [
                {"protocol": "a/official", "split": "test", "group": "in-domain"},
                {"protocol": "b/official", "split": "test", "group": "cross-dataset"},
                {"protocol": "c/official", "split": "test", "group": "cross-dataset"},
            ],
            "aggregates": [{"group": "cross-dataset", "metric": "auc", "how": "mean"}],
        }
    )
    results = {0: {"auc": 0.99}, 1: {"auc": 0.80}, 2: {"auc": 0.90}}
    rows = aggregate_suite(suite, results)
    assert rows == [
        {
            "group": "cross-dataset",
            "metric": "auc",
            "how": "mean",
            "value": pytest.approx(0.85),
            "n_entries": 2,
            "n_expected": 2,
        }
    ]


def test_aggregate_suite_unknown_group_raises():
    suite = Suite.model_validate(
        {
            "name": "demo",
            "entries": [{"protocol": "a/official", "split": "test", "group": "in-domain"}],
            "aggregates": [{"group": "cross-dataset", "metric": "auc"}],
        }
    )
    with pytest.raises(ConfigError, match="no entries"):
        aggregate_suite(suite, {0: {"auc": 0.9}})


def test_aggregate_suite_missing_metric_raises():
    suite = Suite.model_validate(
        {
            "name": "demo",
            "entries": [{"protocol": "a/official", "split": "test", "group": "in-domain"}],
            "aggregates": [{"group": "in-domain", "metric": "auc"}],
        }
    )
    with pytest.raises(ConfigError, match="no result for metric"):
        aggregate_suite(suite, {0: {"ap": 0.9}})
