"""Suites: YAML loading, validation, and rolling per-entry results into aggregate rows."""

from __future__ import annotations

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.eval.suites import Suite, aggregate_suite, list_suites, load_suite, read_suite

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


def test_a_suite_may_describe_itself(tmp_path):
    # a pack documents what a suite is for (its panel, its training reference) in the suite itself
    path = tmp_path / "described.yaml"
    path.write_text(
        "name: described\n"
        "description: Train on ffpp/official at c23, then test on every other dataset.\n"
        "entries:\n  - {protocol: dfdc/official, split: test, group: cross-dataset}\n"
    )
    suite = read_suite(path)
    assert suite.description == "Train on ffpp/official at c23, then test on every other dataset."
    assert read_suite(_write(tmp_path / "plain.yaml", SUITE_YAML)).description is None


def _write(path, text):
    path.write_text(text)
    return path


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


def test_aggregate_suite_group_with_only_undefined_values_is_an_undefined_row():
    """An entry that was scored but whose metric is undefined (``None``: a single-class file under
    a two-class metric) is not an unscored entry: its group gets an undefined row, not an abort."""
    suite = Suite.model_validate(
        {
            "name": "demo",
            "entries": [
                {"protocol": "a/official", "split": "test", "group": "in-domain"},
                {"protocol": "b/official", "split": "test", "group": "cross-dataset"},
            ],
            "aggregates": [
                {"group": "in-domain", "metric": "auc"},
                {"group": "cross-dataset", "metric": "auc"},
            ],
        }
    )
    rows = aggregate_suite(suite, {0: {"auc": 0.9}, 1: {"auc": None}})
    by_group = {row["group"]: row for row in rows}
    assert by_group["in-domain"]["value"] == pytest.approx(0.9)
    assert "undefined" not in by_group["in-domain"]
    undefined = by_group["cross-dataset"]
    assert undefined["value"] is None
    assert undefined["n_entries"] == 0
    assert undefined["n_expected"] == 1
    assert "auc" in undefined["undefined"]


def test_aggregate_suite_mean_leaves_out_an_undefined_entry():
    suite = Suite.model_validate(
        {
            "name": "demo",
            "entries": [
                {"protocol": "a/official", "split": "test", "group": "cross-dataset"},
                {"protocol": "b/official", "split": "test", "group": "cross-dataset"},
            ],
            "aggregates": [{"group": "cross-dataset", "metric": "auc"}],
        }
    )
    (row,) = aggregate_suite(suite, {0: {"auc": 0.8}, 1: {"auc": None}})
    assert row["value"] == pytest.approx(0.8)
    assert (row["n_entries"], row["n_expected"]) == (1, 2)


def test_aggregate_suite_unscored_group_still_raises():
    suite = Suite.model_validate(
        {
            "name": "demo",
            "entries": [{"protocol": "a/official", "split": "test", "group": "in-domain"}],
            "aggregates": [{"group": "in-domain", "metric": "auc"}],
        }
    )
    with pytest.raises(ConfigError, match="no result for metric 'auc'"):
        aggregate_suite(suite, {})


def test_builtin_toyfake_suite_is_registered_and_loadable():
    assert "toyfake" in list_suites()
    suite = load_suite("toyfake")
    assert suite.name == "toyfake"
    protocols = {(e.protocol, e.split) for e in suite.entries}
    assert protocols == {("toyfake/official", "test"), ("toyfake/ident-72-14-14", "test")}
    groups = {e.group for e in suite.entries}
    assert groups == {"in-domain", "cross"}


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
