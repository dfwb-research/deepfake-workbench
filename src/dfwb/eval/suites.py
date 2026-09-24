"""Suites: named, data-defined collections of protocol/split/where entries, with aggregate rows.

A suite says which score files a full evaluation needs (one per entry: a protocol, a split and an
optional ``where`` filter) and how to roll several of them into one number per ``group`` (e.g. the
mean AUC over every entry tagged ``group: cross-dataset``). Packs register suites in the
``eval_suites`` data registry as plain YAML, so a new suite never needs a code change here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dfwb.core.errors import ConfigError, ContractError, did_you_mean, validation_messages
from dfwb.core.plugins import get_registry

__all__ = [
    "Suite",
    "SuiteAggregate",
    "SuiteEntry",
    "aggregate_suite",
    "list_suites",
    "load_suite",
    "read_suite",
]


class SuiteEntry(BaseModel):
    """One score file a suite needs: a protocol, a split, an optional filter and its group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: str
    split: str
    where: dict[str, Any] = Field(default_factory=dict)
    group: str


class SuiteAggregate(BaseModel):
    """One row a suite reports: a metric, reduced by ``how`` over every entry of ``group``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str
    metric: str
    how: Literal["mean"] = "mean"


class Suite(BaseModel):
    """``{name, entries: [...], aggregates: [...]}``, as loaded from one suite YAML file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    entries: list[SuiteEntry]
    aggregates: list[SuiteAggregate] = Field(default_factory=list)


def read_suite(path: str | Path) -> Suite:
    """Read and validate a suite definition file.

    Raises:
        ContractError: the file cannot be read, is not valid YAML, or fails the suite schema.
    """
    source = Path(path)
    try:
        text = source.read_text("utf-8")
    except OSError as exc:
        raise ContractError(
            f"{source}: cannot read ({type(exc).__name__}: {exc})", hint="check the path"
        ) from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(f"{source}: invalid YAML ({exc})", hint="fix the YAML syntax") from None
    try:
        return Suite.model_validate(data)
    except ValidationError as exc:
        raise ContractError(
            f"{source}: " + "; ".join(validation_messages(exc)),
            hint="see the suite schema: {name, entries: [...], aggregates: [...]}",
        ) from None


def list_suites() -> list[str]:
    """Names of every suite registered by an installed pack (the ``eval_suites`` data registry)."""
    return get_registry("eval_suites").keys()


def load_suite(name: str) -> Suite:
    """Load a suite by its registered name.

    Raises:
        UnknownKeyError: ``name`` is not registered.
        ContractError: its YAML file is missing or invalid.
    """
    registry = get_registry("eval_suites")
    path = registry.load(name)
    return read_suite(path)


def aggregate_suite(
    suite: Suite, results: Mapping[int, Mapping[str, float]]
) -> list[dict[str, Any]]:
    """Reduce per-entry metric values into the suite's aggregate rows.

    ``results`` maps an entry's index in ``suite.entries`` to ``{metric: value}`` for the score
    file that entry ran (typically one metric's point estimate; a caller wanting the aggregate of
    a bootstrap CI's bounds too can pass those under different metric names). Every aggregate row
    names the entries it drew from, so a suite that includes an unscored entry is visible rather
    than silently averaging over fewer files than it defined.

    Raises:
        ConfigError: an aggregate names a ``group`` no entry has, or a ``metric`` missing from
            every one of that group's results.
    """
    by_group: dict[str, list[int]] = {}
    for index, entry in enumerate(suite.entries):
        by_group.setdefault(entry.group, []).append(index)
    rows: list[dict[str, Any]] = []
    for agg in suite.aggregates:
        indices = by_group.get(agg.group)
        if not indices:
            raise ConfigError(
                f"suite {suite.name!r}: aggregate group {agg.group!r} has no entries"
                f"{did_you_mean(agg.group, by_group)}",
                hint="groups: " + ", ".join(sorted(by_group)),
            )
        have = [i for i in indices if agg.metric in results.get(i, {})]
        if not have:
            raise ConfigError(
                f"suite {suite.name!r}: no result for metric {agg.metric!r} in group {agg.group!r}",
                hint="score every entry of the suite before aggregating it",
            )
        values = [results[i][agg.metric] for i in have]
        value = sum(values) / len(values)  # "mean" is the only "how" today
        rows.append(
            {
                "group": agg.group,
                "metric": agg.metric,
                "how": agg.how,
                "value": value,
                "n_entries": len(have),
                "n_expected": len(indices),
            }
        )
    return rows
