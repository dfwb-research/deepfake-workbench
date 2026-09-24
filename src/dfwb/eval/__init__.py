"""Metrics, aggregation, uncertainty, suites and reports over score files (contract C5).

Attributes are imported on first use, so ``import dfwb.eval`` never imports numpy until a name is
actually looked up.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dfwb.eval.aggregate import aggregate
    from dfwb.eval.bootstrap import BootstrapResult, SeedSummary, bootstrap_ci, summarize_seeds
    from dfwb.eval.breakdown import BY_FIELDS, group_rows
    from dfwb.eval.compare import (
        CompareResult,
        PairComparison,
        compare,
        delong_test,
        holm_correction,
    )
    from dfwb.eval.coverage import MISSING_POLICIES, Coverage, coverage_of, labels_and_scores
    from dfwb.eval.metrics import Metric, MetricUndefined, compute, parse_metric_spec
    from dfwb.eval.report import EvalResult, evaluate
    from dfwb.eval.suites import (
        Suite,
        SuiteAggregate,
        SuiteEntry,
        aggregate_suite,
        list_suites,
        load_suite,
        read_suite,
    )

__all__ = [
    "BY_FIELDS",
    "MISSING_POLICIES",
    "BootstrapResult",
    "CompareResult",
    "Coverage",
    "EvalResult",
    "Metric",
    "MetricUndefined",
    "PairComparison",
    "SeedSummary",
    "Suite",
    "SuiteAggregate",
    "SuiteEntry",
    "aggregate",
    "aggregate_suite",
    "bootstrap_ci",
    "compare",
    "compute",
    "coverage_of",
    "delong_test",
    "evaluate",
    "group_rows",
    "holm_correction",
    "labels_and_scores",
    "list_suites",
    "load_suite",
    "parse_metric_spec",
    "read_suite",
    "summarize_seeds",
]

_LAZY = {
    "aggregate": "dfwb.eval.aggregate",
    "BootstrapResult": "dfwb.eval.bootstrap",
    "SeedSummary": "dfwb.eval.bootstrap",
    "bootstrap_ci": "dfwb.eval.bootstrap",
    "summarize_seeds": "dfwb.eval.bootstrap",
    "BY_FIELDS": "dfwb.eval.breakdown",
    "group_rows": "dfwb.eval.breakdown",
    "CompareResult": "dfwb.eval.compare",
    "PairComparison": "dfwb.eval.compare",
    "compare": "dfwb.eval.compare",
    "delong_test": "dfwb.eval.compare",
    "holm_correction": "dfwb.eval.compare",
    "MISSING_POLICIES": "dfwb.eval.coverage",
    "Coverage": "dfwb.eval.coverage",
    "coverage_of": "dfwb.eval.coverage",
    "labels_and_scores": "dfwb.eval.coverage",
    "Metric": "dfwb.eval.metrics",
    "MetricUndefined": "dfwb.eval.metrics",
    "compute": "dfwb.eval.metrics",
    "parse_metric_spec": "dfwb.eval.metrics",
    "EvalResult": "dfwb.eval.report",
    "evaluate": "dfwb.eval.report",
    "Suite": "dfwb.eval.suites",
    "SuiteAggregate": "dfwb.eval.suites",
    "SuiteEntry": "dfwb.eval.suites",
    "aggregate_suite": "dfwb.eval.suites",
    "list_suites": "dfwb.eval.suites",
    "load_suite": "dfwb.eval.suites",
    "read_suite": "dfwb.eval.suites",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'dfwb.eval' has no attribute {name!r}")
