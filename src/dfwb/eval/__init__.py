"""Metrics, aggregation, uncertainty, suites and reports over score files (contract C5).

Attributes are imported on first use, so ``import dfwb.eval`` never imports numpy until a name is
actually looked up.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dfwb.eval.aggregate import aggregate
    from dfwb.eval.metrics import Metric, MetricUndefined, compute, parse_metric_spec

__all__ = [
    "Metric",
    "MetricUndefined",
    "aggregate",
    "compute",
    "parse_metric_spec",
]

_LAZY = {
    "aggregate": "dfwb.eval.aggregate",
    "Metric": "dfwb.eval.metrics",
    "MetricUndefined": "dfwb.eval.metrics",
    "compute": "dfwb.eval.metrics",
    "parse_metric_spec": "dfwb.eval.metrics",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'dfwb.eval' has no attribute {name!r}")
