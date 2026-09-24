"""``evaluate()``: coverage-aware metrics with bootstrap CIs, breakdowns, seeds and suites.

This assembles the other modules of this package into the one entry point ``dfwb eval`` needs: read
every score file, apply the ``--missing`` policy, bootstrap every metric, and, if asked, break the
numbers down by a row attribute, fold multiple seeds of "the same run" into one mean +/- sd, and
roll a suite's entries into its aggregate rows.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dfwb.core.errors import ConfigError
from dfwb.core.records import ScoreFile, ScoreMeta, ScoreRow, read_scores
from dfwb.eval.bootstrap import bootstrap_ci, summarize_seeds
from dfwb.eval.breakdown import group_rows
from dfwb.eval.coverage import Coverage, FloatArray, IntArray, coverage_of, labels_and_scores
from dfwb.eval.metrics import MetricUndefined
from dfwb.eval.suites import Suite, aggregate_suite, load_suite

__all__ = ["EvalResult", "evaluate"]

_Identity = tuple[str, str, str, tuple[tuple[str, Any], ...]]


@dataclass(frozen=True)
class EvalResult:
    """The result of :func:`evaluate`: every table it produced, and the coverage exit code.

    ``tables`` always has a ``"files"`` entry (one row per input file); ``"breakdown"``,
    ``"seeds"`` and ``"suite"`` are present only when ``by``, multi-seed input or ``suite``
    respectively produced one. Every row is a plain, JSON-friendly ``dict``.
    """

    metrics: tuple[str, ...]
    by: str | None
    missing: str
    min_coverage: float
    bootstrap: int
    seed: int
    tables: dict[str, list[dict[str, Any]]]
    exit_code: int

    def to_json(self) -> dict[str, Any]:
        """The full result as one JSON-friendly ``dict`` (full precision, no rounding)."""
        return {
            "metrics": list(self.metrics),
            "by": self.by,
            "missing": self.missing,
            "min_coverage": self.min_coverage,
            "bootstrap": self.bootstrap,
            "seed": self.seed,
            "exit_code": self.exit_code,
            "tables": self.tables,
        }


def _identity(meta: ScoreMeta) -> _Identity:
    """Everything about a run that must match for two files to be "the same run, another seed"."""
    return (
        meta.detector.source,
        meta.protocol.id,
        meta.protocol.split,
        tuple(sorted(meta.protocol.where.items())),
    )


def _metric_table(
    metrics: Sequence[str], values: dict[str, float], lo: dict[str, float], hi: dict[str, float]
) -> dict[str, dict[str, float]]:
    return {m: {"value": values[m], "ci_lo": lo[m], "ci_hi": hi[m]} for m in values}


def _score_metrics(
    metrics: Sequence[str],
    y: IntArray,
    p: FloatArray,
    *,
    n_boot: int,
    seed: int,
    skip_undefined: bool,
) -> dict[str, dict[str, float]]:
    values: dict[str, float] = {}
    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    for metric in metrics:
        try:
            result = bootstrap_ci(metric, y, p, n_boot=n_boot, seed=seed)
        except MetricUndefined:
            if skip_undefined:
                continue
            raise
        values[metric] = result.point
        lo[metric] = result.lo
        hi[metric] = result.hi
    return _metric_table(metrics, values, lo, hi)


def _file_row(
    score_file: ScoreFile, coverage: Coverage, n: int, metric_table: dict[str, Any]
) -> dict[str, Any]:
    return {
        "file": score_file.path.name,
        "n": n,
        **coverage.as_dict(),
        "metrics": metric_table,
    }


def _eval_rows_for_group(
    by: str, group_rows_: list[ScoreRow], all_rows: Sequence[ScoreRow]
) -> list[ScoreRow]:
    """The rows one breakdown group is actually evaluated against.

    ``compression`` stays a plain partition: a compression level's own rows already span both
    classes. For a fake-side dimension (``method``, ``family``, ``label_key``), a group that is
    entirely fake (label 1) is evaluated against every real (label 0) row of the same dataset(s)
    too -- the standard per-method-AUC convention (this method's fakes against the *whole* pool of
    reals, not against no reals at all). A group that is not entirely fake (the real group itself,
    or, unusually, a mixed one) is left as its own plain partition.
    """
    if by == "compression":
        return group_rows_
    fakes = [r for r in group_rows_ if r.label == 1]
    if not fakes or len(fakes) != len(group_rows_):
        return group_rows_
    datasets = {r.dataset for r in fakes}
    reals = [r for r in all_rows if r.label == 0 and r.dataset in datasets]
    return fakes + reals


def _breakdown_table(
    files: Sequence[ScoreFile],
    by: str,
    metrics: Sequence[str],
    *,
    missing: str,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score_file in files:
        groups = group_rows(score_file.rows, by, meta=score_file.meta)
        for group_name, group_rows_ in sorted(groups.items()):
            eval_rows = _eval_rows_for_group(by, group_rows_, score_file.rows)
            coverage = coverage_of(eval_rows)
            y, p, kept = labels_and_scores(eval_rows, missing=missing)
            metric_table = _score_metrics(
                metrics, y, p, n_boot=n_boot, seed=seed, skip_undefined=True
            )
            rows.append(
                {
                    "file": score_file.path.name,
                    "group": group_name,
                    "n": len(kept),
                    **coverage.as_dict(),
                    "metrics": metric_table,
                }
            )
    return rows


def _seed_table(
    files: Sequence[ScoreFile], metrics: Sequence[str], points: list[dict[str, float]]
) -> list[dict[str, Any]]:
    groups: dict[_Identity, dict[int, int]] = {}  # identity -> {seed: file index}
    for index, score_file in enumerate(files):
        meta = score_file.meta
        if meta is None or meta.seed is None:
            continue
        groups.setdefault(_identity(meta), {})[meta.seed] = index
    rows: list[dict[str, Any]] = []
    for identity, by_seed in groups.items():
        if len(by_seed) < 2:
            continue
        for metric in metrics:
            per_seed = {s: points[i][metric] for s, i in by_seed.items() if metric in points[i]}
            if len(per_seed) < 2:
                continue
            summary = summarize_seeds(metric, per_seed)
            rows.append(
                {
                    "detector": identity[0],
                    "protocol": identity[1],
                    "split": identity[2],
                    "metric": metric,
                    "seeds": list(summary.seeds),
                    "values": list(summary.values),
                    "mean": summary.mean,
                    "sd": summary.sd,
                }
            )
    return rows


def _suite_table(
    suite_arg: Suite | str, files: Sequence[ScoreFile], points: list[dict[str, float]]
) -> list[dict[str, Any]]:
    suite = load_suite(suite_arg) if isinstance(suite_arg, str) else suite_arg
    entry_results: dict[int, dict[str, float]] = {}
    for entry_index, entry in enumerate(suite.entries):
        for file_index, score_file in enumerate(files):
            meta = score_file.meta
            if (
                meta is not None
                and meta.protocol.id == entry.protocol
                and meta.protocol.split == entry.split
                and meta.protocol.where == entry.where
            ):
                entry_results[entry_index] = points[file_index]
                break
    return aggregate_suite(suite, entry_results)


def evaluate(
    files: Sequence[str | os.PathLike[str]],
    *,
    metrics: Sequence[str],
    by: str | None = None,
    suite: Suite | str | None = None,
    bootstrap: int = 2000,
    seed: int = 0,
    missing: str = "exclude",
    min_coverage: float = 0.99,
) -> EvalResult:
    """Coverage-aware metrics, with bootstrap CIs, over one or more C5 score files.

    Every file gets its own row in ``tables["files"]``: its coverage (``expected``/``ok``/
    ``missing``/``error``, always over every row, unaffected by ``missing``), ``n`` (how many rows
    were actually fed to the metric once ``missing`` was applied -- equal to ``ok`` for
    ``"exclude"``, to every row for the other three policies), and every metric's point estimate
    with its stratified-bootstrap CI. ``exit_code`` is ``3`` (never raised as an exception -- the
    tables are still built and returned) when any file's coverage is below ``min_coverage``.

    ``by`` additionally breaks each file down by ``"method"``, ``"compression"``, ``"label_key"``
    or ``"family"`` (see :func:`~dfwb.eval.breakdown.group_rows`). For the three fake-side
    dimensions, a group that is entirely fake is evaluated against every real row of the same
    dataset(s) too (the usual per-method-AUC convention), so its ``n``/coverage reflect that
    combined set, not just the group's own rows; ``"compression"`` stays a plain partition. A
    metric still undefined for a group (most often a cross-class metric on the real group itself,
    which nothing is added to) is left out of that group's row rather than failing the whole call.

    Several files that agree on everything but ``meta.seed`` get an extra ``tables["seeds"]`` row
    per metric: the per-seed values plus their mean and sample standard deviation.

    ``suite`` (a loaded :class:`~dfwb.eval.suites.Suite`, or a name registered in the
    ``eval_suites`` data registry) matches each of its entries to the input file whose meta
    protocol/split/where agree, and adds a ``tables["suite"]`` row per aggregate.

    Raises:
        ConfigError: ``files`` is empty.
        ContractError: a file cannot be read, or ``by="family"`` is requested for a file with no
            ``.meta.json``.
        MetricUndefined: a top-level metric (not a breakdown group) is undefined for a file.
    """
    if not files:
        raise ConfigError("evaluate needs at least one score file", hint="pass 1 or more files")
    loaded = [read_scores(f) for f in files]

    coverages = [coverage_of(sf.rows) for sf in loaded]
    points: list[dict[str, float]] = []
    file_rows: list[dict[str, Any]] = []
    for score_file, coverage in zip(loaded, coverages, strict=True):
        y, p, kept = labels_and_scores(score_file.rows, missing=missing)
        metric_table = _score_metrics(
            metrics, y, p, n_boot=bootstrap, seed=seed, skip_undefined=False
        )
        points.append({m: v["value"] for m, v in metric_table.items()})
        file_rows.append(_file_row(score_file, coverage, len(kept), metric_table))

    tables: dict[str, list[dict[str, Any]]] = {"files": file_rows}

    if by is not None:
        tables["breakdown"] = _breakdown_table(
            loaded, by, metrics, missing=missing, n_boot=bootstrap, seed=seed
        )

    seed_rows = _seed_table(loaded, metrics, points)
    if seed_rows:
        tables["seeds"] = seed_rows

    if suite is not None:
        tables["suite"] = _suite_table(suite, loaded, points)

    exit_code = 3 if any(not c.meets(min_coverage) for c in coverages) else 0
    return EvalResult(
        metrics=tuple(metrics),
        by=by,
        missing=missing,
        min_coverage=min_coverage,
        bootstrap=bootstrap,
        seed=seed,
        tables=tables,
        exit_code=exit_code,
    )
