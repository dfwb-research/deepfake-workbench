"""``dfwb eval``: turn C5 score files into metrics, comparisons, calibrations and imports.

``dfwb eval <files>`` (no subcommand name) runs the default evaluation; ``compare``, ``calibrate``
and ``import`` are explicit subcommands. This module implements that "runs by default" dispatch
itself (see :class:`_EvalGroup`) because click groups otherwise only ever run a named subcommand.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click

from dfwb.cli._output import emit_json, json_option, table
from dfwb.core.errors import ConfigError, InstallationError
from dfwb.core.hashing import sha256_file

_EXTENSIONS = {"md": "md", "csv": "csv", "latex": "tex"}
_FORMATS = ("md", "csv", "latex", "json")

#: One report table, rendered: its title, column headers, and the already-formatted cell rows.
Block = tuple[str, list[str], list[list[str]]]


class _EvalGroup(click.Group):
    """Routes anything that is not a known subcommand name to the hidden ``run`` command.

    ``dfwb eval a.scores.csv --metrics auc`` and ``dfwb eval --metrics auc a.scores.csv`` both
    become ``dfwb eval run a.scores.csv --metrics auc``; ``dfwb eval compare ...`` and the other
    real subcommands, and ``-h``/``--help``, are left alone.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in (*self.commands, "-h", "--help"):
            args = ["run", *args]
        return super().parse_args(ctx, args)


@click.group(cls=_EvalGroup)
def eval() -> None:
    """Evaluate, compare, calibrate and import score files (contract C5)."""


# --------------------------------------------------------------------------- shared helpers


def _expand_files(paths: Sequence[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(path.glob("*.scores.csv"))
            if not found:
                raise ConfigError(
                    f"{path}: no *.scores.csv files found", hint="check the directory"
                )
            expanded.extend(found)
        else:
            expanded.append(path)
    return expanded


def _metric_columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for name in row.get("metrics", {}):
            if name not in names:
                names.append(name)
    return names


def _metric_cells(row: dict[str, Any], metrics: Sequence[str], *, latex: bool) -> list[str]:
    cells: list[str] = []
    for metric in metrics:
        info = row.get("metrics", {}).get(metric)
        if info is None:
            cells.append("")
            continue
        value, lo, hi = info["value"], info["ci_lo"], info["ci_hi"]
        if value is None:  # undefined on this file's rows; the reason is in the JSON
            cells.append("undefined")
            continue
        if lo is None or hi is None:  # --bootstrap 0: a value, but no confidence interval
            cells.append(f"${value:.4f}$" if latex else f"{value:.4f}")
        elif latex:
            half_width = (hi - lo) / 2
            cells.append(f"${value:.4f} \\pm {half_width:.4f}$")
        else:
            cells.append(f"{value:.4f} [{lo:.4f}, {hi:.4f}]")
    return cells


def _files_block(rows: Sequence[dict[str, Any]], metrics: Sequence[str], *, latex: bool) -> Block:
    headers = ["file", "n", "coverage", *metrics]
    data = [
        [
            row["file"],
            row["n"],
            f"{row['coverage']:.4f}",
            *_metric_cells(row, metrics, latex=latex),
        ]
        for row in rows
    ]
    return "files", headers, data


def _breakdown_block(
    rows: Sequence[dict[str, Any]], metrics: Sequence[str], *, latex: bool
) -> Block:
    headers = ["file", "group", "n", "coverage", *metrics]
    data = [
        [
            row["file"],
            row["group"],
            row["n"],
            f"{row['coverage']:.4f}",
            *_metric_cells(row, metrics, latex=latex),
        ]
        for row in rows
    ]
    return "breakdown", headers, data


def _seeds_block(rows: Sequence[dict[str, Any]]) -> Block:
    headers = ["detector", "protocol", "split", "metric", "seeds", "mean", "sd"]
    data = [
        [
            row["detector"],
            row["protocol"],
            row["split"],
            row["metric"],
            ",".join(map(str, row["seeds"])),
            f"{row['mean']:.4f}",
            f"{row['sd']:.4f}",
        ]
        for row in rows
    ]
    return "seeds", headers, data


def _suite_block(rows: Sequence[dict[str, Any]]) -> Block:
    headers = ["group", "metric", "how", "value", "n_entries", "n_expected"]
    data = [
        [
            row["group"],
            row["metric"],
            row["how"],
            "undefined" if row["value"] is None else f"{row['value']:.4f}",
            row["n_entries"],
            row["n_expected"],
        ]
        for row in rows
    ]
    return "suite", headers, data


def _blocks(
    tables: dict[str, list[dict[str, Any]]], metrics: Sequence[str], *, latex: bool
) -> list[Block]:
    blocks = [_files_block(tables["files"], metrics, latex=latex)]
    if tables.get("breakdown"):
        blocks.append(_breakdown_block(tables["breakdown"], metrics, latex=latex))
    if tables.get("seeds"):
        blocks.append(_seeds_block(tables["seeds"]))
    if tables.get("suite"):
        blocks.append(_suite_block(tables["suite"]))
    return blocks


def _render_md(tables: dict[str, list[dict[str, Any]]], metrics: Sequence[str]) -> str:
    sections = []
    for title, headers, rows in _blocks(tables, metrics, latex=False):
        head = "| " + " | ".join(headers) + " |"
        sep = "| " + " | ".join("---" for _ in headers) + " |"
        body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
        sections.append(f"## {title}\n\n{head}\n{sep}\n{body}")
    return "\n\n".join(sections) + "\n"


def _render_csv(tables: dict[str, list[dict[str, Any]]], metrics: Sequence[str]) -> str:
    buffer = io.StringIO()
    for title, headers, rows in _blocks(tables, metrics, latex=False):
        buffer.write(f"# {title}\n")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)
        buffer.write("\n")
    return buffer.getvalue()


def _render_latex(tables: dict[str, list[dict[str, Any]]], metrics: Sequence[str]) -> str:
    sections = []
    for title, headers, rows in _blocks(tables, metrics, latex=True):
        ncol = len(headers)
        lines = [
            f"% {title}",
            f"\\begin{{tabular}}{{{'l' * ncol}}}",
            "\\toprule",
            " & ".join(headers) + r" \\",
            "\\midrule",
            *(" & ".join(str(c) for c in row) + r" \\" for row in rows),
            "\\bottomrule",
            "\\end{tabular}",
        ]
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def _render(fmt: str, tables: dict[str, list[dict[str, Any]]], metrics: Sequence[str]) -> str:
    if fmt == "csv":
        return _render_csv(tables, metrics)
    if fmt == "latex":
        return _render_latex(tables, metrics)
    return _render_md(tables, metrics)


def _write_plots_for(files: Sequence[Path], out_dir: Path, missing: str) -> None:
    from dfwb.core.records import read_scores
    from dfwb.eval.coverage import labels_and_scores
    from dfwb.eval.plots import write_plots

    plots_dir = out_dir / "plots"
    for path in files:
        score_file = read_scores(path)
        y, p, _kept = labels_and_scores(score_file.rows, missing=missing)
        if p.size == 0:
            continue
        prefix = path.name.removesuffix(".scores.csv")
        try:
            write_plots(y, p, plots_dir, prefix)
        except InstallationError as exc:
            click.echo(f"note: {exc.message} ({exc.hint}); skipping plots", err=True)
            return


# --------------------------------------------------------------------------- run (the default)


@eval.command("run", hidden=True)
@click.argument("files", nargs=-1, required=True, type=click.Path(path_type=Path, exists=True))
@click.option(
    "--metrics", default="auc,eer", show_default=True, help="Comma-separated metric specs."
)
@click.option(
    "--by", "by_field", default=None, help="Break down by method, compression, label_key or family."
)
@click.option("--suite", "suite_name", default=None, help="Name of a registered eval suite.")
@click.option("--bootstrap", default=2000, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--missing",
    default="exclude",
    show_default=True,
    help="exclude, as-real, as-fake or as-chance.",
)
@click.option("--min-coverage", default=0.99, show_default=True, type=float)
@click.option("--out", "out_dir", default=None, type=click.Path(path_type=Path, file_okay=False))
@click.option("--format", "fmt", default="md", type=click.Choice(_FORMATS), show_default=True)
@json_option
def run(
    files: tuple[Path, ...],
    metrics: str,
    by_field: str | None,
    suite_name: str | None,
    bootstrap: int,
    seed: int,
    missing: str,
    min_coverage: float,
    out_dir: Path | None,
    fmt: str,
    as_json: bool,
) -> int:
    """Evaluate one or more C5 score files (this is what bare ``dfwb eval FILES...`` runs)."""
    from dfwb.eval.report import evaluate

    expanded = _expand_files(list(files))
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    result = evaluate(
        expanded,
        metrics=metric_list,
        by=by_field,
        suite=suite_name,
        bootstrap=bootstrap,
        seed=seed,
        missing=missing,
        min_coverage=min_coverage,
    )
    effective_fmt = "json" if as_json else fmt
    payload = result.to_json()
    payload["input_files"] = {str(f): {"sha256": sha256_file(f)} for f in expanded}

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_text = json.dumps(payload, indent=2) + "\n"
        (out_dir / "metrics.json").write_text(metrics_text, encoding="utf-8")
        if effective_fmt != "json":
            report_text = _render(effective_fmt, result.tables, result.metrics)
            ext = _EXTENSIONS[effective_fmt]
            (out_dir / f"report.{ext}").write_text(report_text, encoding="utf-8")
        _write_plots_for(expanded, out_dir, missing)

    if effective_fmt == "json":
        click.echo(json.dumps(payload, indent=2))
    elif effective_fmt == "md":
        # a quick, aligned table for the terminal; the same data as report.md, plainer syntax.
        headers = ["FILE", "N", "COVERAGE", *[m.upper() for m in result.metrics]]
        rows = [
            [
                r["file"],
                r["n"],
                f"{r['coverage']:.4f}",
                *_metric_cells(r, result.metrics, latex=False),
            ]
            for r in result.tables["files"]
        ]
        click.echo(table(headers, rows))
    else:
        click.echo(_render(effective_fmt, result.tables, result.metrics))
    return result.exit_code


# --------------------------------------------------------------------------- compare


@eval.command("compare")
@click.argument(
    "files", nargs=-1, required=True, type=click.Path(path_type=Path, exists=True, dir_okay=False)
)
@click.option("--metrics", default="auc", show_default=True, help="Comma-separated metric specs.")
@click.option("--bootstrap", default=2000, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@json_option
def compare(files: tuple[Path, ...], metrics: str, bootstrap: int, seed: int, as_json: bool) -> int:
    """Paired comparison of two or more C5 score files, on the intersection of their ok rows."""
    from dfwb.eval.compare import compare as run_compare

    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    result = run_compare(files, metrics=metric_list, bootstrap=bootstrap, seed=seed)
    if as_json:
        emit_json(result.to_json())
        return 0
    click.echo(f"holm correction applied: {result.holm_applied}")
    for comparison in result.comparisons:
        click.echo(
            f"{comparison.a} vs {comparison.b} (n={comparison.n}; only in {comparison.a}: "
            f"{comparison.only_a}, only in {comparison.b}: {comparison.only_b})"
        )
        for metric, values in comparison.metrics.items():
            click.echo(f"  {metric}: {values}")
    return 0


# --------------------------------------------------------------------------- calibrate


def _calibrated_name(apply_path: Path, method: str) -> str:
    stem = apply_path.name.removesuffix(".scores.csv")
    return f"{stem}.{method}.scores.csv"


@eval.command("calibrate")
@click.option(
    "--fit",
    "fit_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--apply",
    "apply_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--method", required=True, help="temperature, platt or isotonic.")
@click.option("--out", "out_path", default=None, type=click.Path(path_type=Path))
@json_option
def calibrate(
    fit_path: Path, apply_path: Path, method: str, out_path: Path | None, as_json: bool
) -> int:
    """Fit a calibration on ``--fit`` and apply it to ``--apply``, writing a new C5 file."""
    from dfwb.core.records import write_scores
    from dfwb.eval.calibrate import CALIBRATION_METHODS, calibrate_file

    if method not in CALIBRATION_METHODS:
        raise ConfigError(
            f"unknown --method {method!r}", hint="methods: " + ", ".join(CALIBRATION_METHODS)
        )
    result = calibrate_file(fit_path, apply_path, method)
    target = out_path or apply_path.with_name(_calibrated_name(apply_path, method))
    csv_path, meta_path = write_scores(target, result.rows, result.meta)
    if as_json:
        emit_json(
            {
                "csv": str(csv_path),
                "meta": str(meta_path),
                "calibration": {
                    "method": result.calibration.method,
                    "params": result.calibration.params,
                },
            }
        )
        return 0
    click.echo(f"wrote {csv_path}")
    click.echo(f"calibration: {method} {result.calibration.params}")
    return 0


# --------------------------------------------------------------------------- import


def _default_import_name(protocol_ref: str, split: str) -> str:
    return protocol_ref.replace("/", "-") + f"-{split}.scores.csv"


@eval.command("import")
@click.argument("file", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option(
    "--protocol",
    "protocol_ref",
    required=True,
    help="Protocol reference, e.g. celebdf-v2/official.",
)
@click.option("--split", required=True)
@click.option(
    "--map",
    "map_spec",
    required=True,
    help="key=<col or template>,score=<col>[,label=<col>][,compression=<col>]",
)
@click.option("--polarity", default="fake-high", show_default=True, help="fake-high or real-high.")
@click.option(
    "--status-col", "status_col", default=None, help="Column already carrying ok/missing/error."
)
@click.option("--delimiter", default=",", show_default=True)
@click.option(
    "--labels", default="binary", show_default=True, help="The pack's label mapping to use."
)
@click.option("--out", "out_dir", required=True, type=click.Path(path_type=Path, file_okay=False))
@json_option
def import_(
    file: Path,
    protocol_ref: str,
    split: str,
    map_spec: str,
    polarity: str,
    status_col: str | None,
    delimiter: str,
    labels: str,
    out_dir: Path,
    as_json: bool,
) -> int:
    """Import a foreign score CSV against a protocol split, writing a new C5 file."""
    from dfwb.core.records import write_scores
    from dfwb.eval.importer import import_scores, parse_map

    mapping = parse_map(map_spec)
    result = import_scores(
        file,
        protocol=protocol_ref,
        split=split,
        key=mapping["key"],
        score=mapping["score"],
        label=mapping.get("label"),
        compression=mapping.get("compression"),
        polarity=polarity,
        status_col=status_col,
        delimiter=delimiter,
        labels=labels,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / _default_import_name(protocol_ref, split)
    csv_path, meta_path = write_scores(target, result.rows, result.meta)
    coverage = result.meta.coverage.model_dump()
    if as_json:
        emit_json({"csv": str(csv_path), "meta": str(meta_path), "coverage": coverage})
        return 0
    click.echo(f"wrote {csv_path}")
    click.echo(f"coverage: {coverage}")
    return 0
