"""``dfwb score``: run a detector over a protocol split, or every entry of a suite, and write C5
score files, reusing a cached file unless ``--force``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from dfwb.cli._output import emit_json, json_option, table
from dfwb.core.errors import ConfigError

if TYPE_CHECKING:
    from dfwb.score.harness import ScoreResult


def _parse_where(pairs: tuple[str, ...]) -> dict[str, Any]:
    """``("compression=c23", "identity=000", "identity=002")`` -> ``{"compression": "c23",
    "identity": ["000", "002"]}``: repeating a key collects its values as a list, meaning any of
    them (the same ``where`` semantics :func:`dfwb.preprocess.face.runner.run` uses)."""
    where: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(
                f"--where {pair!r} is not key=value",
                hint="use --where key=value, e.g. "
                "--where compression=c23 (repeat --where to give a key more than one value)",
            )
        if key in where:
            existing = where[key]
            where[key] = [*existing, value] if isinstance(existing, list) else [existing, value]
        else:
            where[key] = value
    return where


@dataclass(frozen=True, slots=True)
class _Entry:
    """One thing to score: an explicit protocol/split/where, or one entry of a ``--suite``."""

    protocol: str
    split: str
    where: dict[str, Any] | None
    group: str | None


def _entries(
    protocol_ref: str | None, split: str | None, where: dict[str, Any], suite_name: str | None
) -> list[_Entry]:
    if suite_name is not None:
        if protocol_ref is not None or split is not None or where:
            raise click.UsageError("--suite cannot be combined with --protocol/--split/--where")
        from dfwb.eval.suites import load_suite

        suite = load_suite(suite_name)
        return [
            _Entry(entry.protocol, entry.split, entry.where or None, entry.group)
            for entry in suite.entries
        ]
    if protocol_ref is None or split is None:
        raise click.UsageError("give --protocol and --split, or --suite")
    return [_Entry(protocol_ref, split, where or None, None)]


def _row(entry: _Entry, result: ScoreResult) -> dict[str, Any]:
    return {
        "protocol": entry.protocol,
        "split": entry.split,
        "where": entry.where or {},
        "group": entry.group,
        "csv": str(result.csv_path),
        "meta": str(result.meta_path),
        "coverage": result.coverage,
        "cached": result.cached,
        "frames": str(result.frames_path) if result.frames_path is not None else None,
    }


@click.command("score")
@click.option(
    "--detector",
    "detector_uri",
    required=True,
    help="Detector URI: run:<dir>[#tag], py:<module>:<factory>, zoo:<name>[@<weights id>].",
)
@click.option(
    "--protocol",
    "protocol_ref",
    default=None,
    help="Protocol reference, e.g. celebdf-v2/official.",
)
@click.option("--split", default=None, help="The protocol split (needs --protocol).")
@click.option(
    "--where",
    "where_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Restrict to videos matching KEY=VALUE (repeatable; repeat a key for any-of).",
)
@click.option(
    "--suite", "suite_name", default=None, help="Score every entry of a registered eval suite."
)
@click.option(
    "--profile",
    default=None,
    help="A locally processed profile id (default: chosen automatically).",
)
@click.option("--clips-per-video", type=int, default=4, show_default=True)
@click.option(
    "--aggregate",
    default="mean-prob",
    show_default=True,
    help="mean-prob, mean-logit, max or median.",
)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--precision", default=None, help="fp16, bf16 or fp32 (default: no autocast).")
@click.option(
    "--allow-input-mismatch",
    is_flag=True,
    help="Accept a derived or otherwise mismatched input adaptation (recorded in the meta).",
)
@click.option(
    "--labels", default="binary", show_default=True, help="The pack's label mapping to use."
)
@click.option(
    "--out",
    "out_dir",
    default=None,
    type=click.Path(path_type=Path, file_okay=False),
    help="Output root (default: $DFWB_RUNS_ROOT/scores).",
)
@click.option("--force", is_flag=True, help="Recompute even if a cached file already matches.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--frames",
    is_flag=True,
    help="Also write per-clip/per-frame scores (needs the [eval] extra: pyarrow).",
)
@json_option
def score(
    detector_uri: str,
    protocol_ref: str | None,
    split: str | None,
    where_pairs: tuple[str, ...],
    suite_name: str | None,
    profile: str | None,
    clips_per_video: int,
    aggregate: str,
    batch_size: int,
    device: str,
    precision: str | None,
    allow_input_mismatch: bool,
    labels: str,
    out_dir: Path | None,
    force: bool,
    seed: int,
    frames: bool,
    as_json: bool,
) -> None:
    """Score a protocol split, or every entry of a suite, with --detector.

    Writes one C5 score file per entry (contract C5), reusing a cached file that already matches
    this request's configuration unless --force is given.
    """
    where = _parse_where(where_pairs)
    entries = _entries(protocol_ref, split, where, suite_name)

    from dfwb.score.harness import score as run_score

    rows: list[dict[str, Any]] = []
    for entry in entries:
        result = run_score(
            detector_uri,
            protocol=entry.protocol,
            split=entry.split,
            where=entry.where,
            profile=profile,
            clips_per_video=clips_per_video,
            aggregate=aggregate,
            batch_size=batch_size,
            device=device,
            precision=precision,
            allow_input_mismatch=allow_input_mismatch,
            labels=labels,
            out=out_dir,
            force=force,
            seed=seed,
            frames=frames,
        )
        rows.append(_row(entry, result))

    if as_json:
        emit_json({"results": rows})
        return
    headers = ["protocol", "split", "group", "csv", "coverage", "cached", "frames"]
    data = [
        [
            r["protocol"],
            r["split"],
            r["group"] or "",
            r["csv"],
            f"ok={r['coverage']['ok']}/{r['coverage']['expected']}",
            r["cached"],
            r["frames"] or "",
        ]
        for r in rows
    ]
    click.echo(table(headers, data))
