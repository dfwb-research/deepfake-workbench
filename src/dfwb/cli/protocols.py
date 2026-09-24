"""``dfwb protocols``: list, inspect, verify, lint and diff protocol packs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import click

from dfwb.cli._output import emit_json, json_option, table


@click.group()
def protocols() -> None:
    """List and inspect installed protocol packs and their schemes."""


def _info_row(info: Any) -> dict[str, Any]:
    return {
        "dataset_id": info.dataset_id,
        "scheme": info.scheme,
        "pack": info.pack,
        "version": info.version,
        "kind": info.kind,
        "default": info.default,
        "counts": info.counts,
        "broken": info.broken,
    }


@protocols.command("list")
@json_option
def list_(as_json: bool) -> None:
    """List every scheme of every dataset in every installed pack (``*`` marks the default)."""
    from dfwb.protocols.protocol import list_protocols

    rows = list_protocols()
    if as_json:
        emit_json([_info_row(r) for r in rows])
        return

    healthy = [r for r in rows if r.broken is None]
    broken = [r for r in rows if r.broken is not None]
    headers = ["PROTOCOL", "PACK", "VERSION", "KIND", "TRAIN", "VAL", "TEST"]
    table_rows = []
    for r in healthy:
        name = f"{r.dataset_id}/{r.scheme}" + ("*" if r.default else "")
        counts = r.counts or {}
        train, val, test = (counts.get(s, "-") for s in ("train", "val", "test"))
        table_rows.append([name, r.pack, r.version, r.kind, train, val, test])
    for r in broken:
        table_rows.append([f"({r.pack})", r.pack, "-", "BROKEN", "-", "-", "-"])
    if table_rows:
        click.echo(table(headers, table_rows))
    else:
        click.echo("no protocol packs installed")
    for r in broken:
        click.echo(f"warning: pack {r.pack!r} is broken: {r.broken}", err=True)


@protocols.command("info")
@click.argument("ref")
@json_option
def info(ref: str, as_json: bool) -> None:
    """Show a protocol's card summary, scheme card and counts per split x compression x label."""
    from dfwb.protocols.protocol import load

    protocol = load(ref)
    split_by_key = {(row.key, row.compression): row.split for row in protocol.split_rows()}
    tallies: Counter[tuple[str, str | None, str]] = Counter()
    for video in protocol.records():
        split = split_by_key[(video.key, video.compression)]
        tallies[(split, video.compression, video.label_key)] += 1
    counts = [
        {"split": split, "compression": compression, "label_key": label_key, "count": count}
        for (split, compression, label_key), count in sorted(
            tallies.items(), key=lambda item: (item[0][0], item[0][1] or "", item[0][2])
        )
    ]

    data = {
        "ref": protocol.ref,
        "pack": protocol.pack.name,
        "pack_version": protocol.pack_version,
        "sha256": protocol.sha256,
        "card": protocol.card.model_dump(mode="json"),
        "scheme_card": protocol.scheme_card.model_dump(mode="json"),
        "counts": counts,
    }
    if as_json:
        emit_json(data)
        return

    click.echo(f"{protocol.ref}  (pack {protocol.pack.name} {protocol.pack_version})")
    click.echo(f"sha256: {protocol.sha256}")
    click.echo(f"dataset: {protocol.card.name}  release {protocol.card.release}")
    click.echo(f"scheme: {protocol.scheme}  kind={protocol.scheme_card.kind}")
    click.echo("")
    rows = [[c["split"], c["compression"] or "-", c["label_key"], c["count"]] for c in counts]
    click.echo(table(["SPLIT", "COMPRESSION", "LABEL", "COUNT"], rows))


@protocols.command("verify")
@click.argument("ref")
@click.option(
    "--inventory",
    "inventory",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Inventory file to check (default: <work root>/<dataset>/inventory.jsonl).",
)
@click.option(
    "--split",
    "splits",
    multiple=True,
    help="A split to require full coverage of (repeatable); default: every split this scheme "
    "assigns, except exclude.",
)
@json_option
def verify(ref: str, inventory: Path | None, splits: tuple[str, ...], as_json: bool) -> int:
    """Join the local inventory to a protocol's pack videos and report coverage.

    Exits 0 on full coverage, 3 if any requested split is short a video, 4 if any video is
    relabelled. A missing inventory is a usage error (exit 2).
    """
    from dfwb.core.paths import require_root, resolve_roots
    from dfwb.protocols.verify import verify as run_verify
    from dfwb.protocols.verify import write_report

    work_root = require_root("work", resolve_roots())
    report = run_verify(ref, inventory=inventory, splits=splits or None, work_root=work_root)
    report_path = write_report(report, work_root)

    if as_json:
        emit_json({**report.to_json(), "report_path": str(report_path)})
        return report.exit_code

    click.echo(f"{report.dataset}/{report.scheme}  (pack {report.pack} {report.pack_version})")
    click.echo("requested splits: " + ", ".join(report.requested_splits))
    for bucket, count in report.counts.items():
        click.echo(f"{bucket}: {count}")
        for sample in report.samples.get(bucket, []):
            click.echo(f"  {sample}")
    for warning in report.warnings:
        click.echo(f"warning: {warning}")
    click.echo(f"report written to {report_path}")
    return report.exit_code


def _lint_issue_row(issue: Any) -> dict[str, Any]:
    return {"severity": issue.severity, "where": issue.where, "message": issue.message}


@protocols.command("lint")
@click.argument("pack", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option(
    "--release", is_flag=True, help="Treat an undecided dataset distribution as an error."
)
@json_option
def lint(pack: Path, release: bool, as_json: bool) -> int:
    """Check a protocol pack directory for problems a pack author needs to fix.

    Re-derives every fact a pack's cards claim about themselves (scheme hashes and counts, cross
    references between videos, splits, pairs and labels) instead of trusting them. Exits 4 if any
    check reports an error; warnings are printed but never fail the run.
    """
    from dfwb.protocols.lint import lint_pack

    issues = lint_pack(pack, release=release)

    if as_json:
        emit_json([_lint_issue_row(issue) for issue in issues])
    elif not issues:
        click.echo("no issues found")
    else:
        for issue in issues:
            click.echo(f"{issue.severity}: {issue.where}: {issue.message}")

    return 4 if any(issue.severity == "error" for issue in issues) else 0


def _scheme_diff_row(scheme_diff: Any) -> dict[str, Any]:
    return {
        "dataset": scheme_diff.dataset,
        "scheme": scheme_diff.scheme,
        "status": scheme_diff.status,
        "added": scheme_diff.added,
        "removed": scheme_diff.removed,
        "moved": scheme_diff.moved,
    }


@protocols.command("diff")
@click.argument("old", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.argument("new", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option(
    "--expect-bump",
    "expect_bump",
    type=click.Choice(["major", "minor", "patch"]),
    default=None,
    help="Fail unless the version change between the two pack.yaml files is at least the bump "
    "these changes require.",
)
@json_option
def diff(old: Path, new: Path, expect_bump: str | None, as_json: bool) -> int:
    """Compare two protocol pack directories and report the SemVer bump the changes require.

    With ``--expect-bump``, also compares that requirement against the actual version change
    between ``OLD/pack.yaml`` and ``NEW/pack.yaml`` (a plain SemVer component comparison, nothing
    to do with the value passed to the option itself) and exits 4 when the actual change is
    smaller than what the changes require -- the check a release pipeline runs before publishing.
    """
    from dfwb.core.records import PackCard
    from dfwb.protocols._yaml import read_model
    from dfwb.protocols.diff import bump_rank, diff_packs, version_bump

    result = diff_packs(old, new)

    exit_code = 0
    actual_bump: str | None = None
    if expect_bump is not None:
        old_card = read_model(old / "pack.yaml", PackCard)
        new_card = read_model(new / "pack.yaml", PackCard)
        actual_bump = version_bump(old_card.version, new_card.version)
        if bump_rank(actual_bump) < bump_rank(result.required_bump):
            exit_code = 4

    if as_json:
        payload: dict[str, Any] = {
            "schemes": [_scheme_diff_row(s) for s in result.schemes],
            "labels_changed": result.labels_changed,
            "required_bump": result.required_bump,
        }
        if actual_bump is not None:
            payload["actual_bump"] = actual_bump
        emit_json(payload)
        return exit_code

    rows = [[s.dataset, s.scheme, s.status, s.added, s.removed, s.moved] for s in result.schemes]
    click.echo(table(["DATASET", "SCHEME", "STATUS", "ADDED", "REMOVED", "MOVED"], rows))
    if result.labels_changed:
        click.echo("labels changed: " + ", ".join(result.labels_changed))
    click.echo(f"required bump: {result.required_bump}")
    if actual_bump is not None:
        click.echo(f"actual bump (pack.yaml version change): {actual_bump}")
        if exit_code:
            click.echo(
                f"error: the version change is only a {actual_bump!r} bump, but these changes "
                f"require {result.required_bump!r}",
                err=True,
            )
    return exit_code
