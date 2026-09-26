"""``dfwb protocols``: list, inspect, verify, lint and diff protocol packs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import click

from dfwb.cli._output import emit_json, hint_line, json_option, table


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
        name = r.dataset_id or f"({r.pack})"
        table_rows.append([name, r.pack, r.version or "-", "BROKEN", "-", "-", "-"])
    if table_rows:
        click.echo(table(headers, table_rows))
    else:
        click.echo("no protocol packs installed")
    for r in broken:
        what = (
            f"dataset {r.dataset_id!r} of pack {r.pack!r}" if r.dataset_id else f"pack {r.pack!r}"
        )
        click.echo(f"warning: {what} is broken: {r.broken}", err=True)


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


def _verify_hint(dataset: str, exit_code: int) -> str:
    if exit_code == 4:
        return (
            f"the local inventory disagrees with the pack about a video's label; re-run "
            f"`dfwb inventory build {dataset}` against the pack this inventory was built for"
        )
    return (
        f"process the missing videos, then re-run `dfwb inventory build {dataset}` to pick them up"
    )


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
    from dfwb.protocols.verification import verify as run_verify
    from dfwb.protocols.verification import write_report

    roots = resolve_roots()
    work_root = require_root("work", roots)
    report = run_verify(ref, inventory=inventory, splits=splits or None, work_root=work_root)
    report_path = write_report(report, work_root, datasets_roots=roots["datasets"].paths)

    if as_json:
        emit_json({**report.to_json(), "report_path": str(report_path)})
        if report.exit_code:
            hint_line(_verify_hint(report.dataset, report.exit_code))
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
    if report.exit_code:
        hint_line(_verify_hint(report.dataset, report.exit_code))
    return report.exit_code


@protocols.command("build")
@click.argument("dataset")
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="The dataset's folder in the pack, normally <pack root>/DATASET.",
)
@click.option(
    "--inventory",
    "inventory",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Inventory to build from (default: <work root>/<dataset>/inventory.jsonl).",
)
@click.option(
    "--scheme",
    "schemes",
    multiple=True,
    help="A scheme to build (repeatable); default: every scheme of the dataset. The default "
    "scheme must be among them.",
)
@click.option(
    "--update-pack-yaml",
    is_flag=True,
    help="Also list DATASET in the pack.yaml next to --out (OUT/../pack.yaml) if it is absent.",
)
@json_option
def build(
    dataset: str,
    out: Path,
    inventory: Path | None,
    schemes: tuple[str, ...],
    update_pack_yaml: bool,
    as_json: bool,
) -> None:
    """Build DATASET's protocol-pack files from its local inventory.

    Writes the videos, one split file per scheme, the pairs, the labels, the dataset card, the
    NOTICE and PROVENANCE.json into --out. Rebuilding from the same inventory gives the same bytes,
    and a rebuild keeps the terms review (distribution and terms) already in --out's dataset.yaml.
    """
    from dfwb.core.errors import ConfigError
    from dfwb.core.paths import absolute
    from dfwb.preprocess.packbuild import PACK_YAML, add_to_pack_yaml, build_dataset

    if update_pack_yaml and absolute(out).name != dataset:
        raise ConfigError(
            f"--update-pack-yaml lists {dataset!r} in the pack, but --out {absolute(out)} is not "
            f"named {dataset!r}",
            hint=f"build into <pack root>/{dataset}",
        )
    result = build_dataset(dataset, out=out, inventory=inventory, schemes=schemes or None)
    pack_yaml: dict[str, Any] | None = None
    if update_pack_yaml:
        added = add_to_pack_yaml(result.out.parent, dataset)
        pack_yaml = {"path": str(result.out.parent / PACK_YAML), "added": added}

    if as_json:
        emit_json(
            {
                "dataset_id": dataset,
                "out": str(result.out),
                "n_videos": result.n_videos,
                "n_pairs": result.n_pairs,
                "schemes": {
                    name: card.model_dump(mode="json") for name, card in result.schemes.items()
                },
                "pack_yaml": pack_yaml,
            }
        )
        return

    click.echo(f"wrote {dataset} to {result.out}: {result.n_videos} videos, {result.n_pairs} pairs")
    rows = []
    for name, card in result.schemes.items():
        counts = card.counts or {}
        train, val, test = (counts.get(s, "-") for s in ("train", "val", "test"))
        rows.append([name, card.rule, train, val, test, card.sha256[:12]])
    click.echo(table(["SCHEME", "RULE", "TRAIN", "VAL", "TEST", "SHA256"], rows))
    if pack_yaml is not None:
        verb = "now lists" if pack_yaml["added"] else "already lists"
        click.echo(f"{pack_yaml['path']} {verb} {dataset}")


@protocols.command("materialize")
@click.argument("ref")
@click.option(
    "--inventory",
    "inventory",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Inventory to recompute from (default: <work root>/<dataset>/inventory.jsonl).",
)
@json_option
def materialize(ref: str, inventory: Path | None, as_json: bool) -> None:
    """Recompute a recipe scheme from the local inventory and check its published hash.

    A pack may publish a scheme as a rule, its parameters and a hash, with no key list. This
    rebuilds the split from your own inventory and, only if it hashes to the published value,
    writes it to the dataset's materialized folder under the work root, where every command that
    loads the protocol finds it. A mismatch exits 4 and writes nothing.
    """
    from dfwb.core.paths import require_root, resolve_roots
    from dfwb.core.records import InventoryRecord, read_jsonl
    from dfwb.preprocess.inventory.runner import get_builder, inventory_path
    from dfwb.preprocess.packbuild import locate_metadata_root
    from dfwb.protocols.materialization import materialize as run_materialize
    from dfwb.protocols.materialization import needs_official
    from dfwb.protocols.refs import parse_ref

    roots = resolve_roots()
    work_root = require_root("work", roots)
    parsed = parse_ref(ref)
    source = inventory if inventory is not None else inventory_path(parsed.dataset, work_root)
    official = None
    if needs_official(parsed) and source.is_file():
        # The publisher's split is dataset knowledge: read it with the dataset's own builder.
        builder = get_builder(parsed.dataset)
        records = read_jsonl(source, InventoryRecord)
        official = builder.official_splits(locate_metadata_root(builder, roots), records)
    result = run_materialize(
        parsed,
        inventory=source,
        official=official,
        work_root=work_root,
        datasets_roots=roots["datasets"].paths,
    )

    if as_json:
        emit_json(
            {
                "ref": result.ref,
                "path": str(result.path),
                "sha256": result.sha256,
                "matched": result.matched,
            }
        )
        return
    click.echo(f"{result.ref}: sha256 {result.sha256} matches the published hash")
    click.echo(f"wrote {result.path}")


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
    errors = [issue for issue in issues if issue.severity == "error"]

    if as_json:
        emit_json([_lint_issue_row(issue) for issue in issues])
    elif not issues:
        click.echo("no issues found")
    else:
        for issue in issues:
            click.echo(f"{issue.severity}: {issue.where}: {issue.message}")

    if errors:
        hint_line("fix each error above, then run `dfwb protocols lint` again")
    return 4 if errors else 0


@protocols.command("new-pack")
@click.argument("directory", type=click.Path(path_type=Path, file_okay=False))
@click.option("--name", required=True, help="Pack distribution and registry name (lower-kebab).")
@click.option(
    "--author", default="Your Name", show_default=True, help="Author name for the generated files."
)
@json_option
def new_pack(directory: Path, name: str, author: str, as_json: bool) -> None:
    """Scaffold a new, dependency-free protocol pack distribution.

    ``DIRECTORY`` is created if it does not exist, and must be empty if it does. The result
    registers itself with the framework through the ``dfwb.plugins`` entry point once installed.
    """
    from dfwb.protocols import newpack

    written = newpack.new_pack(directory, name=name, author=author)

    if as_json:
        emit_json({"directory": str(directory), "written": [str(p) for p in written]})
    else:
        for path in written:
            click.echo(f"wrote {path}")


def _scheme_diff_row(scheme_diff: Any) -> dict[str, Any]:
    return {
        "dataset": scheme_diff.dataset,
        "scheme": scheme_diff.scheme,
        "status": scheme_diff.status,
        "added": scheme_diff.added,
        "removed": scheme_diff.removed,
        "moved": scheme_diff.moved,
    }


# How many relabelled videos a diff names (the count is always given in full).
_RELABELLED_SHOWN = 20


def _report_bump_shortfall(error_line: str, required_bump: str) -> None:
    click.echo(error_line, err=True)
    hint_line(f"bump the version to at least {required_bump!r}")


@protocols.command("diff")
@click.argument("old", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.argument("new", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option(
    "--expect-bump",
    "expect_bump",
    type=click.Choice(["major", "minor", "patch", "none"]),
    default=None,
    help="The bump you claim for this release. Fails unless it is at least the bump these "
    "changes require ('none' when nothing but the version changed).",
)
@json_option
def diff(old: Path, new: Path, expect_bump: str | None, as_json: bool) -> int:
    """Compare two protocol pack directories and report the SemVer bump the changes require.

    With ``--expect-bump``, the value is the bump the pack author claims for the release: the
    command exits 4, naming both, when that claim is smaller than the bump the changes require --
    the check a release pipeline runs before publishing. An unchanged pack requires ``none``,
    which any claim covers. It also reads both ``pack.yaml`` versions and reports their change;
    a malformed version, or one that goes down, exits 4.
    """
    from dfwb.core.records import PackCard
    from dfwb.protocols._yaml import read_model
    from dfwb.protocols.diff import bump_rank, diff_packs, version_bump

    result = diff_packs(old, new)

    exit_code = 0
    actual_bump: str | None = None
    bump_error: str | None = None
    if expect_bump is not None:
        old_card = read_model(old / "pack.yaml", PackCard)
        new_card = read_model(new / "pack.yaml", PackCard)
        actual_bump = version_bump(old_card.version, new_card.version)
        if bump_rank(expect_bump) < bump_rank(result.required_bump):
            exit_code = 4
            bump_error = (
                f"error: --expect-bump {expect_bump!r} does not cover these changes, which "
                f"require {result.required_bump!r}"
            )

    relabelled = result.relabelled[:_RELABELLED_SHOWN]
    if as_json:
        payload: dict[str, Any] = {
            "schemes": [_scheme_diff_row(s) for s in result.schemes],
            "labels_changed": result.labels_changed,
            "labels_added": result.labels_added,
            "relabelled": relabelled,
            "relabelled_count": len(result.relabelled),
            "required_bump": result.required_bump,
        }
        if expect_bump is not None:
            payload["expected_bump"] = expect_bump
            payload["actual_bump"] = actual_bump
        emit_json(payload)
        if bump_error is not None:
            _report_bump_shortfall(bump_error, result.required_bump)
        return exit_code

    rows = [[s.dataset, s.scheme, s.status, s.added, s.removed, s.moved] for s in result.schemes]
    click.echo(table(["DATASET", "SCHEME", "STATUS", "ADDED", "REMOVED", "MOVED"], rows))
    if result.labels_changed:
        click.echo("labels changed: " + ", ".join(result.labels_changed))
    if result.labels_added:
        click.echo("labels added: " + ", ".join(result.labels_added))
    if result.relabelled:
        shown = "" if len(relabelled) == len(result.relabelled) else f" (first {len(relabelled)})"
        click.echo(f"videos relabelled: {len(result.relabelled)}{shown}")
        for key in relabelled:
            click.echo(f"  {key}")
    click.echo(f"required bump: {result.required_bump}")
    if expect_bump is not None:
        click.echo(f"expected bump: {expect_bump}")
        click.echo(f"pack.yaml version change: {actual_bump}")
        if bump_error is not None:
            _report_bump_shortfall(bump_error, result.required_bump)
    return exit_code
