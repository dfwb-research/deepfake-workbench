"""``dfwb preprocess``: run the face pipeline, check its progress, and merge sharded runs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import click

from dfwb.cli._output import emit_json, json_option, table
from dfwb.core.errors import ConfigError

if TYPE_CHECKING:
    from dfwb.core.records.local import CropSpec, SamplingSpec

_SHARD_RE = re.compile(r"^(\d+)/(\d+)$")


@click.group()
def preprocess() -> None:
    """Run the face pipeline, check its progress, and merge sharded runs."""


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


def _parse_shard(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = _SHARD_RE.fullmatch(value)
    if match is None:
        raise ConfigError(f"--shard {value!r} is not i/n", hint="use --shard i/n, e.g. --shard 0/2")
    return int(match.group(1)), int(match.group(2))


def _parse_redo(value: str | None) -> frozenset[str]:
    return frozenset(part.strip() for part in (value or "").split(",") if part.strip())


_dataset_argument = click.argument("dataset")
_profile_option = click.option(
    "--profile", required=True, help="A shipped profile's name, or a profile YAML file's path."
)
_protocol_option = click.option(
    "--protocol",
    default=None,
    help="A protocol reference to take the videos from, e.g. ffpp/official.",
)
_split_option = click.option("--split", default=None, help="The protocol split; needs --protocol.")
_where_option = click.option(
    "--where",
    "where_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Restrict to videos matching KEY=VALUE (repeatable; repeat a key for any-of).",
)


@preprocess.command("run")
@_dataset_argument
@_profile_option
@_protocol_option
@_split_option
@_where_option
@click.option(
    "--shard", "shard_text", default=None, metavar="I/N", help="Process only shard I of N."
)
@click.option(
    "--workers", type=int, default=0, show_default=True, help="Worker processes (0: this process)."
)
@click.option("--device", default="cpu", show_default=True, help="cpu or cuda:<index>.")
@click.option(
    "--redo", "redo_text", default=None, metavar="S1,S2", help="Statuses to process again."
)
@click.option(
    "--limit", type=int, default=None, help="Process at most the first N videos in scope."
)
@click.option(
    "--accept-license",
    is_flag=True,
    help="Record the licence acknowledgement the backend's weights need, if any.",
)
@json_option
def run(
    dataset: str,
    profile: str,
    protocol: str | None,
    split: str | None,
    where_pairs: tuple[str, ...],
    shard_text: str | None,
    workers: int,
    device: str,
    redo_text: str | None,
    limit: int | None,
    accept_license: bool,
    as_json: bool,
) -> None:
    """Process DATASET's videos with PROFILE into its processed store."""
    from dfwb.preprocess.face.runner import run as run_pipeline

    summary = run_pipeline(
        dataset,
        profile=profile,
        protocol=protocol,
        split=split,
        where=_parse_where(where_pairs) or None,
        shard=_parse_shard(shard_text),
        workers=workers,
        device=device,
        redo=_parse_redo(redo_text),
        limit=limit,
        accept_license=accept_license,
    )
    if as_json:
        emit_json(
            {
                "counts_by_status": summary.counts_by_status,
                "n_skipped": summary.n_skipped,
                "store": str(summary.store),
            }
        )
        return
    rows: list[list[Any]] = [[status, count] for status, count in summary.counts_by_status.items()]
    # Videos this run left alone because the store already holds their outcome; named apart from
    # the "skipped" status a video's own row can carry.
    rows.append(["already done", summary.n_skipped])
    click.echo(table(["STATUS", "COUNT"], rows))
    click.echo(f"store: {summary.store}")


@preprocess.command("status")
@_dataset_argument
@_profile_option
@_protocol_option
@_split_option
@json_option
def status(
    dataset: str, profile: str, protocol: str | None, split: str | None, as_json: bool
) -> None:
    """Count DATASET's PROFILE store by status and task (and split with --protocol)."""
    from dfwb.preprocess.face.shard import status as compute_status

    result = compute_status(dataset, profile, protocol, split)
    if as_json:
        emit_json(
            [
                {"task": row.task, "split": row.split, "status": row.status, "count": row.count}
                for row in result.rows
            ]
        )
        return
    if protocol is not None:
        rows = [[row.task, row.split, row.status, row.count] for row in result.rows]
        click.echo(table(["TASK", "SPLIT", "STATUS", "COUNT"], rows))
    else:
        rows = [[row.task, row.status, row.count] for row in result.rows]
        click.echo(table(["TASK", "STATUS", "COUNT"], rows))


@preprocess.command("merge")
@_dataset_argument
@_profile_option
def merge(dataset: str, profile: str) -> None:
    """Combine DATASET's PROFILE store's shard files (from sharded runs) into its index."""
    from dfwb.preprocess.face.shard import merge as merge_store
    from dfwb.preprocess.face.shard import store_root

    root = store_root(dataset, profile)
    summary = merge_store(root)
    if summary.n_shards == 0:
        click.echo(f"nothing to merge in {root}")
        return
    click.echo(
        f"merged {summary.n_shards} shard file(s) into {summary.n_records} record(s) at "
        f"{root / 'index.jsonl'}"
    )


def _sampling_text(spec: SamplingSpec) -> str:
    text: str = spec.mode
    if spec.frames is not None:
        text += f" {spec.frames}f"
    if spec.stride is not None:
        text += f"/{spec.stride}"
    return text


def _crop_text(spec: CropSpec) -> str:
    # Every crop is square (the profile allows nothing else), but the column says so all the same.
    return f"{spec.scale}x/{spec.size}/square"


@preprocess.command("profiles")
@json_option
def profiles(as_json: bool) -> None:
    """List the shipped face-processing profiles."""
    from dfwb.preprocess.face.profiles import builtin_profiles, load_profile

    loaded = [load_profile(name) for name in builtin_profiles()]
    if as_json:
        emit_json(
            [
                {
                    "id": profile.id,
                    "profile_id": profile.profile_id(),
                    "backend": profile.backend.name,
                    "sampling": _sampling_text(profile.sampling),
                    "crop": _crop_text(profile.crop),
                }
                for profile in loaded
            ]
        )
        return
    rows = [
        [
            profile.id,
            profile.profile_id(),
            profile.backend.name,
            _sampling_text(profile.sampling),
            _crop_text(profile.crop),
        ]
        for profile in loaded
    ]
    click.echo(table(["ID", "PROFILE_ID", "BACKEND", "SAMPLING", "CROP"], rows))
