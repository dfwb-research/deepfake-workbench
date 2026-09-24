"""``dfwb plugins``: list and inspect registered components and the plugins that provide them."""

from __future__ import annotations

from typing import Any

import click

from dfwb.cli._output import emit_json, json_option, table


@click.group()
def plugins() -> None:
    """List and inspect plugins and their registered components."""


def _record(record: Any) -> dict[str, Any]:
    return {
        "name": record.name,
        "group": record.group,
        "provider": record.provider,
        "version": record.version,
        "target": record.target,
        "status": record.status.value,
        "reason": record.reason,
        "entries": list(record.entries),
    }


@plugins.command("list")
@click.option(
    "--all", "show_all", is_flag=True, help="Also show failed, skipped and disabled plugins."
)
@json_option
def list_(show_all: bool, as_json: bool) -> None:
    """List registered components (registry/key, provider, summary)."""
    from dfwb.core.plugins import PluginStatus, entries_by_registry, load_plugins

    report = load_plugins()
    grouped = entries_by_registry()
    entries = [entry for name in grouped for entry in grouped[name]]
    records = [r for r in report.records if show_all or r.status is PluginStatus.OK]
    problems = [
        r for r in report.records if r.status in (PluginStatus.FAILED, PluginStatus.SKIPPED)
    ]
    if as_json:
        emit_json(
            {
                "entries": [
                    {
                        "registry": e.registry,
                        "key": e.key,
                        "provider": e.provider,
                        "summary": e.summary,
                    }
                    for e in entries
                ],
                "plugins": [_record(r) for r in records],
            }
        )
        return
    if entries:
        rows = [[f"{e.registry}/{e.key}", e.provider, e.summary] for e in entries]
        click.echo(table(["COMPONENT", "PROVIDER", "SUMMARY"], rows))
    else:
        click.echo("no components registered")
    if show_all:
        click.echo("")
        plugin_rows = [[r.name, r.provider, r.version, r.status.value, r.reason] for r in records]
        click.echo(table(["PLUGIN", "PROVIDER", "VERSION", "STATUS", "REASON"], plugin_rows))
    elif problems:
        click.echo(
            f"note: {len(problems)} plugin(s) failed or were skipped; "
            "run `dfwb plugins list --all`",
            err=True,
        )


@plugins.command("info")
@click.argument("ref", metavar="REGISTRY/KEY")
@json_option
def info(ref: str, as_json: bool) -> None:
    """Show one component, e.g. `dfwb plugins info layers/srm`."""
    from dfwb.core.plugins import get_registry

    registry_name, sep, key = ref.partition("/")
    if not sep or not registry_name or not key:
        raise click.BadParameter(
            "expected REGISTRY/KEY, e.g. layers/srm", param_hint="REGISTRY/KEY"
        )
    entry = get_registry(registry_name).entry(key)
    data = {
        "registry": entry.registry,
        "key": entry.key,
        "qualified": entry.qualified_key,
        "provider": entry.provider,
        "summary": entry.summary,
        "target": entry.target,
        "aliases": list(entry.aliases),
        "requires": list(entry.requires),
        "params": entry.params,
        "meta": dict(entry.meta),
    }
    if as_json:
        emit_json(data)
        return
    for field, value in data.items():
        if isinstance(value, list):
            value = ", ".join(value) or "-"
        elif isinstance(value, dict):
            value = ", ".join(f"{k}={v}" for k, v in value.items()) or "-"
        click.echo(f"{field:<10} {value if value is not None else '-'}")
