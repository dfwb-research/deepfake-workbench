"""``dfwb runs``: list and show training runs (torch is not needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from dfwb.cli._output import emit_json, json_option, table


def _runs_root() -> Path:
    from dfwb.core.paths import require_root, resolve_roots

    return require_root("runs", resolve_roots())


def _best(monitor: dict[str, Any] | None) -> str:
    if not monitor:
        return ""
    best = monitor.get("best")
    return f"{monitor['key']}={'undefined' if best is None else f'{best:.4f}'}"


@click.group()
def runs() -> None:
    """List and show training runs."""


@runs.command("list")
@json_option
def list_(as_json: bool) -> None:
    """List every run under the runs root (the newest of each name is marked *)."""
    from dfwb.train.rundir import list_runs

    root = _runs_root()
    found = list_runs(root)
    if as_json:
        emit_json([summary.to_json() for summary in found])
        return
    if not found:
        click.echo(f"no runs under {root}")
        return
    rows = [
        [
            summary.name,
            summary.path.name + ("*" if summary.latest else ""),
            summary.seed,
            summary.status,
            summary.epochs,
            _best(summary.monitor),
            summary.fingerprint[:12],
        ]
        for summary in found
    ]
    headers = ["NAME", "RUN", "SEED", "STATUS", "EPOCHS", "BEST", "FINGERPRINT"]
    click.echo(table(headers, rows))


def _details(run_dir: Path, root: Path) -> dict[str, Any]:
    from dfwb.train import rundir

    summary = rundir.read_run(run_dir)
    others = [
        other
        for other in rundir.list_runs(root)
        if other.fingerprint == summary.fingerprint and other.path.resolve() != run_dir.resolve()
    ]
    metrics: dict[str, Any] = {}
    if (run_dir / rundir.METRICS_FILE).is_file():
        metrics = rundir.read_json(run_dir / rundir.METRICS_FILE)
    data: dict[str, Any] = {}
    if (run_dir / rundir.DATA_FILE).is_file():
        data = rundir.read_json(run_dir / rundir.DATA_FILE)
    env: dict[str, Any] = {}
    if (run_dir / rundir.ENV_FILE).is_file():
        env = rundir.read_json(run_dir / rundir.ENV_FILE)
    return {
        **summary.to_json(),
        "global_step": metrics.get("global_step"),
        "same_config": [
            {"path": str(other.path), "seed": other.seed, "status": other.status}
            for other in others
        ],
        "data": rundir.data_rows(data),
        "val": metrics.get("val", {}),
        "env": env,
    }


def _environment(env: dict[str, Any]) -> str:
    versions = env.get("env", {})
    parts = [
        f"dfwb {versions.get('dfwb')}",
        f"python {versions.get('python')}",
        f"torch {versions.get('torch')}",
        f"device {versions.get('device')}",
    ]
    git = env.get("git")
    if git and git.get("commit"):
        parts.append(f"git {git['commit'][:7]}" + (" (dirty)" if git.get("dirty") else ""))
    return ", ".join(parts)


def _print(details: dict[str, Any]) -> None:
    status = details["status"]
    epochs = details["epochs"]
    if status == "completed":
        status += f" ({epochs} epoch(s), global step {details['global_step']})"
    else:
        status += f" ({epochs if epochs is not None else 0} epoch(s) done)"
    same = [f"{o['path']} (seed {o['seed']}, {o['status']})" for o in details["same_config"]]
    monitor = details["monitor"]
    lines: list[tuple[str, str]] = [
        ("run", details["path"]),
        ("name", details["name"]),
        ("seed", str(details["seed"])),
        ("status", status),
        ("created", str(details["created"])),
        ("fingerprint", details["fingerprint"]),
        ("same config", same[0] if same else "none"),
        *(("", line) for line in same[1:]),
    ]
    if monitor:
        best = "undefined" if monitor.get("best") is None else f"{monitor['best']:.4f}"
        at = f"best {best} after epoch {monitor['best_epoch']} of {details['epochs']}"
        lines.append(("monitor", f"{monitor['key']} ({monitor['mode']}): {at}"))
    sources = [
        f"{row['role']} {row['name']}: {row['included']} of {row['in_split']} videos"
        for row in details["data"]
    ]
    lines += [("data" if i == 0 else "", text) for i, text in enumerate(sources)]
    lines.append(("environment", _environment(details["env"])))
    values = [f"{key}={value:.4f}" for key, value in details["val"].items() if value is not None]
    if values:
        lines.append(("val", "  ".join(values)))
    for label, text in lines:
        click.echo(f"{label + ':' if label else '':<14}{text}")


@runs.command("show")
@click.argument("run")
@json_option
def show(run: str, as_json: bool) -> None:
    """Show one run: a run directory, a run name (its latest run), or <name>/<run>.

    The fingerprint identifies the experiment: every other run with the same one ran the same
    config (other seeds, or a duplicate).
    """
    from dfwb.train.rundir import find_run

    root = _runs_root()
    details = _details(find_run(run, root), root)
    if as_json:
        emit_json(details)
    else:
        _print(details)
