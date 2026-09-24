"""``dfwb doctor``: what is installed, where the roots are, which plugins loaded."""

from __future__ import annotations

import importlib.util
import platform
import re
from importlib import metadata
from typing import Any

import click

from dfwb.cli._output import emit_json, json_option, table

_EXTRA_MARKER = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")
_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _installed_extras() -> dict[str, bool]:
    """Each declared extra of deepfake-workbench and whether all its requirements are installed."""
    try:
        requirements = metadata.requires("deepfake-workbench") or []
    except metadata.PackageNotFoundError:
        return {}
    needed: dict[str, list[str]] = {}
    for requirement in requirements:
        marker = _EXTRA_MARKER.search(requirement)
        project = _NAME.match(requirement)
        if marker and project:
            needed.setdefault(marker.group(1), []).append(project.group(1))
    status: dict[str, bool] = {}
    for extra, names in sorted(needed.items()):
        present = True
        for name in names:
            if name == "deepfake-workbench":
                continue
            try:
                metadata.version(name)
            except metadata.PackageNotFoundError:
                present = False
        status[extra] = present
    return status


def _torch_status() -> dict[str, Any]:
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return {"installed": False}
    info: dict[str, Any] = {"installed": True, "version": version}
    try:
        if importlib.util.find_spec("torch") is None:
            raise ImportError("torch is not importable")
        import torch

        info["cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # a broken or blocked torch must not break the doctor
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def collect() -> dict[str, Any]:
    """Everything ``dfwb doctor`` reports, as plain data."""
    from dfwb import __version__
    from dfwb.core.paths import resolve_roots
    from dfwb.core.plugins import load_plugins

    roots = resolve_roots()
    report = load_plugins()
    return {
        "dfwb": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "torch": _torch_status(),
        "extras": _installed_extras(),
        "roots": {
            name: {
                "path": None if root.path is None else str(root.path),
                "source": root.source,
                "from": root.detail,
                "warning": root.warning,
            }
            for name, root in roots.items()
        },
        "plugins": [
            {
                "name": r.name,
                "provider": r.provider,
                "version": r.version,
                "status": r.status.value,
                "reason": r.reason,
            }
            for r in report.records
        ],
    }


@click.command()
@json_option
def doctor(as_json: bool) -> None:
    """Check Python, torch, roots, installed extras and plugins."""
    data = collect()
    if as_json:
        emit_json(data)
        return
    torch = data["torch"]
    torch_line = "not installed (install an extra such as [train] when it is needed)"
    if torch["installed"]:
        torch_line = torch["version"] + (f" ({torch['error']})" if "error" in torch else "")
    click.echo(f"dfwb      {data['dfwb']}")
    click.echo(f"python    {data['python']} ({data['platform']})")
    click.echo(f"torch     {torch_line}")
    extras = ", ".join(f"{k}{'' if v else ' (missing)'}" for k, v in data["extras"].items())
    click.echo(f"extras    {extras or 'none declared'}")
    click.echo("")
    rows = [
        [n, r["path"] or "(unset)", f"{r['source']}: {r['from']}"] for n, r in data["roots"].items()
    ]
    click.echo(table(["ROOT", "PATH", "FROM"], rows))
    for name, root in data["roots"].items():
        if root["warning"]:
            click.echo(f"warning: {root['warning']}", err=True)
        if name == "datasets" and root["path"] is None:
            click.echo(
                "note: DFWB_DATASETS_ROOT is unset; inventory and preprocess commands need it",
                err=True,
            )
    click.echo("")
    plugin_rows = [
        [p["name"], p["provider"], p["version"], p["status"], p["reason"] or ""]
        for p in data["plugins"]
    ]
    click.echo(table(["PLUGIN", "PROVIDER", "VERSION", "STATUS", "REASON"], plugin_rows))
