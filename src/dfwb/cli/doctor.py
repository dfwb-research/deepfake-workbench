"""``dfwb doctor``: what is installed, where the roots are, which plugins loaded."""

from __future__ import annotations

import importlib.util
import platform
import re
from collections.abc import Mapping
from importlib import metadata
from typing import TYPE_CHECKING, Any

import click

from dfwb.cli._output import emit_json, json_option, table

if TYPE_CHECKING:
    from dfwb.core.paths import ResolvedRoot, RootName

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


def _datasets(roots: Mapping[RootName, ResolvedRoot]) -> list[dict[str, Any]]:
    """Every registered inventory builder's folder, and where it resolved on this machine.

    Reads only registry metadata, so no builder module is imported.
    """
    from dfwb.core.paths import dataset_overrides
    from dfwb.core.plugins import get_registry
    from dfwb.preprocess.inventory.runner import folder_status

    overrides = dataset_overrides()
    rows: list[dict[str, Any]] = []
    for entry in get_registry("inventory_builders").entries():
        folder = entry.meta.get("folder")
        folder = folder if isinstance(folder, str) and folder else None
        status = folder_status(entry.key, folder, roots, overrides)
        location = status.location
        warning = None
        if location is None and entry.key in overrides:
            warning = status.problem  # an override that points nowhere
        elif location is not None and location.also_found:
            others = ", ".join(str(p) for p in location.also_found)
            warning = (
                f"{entry.key} is in several datasets roots; using {location.path} "
                f"({status.source}), also in {others}"
            )
        rows.append(
            {
                "id": entry.key,
                "folder": folder,
                "path": None if location is None else str(location.path),
                "source": status.source,
                "also_found": [] if location is None else [str(p) for p in location.also_found],
                "warning": warning,
            }
        )
    return rows


def _roots() -> tuple[Mapping[RootName, ResolvedRoot], dict[str, str] | None]:
    """The resolved roots, and the problem that stopped them being read (``{"message",
    "hint"}``), if any.

    A broken ``dfwb.toml`` (or user config) must not abort the whole report: it is shown as this
    one failed check, and every other section -- torch, extras, plugins, licences -- still runs.
    """
    from dfwb.core.errors import ConfigError
    from dfwb.core.paths import resolve_roots

    try:
        return resolve_roots(), None
    except ConfigError as exc:
        return {}, {"message": exc.message, "hint": exc.hint}


def _licenses() -> tuple[dict[str, dict[str, str]], dict[str, str] | None]:
    """Every licence acknowledgement recorded on this machine, keyed by name, and the problem
    that stopped the store being read (``{"message", "hint"}``), if any.

    A store that cannot be read is reported, not raised: the doctor is where to find out about it,
    so it must still report everything else.
    """
    from dfwb.core import licenses
    from dfwb.core.errors import ContractError

    try:
        accepted = licenses.all_accepted()
    except ContractError as exc:
        return {}, {"message": exc.message, "hint": exc.hint}
    return {
        name: {"license": entry.license, "accepted_at": entry.accepted_at}
        for name, entry in sorted(accepted.items())
    }, None


def collect() -> dict[str, Any]:
    """Everything ``dfwb doctor`` reports, as plain data."""
    from dfwb import __version__
    from dfwb.core.envfile import last_applied
    from dfwb.core.plugins import load_plugins

    roots, roots_error = _roots()
    report = load_plugins()
    env_file = last_applied()
    accepted, licenses_error = _licenses()
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
                **({"paths": [str(p) for p in root.paths]} if name == "datasets" else {}),
            }
            for name, root in roots.items()
        },
        "env_file": None
        if env_file is None
        else {
            "path": str(env_file.file),
            "applied": list(env_file.applied),
            "skipped": list(env_file.skipped),
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
        "roots_error": roots_error,
        # dataset_overrides() re-reads the same project/user config as resolve_roots(): once that
        # has already failed, re-reading it would only raise the same error again.
        "datasets": [] if roots_error is not None else _datasets(roots),
        "licenses": accepted,
        "licenses_error": licenses_error,
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
    if data["roots_error"] is not None:
        click.echo(data["roots_error"]["message"])
        click.echo(f"hint: {data['roots_error']['hint']}")
    else:
        rows = [
            [n, r["path"] or "(unset)", f"{r['source']}: {r['from']}"]
            for n, r in data["roots"].items()
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
        for path in data["roots"]["datasets"]["paths"]:
            click.echo(f"datasets root: {path}")
    if data["env_file"] is not None:
        env_file = data["env_file"]
        click.echo(f"env file: {env_file['path']} ({len(env_file['applied'])} keys applied)")
    click.echo("")
    plugin_rows = [
        [p["name"], p["provider"], p["version"], p["status"], p["reason"] or ""]
        for p in data["plugins"]
    ]
    click.echo(table(["PLUGIN", "PROVIDER", "VERSION", "STATUS", "REASON"], plugin_rows))
    click.echo("")
    if not data["datasets"]:
        click.echo("datasets: no inventory builders are registered")
    else:
        dataset_rows = [
            [d["id"], d["folder"] or "-", d["source"], d["path"] or ""] for d in data["datasets"]
        ]
        click.echo(table(["DATASET", "FOLDER", "RESOLVED", "PATH"], dataset_rows))
        for dataset in data["datasets"]:
            if dataset["warning"]:
                click.echo(f"warning: {dataset['warning']}", err=True)
    click.echo("")
    click.echo("LICENCES")
    if data["licenses_error"] is not None:
        click.echo(data["licenses_error"]["message"])
        click.echo(f"hint: {data['licenses_error']['hint']}")
    elif not data["licenses"]:
        click.echo("no licences acknowledged yet")
    else:
        licence_rows = [
            [name, entry["license"], entry["accepted_at"]]
            for name, entry in data["licenses"].items()
        ]
        click.echo(table(["NAME", "LICENSE", "ACCEPTED"], licence_rows))
