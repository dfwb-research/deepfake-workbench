"""``dfwb config``: templates, starter files, and composed/validated configs."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import click

from dfwb.cli._output import emit_json, json_option, table

STARTER = """\
# Starter experiment written by `dfwb config init`. Edit the values, then check them with
#   dfwb config show -c {out}
# (`dfwb config validate` also checks each component against the installed plugins.)
schema: dfwb.train/1
extends: [dfwb://templates/{template}.yaml]

run:
  name: my-experiment
  seeds: [42]
  output_root: ${{env:DFWB_RUNS_ROOT,./runs}}

data:
  processing: face-256-1.3x-32f
  train: [{{protocol: ffpp/official, split: train, where: {{compression: c23}}}}]
  val:   [{{protocol: ffpp/official, split: val,   where: {{compression: c23}}}}]

model:
  backbone: {{name: timm, model: vit_base_patch16_224.augreg_in21k, pretrained: true}}
"""


def _templates() -> dict[str, str]:
    """Shipped template name -> first comment line."""
    folder = resources.files("dfwb").joinpath("templates")
    found: dict[str, str] = {}
    for item in sorted(folder.iterdir(), key=lambda t: t.name):
        if item.name.endswith(".yaml"):
            first = item.read_text("utf-8").splitlines()[0] if item.is_file() else ""
            found[item.name[: -len(".yaml")]] = first.lstrip("# ").strip()
    return found


@click.group()
def config() -> None:
    """Compose, show and validate configs."""


@config.command("templates")
@json_option
def templates(as_json: bool) -> None:
    """List the shipped templates (usable as dfwb://templates/<name>.yaml)."""
    found = _templates()
    if as_json:
        emit_json([{"name": k, "summary": v} for k, v in found.items()])
    else:
        click.echo(table(["TEMPLATE", "SUMMARY"], list(found.items())))


@config.command("init")
@click.option("--template", default="binary-frame", show_default=True, help="Template to extend.")
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("experiment.yaml"),
    show_default=True,
)
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
@json_option
def init(template: str, out: Path, force: bool, as_json: bool) -> None:
    """Write a starter config that extends a shipped template."""
    from dfwb.core.errors import ConfigError, did_you_mean

    found = _templates()
    if template not in found:
        raise ConfigError(
            f"unknown template {template!r}{did_you_mean(template, found)}",
            hint="run `dfwb config templates`",
        )
    if out.exists() and not force:
        raise ConfigError(
            f"{out} already exists", hint="pass --force to overwrite it, or choose --out"
        )
    out.write_text(STARTER.format(template=template, out=out), encoding="utf-8")
    if as_json:
        emit_json({"written": str(out), "template": template})
    else:
        click.echo(f"wrote {out} (extends dfwb://templates/{template}.yaml)")


_config_option = click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Config file.",
)
_overrides_argument = click.argument("overrides", nargs=-1, metavar="[KEY=VALUE]...")


def _summary(loaded: Any) -> dict[str, Any]:
    return {
        "schema": loaded.schema,
        "fingerprint": loaded.fingerprint,
        "sources": list(loaded.sources),
    }


@config.command("show")
@_config_option
@_overrides_argument
@json_option
def show(config_path: Path, overrides: tuple[str, ...], as_json: bool) -> None:
    """Print the fully resolved config (extends merged, overrides and interpolation applied)."""
    from dfwb.core.config import dump_yaml, load_config

    loaded = load_config(config_path, overrides)
    if as_json:
        emit_json({**_summary(loaded), "config": loaded.data})
        return
    click.echo(f"# fingerprint: {loaded.fingerprint}")
    click.echo(f"# extends: {' -> '.join(loaded.sources)}")
    click.echo(dump_yaml(loaded.data), nl=False)


@config.command("validate")
@_config_option
@_overrides_argument
@json_option
def validate(config_path: Path, overrides: tuple[str, ...], as_json: bool) -> None:
    """Check a config, including every component and metric against the installed plugins.

    Protocol packs and processed stores are not needed here: training checks them, before it
    starts, once it joins the data.
    """
    from dfwb.core.config import load_config
    from dfwb.core.config.schema import TrainConfig
    from dfwb.eval.aggregate import check_eval_config

    loaded = load_config(config_path, overrides, check_registries=True)
    if isinstance(loaded.model, TrainConfig):
        check_eval_config(loaded.model.eval.metrics, loaded.model.eval.aggregate)
    if as_json:
        emit_json({**_summary(loaded), "valid": True})
    else:
        click.echo(f"ok: {config_path} ({loaded.schema}, fingerprint {loaded.fingerprint[:12]})")
