"""``dfwb train``: train a config (one run per seed), or resume an interrupted run."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import click

from dfwb.cli._output import emit_json, json_option

_TRAIN_MODULES = ("torch", "lightning")


def _require_train_extra() -> None:
    from dfwb.core.errors import InstallationError

    for name in _TRAIN_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            raise InstallationError(
                f"dfwb train needs {name!r}, which is not installed",
                hint='pip install "deepfake-workbench[train]" (with the torch build you need)',
            )


@click.command("train")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Config file (schema dfwb.train/1).",
)
@click.argument("overrides", nargs=-1, metavar="[KEY=VALUE]...")
@click.option(
    "--resume",
    "resume",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Carry on the run in this run directory: only an interrupted run; a finished run "
    "cannot be extended -- more epochs is a new experiment.",
)
@click.option(
    "--device",
    default=None,
    help="cpu, cuda, or cuda:<index> (default: the config's, else any GPU, else the CPU).",
)
@json_option
def train(
    config_path: Path | None,
    overrides: tuple[str, ...],
    resume: Path | None,
    device: str | None,
    as_json: bool,
) -> None:
    """Train a config: one run per seed, each in its own run directory.

    Every config problem is reported before any data is read. With --resume, an interrupted run
    carries on from the end of its last finished epoch, with its own resolved config (-c, if
    also given, must be the same experiment).
    """
    if config_path is None and resume is None:
        raise click.UsageError(
            "give -c/--config to start runs, or --resume <run dir> to carry one on"
        )
    if overrides and config_path is None:
        raise click.UsageError("KEY=VALUE overrides need -c/--config")
    _require_train_extra()

    from dfwb.core.config import load_config
    from dfwb.train.run import resume_run, run_experiment

    loaded = load_config(config_path, overrides) if config_path is not None else None
    if resume is not None:
        results = [resume_run(resume, config=loaded, device=device, progress=not as_json)]
    else:
        assert loaded is not None
        results = run_experiment(loaded, device=device, progress=not as_json)

    if as_json:
        emit_json([result.to_json() for result in results])
        return
    for result in results:
        metrics = result.metrics
        line = f"{result.run_dir}: {metrics['epochs']} epoch(s)"
        monitor = metrics["monitor"]
        if monitor is None:
            line += ", no validation (checkpoints/best is the last epoch's)"
        else:
            best = "undefined" if monitor["best"] is None else f"{monitor['best']:.4f}"
            at = f"after epoch {monitor['best_epoch']} of {metrics['epochs']}"
            line += f", best {monitor['key']} = {best} ({at})"
        click.echo(line)
