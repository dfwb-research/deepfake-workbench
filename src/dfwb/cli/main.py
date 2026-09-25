"""The ``dfwb`` command.

Subcommands are imported only when they run, and this module imports nothing but click, so
``dfwb --help`` is fast and never needs torch. Errors never show a traceback unless ``--debug``
(or ``DFWB_DEBUG=1``) is set: they print ``error:`` and ``hint:`` lines and exit with the code of
the error class (see ``dfwb.core.errors``).
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import click

__all__ = ["COMMANDS", "LazyGroup", "cli", "main"]

# name -> ("module:attribute", one-line help shown by `dfwb --help`)
COMMANDS: dict[str, tuple[str, str]] = {
    "completion": ("dfwb.cli.completion:completion", "Print the shell completion snippet."),
    "config": ("dfwb.cli.config:config", "Compose, show and validate configs."),
    "datasets": ("dfwb.cli.datasets:datasets", "List supported datasets and where they are."),
    "doctor": ("dfwb.cli.doctor:doctor", "Check Python, roots, extras and plugins."),
    "eval": (
        "dfwb.cli.eval:eval",
        "Evaluate, compare, calibrate and import score files.",
    ),
    "inventory": ("dfwb.cli.inventory:inventory", "Build and summarise dataset inventories."),
    "plugins": ("dfwb.cli.plugins:plugins", "List and inspect plugins and their components."),
    "preprocess": (
        "dfwb.cli.preprocess:preprocess",
        "Run the face pipeline, check its progress, and merge sharded runs.",
    ),
    "protocols": ("dfwb.cli.protocols:protocols", "List and inspect installed protocol packs."),
    "runs": ("dfwb.cli.runs:runs", "List and show training runs."),
    "schema": ("dfwb.cli.schema:schema", "Export the contract JSON Schemas."),
    "score": (
        "dfwb.cli.score:score",
        "Run a detector over a protocol split (or a suite) and write score files.",
    ),
    "train": ("dfwb.cli.train:train", "Train a config, or resume an interrupted run."),
}

ISSUES_URL = "https://github.com/dfwb-research/deepfake-workbench/issues"


class LazyGroup(click.Group):
    """A group whose subcommands are imported on use; help text comes from the table."""

    def __init__(
        self,
        *args: Any,
        lazy_subcommands: Mapping[str, tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lazy_subcommands = dict(lazy_subcommands or {})

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self.lazy_subcommands})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name not in self.lazy_subcommands:
            return super().get_command(ctx, cmd_name)
        module_name, _, attribute = self.lazy_subcommands[cmd_name][0].partition(":")
        command = getattr(importlib.import_module(module_name), attribute)
        if not isinstance(command, click.Command):
            raise TypeError(f"{module_name}:{attribute} is not a click command")
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [(name, self.lazy_subcommands[name][1]) for name in sorted(self.lazy_subcommands)]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


def _print_version(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    from dfwb import __version__
    from dfwb.core.plugins import PluginStatus, load_plugins

    click.echo(f"dfwb {__version__}")
    for record in load_plugins().records:
        if record.status is PluginStatus.OK and record.group == "dfwb.plugins":
            click.echo(f"  {record.provider} {record.version or '(unknown version)'}")
    ctx.exit(0)


@click.group(
    cls=LazyGroup,
    lazy_subcommands=COMMANDS,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Deepfake Workbench: datasets, protocols, training, scoring and evaluation.",
)
@click.option("--debug", is_flag=True, envvar="DFWB_DEBUG", help="Show full tracebacks on errors.")
@click.option(
    "--version",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=_print_version,
    help="Show the dfwb version and the versions of installed plugins.",
)
@click.option(
    "--env-file",
    "env_file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Load environment variables from this file instead of discovering one.",
)
@click.option(
    "--no-env-file",
    "no_env_file",
    is_flag=True,
    help="Never load a .env file, even one that would otherwise be discovered.",
)
def cli(debug: bool, env_file: Path | None, no_env_file: bool) -> None:
    from dfwb.core.log import setup_logging

    setup_logging("DEBUG" if debug else "WARNING")
    _load_env_file(env_file, no_env_file)


def _load_env_file(env_file: Path | None, no_env_file: bool) -> None:
    """Find and apply this machine's ``.env`` before any subcommand runs (never at import)."""
    from dfwb.core import envfile

    if no_env_file:
        envfile._remember(None)
        return

    path: Path | None
    if env_file is not None:
        if not env_file.is_file():
            from dfwb.core.errors import ConfigError

            raise ConfigError(f"{env_file}: no such file", hint="check --env-file")
        path = env_file
    else:
        path = envfile.find_env_file(Path.cwd(), os.environ)

    if path is None:
        envfile._remember(None)
        return

    from dfwb.core.errors import ConfigError

    try:
        applied = envfile.apply_env_file(path, os.environ)
    except ConfigError as exc:
        # Another tool's .env (docker-compose, say) may sit where dfwb looks for its own.
        raise ConfigError(
            exc.message,
            hint=f"{exc.hint}; if {path} is not meant for dfwb, run dfwb --no-env-file ..., or "
            f"point {envfile.ENV_FILE_VAR} at a dfwb .env",
        ) from None
    envfile._remember(applied)


def _debug_requested(args: Sequence[str]) -> bool:
    return "--debug" in args or os.environ.get("DFWB_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _report(message: str, hint: str) -> None:
    first, *rest = message.splitlines() or [""]
    click.echo(f"error: {first}", err=True)
    for line in rest:
        click.echo(line, err=True)
    click.echo(f"hint: {hint}", err=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code (the console script exits with it)."""
    args = list(sys.argv[1:] if argv is None else argv)
    debug = _debug_requested(args)
    try:
        result = cli.main(args=args, prog_name="dfwb", standalone_mode=False)
    except click.exceptions.NoArgsIsHelpError as exc:  # `dfwb`, `dfwb config`: show the help
        click.echo(exc.format_message())
        return 0
    except click.UsageError as exc:
        path = exc.ctx.command_path if exc.ctx is not None else "dfwb"
        if exc.ctx is not None:
            click.echo(exc.ctx.get_usage(), err=True)
        _report(exc.format_message(), f"run `{path} --help` for usage")
        return exc.exit_code
    except click.ClickException as exc:
        _report(exc.format_message(), "run `dfwb --help` for usage")
        return exc.exit_code
    except click.Abort:
        click.echo("aborted", err=True)
        return 1
    except Exception as exc:
        from dfwb.core.errors import DFWBError

        if debug:
            traceback.print_exc()
        if isinstance(exc, DFWBError):
            _report(exc.message, exc.hint)
            return exc.exit_code
        _report(
            f"unexpected {type(exc).__name__}: {exc}",
            f"re-run with --debug for the traceback, and report it at {ISSUES_URL}",
        )
        return 1
    return result if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
