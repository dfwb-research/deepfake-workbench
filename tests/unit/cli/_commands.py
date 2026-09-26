"""Commands used by the CLI tests to exercise error handling."""

import click

from dfwb.core.errors import ContractError


@click.command()
def contract_error() -> None:
    raise ContractError("bad file\n  detail line", hint="fix it")


@click.command()
def crash() -> None:
    raise RuntimeError("kaput")


@click.command()
def abort() -> None:
    # What click itself raises when a command reading input hits EOF or Ctrl+C; simulated
    # directly here since neither actually happens in a non-interactive test.
    raise click.Abort()


NOT_A_COMMAND = "just a string"
