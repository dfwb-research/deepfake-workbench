"""Commands used by the CLI tests to exercise error handling."""

import click

from dfwb.core.errors import ContractError


@click.command()
def contract_error() -> None:
    raise ContractError("bad file\n  detail line", hint="fix it")


@click.command()
def crash() -> None:
    raise RuntimeError("kaput")


NOT_A_COMMAND = "just a string"
