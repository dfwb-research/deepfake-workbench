"""``dfwb completion``: print the line that enables shell completion."""

from __future__ import annotations

import click

from dfwb.cli._output import emit_json, json_option

SNIPPETS = {
    "bash": 'eval "$(_DFWB_COMPLETE=bash_source dfwb)"',
    "zsh": 'eval "$(_DFWB_COMPLETE=zsh_source dfwb)"',
    "fish": "_DFWB_COMPLETE=fish_source dfwb | source",
}


@click.command()
@click.argument("shell", type=click.Choice(sorted(SNIPPETS)))
@json_option
def completion(shell: str, as_json: bool) -> None:
    """Print the line to add to your shell's startup file to enable completion."""
    if as_json:
        emit_json({"shell": shell, "snippet": SNIPPETS[shell]})
    else:
        click.echo(SNIPPETS[shell])
