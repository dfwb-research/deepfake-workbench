"""Generate ``reference/cli.md``: every command and subcommand of the ``dfwb`` click tree.

Run by the ``gen-files`` MkDocs plugin at build time; nothing here is committed. Walks the real
command tree through click's own introspection, including the root's lazily-loaded groups
(:class:`dfwb.cli.main.LazyGroup`), so it never drifts from what ``dfwb --help`` actually shows.
Every subcommand module dfwb ships imports its heavy optional dependencies only inside functions
(never at module import time), so walking the whole tree -- which imports every subcommand module
once, to read its ``click.Command`` objects -- needs no optional extra at all, let alone torch.
"""

from __future__ import annotations

import click
import mkdocs_gen_files

from dfwb.cli.main import cli

_OUT = "reference/cli.md"


def _escape(text: str) -> str:
    """Neutralise Markdown/HTML-significant characters in plain help text.

    Click's own metavars and choice lists are full of ``<...>``, ``[...]`` and, at least once,
    ``|...|``: read as Markdown, a bracket pair can look like a reference-style link, an
    angle-bracket pair like ``<col>`` can look like an (unknown, and so silently dropped) inline
    HTML tag, and an unescaped ``|`` breaks out of the table cell it is written in. Escaping all
    three keeps the literal text on the page instead.
    """
    return (
        text.replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("|", "\\|")
    )


def _options_table(command: click.Command, ctx: click.Context) -> list[str]:
    rows = []
    for param in command.get_params(ctx):
        record = param.get_help_record(ctx)
        if record is None:  # hidden, or an argument (listed in the usage line instead)
            continue
        opt, help_text = record
        rows.append(f"| `{opt}` | {_escape(help_text or '')} |")
    if not rows:
        return []
    return ["", "| Option | Description |", "|---|---|", *rows]


def _arguments_line(command: click.Command, ctx: click.Context) -> str | None:
    names = [
        param.human_readable_name
        for param in command.get_params(ctx)
        if isinstance(param, click.Argument)
    ]
    return f"Arguments: {', '.join(names)}." if names else None


def _walk(command: click.Command, path: str, depth: int, lines: list[str]) -> None:
    ctx = click.Context(command, info_name=path)
    heading = "#" * min(depth + 2, 6)
    lines.append(f"{heading} `{path}`")
    lines.append("")
    help_text = command.help or command.get_short_help_str() or "*(no help text)*"
    lines.append(_escape(help_text.strip()))
    arguments = _arguments_line(command, ctx)
    if arguments:
        lines.append("")
        lines.append(_escape(arguments))
    lines.extend(_options_table(command, ctx))
    lines.append("")
    if isinstance(command, click.Group):
        for name in sorted(command.list_commands(ctx)):
            sub = command.get_command(ctx, name)
            if sub is None or sub.hidden:
                continue
            _walk(sub, f"{path} {name}", depth + 1, lines)


def generate() -> None:
    lines = [
        "# CLI reference",
        "",
        "Every `dfwb` command and subcommand, generated from the command tree itself (the same "
        "help text and options `dfwb <command> --help` prints).",
        "",
    ]
    _walk(cli, "dfwb", 0, lines)
    with mkdocs_gen_files.open(_OUT, "w") as handle:
        handle.write("\n".join(lines))


generate()
