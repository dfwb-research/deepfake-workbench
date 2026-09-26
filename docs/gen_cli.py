"""Generate ``reference/cli.md``: every command and subcommand of the ``dfwb`` click tree.

Run by the ``gen-files`` MkDocs plugin at build time; nothing here is committed. Walks the real
command tree through click's own introspection, including the root's lazily-loaded groups
(:class:`dfwb.cli.main.LazyGroup`), so it never drifts from what ``dfwb --help`` actually shows.
Every subcommand module dfwb ships imports its heavy optional dependencies only inside functions
(never at module import time), so walking the whole tree -- which imports every subcommand module
once, to read its ``click.Command`` objects -- needs no optional extra at all, let alone torch.

Every child context is built with its real parent (see :func:`_walk`), so a setting the root
group declares (``help_option_names=["-h", "--help"]``, in :mod:`dfwb.cli.main`) is inherited all
the way down, exactly as it is for the live CLI -- not just at the root, which a parentless
``click.Context`` would otherwise silently fall back to.
"""

from __future__ import annotations

import click
import mkdocs_gen_files

from dfwb.cli.main import cli

_OUT = "reference/cli.md"

# A group whose *bare* invocation (no subcommand name) dispatches to one of its own hidden
# subcommands, mapped to that subcommand's name. `dfwb eval FILES...` is the only one: `_EvalGroup`
# (src/dfwb/cli/eval.py) routes anything that is not a known subcommand name to its hidden `run`,
# and `run`'s own docstring says so ("this is what bare `dfwb eval FILES...` runs"). Documented as
# that group's own default form below, not as a `run` subcommand a user would type.
_DEFAULT_DISPATCH: dict[str, str] = {"dfwb eval": "run"}

# Hidden commands to leave out of the reference entirely, named explicitly here -- never a blanket
# "skip every hidden command" filter, which would just as silently have dropped `eval run` (and
# every option `dfwb eval FILES...` actually takes) along with anything genuinely internal. Empty
# today: nothing currently hidden has no documentation value. Add a full path (e.g. "dfwb foo bar")
# here, with a comment saying why, the day one does.
_HIDDEN_DENYLIST: frozenset[str] = frozenset()


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
        # `opt` (e.g. `--format [md|csv|latex|json]` for a Choice) is never escaped: it is always
        # wrapped in backticks below, and a Markdown table splits cells on `|` outside a code span,
        # never inside one -- escaping here would print the backslashes literally instead.
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


def _walk(
    command: click.Command,
    path: str,
    depth: int,
    lines: list[str],
    *,
    parent_ctx: click.Context | None = None,
    heading_override: str | None = None,
) -> None:
    # Mirrors what `Command.make_context` does: merge the command's own `context_settings` (only
    # the root `cli` declares any, `help_option_names=["-h", "--help"]`) into the context, on top
    # of whatever it would otherwise inherit from `parent_ctx`. Building `click.Context(...)`
    # directly, as here, skips that merge -- unlike a real invocation, which always goes through
    # `make_context` -- so without it, even the root page loses `-h`, not only its children.
    ctx = click.Context(command, info_name=path, parent=parent_ctx, **command.context_settings)
    heading = "#" * min(depth + 2, 6)
    lines.append(f"{heading} {heading_override or f'`{path}`'}")
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
        default_name = _DEFAULT_DISPATCH.get(path)
        for name in sorted(command.list_commands(ctx)):
            sub = command.get_command(ctx, name)
            if sub is None:
                continue
            sub_path = f"{path} {name}"
            if sub.hidden and name == default_name:
                # Document it as `path`'s own default form (its usage/options are still `sub`'s),
                # not as a `path name` subcommand nobody types.
                _walk(
                    sub,
                    path,
                    depth + 1,
                    lines,
                    parent_ctx=ctx,
                    heading_override=f"`{path}` (default: no subcommand name)",
                )
                continue
            if sub.hidden and sub_path in _HIDDEN_DENYLIST:
                continue
            _walk(sub, sub_path, depth + 1, lines, parent_ctx=ctx)


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
