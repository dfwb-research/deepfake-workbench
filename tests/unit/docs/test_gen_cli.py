"""``docs/gen_cli.py`` documents the real CLI tree, including what ``--help`` cannot show you.

Skipped unless the ``docs`` dependency group is installed (``mkdocs_gen_files``), the same way any
other optional-extra test skips: this script is never run by the standard, torch-free or
torch-installed test jobs, only where the docs group is present (a full dev environment, or a docs
build).
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

mkdocs_gen_files = pytest.importorskip("mkdocs_gen_files")

REPO = Path(__file__).resolve().parents[3]
GEN_CLI = REPO / "docs" / "gen_cli.py"


class _FakeHandle:
    def __init__(self, written: dict[str, str], path: str) -> None:
        self._written = written
        self._path = path
        self._buffer: list[str] = []

    def __enter__(self) -> _FakeHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self._written[self._path] = "".join(self._buffer)

    def write(self, text: str) -> None:
        self._buffer.append(text)


def _generated_text(monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the real generator with ``mkdocs_gen_files.open`` captured in memory, not on disk."""
    written: dict[str, str] = {}
    monkeypatch.setattr(
        mkdocs_gen_files, "open", lambda path, mode="r", *a, **k: _FakeHandle(written, path)
    )
    runpy.run_path(str(GEN_CLI), run_name="gen_cli_under_test")
    assert "reference/cli.md" in written, "the generator did not write reference/cli.md"
    return written["reference/cli.md"]


def test_reference_documents_bare_eval_and_its_options(monkeypatch):
    """``dfwb eval FILES...`` (the hidden ``run`` command ``_EvalGroup`` dispatches bare args to,
    src/dfwb/cli/eval.py) must be documented, with its real options -- not silently dropped for
    being ``hidden=True``.

    A few of ``run``'s option names (``--suite``, ``--metrics``, ``--min-coverage``) also appear
    on ``dfwb score``/``compare``/``zoo`` commands, with different help text, defaults or types --
    each fragment below is the exact row text that can only come from ``run``, not a bare flag name
    that a collision elsewhere in the CLI could satisfy by accident.
    """
    text = _generated_text(monkeypatch)
    assert "dfwb eval run" not in text, "should read as dfwb eval's default form, not a subcommand"
    assert "`dfwb eval` (default: no subcommand name)" in text
    for fragment in (
        # unique outright: no other command in the CLI takes either of these.
        "`--missing TEXT` | exclude, as-real, as-fake or as-chance.",
        "`--format [md|csv|latex|json]` | ",
        # disambiguated from `dfwb score --suite` (different help text).
        "`--suite TEXT` | Name of a registered eval suite.",
        # disambiguated from `dfwb eval compare --metrics` (default `auc`, not `auc,eer`).
        "`--metrics TEXT` | Comma-separated metric specs.  \\[default: auc,eer\\]",
        # disambiguated from `dfwb score`/`dfwb zoo parity --min-coverage` (FloatRange, not float).
        "`--min-coverage FLOAT` |",
        "`--by TEXT` | Break down by method, compression, label_key or family.",
    ):
        assert fragment in text, f"missing bare `dfwb eval`'s own option row: {fragment!r}"


def test_every_generated_command_shows_the_h_alias(monkeypatch):
    """The root group's ``context_settings={"help_option_names": ["-h", "--help"]}`` must reach
    every child context, the way it does for the real, live CLI."""
    text = _generated_text(monkeypatch)
    assert "| `-h, --help`" in text
    assert "| `--help`" not in text, "a child context that lost -h is missing the root's alias"


def test_multi_paragraph_help_renders_as_paragraphs_not_a_code_block(monkeypatch):
    """A command's later docstring paragraphs must render as ordinary text, not an indented code
    block.

    Click stores a command's ``help`` as the raw, un-dedented docstring: only its first line has no
    leading whitespace, every later line keeps its original source indentation (four spaces, from
    the function body). A plain ``.strip()`` only trims the ends of the whole string, so those later
    lines stay four-space indented -- which Markdown renders as a literal code block, not a
    paragraph. ``dfwb train``'s own docstring has exactly this shape.
    """
    text = _generated_text(monkeypatch)
    assert "Every config problem is reported before any data is read." in text
    assert "\n    Every config problem is reported before any data is read." not in text, (
        "a docstring's later paragraphs must be dedented, not left at their source indentation"
    )
    assert "<pre>" not in text
    assert "```" not in text
