import subprocess
import sys

import pytest

from dfwb.cli import main as main_module


def test_help_lists_commands_without_importing_them(run):
    result = run("--help")
    assert result.code == 0
    for name in ("config", "doctor", "plugins", "schema", "completion"):
        assert name in result.out


def test_help_does_not_import_heavy_modules():
    code = (
        "import sys; from dfwb.cli.main import main; main(['--help']); "
        "heavy = ('pydantic', 'yaml', 'numpy', 'torch', 'dfwb.core.plugins'); "
        "print(sorted(m for m in heavy if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip().splitlines()[-1] == "[]"


def test_no_arguments_shows_the_help(run):
    result = run()
    assert result.code == 0
    assert "Commands:" in result.out
    assert result.err == ""


def test_version(run):
    result = run("--version")
    assert result.code == 0
    assert result.out.startswith("dfwb 0.1.0a2\n")


def test_usage_error_prints_hint_and_exits_2(run):
    result = run("nope")
    assert result.code == 2
    assert "error: No such command 'nope'." in result.err
    assert "hint: run `dfwb --help` for usage" in result.err


def test_subcommand_usage_error_hint_names_the_subcommand(run):
    result = run("completion")
    assert result.code == 2
    assert "hint: run `dfwb completion --help` for usage" in result.err


TEST_COMMANDS = {
    "contract-error": "tests.unit.cli._commands:contract_error",
    "crash": "tests.unit.cli._commands:crash",
    "not-a-command": "tests.unit.cli._commands:NOT_A_COMMAND",
}


@pytest.fixture
def test_commands(monkeypatch):
    for name, target in TEST_COMMANDS.items():
        monkeypatch.setitem(main_module.cli.lazy_subcommands, name, (target, "test command"))


def test_dfwb_error_prints_error_and_hint_with_its_exit_code(run, test_commands):
    result = run("contract-error")
    assert result.code == 4
    assert result.err.splitlines() == ["error: bad file", "  detail line", "hint: fix it"]
    assert "Traceback" not in result.err


def test_unexpected_error_hint_and_debug_traceback(run, test_commands, monkeypatch):
    result = run("crash")
    assert result.code == 1
    assert "error: unexpected RuntimeError: kaput" in result.err
    assert "hint: re-run with --debug" in result.err
    assert "Traceback" not in result.err
    assert "Traceback" in run("--debug", "crash").err
    monkeypatch.setenv("DFWB_DEBUG", "1")
    assert "Traceback" in run("crash").err


def test_lazy_entry_that_is_not_a_command(run, test_commands):
    result = run("not-a-command")
    assert result.code == 1
    assert "is not a click command" in result.err


def test_unknown_lazy_name_falls_back_to_click():
    import click

    group = main_module.LazyGroup(lazy_subcommands={})
    assert group.get_command(click.Context(group), "missing") is None


@pytest.mark.parametrize(
    ("shell", "snippet"), [("bash", "bash_source"), ("zsh", "zsh_source"), ("fish", "fish_source")]
)
def test_completion(run, shell, snippet):
    result = run("completion", shell)
    assert result.code == 0
    assert snippet in result.out
    assert '"snippet"' in run("completion", shell, "--json").out


def test_console_script_starts_fast():
    import shutil
    from pathlib import Path

    exe = shutil.which("dfwb") or str(Path(sys.executable).parent / "dfwb")
    best = min(_time(exe) for _ in range(5))
    assert best < 0.3, f"dfwb --help took {best:.3f}s"


def _time(exe):
    import time

    start = time.perf_counter()
    subprocess.run([exe, "--help"], check=True, capture_output=True)
    return time.perf_counter() - start
