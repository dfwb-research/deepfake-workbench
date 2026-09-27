import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dfwb.core.log import get_logger, setup_logging
from dfwb.core.paths import resolve_roots
from dfwb.core.runmeta import capture_git, collect_run_info, sanitize_command, utc_now
from dfwb.core.seed import seed_everything


def test_seed_everything_is_reproducible(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    seed_everything(7)
    first = (random.random(), np.random.rand())  # noqa: NPY002 - seed_everything seeds this
    seed_everything(7)
    assert (random.random(), np.random.rand()) == first  # noqa: NPY002


def test_seed_everything_skips_missing_torch():
    # A fresh interpreter, so "torch" starts absent from sys.modules regardless of whatever
    # other, unrelated tests in this worker process have legitimately imported by now (the
    # models layer imports torch at module level; see dfwb.models.backbone).
    code = (
        "import sys\n"
        "from dfwb.core import seed\n"
        "seed._installed = lambda name: name == 'numpy'\n"
        "seed.seed_everything(1, deterministic=True)\n"  # no torch: nothing to do, no error
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_setup_logging_is_idempotent_and_leaves_root_alone(capsys):
    root_handlers = list(logging.getLogger().handlers)
    setup_logging("DEBUG", use_rich=False)
    logger = setup_logging("INFO", use_rich=False)
    assert len([h for h in logger.handlers if getattr(h, "_dfwb_handler", False)]) == 1
    assert logging.getLogger().handlers == root_handlers
    get_logger("train").info("hello")
    get_logger("train").debug("hidden")
    err = capsys.readouterr().err
    assert "dfwb.train: hello" in err
    assert "hidden" not in err
    assert get_logger("dfwb.core").name == "dfwb.core"


def test_setup_logging_can_use_rich():
    pytest.importorskip("rich")
    handler = setup_logging(use_rich=True).handlers[-1]
    assert type(handler).__name__ == "RichHandler"


def test_utc_now_format():
    stamp = utc_now()
    assert len(stamp) == 20
    assert stamp.endswith("Z")


def test_capture_env_does_not_import_torch():
    # A fresh interpreter: see test_seed_everything_skips_missing_torch for why this can't check
    # sys.modules in-process once anything else in the suite has legitimately imported torch.
    code = (
        "import sys\n"
        "from dfwb.core.runmeta import capture_env\n"
        "env = capture_env()\n"
        "assert env['python'].count('.') == 2\n"
        "assert env['dfwb']\n"
        "assert 'torch' not in sys.modules\n"
        "assert env['device'] is None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_capture_git(tmp_path):
    assert capture_git(tmp_path) is None
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.org",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    state = capture_git(tmp_path)
    assert state is not None
    assert len(state.commit) == 40
    assert state.dirty is False


def test_sanitize_command_removes_absolute_paths(tmp_path):
    roots = resolve_roots(
        env={"DFWB_RUNS_ROOT": "/r", "DFWB_DATASETS_ROOT": "/d"},
        cwd=tmp_path,
        user_config=tmp_path / "none.toml",
    )
    argv = [
        "/venv/bin/dfwb",
        "score",
        "--detector",
        "run:/r/vit/2026",
        "--out=/d/x",
        "/elsewhere/cfg.yaml",
        "optim.lr=3e-4",
        "https://example.org/a",
        "/r",
    ]
    assert sanitize_command(argv, roots).split(" ") == [
        "dfwb",
        "score",
        "--detector",
        "'run:$DFWB_RUNS_ROOT/vit/2026'",
        "'--out=$DFWB_DATASETS_ROOT/x'",
        "'<abs>/cfg.yaml'",
        "optim.lr=3e-4",
        "https://example.org/a",
        "'$DFWB_RUNS_ROOT'",
    ]
    assert sanitize_command([], roots) == ""


def test_collect_run_info(tmp_path):
    info = collect_run_info(
        ["dfwb", "doctor"],
        cwd=tmp_path,
        roots=resolve_roots(env={}, cwd=tmp_path, user_config=tmp_path / "x"),
    )
    assert info.command == "dfwb doctor"
    assert info.git is None
    assert "dfwb" in info.plugins  # the built-ins entry point is installed with the package


# Tokens the whole-token rules alone left with an absolute path inside them, so a record holding
# the command (a score file's meta, say) failed the absolute-path guard when it was written.
_EMBEDDED_PATH_TOKENS = [
    "note=a /elsewhere/x",  # a path after a space, inside one argument
    "a,/elsewhere/x",  # after a comma
    "a;/elsewhere/x",  # after a semicolon
    "(/elsewhere/x",  # after a parenthesis
    "file:///elsewhere/x",  # a file URL
    "file:////elsewhere/x",  # a file URL with an extra slash
    "a/b=/elsewhere/x",  # key=value whose key holds a slash
    ":/elsewhere/x",  # a bare colon prefix
    "1:/elsewhere/x",  # a prefix that does not start with a letter
    "x'/elsewhere/y",  # after a quote
    "k=v:/elsewhere/x,/r/y",  # several, one of them under a root
]


@pytest.mark.parametrize("token", _EMBEDDED_PATH_TOKENS)
def test_sanitize_command_output_always_passes_the_absolute_path_guard(tmp_path, token):
    from dfwb.core.records import assert_no_absolute_paths

    roots = resolve_roots(env={"DFWB_RUNS_ROOT": "/r"}, cwd=tmp_path, user_config=tmp_path / "x")

    command = sanitize_command(["/venv/bin/dfwb", "score", token], roots)

    assert_no_absolute_paths(command, where="command")
    assert "elsewhere" not in command  # only a path's final component survives


def test_sanitize_command_keeps_a_root_relative_path_inside_a_token(tmp_path):
    roots = resolve_roots(env={"DFWB_RUNS_ROOT": "/r"}, cwd=tmp_path, user_config=tmp_path / "x")

    command = sanitize_command(["dfwb", "x", "note=see /r/vit/2026"], roots)

    assert command == "dfwb x 'note=see $DFWB_RUNS_ROOT/vit/2026'"


@settings(max_examples=300, deadline=None)
@given(
    st.lists(
        st.text(alphabet=st.sampled_from(list("/~:=,;'\"( \tabcfile.x0-_$<>")), max_size=24),
        max_size=6,
    )
)
def test_sanitize_command_and_the_absolute_path_guard_agree_on_any_argv(tokens):
    from dfwb.core.records import assert_no_absolute_paths

    roots = resolve_roots(env={"DFWB_RUNS_ROOT": "/r"}, cwd=Path("/w"), user_config=Path("/nx"))

    assert_no_absolute_paths(sanitize_command(["dfwb", *tokens], roots), where="command")


def test_sanitize_command_cleans_config_overrides(tmp_path):
    roots = resolve_roots(env={"DFWB_RUNS_ROOT": "/r"}, cwd=tmp_path, user_config=tmp_path / "x")
    argv = [
        "dfwb",
        "train",
        "-c",
        "exp.yaml",
        "run.output_root=/r/vit",
        "data.x=/elsewhere/y",
    ]
    assert sanitize_command(argv, roots).split(" ") == [
        "dfwb",
        "train",
        "-c",
        "exp.yaml",
        "'run.output_root=$DFWB_RUNS_ROOT/vit'",
        "'data.x=<abs>/y'",
    ]
