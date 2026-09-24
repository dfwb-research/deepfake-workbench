import pytest

from dfwb.core.envfile import apply_env_file, find_env_file, parse_env_file
from dfwb.core.errors import ConfigError


def write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text)
    return path


def test_parse_the_supported_grammar(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME_BASE", "/h")
    path = write(
        tmp_path,
        "# comment\n\nexport A=1\nB = spaced value  # trailing\nC='lit ${A} #x'\n"
        'D="two\\nlines \\"q\\""\nE=${A}/x:${HOME_BASE}/y\n',
    )
    assert parse_env_file(path) == {
        "A": "1",
        "B": "spaced value",
        "C": "lit ${A} #x",
        "D": 'two\nlines "q"',
        "E": "1/x:/h/y",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("NOEQUALS\n", ".env:1: expected KEY=VALUE"),
        ("1BAD=x\n", ".env:1: expected KEY=VALUE"),
        ("A=${NOPE}\n", ".env:1: unknown variable 'NOPE'"),
        ('A="open\n', ".env:1: unterminated quote"),
    ],
)
def test_parse_errors_name_the_line(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        parse_env_file(write(tmp_path, text))


def test_the_real_environment_wins(tmp_path):
    environ = {"A": "from-shell"}
    applied = apply_env_file(write(tmp_path, "A=file\nB=file\n"), environ)
    assert environ == {"A": "from-shell", "B": "file"}
    assert applied.applied == ("B",)
    assert applied.skipped == ("A",)


def test_find_env_file(tmp_path):
    assert find_env_file(tmp_path, {}) is None
    write(tmp_path, "A=1\n")
    assert find_env_file(tmp_path, {}) == tmp_path / ".env"
    other = tmp_path / "hades.env"
    other.write_text("A=2\n")
    assert find_env_file(tmp_path, {"DFWB_ENV_FILE": str(other)}) == other
    with pytest.raises(ConfigError, match="DFWB_ENV_FILE"):
        find_env_file(tmp_path, {"DFWB_ENV_FILE": str(tmp_path / "missing.env")})
