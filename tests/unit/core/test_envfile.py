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
        ("A='value'x\n", ".env:1: unexpected text after the closing quote"),
        ('A="value" # note\n', ".env:1: unexpected text after the closing quote"),
    ],
)
def test_parse_errors_name_the_line(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        parse_env_file(write(tmp_path, text))


def test_trailing_whitespace_after_a_closing_quote_is_fine(tmp_path):
    assert parse_env_file(write(tmp_path, "A='value'   \n")) == {"A": "value"}


def test_a_leading_byte_order_mark_does_not_break_line_1(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"\xef\xbb\xbfA=1\nB=2\n")
    assert parse_env_file(path) == {"A": "1", "B": "2"}


def test_the_real_environment_wins(tmp_path):
    environ = {"A": "from-shell"}
    applied = apply_env_file(write(tmp_path, "A=file\nB=file\n"), environ)
    assert environ == {"A": "from-shell", "B": "file"}
    assert applied.applied == ("B",)
    assert applied.skipped == ("A",)


def test_expansion_sees_the_real_environment_first(tmp_path):
    # A key the shell sets wins over the file's value, and ${KEY} expands to the shell's value
    # too, so the file never mixes the two.
    environ = {"BASE": "/shell"}
    path = write(tmp_path, 'BASE=/file\nDERIVED=${BASE}/x\nQUOTED="${BASE}/y"\n')
    apply_env_file(path, environ)
    assert environ == {"BASE": "/shell", "DERIVED": "/shell/x", "QUOTED": "/shell/y"}


def test_expansion_falls_back_to_earlier_keys_of_the_file(tmp_path):
    environ: dict[str, str] = {}
    apply_env_file(write(tmp_path, "BASE=/file\nDERIVED=${BASE}/x\n"), environ)
    assert environ == {"BASE": "/file", "DERIVED": "/file/x"}


def test_find_env_file(tmp_path):
    assert find_env_file(tmp_path, {}) is None
    write(tmp_path, "A=1\n")
    assert find_env_file(tmp_path, {}) == tmp_path / ".env"
    other = tmp_path / "gpu-node-1.env"
    other.write_text("A=2\n")
    assert find_env_file(tmp_path, {"DFWB_ENV_FILE": str(other)}) == other
    with pytest.raises(ConfigError, match="DFWB_ENV_FILE"):
        find_env_file(tmp_path, {"DFWB_ENV_FILE": str(tmp_path / "missing.env")})
