"""What the docs say about the release: its status lines, and the framework a pack needs.

While ``__version__`` is a pre-release, README.md (also the package's PyPI description),
docs/index.md and docs/install.md say so, and that the package is not on PyPI yet. The commit
that bumps the version to a final release rewrites those lines too; the test below runs, and
fails until it has, on that commit's pull request (a final version is one with no a, b, rc or
dev segment). For any other version it is skipped.

Each page marks its status text with ``<!-- release-status:start -->`` /
``<!-- release-status:end -->`` comments, and the check reads only what is between them. That
keeps prose elsewhere on the page -- such as this repository's own explanation of the check, which
names the very phrases it looks for -- from tripping a false positive.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.version import Version

import dfwb._version

REPO = Path(__file__).resolve().parents[3]
STATUS_PAGES = ("README.md", "docs/index.md", "docs/install.md")

# Anything that tells a reader the package is not released yet.
_UNRELEASED = re.compile(
    r"\bpre-release\b|\bunreleased\b|\bin development\b|\bon PyPI yet\b"
    r"|\bnot (?:yet )?(?:published|released) on PyPI\b"
    r"|\bnothing is (?:yet )?(?:published|released) on PyPI\b|\bnot on PyPI\b",
    re.IGNORECASE,
)

_STATUS_BLOCK = re.compile(
    r"<!-- release-status:start -->(.*?)<!-- release-status:end -->", re.DOTALL
)


def status_block(text: str) -> str:
    """The text between a page's ``release-status`` markers: the only part the check reads."""
    match = _STATUS_BLOCK.search(text)
    assert match, "no release-status:start/end markers found"
    return match.group(1)


# The one statement of the framework a protocol pack with the recipe hash fields needs.
FRAMEWORK_REQUIREMENT = (
    "a Deepfake Workbench newer than 0.1.0b2, one that reads the recipe hash fields"
)


def is_final(version: str) -> bool:
    return not Version(version).is_prerelease


def unreleased_claims(text: str) -> list[str]:
    return [match.group(0) for match in _UNRELEASED.finditer(" ".join(text.split()))]


@pytest.mark.parametrize(
    ("version", "final"),
    [
        ("0.1.0", True),
        ("1.2.3.post1", True),
        ("0.1.0b2", False),
        ("0.1.0b3.dev0", False),
        ("0.1.0rc1", False),
        ("0.2.0a1", False),
        ("0.2.0.dev1", False),
    ],
)
def test_which_versions_are_final(version, final):
    assert is_final(version) is final


@pytest.mark.parametrize(
    "line",
    [
        "> **Status:** pre-release (`0.1.0b3.dev0`, in development after `0.1.0b2`). Nothing is\n"
        "> released on PyPI yet; install from a clone.",
        '!!! note "Pre-release"\n    `dfwb` is at `0.1.0b3.dev0`. Nothing is published on PyPI '
        "yet -- install from a clone",
        '!!! note "Not on PyPI yet"\n    dfwb is not published on PyPI: install from a clone',
    ],
    ids=["readme", "index", "install"],
)
def test_the_pre_release_status_lines_are_what_the_check_catches(line):
    assert unreleased_claims(line)


def test_a_final_release_does_not_call_itself_unreleased():
    version = dfwb._version.__version__
    if not is_final(version):
        pytest.skip(f"{version} is a pre-release, so its status lines may say so")
    for page in STATUS_PAGES:
        block = status_block((REPO / page).read_text("utf-8"))
        claims = unreleased_claims(block)
        assert not claims, (
            f"{page}'s release-status block still says {claims} but the version is {version}: "
            "rewrite its status lines as part of the version bump"
        )


@pytest.mark.parametrize("page", STATUS_PAGES)
def test_every_status_page_marks_its_release_status_block(page):
    # Runs at every version, not only a final one, so the markers can never quietly go missing.
    status_block((REPO / page).read_text("utf-8"))


def test_the_release_status_block_ignores_explanatory_prose_elsewhere():
    text = (
        "# Deepfake Workbench\n\n"
        "<!-- release-status:start -->\n"
        "> **Status:** 0.1.0. Linux is the only supported and tested OS. Python >= 3.12.\n"
        "<!-- release-status:end -->\n\n"
        "**Releasing.** ... also rewrites the status lines that describe a pre-release, and the "
        '"Not on PyPI yet" note in docs/install.md.\n'
    )
    # The page as a whole does say "pre-release" and "Not on PyPI yet" -- in the explanatory
    # paragraph, not the status block -- so an unscoped check would wrongly flag it.
    assert unreleased_claims(text)
    assert not unreleased_claims(status_block(text))


@pytest.mark.parametrize(
    "path",
    ["README.md", "CHANGELOG.md", "src/dfwb/protocols/_newpack/README.md.tmpl"],
)
def test_the_framework_a_recipe_card_needs_is_stated_one_way(path):
    text = " ".join((REPO / path).read_text("utf-8").split())
    assert FRAMEWORK_REQUIREMENT in text
    assert "0.1.0b2 or later" not in text
    assert "0.1.0b3 or later" not in text
