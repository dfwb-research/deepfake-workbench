"""Pack versions: ``MAJOR.MINOR.PATCH`` with PEP 440 pre-, post- and dev-release suffixes.

A pack is versioned like a Python distribution, so ``0.1.0a2``, ``1.0.0rc1``, ``1.0.0.post1``
and ``1.0.0.dev3`` are all pack versions, parsed and ordered here exactly as PEP 440 orders them
(``1.0.0.dev1 < 1.0.0a1 < 1.0.0b1 < 1.0.0rc1 < 1.0.0 < 1.0.0.post1``), with the standard library
only. The alternative spellings and separators PEP 440 normalises (``1.0.0-rc.1``,
``1.0.0alpha2``, ``1.0.0-1``) are accepted and compare equal to their normal form. The release is
always three numbers; epochs (``1!``) and local labels (``+abc``) are not pack versions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = ["VERSION_PATTERN", "PackVersion", "parse_version"]

_SEP = r"[-_.]?"
_PRE_WORDS = r"alpha|beta|preview|pre|rc|a|b|c"
_POST_WORDS = r"post|rev|r"

# The grammar as one case-insensitive regex with no groups, to embed in other grammars.
VERSION_PATTERN: Final = (
    rf"(?i:\d+\.\d+\.\d+"
    rf"(?:{_SEP}(?:{_PRE_WORDS}){_SEP}\d*)?"
    rf"(?:-\d+|{_SEP}(?:{_POST_WORDS}){_SEP}\d*)?"
    rf"(?:{_SEP}dev{_SEP}\d*)?)"
)

_PARSE = re.compile(
    rf"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    rf"(?:{_SEP}(?P<pre_l>{_PRE_WORDS}){_SEP}(?P<pre_n>\d*))?"
    rf"(?:-(?P<post_implicit>\d+)|{_SEP}(?P<post_l>{_POST_WORDS}){_SEP}(?P<post_n>\d*))?"
    rf"(?:{_SEP}(?P<dev_l>dev){_SEP}(?P<dev_n>\d*))?$",
    re.IGNORECASE,
)

# Each pre-release spelling's phase, in PEP 440 order: 0 is a, 1 is b, 2 is rc.
_PHASE_OF: Final = {
    "a": 0,
    "alpha": 0,
    "b": 1,
    "beta": 1,
    "c": 2,
    "rc": 2,
    "pre": 2,
    "preview": 2,
}


@dataclass(frozen=True)
class PackVersion:
    """A parsed pack version. Compare versions with :attr:`sort_key` (PEP 440 order)."""

    release: tuple[int, int, int]
    pre: tuple[int, int] | None = None  # (phase: 0 a, 1 b, 2 rc; number)
    post: int | None = None
    dev: int | None = None

    @property
    def sort_key(self) -> tuple[tuple[int, int, int], tuple[int, int, int], int, tuple[int, int]]:
        """PEP 440 order: dev < pre-releases < the release < post-releases, dev before each."""
        if self.pre is not None:
            pre = (0, *self.pre)
        elif self.dev is not None and self.post is None:
            pre = (-1, 0, 0)  # 1.0.0.dev1 sorts before 1.0.0a1
        else:
            pre = (1, 0, 0)
        post = -1 if self.post is None else self.post
        dev = (1, 0) if self.dev is None else (0, self.dev)
        return (self.release, pre, post, dev)


def _number(text: str | None) -> int:
    return int(text) if text else 0


def parse_version(text: str) -> PackVersion | None:
    """``text`` as a :class:`PackVersion`, or ``None`` when it is not a pack version."""
    match = _PARSE.match(text)
    if match is None:
        return None
    release = (int(match["major"]), int(match["minor"]), int(match["patch"]))
    pre = None
    if match["pre_l"] is not None:
        pre = (_PHASE_OF[match["pre_l"].lower()], _number(match["pre_n"]))
    post = None
    if match["post_implicit"] is not None:
        post = int(match["post_implicit"])
    elif match["post_l"] is not None:
        post = _number(match["post_n"])
    dev = _number(match["dev_n"]) if match["dev_l"] is not None else None
    return PackVersion(release, pre, post, dev)
