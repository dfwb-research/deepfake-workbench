"""Committed code, tests and docs never cite the private planning material.

This package is published; the documents that drove its design are not. A decision id, a task
number, a milestone reference, or a path into that private material would either mean nothing to
an outside reader or, worse, describe unpublished work. Every rationale that used to be a citation
must instead be spelled out in plain words at the point it is needed.

Every git-tracked file is scanned -- not just the source and test trees -- because packaging and
citation metadata (``pyproject.toml``, ``CITATION.cff``, the pre-commit config, ...) is published
too; only generated lockfiles and binary/compressed files are excluded, by an explicit list below,
and a file that fails to decode as UTF-8 is skipped rather than erroring.

Patterns are deliberately narrow so ordinary prose survives: a decision id needs a digit directly
against its letter, with nothing (not even an equals sign) in between, and a milestone id needs its
own word boundaries so it cannot match inside a longer identifier. A single letter immediately
followed by a digit is common enough elsewhere (a norm's name, for instance) that a genuine clash
would be handled through the allowlist below rather than by loosening the pattern.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Generated, not prose: it churns on every dependency bump and its content is not anyone's words.
_EXCLUDED_NAMES = {"uv.lock"}

# Binary or compressed: not text, and may not even decode as UTF-8 (belt and suspenders with the
# decode-failure skip below, which also covers any such file this list misses).
_EXCLUDED_SUFFIXES = {".gz", ".png", ".jpg", ".jpeg", ".mkv", ".mp4", ".safetensors"}

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:J|K|L|P|DV)\d{1,2}\b"),
    re.compile(r"\bT[1-6]\b"),
    re.compile(r"\bTask \d+\b"),
    re.compile(r"\b(?i:the|task) brief\b"),
    re.compile(r"\breview focus\b"),
    re.compile(r"\bpackage plan\b"),
    re.compile(r"\bimpl/M\d"),
    re.compile(r"\bplan/"),
    re.compile(r"\bPROGRESS\.md\b"),
    re.compile(r"\bM[0-7]\b"),
    # A bare mention of this one filename is as much a citation as a directory path would be: it
    # names a private per-package document by its own filename rather than by directory. But a
    # *path-qualified* mention (something/protocols.md) is not this citation -- a later task adds
    # a public docs page at that same bare filename under docs/, and that page must be citable by
    # its own path. The negative lookbehind only blocks a match when the filename is unqualified.
    re.compile(r"(?<![/\w])protocols\.md\b"),
    # The unpublished research these packages grew out of is not described here, not even by
    # naming the kind of document it is; a regime, profile or fixture is described by what it does.
    re.compile(r"\b[Tt][Hh][Ee][Ss][Ii][Ss]\b"),  # any casing, without spelling the word here
)

# (relative path, pattern index) pairs known to be false positives, kept explicit so a new one
# cannot be added silently.
_ALLOWLIST: frozenset[tuple[str, int]] = frozenset(
    {
        # The hero SVGs are copied byte-identical from the organisation's own generator (never
        # edited here); their embedded woff2 fonts are base64, and two of those characters -- the
        # letter this pattern's third alternative is, immediately followed by a single digit --
        # happen to fall on a word boundary in that data in both files. A coincidence of the font
        # bytes, not a decision id, and not something editing the SVG (which must stay
        # byte-identical) could fix.
        ("docs/assets/hero-dark.svg", 0),
        ("docs/assets/hero-light.svg", 0),
    }
)


def _scanned_files() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    paths = []
    for line in listing.splitlines():
        if not line:
            continue
        path = Path(line)
        if path.name in _EXCLUDED_NAMES or path.suffix in _EXCLUDED_SUFFIXES:
            continue
        paths.append(REPO / line)
    return paths


def test_no_references_to_the_planning_material():
    offenders = []
    for path in _scanned_files():
        try:
            text = path.read_text("utf-8")
        except UnicodeDecodeError:
            continue  # a binary fixture, not prose
        rel = str(path.relative_to(REPO))
        for lineno, line in enumerate(text.splitlines(), start=1):
            for index, pattern in enumerate(_PATTERNS):
                if (rel, index) in _ALLOWLIST:
                    continue
                if pattern.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
                    break
    assert offenders == []


def test_the_check_catches_the_patterns_it_targets():
    # Built at runtime, by concatenation, so this file's own source does not trip the check above.
    decision_id = "J" + "10"
    t_series = "T" + "3"
    task_number = "Task" + " 7"
    brief = "the" + " brief"
    reviewer_phrase = "review" + " focus"
    plan_doc = "package" + " plan"
    per_milestone_path = "impl" + "/M" + "2"
    milestone = "M" + "2"
    progress = "PROGRESS" + ".md"
    plan_path = "plan" + "/06-migration-map.md"
    protocols_bare = "per " + "protocols" + ".md"

    assert any(p.search(decision_id) for p in _PATTERNS)
    assert any(p.search(t_series) for p in _PATTERNS)
    assert any(p.search(task_number) for p in _PATTERNS)
    assert any(p.search(brief) for p in _PATTERNS)
    assert any(p.search(reviewer_phrase) for p in _PATTERNS)
    assert any(p.search(plan_doc) for p in _PATTERNS)
    assert any(p.search(per_milestone_path) for p in _PATTERNS)
    assert any(p.search(milestone) for p in _PATTERNS)
    assert any(p.search(progress) for p in _PATTERNS)
    assert any(p.search(plan_path) for p in _PATTERNS)
    assert any(p.search(protocols_bare) for p in _PATTERNS)

    # And it must not fire on the neighbouring, legitimate text the patterns are tuned to spare:
    # an assignment (not a decision id), and a path-qualified mention of the one filename that a
    # later task's public docs page is expected to share.
    protocols_qualified = "see docs/concepts/" + "protocols" + ".md"
    assert not any(p.search("T=1") for p in _PATTERNS)
    assert not any(p.search(protocols_qualified) for p in _PATTERNS)
