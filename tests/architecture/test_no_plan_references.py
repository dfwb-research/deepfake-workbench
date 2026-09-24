"""Committed code, tests and docs never cite the private planning material.

This package is published; the documents that drove its design are not. A decision id, a task
number, a milestone reference, or a path into that private material would either mean nothing to
an outside reader or, worse, describe unpublished work. Every rationale that used to be a citation
must instead be spelled out in plain words at the point it is needed.

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

_SCANNED_TOP_LEVEL = {"src", "tests", "docs", ".github", "scripts"}
_SCANNED_NAMES = {"README.md", "CHANGELOG.md"}

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
    # names a private per-package document by its own filename rather than by directory.
    re.compile(r"\bprotocols\.md\b"),
)

# (relative path, pattern index) pairs known to be false positives, kept explicit so a new one
# cannot be added silently. Empty: nothing in this tree has needed one so far.
_ALLOWLIST: frozenset[tuple[str, int]] = frozenset()


def _scanned_files() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    paths = []
    for line in listing.splitlines():
        if not line:
            continue
        top = line.split("/", 1)[0]
        if top in _SCANNED_TOP_LEVEL or line in _SCANNED_NAMES:
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
    milestone = "M" + "2"
    progress = "PROGRESS" + ".md"
    plan_path = "plan" + "/06-migration-map.md"
    protocols_doc = "protocols" + ".md"

    assert any(p.search(decision_id) for p in _PATTERNS)
    assert any(p.search(t_series) for p in _PATTERNS)
    assert any(p.search(task_number) for p in _PATTERNS)
    assert any(p.search(brief) for p in _PATTERNS)
    assert any(p.search(milestone) for p in _PATTERNS)
    assert any(p.search(progress) for p in _PATTERNS)
    assert any(p.search(plan_path) for p in _PATTERNS)
    assert any(p.search(protocols_doc) for p in _PATTERNS)

    # And it must not fire on the neighbouring, legitimate text the T-series pattern is tuned to
    # spare (an assignment, not a decision id).
    assert not any(p.search("T=1") for p in _PATTERNS)
