#!/usr/bin/env python3
"""Evaluate score files your own code already produced, with no ``torch`` install at all.

Mirrors the "I only have score files from my own code" journey: a CSV of six scores is imported
against dfwb's own built-in ``toyfake/official`` protocol (shipped inside dfwb itself, so this
needs no dataset, no inventory and no ``protocols verify`` first) with ``dfwb eval import``, then
evaluated with ``dfwb eval``. Needs only the base install plus the ``eval`` extra for confidence
intervals (``uv sync --extra eval``) -- no ``preprocess``, ``train`` or ``zoo`` extra, and
``python -c "import torch"`` never has to succeed for this to work.

``dfwb eval import`` reports the six scored videos honestly against the other 35 videos of
``toyfake/official``'s test split that this file does not cover (``missing`` rows, not silently
dropped from the split) and still exits ``0``; it is the ``dfwb eval`` step after it that checks
coverage against ``--min-coverage`` and exits ``3`` here -- expected for a deliberately partial
example, not a failure.

Writes only under the current directory; never touches a real datasets root, ``~/.cache/dfwb`` or
``~/.local/state/dfwb``.

    python examples/import_foreign_scores.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dfwb.cli.main import main

_MY_SCORES = """\
video_id,prob
REAL/p006,0.10
REAL/p009,0.22
REAL/p015,0.05
BLEND_A/p006_p058,0.81
BLEND_A/p009_p011,0.74
BLEND_A/p015_p013,0.63
"""


def _run(*args: str) -> int:
    print(f"$ dfwb {' '.join(args)}")
    return main(["--no-env-file", *args])


def main_() -> None:
    Path("my_scores.csv").write_text(_MY_SCORES, encoding="utf-8")

    imported = _run(
        "eval",
        "import",
        "my_scores.csv",
        "--protocol",
        "toyfake/official",
        "--split",
        "test",
        "--map",
        "key=video_id,score=prob",
        "--polarity",
        "fake-high",
        "--out",
        "imported",
    )
    if imported != 0:
        raise SystemExit(f"dfwb eval import exited {imported}")

    scores = sorted(Path("imported").glob("*.scores.csv"))
    if not scores:
        raise SystemExit("dfwb eval import wrote no score file")
    evaluated = _run("eval", str(scores[0]), "--bootstrap", "500")
    if evaluated != 3:  # 3: below --min-coverage, expected -- only 6 of 41 videos were scored
        raise SystemExit(f"dfwb eval exited {evaluated}, expected 3 (partial coverage)")


if __name__ == "__main__":
    main_()
    sys.exit(0)
