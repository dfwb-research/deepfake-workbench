"""Coverage policy over C5 score rows: the four ``--missing`` options and coverage accounting.

Videos that could not be scored are rows with ``status`` ``missing`` or ``error``, never dropped
from the file. This module turns those rows, plus a chosen policy, into the ``(label, score)``
arrays a metric actually consumes, and reports how much of the split was really scored.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from dfwb.core.errors import ConfigError, did_you_mean
from dfwb.core.records import ScoreRow

__all__ = ["MISSING_POLICIES", "Coverage", "coverage_of", "labels_and_scores"]

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

#: The four ``--missing`` policies from the C5 contract, in the order the CLI documents them.
MISSING_POLICIES: Final = ("exclude", "as-real", "as-fake", "as-chance")

# Score substituted for a non-ok row under each fill policy (``exclude`` drops the row instead).
_FILL_SCORE: Final[dict[str, float]] = {"as-real": 0.0, "as-fake": 1.0, "as-chance": 0.5}


@dataclass(frozen=True)
class Coverage:
    """How many of a split's expected rows actually carry a score."""

    expected: int
    ok: int
    missing: int
    error: int

    @property
    def fraction(self) -> float:
        """``ok / expected``; ``1.0`` for an empty file (nothing was expected, nothing missing)."""
        return self.ok / self.expected if self.expected else 1.0

    def meets(self, min_coverage: float) -> bool:
        """Whether this coverage is at or above ``min_coverage``."""
        return self.fraction >= min_coverage

    def as_dict(self) -> dict[str, int | float]:
        """A JSON-friendly mapping: ``expected``, ``ok``, ``missing``, ``error``, ``coverage``."""
        return {
            "expected": self.expected,
            "ok": self.ok,
            "missing": self.missing,
            "error": self.error,
            "coverage": self.fraction,
        }


def coverage_of(rows: Sequence[ScoreRow]) -> Coverage:
    """Count ``rows`` by status: ``expected`` is simply how many there are."""
    counts = Counter(row.status for row in rows)
    return Coverage(len(rows), counts["ok"], counts["missing"], counts["error"])


def labels_and_scores(
    rows: Sequence[ScoreRow], *, missing: str = "exclude"
) -> tuple[IntArray, FloatArray, list[ScoreRow]]:
    """Turn ``rows`` into ``(labels, scores, kept_rows)`` under one ``--missing`` policy.

    - ``"exclude"``: rows whose ``status`` is not ``ok`` are dropped.
    - ``"as-real"``, ``"as-fake"``, ``"as-chance"``: a non-``ok`` row is kept with its recorded
      label and a substituted score of ``0.0``, ``1.0`` or ``0.5`` respectively (P(fake)).

    Every kept row's label is its own recorded ``label`` regardless of status; only the score
    (and, for ``ok`` rows, nothing) is ever substituted. The three arrays/lists share one order:
    ``kept_rows[i]`` is the row ``labels[i]``/``scores[i]`` came from.

    Raises:
        ConfigError: ``missing`` is not one of :data:`MISSING_POLICIES`.
    """
    if missing not in MISSING_POLICIES:
        raise ConfigError(
            f"unknown --missing policy {missing!r}{did_you_mean(missing, MISSING_POLICIES)}",
            hint="use one of " + ", ".join(MISSING_POLICIES),
        )
    fill = _FILL_SCORE.get(missing)
    labels: list[int] = []
    scores: list[float] = []
    kept: list[ScoreRow] = []
    for row in rows:
        if row.status == "ok":
            assert row.score is not None  # invariant of ScoreRow: ok always carries a score
            labels.append(row.label)
            scores.append(row.score)
            kept.append(row)
        elif fill is not None:
            labels.append(row.label)
            scores.append(fill)
            kept.append(row)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        kept,
    )
