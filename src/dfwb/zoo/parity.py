"""The parity harness: compares numbers DFWB has reproduced against an adapter card's own
``reported`` claims, and writes the result to a local overlay file
(``$DFWB_CACHE_ROOT/zoo/<name>/parity.json``) shaped exactly like the card's own ``parity:`` list,
so its entries can be pasted straight into the card in a PR.

Actually running the adapter over its parity set and computing the metric is the CLI's job (that
needs ``dfwb.score`` and ``dfwb.eval``, both off limits to this layer): this module only compares
already-computed values against the card's claims and reads/writes the overlay file.
"""

from __future__ import annotations

import datetime
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dfwb import __version__
from dfwb.core.errors import ContractError
from dfwb.core.paths import require_root, resolve_roots
from dfwb.zoo.card import AdapterCard, ParityMetric

__all__ = [
    "ParityCheck",
    "compare_parity",
    "parity_path",
    "read_parity_overlay",
    "write_parity_overlay",
]

MetricKey = tuple[str, str, str]  # (protocol, split, metric)


@dataclass(frozen=True)
class ParityCheck:
    """One reproduced number, checked against what the card's ``reported`` claims for it."""

    protocol: str
    split: str
    metric: str
    reported: float
    measured: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return abs(self.measured - self.reported) <= self.tolerance

    def as_metric(
        self, *, dfwb_version: str | None = None, date: str | None = None
    ) -> ParityMetric:
        """This check, in the shape the card's own ``parity:`` list uses."""
        return ParityMetric(
            protocol=self.protocol,
            split=self.split,
            metric=self.metric,
            value=self.measured,
            tolerance=self.tolerance,
            dfwb_version=dfwb_version or __version__,
            date=date or datetime.datetime.now(datetime.UTC).date().isoformat(),
        )


def compare_parity(
    card: AdapterCard, measured: Mapping[MetricKey, float], *, tolerance: float = 0.01
) -> list[ParityCheck]:
    """One :class:`ParityCheck` per entry of ``card.reported``, looking up its measured value.

    Raises:
        ContractError: ``measured`` has no value for one of ``card.reported``'s
            ``(protocol, split, metric)``.
    """
    checks: list[ParityCheck] = []
    for claim in card.reported:
        key: MetricKey = (claim.protocol, claim.split, claim.metric)
        if key not in measured:
            raise ContractError(
                f"zoo:{card.name}: no measured value for {claim.protocol} {claim.split} "
                f"{claim.metric}",
                hint="score and evaluate the adapter's parity set for every reported metric first",
            )
        checks.append(
            ParityCheck(
                protocol=claim.protocol,
                split=claim.split,
                metric=claim.metric,
                reported=claim.value,
                measured=measured[key],
                tolerance=tolerance,
            )
        )
    return checks


def parity_path(name: str) -> Path:
    """``$DFWB_CACHE_ROOT/zoo/<name>/parity.json``."""
    roots = resolve_roots()
    cache_root = require_root("cache", roots)
    return cache_root / "zoo" / name / "parity.json"


def write_parity_overlay(name: str, metrics: Sequence[ParityMetric]) -> Path:
    """Write ``metrics`` (typically from :meth:`ParityCheck.as_metric`) to the local overlay
    file, written atomically. Overwrites whatever the previous run left there."""
    path = parity_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"parity": [metric.model_dump(mode="json") for metric in metrics]}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_parity_overlay(name: str) -> list[ParityMetric]:
    """The overlay file's own ``parity:`` entries, or ``[]`` if none has been written yet."""
    path = parity_path(name)
    if not path.is_file():
        return []
    data = json.loads(path.read_text("utf-8"))
    return [ParityMetric.model_validate(entry) for entry in data.get("parity", [])]
