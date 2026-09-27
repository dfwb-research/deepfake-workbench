"""Breakdowns of C5 score rows by method, compression, label key or method family.

``method`` and ``compression`` are plain row columns; ``label_key`` groups rows by their raw
per-method label (the same thing ``family`` groups more coarsely). ``family`` is not stored on the
row at all -- it comes from the protocol pack's ``labels.yaml``, via the same ``family`` mapping
:meth:`~dfwb.protocols.protocol.Protocol.labels` already exposes for training code, so a breakdown
never needs its own copy of that vocabulary.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Final

from dfwb.core.errors import ConfigError, ContractError, did_you_mean
from dfwb.core.records import ScoreMeta, ScoreRow
from dfwb.protocols.protocol import load as load_protocol

__all__ = ["BY_FIELDS", "group_rows"]

#: The breakdown dimensions ``dfwb eval --by`` accepts.
BY_FIELDS: Final = ("method", "compression", "label_key", "family")

_ROW_FIELDS = frozenset({"method", "compression", "label_key"})


def _family_of_row(row: ScoreRow, meta: ScoreMeta | None) -> str | None:
    if row.label_key is None:
        return None
    if meta is None:
        raise ContractError(
            "breakdown by 'family' needs the score file's .meta.json (for its protocol)",
            hint="keep .meta.json alongside .scores.csv, or break down by label_key instead",
        )
    protocol = load_protocol(meta.protocol.id)
    family = protocol.labels("family")(row.label_key)
    return str(family)


def group_rows(
    rows: Sequence[ScoreRow], by: str, *, meta: ScoreMeta | None = None
) -> dict[str, list[ScoreRow]]:
    """Group ``rows`` by ``by``, dropping rows that carry no value for it.

    ``by`` is one of :data:`BY_FIELDS`: ``"method"``, ``"compression"`` and ``"label_key"`` read
    the row's own column of that name; ``"family"`` looks each row's ``label_key`` up in its
    protocol's ``labels.yaml`` (see :func:`~dfwb.protocols.protocol.load`), so it needs ``meta``.

    Raises:
        ConfigError: ``by`` is not one of :data:`BY_FIELDS`.
        ContractError: ``by="family"`` and ``meta`` is ``None``.
    """
    if by not in BY_FIELDS:
        raise ConfigError(
            f"unknown breakdown {by!r}{did_you_mean(by, BY_FIELDS)}",
            hint="breakdowns: " + ", ".join(BY_FIELDS),
        )
    groups: dict[str, list[ScoreRow]] = defaultdict(list)
    for row in rows:
        value = _family_of_row(row, meta) if by == "family" else getattr(row, by)
        if value is not None:
            groups[str(value)].append(row)
    return dict(groups)
