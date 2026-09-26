"""The ``chance`` dummy adapter: always predicts ``P(fake) = 0.5``.

No weights, no learned behaviour: a sanity floor that exercises the whole scoring pipeline (card,
detector source, harness) without needing any real model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dfwb.core.detector import DetectorOutput
from dfwb.zoo.adapter import meta_from_card, read_builtin_card

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["ChanceAdapter", "ChanceDetector"]

_NAME = "chance"
_SCORE = 0.5


class ChanceDetector:
    """Always predicts 0.5. Contract C4's ``Detector``: no state, no weights."""

    def __init__(self, meta: Any) -> None:
        self.meta = meta

    def to(self, device: Any) -> ChanceDetector:
        return self

    def predict(self, batch: Any) -> DetectorOutput:
        import torch

        return DetectorOutput(score=torch.full((len(batch.keys),), _SCORE))


class ChanceAdapter:
    """Builds :class:`ChanceDetector` from the ``chance`` card."""

    card = read_builtin_card(_NAME)

    def load(
        self,
        weights: Path | None,
        device: str,
        *,
        seed: int | None = None,
        code_root: Path | None = None,
    ) -> ChanceDetector:
        return ChanceDetector(meta_from_card(self.card, source=f"zoo:{self.card.name}"))
