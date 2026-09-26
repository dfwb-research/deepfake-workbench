"""A fixture zoo adapter for ``dfwb.zoo.source`` tests: unlike ``chance``/``random``, it declares
weight variants, so ``load_zoo`` actually exercises weight resolution, fetching and
``checkpoint_sha256`` -- without shipping any real third-party adapter. Its own card is set per
test (``monkeypatch.setattr(WeightedTestAdapter, "card", ...)``), since it needs a test's own
local HTTP server's port baked into the weight URLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from dfwb.zoo.adapter import meta_from_card
from dfwb.zoo.card import AdapterCard

__all__ = ["WeightedTestAdapter", "WeightedTestDetector"]


class WeightedTestDetector:
    def __init__(self, meta: Any, *, weights_path: Path | None) -> None:
        self.meta = meta
        self.weights_path = weights_path

    def to(self, device: Any) -> WeightedTestDetector:
        return self

    def predict(self, batch: Any) -> Any:  # pragma: no cover - not exercised by these tests
        raise NotImplementedError


class WeightedTestAdapter:
    card: ClassVar[AdapterCard]

    def load(
        self, weights: Path | None, device: str, *, seed: int | None = None
    ) -> WeightedTestDetector:
        meta = meta_from_card(self.card, source=f"zoo:{self.card.name}")
        return WeightedTestDetector(meta, weights_path=weights)
