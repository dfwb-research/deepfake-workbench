"""The ``random`` dummy adapter: seeded uniform scores, independent of any batch or clip order.

Every score is ``sha256(seed, dataset, key, compression)``, scaled to ``[0, 1)``: a pure function
of the video's own identity and the seed, so it never depends on batching, worker order or how
many clips a video happens to have (every clip of the same video gets the same score). Two
different seeds give two different, but each internally reproducible, sets of scores -- a second
sanity floor next to ``chance``, and the only zoo adapter whose own identity includes a seed.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from dfwb.core.detector import DetectorOutput
from dfwb.zoo.adapter import meta_from_card, read_builtin_card

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["DEFAULT_SEED", "RandomAdapter", "RandomDetector"]

_NAME = "random"
DEFAULT_SEED = 0


def _uniform(seed: int, dataset: str, key: str, compression: str | None) -> float:
    """A value in ``[0, 1)``, deterministic in ``(seed, dataset, key, compression)`` alone."""
    text = f"{seed}|{dataset}|{key}|{compression or ''}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class RandomDetector:
    """Seeded uniform scores. Contract C4's ``Detector``: no weights, one integer of state."""

    def __init__(self, meta: Any, *, seed: int) -> None:
        self.meta = meta
        # Duck-typed by dfwb.score.cache.DetectorIdentity: the seed is part of this detector's
        # own identity, so scoring the same protocol with two different seeds writes two
        # different score files, and each one records the seed it actually used.
        self.training_seed = seed

    def to(self, device: Any) -> RandomDetector:
        return self

    def predict(self, batch: Any) -> DetectorOutput:
        import torch

        values = [
            _uniform(self.training_seed, batch.dataset_ids[i], batch.keys[i], batch.compressions[i])
            for i in range(len(batch.keys))
        ]
        return DetectorOutput(score=torch.tensor(values, dtype=torch.float32))


class RandomAdapter:
    """Builds :class:`RandomDetector` from the ``random`` card, seeded by whichever seed the
    detector source was given (``--seed``, default 0)."""

    card = read_builtin_card(_NAME)

    def load(self, weights: Path | None, device: str, *, seed: int | None = None) -> RandomDetector:
        effective_seed = DEFAULT_SEED if seed is None else seed
        meta = meta_from_card(self.card, source=f"zoo:{self.card.name}")
        return RandomDetector(meta, seed=effective_seed)
