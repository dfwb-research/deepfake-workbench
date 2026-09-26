"""The ``zoo:`` detector source: ``zoo:<name>[@<weights id>]`` builds a C4 detector from a
registered zoo adapter.

Registered under ``detector_sources`` as ``zoo``; the score layer resolves it with
``get_registry("detector_sources").load("zoo")(ref, seed=...)``, never importing this module (or
any adapter module) directly. Looks up ``<name>`` in the ``detectors`` registry, gates on the
adapter card's own licence terms, resolves and verifies its weights (if it has any), and hands the
adapter a local path to score from. Sets ``checkpoint_sha256`` on the detector it returns (the
verified weights file's sha256; ``None`` for an adapter with no weights) and always rewrites
``meta.source`` to ``zoo:<name>`` or ``zoo:<name>@<weights id>`` -- whichever the adapter itself
does not reliably know how to spell, since only this module knows which weights id a bare
``zoo:<name>`` (no id given) actually resolved to.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from dfwb.core.errors import ConfigError, UnknownKeyError, did_you_mean
from dfwb.core.hashing import sha256_file
from dfwb.core.plugins import get_registry
from dfwb.zoo.adapter import require_license_accepted
from dfwb.zoo.weights import ensure_weights

if TYPE_CHECKING:
    from pathlib import Path

    from dfwb.core.detector import Detector
    from dfwb.zoo.card import AdapterCard, WeightSpec

__all__ = ["load_zoo"]


def _select_weight(card: AdapterCard, weights_id: str | None) -> WeightSpec:
    if weights_id is None:
        if len(card.weights) == 1:
            return card.weights[0]
        ids = [w.id for w in card.weights]
        raise ConfigError(
            f"zoo:{card.name}: {len(card.weights)} weight variants are available "
            f"({', '.join(ids)}); a weights id must be given",
            hint=f"e.g. zoo:{card.name}@{ids[0]}",
        )
    for spec in card.weights:
        if spec.id == weights_id:
            return spec
    ids = [w.id for w in card.weights]
    raise UnknownKeyError(
        f"zoo:{card.name}: unknown weights id {weights_id!r}{did_you_mean(weights_id, ids)}",
        hint="weight ids: " + (", ".join(ids) or "(none)"),
    )


def load_zoo(ref: str, *, seed: int | None = None) -> Detector:
    """The ``zoo:`` detector source: ``<name>[@<weights id>]``.

    Raises:
        ConfigError: ``ref`` has no adapter name, names a weights id for an adapter with no
            weights, or leaves an ambiguous choice among several weight variants.
        UnknownKeyError: ``<name>`` names no registered adapter, or ``<weights id>`` is not one of
            the card's own weight variants.
        InstallationError: the adapter's licence must be acknowledged first
            (:func:`~dfwb.zoo.adapter.require_license_accepted`, exit code 5), or its weights are
            not cached and cannot be downloaded (offline, or the download itself failed).
        ContractError: a cached weights file does not match the card and cannot be re-downloaded
            offline.
    """
    name, has_id, weights_id = ref.partition("@")
    if not name:
        raise ConfigError(f"zoo: {ref!r} must be <name>[@<weights id>]", hint="example: zoo:chance")
    adapter_cls = get_registry("detectors").load(name)
    adapter: Any = adapter_cls()
    card: AdapterCard = adapter.card
    require_license_accepted(card)

    weights_path: Path | None = None
    checkpoint_sha256: str | None = None
    resolved_id: str | None = None
    if card.weights:
        spec = _select_weight(card, weights_id if has_id else None)
        weights_path = ensure_weights(card.name, spec)
        # The measured hash of the file actually on disk, not just an echo of the card's own
        # declared string -- correct by construction once `ensure_weights` has verified it, and
        # still correct even if that guarantee were ever violated by a bug.
        checkpoint_sha256 = sha256_file(weights_path)
        resolved_id = spec.id
    elif has_id:
        raise ConfigError(
            f"zoo:{card.name}: has no weights, but {ref!r} names one",
            hint=f"use zoo:{card.name} (no @<weights id>)",
        )

    detector: Detector = adapter.load(weights_path, "cpu", seed=seed)
    source = f"zoo:{card.name}" + (f"@{resolved_id}" if resolved_id else "")
    detector.meta = dataclasses.replace(detector.meta, source=source)
    detector.checkpoint_sha256 = checkpoint_sha256  # type: ignore[attr-defined]
    return detector
