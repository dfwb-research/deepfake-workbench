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
from typing import TYPE_CHECKING, Any, Final

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError, did_you_mean
from dfwb.core.plugins import get_registry
from dfwb.zoo.adapter import require_license_accepted
from dfwb.zoo.strategies import ensure_clone
from dfwb.zoo.weights import ensure_weights

if TYPE_CHECKING:
    from pathlib import Path

    from dfwb.core.detector import Detector
    from dfwb.zoo.card import AdapterCard, WeightSpec

__all__ = ["load_zoo", "select_weight"]

# What contract C4 requires of anything an adapter's load() hands back, and what the `detectors`
# registry requires of an adapter class itself (mirrors dfwb.score.sources's own py: check).
_DETECTOR_ATTRS: Final = ("meta", "predict", "to")
_ADAPTER_ATTRS: Final = ("card", "load")


def _require_adapter_contract(adapter: Any, key: str) -> None:
    missing = [name for name in _ADAPTER_ATTRS if not hasattr(adapter, name)]
    if missing:
        raise ContractError(
            f"detectors/{key}: adapter class is missing {missing}",
            hint="an adapter needs a `card` attribute and a `load()` method",
        )


def _require_detector_contract(detector: Any, ref: str) -> None:
    missing = [name for name in _DETECTOR_ATTRS if not hasattr(detector, name)]
    if missing:
        raise ContractError(
            f"zoo:{ref}: adapter.load() returned a {type(detector).__name__!r} object, missing "
            f"{missing} required by the detector contract",
            hint="load() must return an object with meta, predict() and to()",
        )


def select_weight(card: AdapterCard, weights_id: str | None) -> WeightSpec:
    """The weight variant ``weights_id`` names, or ``card``'s only one when ``weights_id`` is
    ``None`` and there is exactly one -- the same resolution :func:`load_zoo` itself uses for a
    bare ``zoo:<name>``, exposed so a caller (``dfwb zoo fetch``) can resolve a variant the same
    way without downloading it.

    Raises:
        ConfigError: ``weights_id`` is ``None`` and ``card`` has more than one weight variant.
        UnknownKeyError: ``weights_id`` names none of ``card``'s own weight variants.
    """
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
            offline, the registry target is not a well-formed adapter (missing `card` or `load`),
            the card's own `name` does not match the registry key it is added under, or `load()`
            returned something that fails the detector contract (missing `meta`, `predict` or
            `to`).
    """
    name, has_id, weights_id = ref.partition("@")
    if not name:
        raise ConfigError(f"zoo: {ref!r} must be <name>[@<weights id>]", hint="example: zoo:chance")
    registry = get_registry("detectors")
    entry = registry.entry(name)
    adapter_cls = registry.load(name)
    adapter: Any = adapter_cls()
    _require_adapter_contract(adapter, entry.key)
    card: AdapterCard = adapter.card
    if card.name != entry.key:
        raise ContractError(
            f"detectors/{entry.key}: its card declares name {card.name!r}, which does not match "
            "the registry key it is added under",
            hint="the card's `name` field and the registry key it is registered under must agree",
        )
    require_license_accepted(card)

    weights_path: Path | None = None
    checkpoint_sha256: str | None = None
    resolved_id: str | None = None
    if card.weights:
        spec = select_weight(card, weights_id if has_id else None)
        weights_path = ensure_weights(card.name, spec)
        # `ensure_weights` has already hashed this exact file and confirmed it matches
        # `spec.sha256` (or raised) -- reuse that already-verified value instead of hashing a
        # (potentially large) weights file a second time here.
        checkpoint_sha256 = spec.sha256
        resolved_id = spec.id
    elif has_id:
        raise ConfigError(
            f"zoo:{card.name}: has no weights, but {ref!r} names one",
            hint=f"use zoo:{card.name} (no @<weights id>)",
        )

    # code_root is only ever passed to a pinned-clone adapter's load(); every other strategy's
    # adapter keeps the plain load(weights, device, *, seed=None) shape -- it never has to know
    # about a keyword that means nothing for it.
    load_kwargs: dict[str, Any] = {"seed": seed}
    if card.code_strategy == "pinned-clone":
        load_kwargs["code_root"] = ensure_clone(card)

    detector: Detector = adapter.load(weights_path, "cpu", **load_kwargs)
    _require_detector_contract(detector, ref)
    source = f"zoo:{card.name}" + (f"@{resolved_id}" if resolved_id else "")
    detector.meta = dataclasses.replace(detector.meta, source=source)
    detector.checkpoint_sha256 = checkpoint_sha256  # type: ignore[attr-defined]
    return detector
