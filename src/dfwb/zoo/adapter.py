"""The adapter contract: what every zoo adapter implements, and the licence gate every one of
them is resolved through.

``Adapter.load`` is deliberately narrow: given a (possibly ``None``, for a weight-free adapter)
local, already-verified weights path and a device, it returns a C4
:class:`~dfwb.core.detector.Detector`. Everything else -- resolving ``zoo:<name>[@<weights id>]``,
verifying and caching weights, gating on a required licence acknowledgement -- is
:mod:`dfwb.zoo.source`'s job, not the adapter's.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dfwb import __version__ as _FRAMEWORK_VERSION
from dfwb.core.detector import DetectorMeta
from dfwb.core.errors import InstallationError
from dfwb.core.licenses import accept as _accept_licence
from dfwb.core.licenses import is_accepted, require_accepted
from dfwb.zoo.card import AdapterCard, read_card

if TYPE_CHECKING:
    from dfwb.core.detector import Detector

__all__ = [
    "Adapter",
    "accept_license",
    "license_text_for",
    "meta_from_card",
    "read_builtin_card",
    "require_license_accepted",
]

_CARDS_DIR = Path(__file__).resolve().parent / "cards"


def read_builtin_card(name: str) -> AdapterCard:
    """The adapter card shipped at ``dfwb/zoo/cards/<name>.yaml``."""
    return read_card(_CARDS_DIR / f"{name}.yaml")


@runtime_checkable
class Adapter(Protocol):
    """What every zoo adapter implements: its own card, and how to build a detector from
    (optionally) verified local weights.

    An adapter with weights loads them with :func:`dfwb.zoo.weights.load_weights` (safetensors or
    ``torch.load(weights_only=True)`` only, dispatched from the card's own weight format), never
    its own ad hoc pickle or checkpoint loading.

    A ``code_strategy: pinned-clone`` adapter's own ``load()`` additionally declares a
    ``code_root: Path | None = None`` keyword -- :mod:`dfwb.zoo.source` passes it (the pinned
    clone directory, from :func:`dfwb.zoo.strategies.ensure_clone`) only to such an adapter, and
    the adapter imports its entry point from it with
    :func:`dfwb.zoo.strategies.import_pinned_entry`. This is not part of the base signature below:
    every other strategy's adapter never receives it, and never has to know about a keyword that
    would mean nothing for it.
    """

    card: AdapterCard

    def load(self, weights: Path | None, device: str, *, seed: int | None = None) -> Detector:
        """Build the detector, from a verified local weights file (``None`` for an adapter with
        no weights) and the device it should end up on. ``seed`` is given whenever a caller named
        one explicitly (``score(..., seed=...)``, or its own default); most adapters ignore it."""
        ...


def meta_from_card(card: AdapterCard, *, version: str | None = None, source: str) -> DetectorMeta:
    """A :class:`~dfwb.core.detector.DetectorMeta` built from ``card``: its own input spec,
    licence and citation, with ``source`` set as the caller (:mod:`dfwb.zoo.source`) decides."""
    return DetectorMeta(
        name=card.name,
        version=version or _FRAMEWORK_VERSION,
        contract_version=card.contract_version,
        input=card.input.to_input_spec(),
        license=card.license.code,
        weights_license=card.license.weights,
        citation=card.upstream.bibtex if card.upstream else None,
        source=source,
    )


def license_text_for(card: AdapterCard) -> str:
    """The licence text ``card``'s acknowledgement is about: its *code* licence for a
    ``"pinned-clone"`` strategy (the usual reason a card would use that strategy at all), or its
    *weights* licence (falling back to its code licence when it declares none) for every other
    strategy. :func:`require_license_accepted` gates on exactly this text, and
    :func:`accept_license` records exactly this text, so a caller (``dfwb zoo fetch``) never has
    to work out which licence a card's acknowledgement means -- and can never accidentally record
    a different one than the gate itself checks."""
    if card.code_strategy == "pinned-clone":
        return card.license.code
    return card.license.weights or card.license.code


def require_license_accepted(card: AdapterCard) -> None:
    """Gate use of ``card`` on its own licence terms -- the same gate the face backends use
    (:func:`dfwb.core.licenses.require_accepted`), so there is exactly one implementation of
    "has this licence been acknowledged" in the framework.

    Raises:
        InstallationError: ``card.license.requires_ack`` is set and ``card.name`` has not yet
            been acknowledged (exit code 5); the hint names the exact command to run.
    """
    if not card.license.requires_ack:
        return
    what = (
        "this adapter's code licence"
        if card.code_strategy == "pinned-clone"
        else "these model weights"
    )
    licence = license_text_for(card)
    try:
        require_accepted(card.name, terms=f"licensed under {licence}", what=what)
    except InstallationError as exc:
        raise InstallationError(
            exc.message,
            hint=f"licensed under {licence}; run `dfwb zoo fetch {card.name} "
            "--accept-license` once (this machine will not ask again)",
        ) from None


def accept_license(card: AdapterCard) -> None:
    """Record ``card``'s licence acknowledgement, under the exact text
    :func:`require_license_accepted` gates on (:func:`license_text_for`) -- so whatever records an
    acceptance and whatever later checks it always agree, and ``dfwb zoo licenses`` shows the
    licence the user actually agreed to (a pinned-clone adapter's *code* licence, not its distinct
    weights licence, say). A no-op if ``card`` needs no acknowledgement at all, or one has already
    been recorded (the existing acknowledgement's timestamp is left untouched)."""
    if not card.license.requires_ack or is_accepted(card.name):
        return
    _accept_licence(card.name, license=license_text_for(card))
