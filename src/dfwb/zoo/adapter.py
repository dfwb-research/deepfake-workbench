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
from dfwb.core.errors import ContractError
from dfwb.core.licenses import is_accepted
from dfwb.zoo.card import AdapterCard, read_card

if TYPE_CHECKING:
    from dfwb.core.detector import Detector

__all__ = ["Adapter", "meta_from_card", "read_builtin_card", "require_license_accepted"]

_CARDS_DIR = Path(__file__).resolve().parent / "cards"


def read_builtin_card(name: str) -> AdapterCard:
    """The adapter card shipped at ``dfwb/zoo/cards/<name>.yaml``."""
    return read_card(_CARDS_DIR / f"{name}.yaml")


@runtime_checkable
class Adapter(Protocol):
    """What every zoo adapter implements: its own card, and how to build a detector from
    (optionally) verified local weights."""

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


def require_license_accepted(card: AdapterCard) -> None:
    """Gate use of ``card`` on its own licence terms.

    Raises:
        ContractError: ``card.license.requires_ack`` is set and ``card.name`` has not yet been
            acknowledged (:func:`dfwb.core.licenses.is_accepted`).
    """
    if not card.license.requires_ack or is_accepted(card.name):
        return
    licence = card.license.weights or card.license.code
    raise ContractError(
        f"zoo:{card.name}: its licence ({licence}) must be acknowledged before use",
        hint=f"run `dfwb zoo fetch {card.name} --accept-license` once",
    )
