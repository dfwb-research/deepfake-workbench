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
from dfwb.core.licenses import require_accepted
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
    (optionally) verified local weights.

    An adapter with weights loads them with :func:`dfwb.zoo.weights.load_weights` (safetensors or
    ``torch.load(weights_only=True)`` only, dispatched from the card's own weight format), never
    its own ad hoc pickle or checkpoint loading.
    """

    card: AdapterCard

    def load(
        self,
        weights: Path | None,
        device: str,
        *,
        seed: int | None = None,
        code_root: Path | None = None,
    ) -> Detector:
        """Build the detector, from a verified local weights file (``None`` for an adapter with
        no weights) and the device it should end up on. ``seed`` is given whenever a caller named
        one explicitly (``score(..., seed=...)``, or its own default); most adapters ignore it.
        ``code_root`` is the adapter's own pinned clone (``code_strategy: pinned-clone``, set by
        :func:`dfwb.zoo.strategies.ensure_clone`), ``None`` for every other strategy; an adapter
        that uses it imports its entry point with
        :func:`dfwb.zoo.strategies.import_pinned_entry`."""
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
    """Gate use of ``card`` on its own licence terms -- the same gate the face backends use
    (:func:`dfwb.core.licenses.require_accepted`), so there is exactly one implementation of
    "has this licence been acknowledged" in the framework.

    Raises:
        InstallationError: ``card.license.requires_ack`` is set and ``card.name`` has not yet
            been acknowledged (exit code 5).
    """
    if not card.license.requires_ack:
        return
    licence = card.license.weights or card.license.code
    require_accepted(card.name, terms=f"licensed under {licence}")
