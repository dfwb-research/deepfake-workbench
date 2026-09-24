"""Protocol pack discovery (contract C3a): installed packs and dataset lookup.

Packs are registered in the ``protocol_packs`` data registry (``dfwb.core.plugins``); ``.load()``
locates each one's root directory without importing any of its code. ``installed_packs()`` builds
its list fresh on every call (no module-level cache), so a pack whose ``pack.yaml`` is missing or
invalid never hides the others (J17): it is returned with ``card=None`` and ``error`` set, and
using it raises :class:`ContractError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dfwb.core.errors import AmbiguousKeyError, ContractError, UnknownKeyError, did_you_mean
from dfwb.core.plugins import get_registry
from dfwb.core.records import PackCard
from dfwb.core.registry import catalogue_requirement
from dfwb.protocols._yaml import read_model

__all__ = ["Pack", "find_dataset", "installed_packs"]


@dataclass(frozen=True)
class Pack:
    """One installed protocol pack: its ``pack.yaml`` card, or why it could not be read (J17)."""

    name: str
    provider: str
    root: Path
    card: PackCard | None
    error: str | None

    @property
    def version(self) -> str:
        """The pack's version. Raises :class:`ContractError` if the pack is broken."""
        if self.card is None:
            raise ContractError(
                f"protocol pack {self.name!r} is broken: {self.error}",
                hint=f"fix or reinstall the pack {self.name!r} (provided by {self.provider})",
            )
        return self.card.version

    def dataset_dir(self, dataset_id: str) -> Path:
        """Where ``dataset_id``'s files live inside this pack (whether or not it exists)."""
        return self.root / dataset_id


def installed_packs() -> list[Pack]:
    """Every registered protocol pack, sorted by name. Rebuilt on every call (no caching)."""
    registry = get_registry("protocol_packs")
    packs: list[Pack] = []
    for entry in registry.entries():
        root = registry.load(entry.qualified_key)
        try:
            card = read_model(root / "pack.yaml", PackCard)
        except ContractError as exc:
            packs.append(Pack(entry.key, entry.provider, root, None, exc.message))
        else:
            packs.append(Pack(entry.key, entry.provider, root, card, None))
    return sorted(packs, key=lambda p: p.name)


def _dataset_ids(pack: Pack) -> frozenset[str]:
    """Dataset ids a healthy pack publishes (empty for a broken one)."""
    return frozenset(pack.card.datasets) if pack.card is not None else frozenset()


def find_dataset(dataset: str, pack: str | None = None) -> Pack:
    """The pack that publishes ``dataset``.

    Raises:
        UnknownKeyError: ``dataset`` is in no installed pack (with did-you-mean over every
            dataset id, and an install hint when the id is in the catalogue), or ``pack`` names a
            pack that is not installed or does not publish ``dataset``.
        AmbiguousKeyError: ``dataset`` is published by more than one pack and ``pack`` is ``None``.
        ContractError: the named pack is broken.
    """
    packs = installed_packs()
    if pack is not None:
        named = next((p for p in packs if p.name == pack), None)
        if named is None:
            raise UnknownKeyError(
                f"unknown protocol pack {pack!r}{did_you_mean(pack, [p.name for p in packs])}",
                hint="run `dfwb plugins list --all` to see installed protocol packs",
            )
        if named.error is not None:
            raise ContractError(
                f"protocol pack {pack!r} is broken: {named.error}",
                hint=f"fix or reinstall the pack {pack!r} (provided by {named.provider})",
            )
        if dataset not in _dataset_ids(named):
            raise UnknownKeyError(
                f"{pack!r} does not publish dataset {dataset!r}"
                f"{did_you_mean(dataset, _dataset_ids(named))}",
                hint=f"run `dfwb protocols list --pack {pack}` to see its datasets",
            )
        return named
    matches = [p for p in packs if dataset in _dataset_ids(p)]
    if not matches:
        all_ids = sorted({d for p in packs for d in _dataset_ids(p)})
        requirement = catalogue_requirement("protocol_packs", dataset)
        hint = (
            f"pip install {requirement}"
            if requirement is not None
            else "run `dfwb plugins list --all` to see installed protocol packs"
        )
        raise UnknownKeyError(
            f"unknown dataset {dataset!r}{did_you_mean(dataset, all_ids)}", hint=hint
        )
    if len(matches) > 1:
        qualified = sorted(f"{p.name}:{dataset}" for p in matches)
        raise AmbiguousKeyError(
            f"dataset {dataset!r} is provided by more than one pack: {', '.join(qualified)}",
            hint=f"qualify it, e.g. {qualified[0]!r}",
        )
    return matches[0]
