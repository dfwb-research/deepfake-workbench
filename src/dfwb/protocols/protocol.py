"""Load, query and enumerate protocols (contract C3a): ``Protocol``, ``load``, ``list_protocols``.

A :class:`Protocol` is one hash-checked ``dataset/scheme`` inside one installed pack. ``load``
resolves the reference (defaulting the scheme, checking a pin, and hashing the split file to catch
drift between an installed pack and its card), falling back to a materialized recipe scheme under
the work root when the pack itself carries no split file. ``list_protocols`` enumerates every
scheme of every dataset of every installed pack, without reading any split file (J17, J4).
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError, did_you_mean
from dfwb.core.paths import require_root, resolve_roots
from dfwb.core.records import (
    DatasetCard,
    PairRecord,
    SchemeCard,
    SplitRow,
    VideoRecord,
    iter_jsonl_dicts,
    read_jsonl,
    read_split_tsv,
    split_sha256,
)
from dfwb.protocols._yaml import read_card, read_labels
from dfwb.protocols.packs import Pack, find_dataset, installed_packs
from dfwb.protocols.refs import ProtocolRef, parse_ref

__all__ = ["LabelMapping", "Protocol", "ProtocolInfo", "list_protocols", "load"]

_VERSION_PIN = re.compile(r"^\d+\.\d+\.\d+$")

# Plain VideoRecord fields the ``where`` filter may query, plus the derived "task" key (the key
# prefix before the first "/"). ``attrs.<name>`` is handled separately (its vocabulary is per
# dataset, not fixed).
_WHERE_FIELDS = frozenset(
    {
        "key",
        "compression",
        "label_key",
        "method",
        "identity",
        "source_id",
        "target_id",
        "pair_key",
        "task",
    }
)
_ATTR_PREFIX = "attrs."


def _task_of(key: str) -> str:
    return key.split("/", 1)[0]


@dataclass(frozen=True)
class LabelMapping:
    """A named label mapping: ``label_key -> int | str`` (a string value may be ``"exclude"``)."""

    name: str
    values: Mapping[str, int | str]

    def __call__(self, label_key: str) -> int | str:
        if label_key not in self.values:
            raise ContractError(
                f"label key {label_key!r} is not in mapping {self.name!r}",
                hint="check the dataset's labels.yaml vocab",
            )
        return self.values[label_key]


@dataclass(frozen=True)
class ProtocolInfo:
    """One row of :func:`list_protocols`: a scheme of a dataset, or a broken pack."""

    dataset_id: str
    scheme: str
    pack: str
    version: str
    kind: str
    default: bool
    counts: dict[str, int] | None
    broken: str | None


@dataclass(frozen=True)
class Protocol:
    """A loaded, hash-checked protocol: ``dataset/scheme`` inside one installed pack.

    Construct through :func:`load`, never directly -- the trailing fields are the engine's own
    state (the scheme's split rows, and where to read ``videos.jsonl.gz``/``pairs.jsonl.gz`` from,
    which may be the pack itself or a materialized recipe copy under the work root).
    """

    pack: Pack
    dataset: str
    scheme: str
    card: DatasetCard
    scheme_card: SchemeCard
    sha256: str
    pack_version: str
    ref: str
    _rows: tuple[SplitRow, ...] = field(repr=False, compare=False)
    _videos_path: Path = field(repr=False, compare=False)
    _pairs_path: Path | None = field(repr=False, compare=False)

    def split_rows(self) -> list[SplitRow]:
        """Every row of this scheme's split file."""
        return list(self._rows)

    def records(
        self,
        split: str | Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[VideoRecord]:
        """Videos assigned in this scheme, optionally narrowed by ``split`` and ``where``.

        A video absent from the scheme's split rows (J4: unassigned, not ``exclude``) is never
        returned, regardless of ``split``/``where``. ``where`` keys are ``VideoRecord`` fields,
        ``"task"`` (the key prefix before ``/``), or ``"attrs.<name>"``; a scalar value means
        equality (``None`` matches a null value), a list means membership.

        Builds a ``(key, compression) -> split`` dict from the split rows and walks
        ``videos.jsonl.gz`` once (:func:`~dfwb.core.records.iter_jsonl_dicts`, which owns the
        file's open/gzip/decode/JSON-error handling), filtering each row as a plain dict before
        building a ``VideoRecord`` for it, so a selective query stays linear without paying to
        construct rows it is about to discard (the 250k-row performance budget, protocols.md).

        Raises:
            ConfigError: a ``where`` key is not one of those, with a did-you-mean suggestion.
            ContractError: a row has no string ``"key"``, or a matched row does not build a
                ``VideoRecord`` (both name ``videos.jsonl.gz:<lineno>``).
        """
        wanted = None if split is None else ({split} if isinstance(split, str) else set(split))
        where = where or {}
        attr_keys = self._check_where_fields(where)
        split_index = {(row.key, row.compression): row.split for row in self._rows}
        name = self._videos_path.name

        result: list[VideoRecord] = []
        seen_attrs: set[str] = set()
        for lineno, data in iter_jsonl_dicts(self._videos_path):
            if attr_keys:
                seen_attrs.update(data.get("attrs") or {})
            key = data.get("key")
            if not isinstance(key, str):
                raise ContractError(
                    f"{name}:{lineno}: missing 'key'", hint="the file is corrupt; rebuild it"
                )
            row_split = split_index.get((key, data.get("compression")))
            if row_split is None:
                continue
            if wanted is not None and row_split not in wanted:
                continue
            if _matches_where(data, where):
                result.append(_build_video_record(data, name, lineno))

        unknown_attrs = attr_keys - seen_attrs
        if unknown_attrs:
            attr_name = sorted(unknown_attrs)[0]
            raise ConfigError(
                f"unknown attribute {attr_name!r} for dataset {self.dataset!r}"
                f"{did_you_mean(attr_name, seen_attrs)}",
                hint=f"run `dfwb protocols info {self.ref}` to see this dataset's attrs",
            )
        return result

    @staticmethod
    def _check_where_fields(where: Mapping[str, Any]) -> set[str]:
        """Reject an unknown plain field now; return the ``attrs.<name>`` suffixes to check."""
        attr_keys: set[str] = set()
        for key in where:
            if key in _WHERE_FIELDS:
                continue
            if key.startswith(_ATTR_PREFIX):
                attr_keys.add(key[len(_ATTR_PREFIX) :])
                continue
            raise ConfigError(
                f"unknown where field {key!r}{did_you_mean(key, _WHERE_FIELDS)}",
                hint="fields: " + ", ".join(sorted(_WHERE_FIELDS)) + ", or attrs.<name>",
            )
        return attr_keys

    def labels(self, mapping: str = "binary") -> LabelMapping:
        """The named label mapping: ``LabelVocab.mappings[mapping]``'s ``from`` attr + overrides.

        Raises:
            UnknownKeyError: ``mapping`` is not one of this dataset's mappings.
        """
        vocab = read_labels(self.pack.dataset_dir(self.dataset))
        if mapping not in vocab.mappings:
            raise UnknownKeyError(
                f"unknown label mapping {mapping!r} for dataset {self.dataset!r}"
                f"{did_you_mean(mapping, vocab.mappings)}",
                hint=f"run `dfwb protocols info {self.ref}` to see its label mappings",
            )
        spec = vocab.mappings[mapping]
        values: dict[str, int | str] = {}
        for label_key, attrs in vocab.vocab.items():
            value: Any = spec.override.get(label_key, attrs.get(spec.from_))
            if not isinstance(value, (int, str)):
                raise ContractError(
                    f"{self.dataset}/labels.yaml: {label_key}.{spec.from_} is not an int or string",
                    hint="fix the vocab entry",
                )
            values[label_key] = value
        return LabelMapping(mapping, values)

    def pairs(self, split: str | None = None) -> list[tuple[str, str]]:
        """``(real_key, fake_key)`` pairs, or ``[]`` if the pack has none.

        When ``split`` is given, a pair is kept if the real record's split in this scheme equals
        ``split``, or, when the real is absent from the scheme, the fake's split does (J7).
        """
        if self._pairs_path is None:
            return []
        records = read_jsonl(self._pairs_path, PairRecord)
        if split is None:
            return [(r.real_key, r.fake_key) for r in records]
        split_by_key: dict[str, str] = {row.key: row.split for row in self._rows}
        result: list[tuple[str, str]] = []
        for r in records:
            effective = split_by_key.get(r.real_key, split_by_key.get(r.fake_key))
            if effective == split:
                result.append((r.real_key, r.fake_key))
        return result

    def summary(self) -> dict[str, Any]:
        """A small, JSON-friendly summary: ref, pack, hash and per-split counts."""
        counts: dict[str, int] = (
            {str(k): v for k, v in self.scheme_card.counts.items()}
            if self.scheme_card.counts
            else dict(Counter(row.split for row in self._rows))
        )
        return {
            "ref": self.ref,
            "dataset": self.dataset,
            "scheme": self.scheme,
            "pack": self.pack.name,
            "pack_version": self.pack_version,
            "sha256": self.sha256,
            "kind": self.scheme_card.kind,
            "counts": counts,
        }


_VIDEO_RECORD_FIELDS = frozenset(f.name for f in dataclasses.fields(VideoRecord))


def _build_video_record(data: Mapping[str, Any], name: str, lineno: int) -> VideoRecord:
    """Build a ``VideoRecord`` from an already-matched row.

    Raises the same :class:`ContractError` shape as ``io.py``'s ``iter_jsonl`` for a row that does
    not fit the dataclass (an unknown, missing or extra field), naming ``name:lineno``. ``name``
    and ``lineno`` are formatted only on this (rare) error path, not on every matched row.
    """
    try:
        return VideoRecord(**data)
    except TypeError as exc:
        unknown = sorted(set(data) - _VIDEO_RECORD_FIELDS)
        detail = f"unexpected field(s) {unknown}" if unknown else str(exc).split(") ", 1)[-1]
        raise ContractError(
            f"{name}:{lineno}: not a valid VideoRecord: {detail}",
            hint="a file written by a newer dfwb needs a newer dfwb; otherwise rebuild it",
        ) from None


def _matches_where(data: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    for key, expected in where.items():
        if key == "task":
            actual: Any = _task_of(data["key"])
        elif key.startswith(_ATTR_PREFIX):
            actual = (data.get("attrs") or {}).get(key[len(_ATTR_PREFIX) :])
        else:
            actual = data.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _check_pin(ref: str, pin: str, pack_version: str, scheme_sha256: str) -> None:
    if _VERSION_PIN.match(pin):
        if pin != pack_version:
            raise ContractError(
                f"{ref} pinned @{pin} but installed pack version is {pack_version}",
                hint="install the pinned pack version, or update the pin",
            )
        return
    if not scheme_sha256.startswith(pin):
        raise ContractError(
            f"{ref} pinned @{pin} but installed scheme hash is {scheme_sha256}",
            hint="install the pinned pack version, or update the pin",
        )


def load(ref: str | ProtocolRef, *, work_root: Path | None = None) -> Protocol:
    """Load and hash-check one protocol.

    The scheme defaults to the dataset card's ``default_scheme``. When the pack carries no split
    file for the scheme (a recipe scheme), this reads
    ``<work_root>/<dataset>/materialized/{splits/<scheme>.tsv.gz,videos.jsonl.gz}`` instead
    (``work_root`` defaults to the resolved ``work`` root); if those are missing too, it raises
    :class:`ContractError` with a ``materialize`` hint.

    Raises:
        UnknownKeyError: the dataset or pack is unknown, or the scheme is not one of the card's.
        ContractError: a pin does not match, the split file has drifted from the card, the pack
            is broken, or a recipe scheme has nothing materialized yet.
    """
    parsed = parse_ref(ref) if isinstance(ref, str) else ref
    pack = find_dataset(parsed.dataset, pack=parsed.pack)
    dataset_dir = pack.dataset_dir(parsed.dataset)
    card = read_card(dataset_dir)
    scheme = parsed.scheme or card.default_scheme
    if scheme not in card.schemes:
        raise UnknownKeyError(
            f"unknown scheme {scheme!r} for dataset {parsed.dataset!r}"
            f"{did_you_mean(scheme, card.schemes)}",
            hint=f"run `dfwb protocols info {parsed.dataset}` to see its schemes",
        )
    scheme_card = card.schemes[scheme]
    canonical_ref = f"{parsed.dataset}/{scheme}"

    if parsed.pin is not None:
        _check_pin(canonical_ref, parsed.pin, pack.version, scheme_card.sha256)

    split_path = dataset_dir / "splits" / f"{scheme}.tsv.gz"
    videos_path = dataset_dir / "videos.jsonl.gz"
    if not split_path.is_file():
        resolved_work_root = (
            work_root if work_root is not None else require_root("work", resolve_roots())
        )
        materialized = resolved_work_root / parsed.dataset / "materialized"
        split_path = materialized / "splits" / f"{scheme}.tsv.gz"
        videos_path = materialized / "videos.jsonl.gz"
        if not split_path.is_file() or not videos_path.is_file():
            raise ContractError(
                f"{canonical_ref}: no materialized split for this recipe scheme",
                hint=f"run: dfwb protocols materialize {canonical_ref}",
            )

    rows = read_split_tsv(split_path)
    sha256 = split_sha256(rows)
    if sha256 != scheme_card.sha256:
        raise ContractError(
            f"{canonical_ref}: split file hash {sha256} does not match the card's "
            f"{scheme_card.sha256}",
            hint="the installed pack drifted from its card; reinstall or rebuild the pack",
        )

    pairs_file = dataset_dir / "pairs.jsonl.gz"
    pairs_path: Path | None = pairs_file if pairs_file.is_file() else None

    return Protocol(
        pack=pack,
        dataset=parsed.dataset,
        scheme=scheme,
        card=card,
        scheme_card=scheme_card,
        sha256=sha256,
        pack_version=pack.version,
        ref=canonical_ref,
        _rows=tuple(rows),
        _videos_path=videos_path,
        _pairs_path=pairs_path,
    )


def list_protocols() -> list[ProtocolInfo]:
    """One row per scheme per dataset per healthy pack, plus one row per broken pack (J17, J4).

    ``counts`` comes straight from :attr:`SchemeCard.counts`, so no split file is read.
    """
    rows: list[ProtocolInfo] = []
    for pack in installed_packs():
        if pack.error is not None:
            rows.append(ProtocolInfo("", "", pack.name, "", "", False, None, pack.error))
            continue
        assert pack.card is not None  # invariant: card is None only when error is set
        for dataset_id in pack.card.datasets:
            card = read_card(pack.dataset_dir(dataset_id))
            for scheme_name, scheme_card in card.schemes.items():
                rows.append(
                    ProtocolInfo(
                        dataset_id=dataset_id,
                        scheme=scheme_name,
                        pack=pack.name,
                        version=pack.version,
                        kind=scheme_card.kind,
                        default=(scheme_name == card.default_scheme),
                        counts={str(k): v for k, v in scheme_card.counts.items()}
                        if scheme_card.counts
                        else None,
                        broken=None,
                    )
                )
    return sorted(rows, key=lambda r: (r.dataset_id, r.scheme, r.pack, r.broken or ""))
