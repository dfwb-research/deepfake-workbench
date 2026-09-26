"""Rebuild a recipe's lists from a local inventory and check them against their published hashes.

A pack can publish a scheme as a *recipe*: its rule, the rule's parameters and the ``sha256`` of
the split rows the rule produces, but no key list. That is how a dataset whose terms forbid
redistributing even file names can still have a comparable split. :func:`materialize` rebuilds the
rows from the user's own inventory and keeps them only when they hash to the published value, so a
local split is either exactly the one the pack describes or it is refused.

A recipe *dataset* goes further and ships no key list at all: no videos, no pairs, no split rows,
only its card, which records every scheme's rule, parameters and hash plus the hashes of the video
and pair lists (``videos_sha256``, ``pairs_sha256``) and the pairing rule.
:func:`materialize_dataset` rebuilds all of those lists from the inventory at once, and writes
them only when every hash matches.

:func:`assign_rule` runs a scheme's rule from its card's ``rule`` and ``params``. Pack building
(``dfwb protocols build``) produces every scheme through this same function, so a published scheme
and its materialized copy can never be computed two different ways.

The publisher's own split (the ``official`` rules, and a benchmark drawn from the official test)
comes from files only the dataset's inventory builder knows how to read, and so do the pairs (a
builder's pairing rule is code), and this layer never imports builders: the caller passes that
split in as ``official``, and the pairs as ``pairs``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError, did_you_mean
from dfwb.core.records import (
    DatasetCard,
    InventoryRecord,
    LabelVocab,
    PackProvenance,
    PairRecord,
    SchemeCard,
    SplitRow,
    VideoRecord,
    read_jsonl,
    records_sha256,
    split_sha256,
    to_video_record,
    write_jsonl,
    write_split_tsv,
)
from dfwb.protocols._materialized import HASHES_FILE, materialized_dir
from dfwb.protocols._rawdata import check_outside_datasets_roots
from dfwb.protocols._yaml import read_card, read_labels, read_model
from dfwb.protocols.packs import find_dataset
from dfwb.protocols.protocol import _check_pin
from dfwb.protocols.refs import ProtocolRef, parse_ref
from dfwb.protocols.rules import (
    Assignment,
    BenchmarkSpec,
    RuleRecord,
    Split,
    assign_72_14_14,
    assign_all_test,
    assign_benchmark,
    assign_official,
    assign_official_plus_80_20,
)
from dfwb.protocols.writer import rows_from_assignment

__all__ = [
    "BENCHMARK_POOLS",
    "OFFICIAL_RULES",
    "RULES",
    "DatasetMaterialization",
    "MaterializeResult",
    "assign_rule",
    "benchmark_params",
    "check_recipe_card",
    "materialize",
    "materialize_dataset",
    "needs_official",
    "release_rows",
    "rule_needs_official",
    "ships_key_lists",
]

# Every rule a scheme card may name and still be recomputed.
RULES: Final = ("official", "official+ident-80-20", "ident-72-14-14", "all-test", "benchmark")
# The rules that read the publisher's own split.
OFFICIAL_RULES: Final = frozenset({"official", "official+ident-80-20"})
# Where a benchmark draws from: the dataset's official test, or every record.
BENCHMARK_POOLS: Final = ("official-test", "all")

_BENCHMARK_PARAMS: Final = ("k_fake", "task_order", "pool")
_VERIFY_HINT: Final = (
    "your local copy differs from the release the pack describes; run dfwb protocols verify"
)
# The order split counts are given in.
_SPLIT_ORDER: Final = ("train", "val", "test", "exclude")

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaterializeResult:
    """A materialized scheme: its reference, the split file written, and its hash."""

    ref: str
    path: Path
    sha256: str
    matched: bool


@dataclass(frozen=True)
class DatasetMaterialization:
    """A materialised recipe dataset: the folder written, and every hash its lists matched.

    ``schemes`` maps each scheme of the card to its split's hash.
    """

    dataset: str
    path: Path
    videos_sha256: str
    pairs_sha256: str
    schemes: dict[str, str]
    n_videos: int
    n_pairs: int


# ---------------------------------------------------------------------------------------------
# Rules from a scheme card
# ---------------------------------------------------------------------------------------------


def benchmark_params(
    spec: BenchmarkSpec, *, task_order: Sequence[str], pool: Literal["official-test", "all"]
) -> dict[str, Any]:
    """The benchmark rule's parameters, as a scheme card records them.

    ``task_order`` lists the dataset's tasks in rank order (it breaks ties between records that
    share a local key); ``pool`` says whether the subset is drawn from the dataset's official test
    (``official-test``) or from every record (``all``). Together with the spec's own fields
    (``compressions`` is ``None`` when the benchmark is drawn across every compression) that is
    everything :func:`assign_rule` needs to redraw exactly the same subset.

    Raises:
        ContractError: ``pool`` is not one of :data:`BENCHMARK_POOLS`.
    """
    if pool not in BENCHMARK_POOLS:
        raise ContractError(
            f"unknown benchmark pool {pool!r}", hint="pools: " + ", ".join(BENCHMARK_POOLS)
        )
    return {
        "k_fake": spec.k_fake,
        "strata": list(spec.strata),
        "k_real_cap": spec.k_real_cap,
        "exclude_tasks": list(spec.exclude_tasks),
        "seed": spec.seed,
        "task_order": list(task_order),
        "pool": pool,
        "compressions": None if spec.compressions is None else list(spec.compressions),
    }


def rule_needs_official(rule: str | None, params: Mapping[str, Any]) -> bool:
    """Whether ``rule`` with ``params`` reads the publisher's official split."""
    return rule in OFFICIAL_RULES or (rule == "benchmark" and params.get("pool") == "official-test")


def _required_official(official: Mapping[str, Split] | None, rule: str) -> Mapping[str, Split]:
    if official is None:
        raise ConfigError(
            f"the {rule} rule needs the publisher's official split, and none was given",
            hint="dfwb protocols materialize reads it with the dataset's inventory builder; from "
            "Python, pass official=builder.official_splits(dataset folder, inventory records)",
        )
    return official


def _benchmark_spec(params: Mapping[str, Any]) -> tuple[BenchmarkSpec, dict[str, int], str]:
    missing = [name for name in _BENCHMARK_PARAMS if name not in params]
    if missing:
        raise ContractError(
            f"the benchmark parameters lack {', '.join(repr(name) for name in missing)}",
            hint="rebuild the pack with dfwb protocols build, which records every parameter",
        )
    pool = str(params["pool"])
    if pool not in BENCHMARK_POOLS:
        raise ContractError(
            f"unknown benchmark pool {pool!r}", hint="pools: " + ", ".join(BENCHMARK_POOLS)
        )
    # A card written before benchmarks recorded their compressions drew from every record.
    compressions = params.get("compressions")
    if compressions is not None and (
        not isinstance(compressions, list) or not all(isinstance(c, str) for c in compressions)
    ):
        raise ContractError(
            f"the benchmark's compressions {compressions!r} are not a list of names",
            hint="rebuild the pack with dfwb protocols build, which records every parameter",
        )
    spec = BenchmarkSpec(
        k_fake=int(params["k_fake"]),
        strata=tuple(params.get("strata") or ()),
        k_real_cap=params.get("k_real_cap"),
        exclude_tasks=tuple(params.get("exclude_tasks") or ()),
        seed=int(params.get("seed", 0)),
        compressions=None if compressions is None else tuple(compressions),
    )
    task_rank = {str(task): index for index, task in enumerate(params["task_order"])}
    return spec, task_rank, pool


def assign_rule[R: RuleRecord](
    rule: str | None,
    params: Mapping[str, Any],
    records: Iterable[R],
    *,
    official: Mapping[str, Split] | None,
    is_real: Callable[[R], bool],
) -> Assignment:
    """Run the generic split rule ``rule``, with a scheme card's ``params``, over ``records``.

    ``official`` and ``official+ident-80-20`` apply the publisher's split, ``official`` (record
    key -> split); the second takes its ``policy`` from ``params``. ``ident-72-14-14`` and
    ``all-test`` need nothing else. ``benchmark`` takes its spec, task order and pool from
    ``params`` (see :func:`benchmark_params`), tells reals from fakes with ``is_real`` and, when
    its pool is ``official-test``, draws only from the records ``official`` puts in ``test``.

    Raises:
        ConfigError: the rule needs ``official`` and it is ``None``.
        ContractError: ``rule`` is not one of :data:`RULES`, or its parameters are incomplete.
    """
    records = list(records)
    if rule == "official":
        return assign_official(records, _required_official(official, rule))
    if rule == "official+ident-80-20":
        return assign_official_plus_80_20(
            records,
            _required_official(official, rule),
            policy=params.get("policy"),  # type: ignore[arg-type]
        )
    if rule == "ident-72-14-14":
        return assign_72_14_14(records)
    if rule == "all-test":
        return assign_all_test(records)
    if rule == "benchmark":
        spec, task_rank, pool = _benchmark_spec(params)
        pool_keys: set[str] | None = None
        if pool == "official-test":
            matched = assign_official(records, _required_official(official, "benchmark"))
            pool_keys = {key for (key, _), split in matched.items() if split == "test"}
        return assign_benchmark(
            records, spec=spec, is_real=is_real, task_rank=task_rank, pool_keys=pool_keys
        )
    raise ContractError(f"rule {rule!r} cannot be recomputed", hint="rules: " + ", ".join(RULES))


# ---------------------------------------------------------------------------------------------
# Materializing a pack's scheme
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scheme:
    ref: str
    dataset: str
    name: str
    card: SchemeCard
    labels: Callable[[], LabelVocab]
    dataset_card: DatasetCard
    ships_videos: bool
    dataset_dir: Path


def _resolve(ref: str | ProtocolRef) -> _Scheme:
    parsed = parse_ref(ref) if isinstance(ref, str) else ref
    pack = find_dataset(parsed.dataset, pack=parsed.pack)
    dataset_dir = pack.dataset_dir(parsed.dataset)
    card = read_card(dataset_dir)
    name = parsed.scheme or card.default_scheme
    if name not in card.schemes:
        raise UnknownKeyError(
            f"unknown scheme {name!r} for dataset {parsed.dataset!r}"
            f"{did_you_mean(name, card.schemes)}",
            hint=f"run `dfwb protocols list` to see {parsed.dataset}'s schemes",
        )
    scheme_card = card.schemes[name]
    canonical = f"{parsed.dataset}/{name}"
    if parsed.pin is not None:
        _check_pin(canonical, parsed.pin, pack.version, scheme_card.sha256)
    return _Scheme(
        canonical,
        parsed.dataset,
        name,
        scheme_card,
        lambda: read_labels(dataset_dir),
        card,
        (dataset_dir / "videos.jsonl.gz").is_file(),
        dataset_dir,
    )


def ships_key_lists(ref: str | ProtocolRef) -> bool:
    """Whether the pack ships ``ref``'s dataset with its key lists (its ``videos.jsonl.gz``).

    ``False`` for a recipe dataset published without them, which :func:`materialize_dataset`
    rebuilds from an inventory as a whole.

    Raises:
        UnknownKeyError, ContractError: as :func:`materialize` does when resolving ``ref``.
    """
    return _resolve(ref).ships_videos


def needs_official(ref: str | ProtocolRef) -> bool:
    """Whether materializing ``ref`` needs the publisher's official split (see :func:`materialize`).

    Raises:
        UnknownKeyError, ContractError: as :func:`materialize` does when resolving ``ref``.
    """
    scheme = _resolve(ref)
    return rule_needs_official(scheme.card.rule, scheme.card.params)


def _is_real_from(
    labels: Callable[[], LabelVocab], ref: str, hint: str = _VERIFY_HINT
) -> Callable[[VideoRecord], bool]:
    """Real under the visual ``binary`` label of the pack's vocabulary (read on first use)."""
    vocab: dict[str, Mapping[str, Any]] = {}

    def is_real(record: VideoRecord) -> bool:
        if not vocab:
            vocab.update(labels().vocab)
        entry = vocab.get(record.label_key)
        if entry is None:
            raise ContractError(
                f"{ref}: label key {record.label_key!r} of {record.key!r} is not in the pack's "
                "labels.yaml",
                hint=hint,
            )
        return entry.get("binary") == 0

    return is_real


def release_rows(rows: Iterable[InventoryRecord], card: DatasetCard) -> list[InventoryRecord]:
    """The inventory rows of the release the card describes: those of no compression variant, or
    of a compression the card's ``compressions`` lists.

    A local copy may hold compressions a pack does not publish; they are no part of its lists.
    """
    listed = frozenset(card.compressions or ())
    return [row for row in rows if row.compression is None or row.compression in listed]


def _video_records(
    rows: Sequence[InventoryRecord], inventory: Path, dataset: str, local_attrs: Collection[str]
) -> list[VideoRecord]:
    records = [to_video_record(row, local_attrs=local_attrs) for row in rows]
    seen: set[tuple[str, str | None]] = set()
    for record in records:
        ident = (record.key, record.compression)
        if ident in seen:
            raise ContractError(
                f"{inventory.name}: {record.key!r} (compression {record.compression!r}) is "
                "listed twice",
                hint=f"rebuild the inventory: dfwb inventory build {dataset}",
            )
        seen.add(ident)
    return sorted(records, key=lambda r: (r.key, r.compression or ""))


def _read_records(
    inventory: Path, card: DatasetCard, local_attrs: Collection[str]
) -> list[VideoRecord]:
    """The inventory's records for one recipe scheme of a pack that ships its videos.

    A card that lists no ``compressions`` filters nothing here: a pack built before cards had to
    list them may carry compressions in its videos all the same.
    """
    rows = read_jsonl(inventory, InventoryRecord)
    if card.compressions is not None:
        rows = release_rows(rows, card)
    return _video_records(rows, inventory, card.id, local_attrs)


def _check_same_videos(path: Path, records: Sequence[VideoRecord], dataset: str) -> None:
    """Refuse to go on when ``path`` holds other records than ``records``."""
    existing = read_jsonl(path, VideoRecord)
    if existing != list(records):
        raise ContractError(
            f"{path} holds {len(existing)} videos written from another inventory, which differ "
            f"from this inventory's {len(records)}; replacing them would change what the schemes "
            "already materialized from it describe",
            hint=f"delete {path} (and the split files next to it) and materialize again, or "
            f"materialize from the same inventory that wrote it (dfwb inventory build {dataset} "
            "rebuilds the default one)",
        )


def materialize(
    ref: str | ProtocolRef,
    *,
    inventory: Path,
    official: Mapping[str, Split] | None,
    work_root: Path,
    datasets_roots: Sequence[Path] | None = None,
    local_attrs: Collection[str] = (),
) -> MaterializeResult:
    """Recompute ``ref``'s split from ``inventory`` and keep it only if it matches the pack.

    The rule and its parameters come from the pack's scheme card (the scheme defaults to the
    card's default scheme); the records are ``inventory``'s rows as
    :class:`~dfwb.core.records.VideoRecord`, without the attributes named in ``local_attrs``
    (those the dataset's builder declares as facts about the local copy, which a pack never
    publishes); when the card lists its ``compressions``, rows of any other compression are left
    out (a card that lists none filters nothing). The rows are hashed exactly as a split file is,
    and only a match with the card's ``sha256`` writes
    ``<work_root>/<dataset>/materialized/{videos.jsonl.gz,splits/<scheme>.tsv.gz}``, where
    :func:`~dfwb.protocols.protocol.load` then finds them. A mismatch writes nothing, so the work
    root is left exactly as it was.

    Every materialized scheme of a dataset shares one ``videos.jsonl.gz``. An existing one that
    holds exactly this inventory's records is left untouched; one that holds different records
    (written from another inventory) is never replaced, because the schemes materialized with it
    would then describe videos that are not there.

    ``official`` is the publisher's split (record key -> split) for the rules that need it (see
    :func:`needs_official`); ``None`` otherwise. Nothing is ever written inside a datasets root
    (``datasets_roots``, default the resolved ones), even when ``work_root`` points there.

    Raises:
        UnknownKeyError: the dataset, pack or scheme is unknown.
        ConfigError: the pack ships no ``videos.jsonl.gz`` for the dataset (use
            :func:`materialize_dataset`), there is no inventory at ``inventory``, the rule needs
            ``official`` and it is ``None``, or the materialized folder would be inside a
            datasets root.
        ContractError: the scheme's rule cannot be recomputed, a pin does not match, the
            inventory repeats a video, the recomputed rows do not hash to the published value,
            or the materialized ``videos.jsonl.gz`` holds different records.
    """
    scheme = _resolve(ref)
    if not scheme.ships_videos:
        raise ConfigError(
            f"{scheme.ref}: the pack ships no key list for {scheme.dataset}, so every list of it "
            "is materialised at once, never one scheme alone",
            hint=f"run: dfwb protocols materialize {scheme.dataset} (from Python, "
            "materialize_dataset)",
        )
    materialized = materialized_dir(work_root, scheme.dataset)
    check_outside_datasets_roots(materialized, datasets_roots, what="the materialized split")
    rule, params = scheme.card.rule, scheme.card.params
    if rule not in RULES:
        raise ContractError(
            f"{scheme.ref}: its rule {rule!r} cannot be recomputed from an inventory",
            hint="only a scheme built by dfwb protocols build can be materialized "
            f"(rules: {', '.join(RULES)})",
        )
    if not inventory.is_file():
        raise ConfigError(
            f"{scheme.dataset}: no inventory at {inventory}",
            hint=f"run: dfwb inventory build {scheme.dataset}",
        )
    if rule_needs_official(rule, params) and official is None:
        raise ConfigError(
            f"{scheme.ref}: the {rule} rule needs the publisher's official split, and none was "
            "given",
            hint="run dfwb protocols materialize, which reads it with the dataset's inventory "
            "builder",
        )
    records = _read_records(inventory, scheme.dataset_card, local_attrs)
    assignment = assign_rule(
        rule, params, records, official=official, is_real=_is_real_from(scheme.labels, scheme.ref)
    )
    rows = rows_from_assignment(assignment)
    sha256 = split_sha256(rows)
    if sha256 != scheme.card.sha256:
        raise ContractError(
            f"{scheme.ref}: materialized hash {sha256} != published {scheme.card.sha256}",
            hint=_VERIFY_HINT,
        )

    videos = materialized / "videos.jsonl.gz"
    if videos.is_file():
        _check_same_videos(videos, records, scheme.dataset)
    (materialized / "splits").mkdir(parents=True, exist_ok=True)
    if not videos.is_file():
        write_jsonl(videos, records)
    path = materialized / "splits" / f"{scheme.name}.tsv.gz"
    write_split_tsv(path, rows)
    return MaterializeResult(scheme.ref, path, sha256, True)


# ---------------------------------------------------------------------------------------------
# Materialising a recipe dataset that ships no key list
# ---------------------------------------------------------------------------------------------


def check_recipe_card(card: DatasetCard) -> None:
    """Check that a recipe card holds everything its lists are rebuilt and checked with.

    Raises:
        ContractError: the card lacks ``videos_sha256`` or ``pairs_sha256``, or a scheme's rule
            cannot be recomputed from an inventory.
    """
    missing = [name for name in ("videos_sha256", "pairs_sha256") if getattr(card, name) is None]
    if missing:
        raise ContractError(
            f"{card.id}: its card records no {' or '.join(missing)}, so lists rebuilt from an "
            "inventory could not be checked",
            hint="the pack was built before a recipe dataset could ship without its key lists; "
            "rebuild it with dfwb protocols build, which records these hashes",
        )
    for name, scheme in sorted(card.schemes.items()):
        if scheme.rule not in RULES:
            raise ContractError(
                f"{card.id}/{name}: its rule {scheme.rule!r} cannot be recomputed from an "
                "inventory",
                hint="only a scheme built by dfwb protocols build can be materialised "
                f"(rules: {', '.join(RULES)})",
            )


def _check_pairing_rule(pairs: Sequence[PairRecord], card: DatasetCard) -> None:
    used = sorted({pair.rule for pair in pairs})
    if used and used != [card.pairing_rule]:
        raise ContractError(
            f"{card.id}: the pairs given were drawn by {', '.join(repr(r) for r in used)}, and "
            f"the pack declares the pairing rule {card.pairing_rule!r}",
            hint="draw the pairs with the pairing rule the pack declares, as dfwb protocols "
            "materialize does with the dataset's inventory builder",
        )


def _running(pid: int) -> bool:
    """Whether ``pid`` is a live process on this host, other than this one."""
    if pid == os.getpid():
        return False  # an earlier process's leftover, under a pid now reused by this one
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, but another user's
    except (OverflowError, ValueError):
        return False
    return True


def _remove(path: Path, *, what: str) -> None:
    """Remove the folder ``path``; a failure is a warning, never silent and never fatal."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        _log.warning("cannot remove %s, %s: %s", path, what, exc.strerror or exc)


def _clear_leftovers(target: Path) -> None:
    """Remove the hidden siblings of ``target`` a killed run left behind.

    Each carries the pid of the run that made it: a folder whose run is still going on this host
    (a concurrent materialize) is left alone, and so is one whose name carries no pid.
    """
    for pattern in (f".{target.name}.tmp-*", f".{target.name}.old-*"):
        for leftover in target.parent.glob(pattern):
            pid = leftover.name.rsplit("-", 1)[1]
            if pid.isdigit() and not _running(int(pid)):
                _remove(leftover, what="left by an earlier run")


def _write_materialized(
    target: Path,
    *,
    videos: Sequence[VideoRecord],
    pairs: Sequence[PairRecord],
    schemes: Mapping[str, list[SplitRow]],
    hashes: Mapping[str, Any],
) -> None:
    """Write every list into a hidden sibling of ``target``, then swap it in for ``target``.

    Whatever was materialised before is replaced as a whole, so the folder never mixes lists
    from two versions of a pack; a failure part way, the swap included, leaves the previous
    folder in place. The hidden siblings a killed run left behind are removed first (see
    :func:`_clear_leftovers`).

    Raises:
        ConfigError: the folder cannot be written (no permission, no space left, ...).
    """
    _clear_leftovers(target)
    staging = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    retired = target.with_name(f".{target.name}.old-{os.getpid()}")
    try:
        (staging / "splits").mkdir(parents=True)
        write_jsonl(staging / "videos.jsonl.gz", videos)
        if pairs:
            write_jsonl(staging / "pairs.jsonl.gz", pairs)
        for name, rows in schemes.items():
            write_split_tsv(staging / "splits" / f"{name}.tsv.gz", rows)
        (staging / HASHES_FILE).write_text(
            json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if target.exists():
            target.rename(retired)
        try:
            staging.rename(target)
        except BaseException:
            if retired.exists() and not target.exists():
                retired.rename(target)
            raise
    except OSError as exc:
        raise ConfigError(
            f"cannot write the materialised lists to {target}: {exc.strerror or exc}",
            hint="check that you have permission to write under the work root (DFWB_WORK_ROOT) "
            "and that its disk has space left",
        ) from None
    finally:
        _remove(staging, what="the folder the lists were written to")
        _remove(retired, what="the folder the lists replaced")


def _counts_text(rebuilt: Mapping[str, int], published: Mapping[str, int]) -> str:
    """``train 121/val 37/test 41, published 122/37/41``."""
    names = [name for name in _SPLIT_ORDER if name in rebuilt or name in published]
    mine = "/".join(f"{name} {rebuilt.get(name, 0)}" for name in names)
    theirs = "/".join(str(published.get(name, 0)) for name in names)
    return f"{mine}, published {theirs}"


def _names_text(names: Iterable[str]) -> str:
    return ", ".join(sorted(names)) or "none"


def _read_provenance(dataset_dir: Path) -> PackProvenance | None:
    try:
        return read_model(dataset_dir / "PROVENANCE.json", PackProvenance)
    except ContractError:
        return None


def _mismatch(
    scheme: _Scheme,
    rows: Sequence[InventoryRecord],
    schemes: Mapping[str, list[SplitRow]],
    failed: Sequence[str],
    checked: int,
) -> ContractError:
    """What differs, and the likeliest cause, told from counts, compressions and versions alone.

    ``failed`` names the lists whose hashes differ: ``videos``, ``pairs`` or a scheme.
    """
    card, dataset = scheme.dataset_card, scheme.dataset
    parts: list[str] = []
    fewer = more = False
    for name in failed:
        counts = card.schemes[name].counts if name in card.schemes else None
        if counts is None:
            parts.append(name)
            continue
        published = {str(split): count for split, count in counts.items()}
        rebuilt = Counter(str(row.split) for row in schemes[name])
        parts.append(f"{name} ({_counts_text(rebuilt, published)})")
        fewer = fewer or sum(rebuilt.values()) < sum(published.values())
        more = more or sum(rebuilt.values()) > sum(published.values())
    held = {row.compression for row in rows if row.compression is not None}
    listed = list(card.compressions or ())
    message = (
        f"{dataset}: {len(failed)} of {checked} published hashes differ from what your inventory "
        f"gives: {', '.join(parts)}. Your inventory gives {len(rows)} videos; compressions: "
        f"{_names_text(held)} (the card lists {_names_text(listed)})"
    )
    rebuild = f"dfwb inventory build {dataset}"
    provenance = _read_provenance(scheme.dataset_dir)
    built_with = provenance.builder.get("version") if provenance is not None else None
    versions = sorted({row.builder.version for row in rows})
    if provenance is not None and built_with is not None and versions not in ([], [built_with]):
        message += (
            f"; the inventory was built by version {', '.join(versions)} of the {dataset} "
            f"inventory builder, the pack by version {built_with} (dfwb {provenance.dfwb})"
        )
        hint = (
            "rebuild the inventory with the dfwb version the pack's PROVENANCE.json records "
            f"({provenance.dfwb}) or a later one whose {dataset} inventory builder is still "
            f"version {built_with}: {rebuild}"
        )
    elif missing := [name for name in listed if name not in held]:
        hint = (
            f"your inventory has no {', '.join(missing)} video: a recipe needs every video of "
            f"the release in every compression the card lists ({', '.join(listed)}); add them, "
            f"then rebuild the inventory: {rebuild}"
        )
    elif fewer and not more:
        hint = (
            f"your copy lacks videos of the release ({card.release}): the rebuilt splits hold "
            f"fewer videos than the published ones; get every video of it, then rebuild the "
            f"inventory: {rebuild}"
        )
    elif more and not fewer:
        hint = (
            f"your copy holds videos the release ({card.release}) does not: the rebuilt splits "
            "hold more videos than the published ones; take out the videos that are not part of "
            f"it (stray files, or a later release's), then rebuild the inventory: {rebuild}"
        )
    elif more and fewer:
        hint = (
            f"your copy is not the release the pack describes ({card.release}): the rebuilt "
            "splits hold other videos than the published ones; get that release, then rebuild "
            f"the inventory: {rebuild}"
        )
    else:
        hint = (
            "your inventory has the release's videos in the published numbers, but some differ "
            "in label, method, identity or attributes: the copy may be another upstream release "
            f"than the card's ({card.release}), or its metadata files differ; check them, then "
            f"rebuild the inventory: {rebuild}"
        )
    return ContractError(message, hint=hint)


def materialize_dataset(
    ref: str | ProtocolRef,
    *,
    inventory: Path,
    official: Mapping[str, Split] | None,
    pairs: Sequence[PairRecord],
    work_root: Path,
    datasets_roots: Sequence[Path] | None = None,
    local_attrs: Collection[str] = (),
) -> DatasetMaterialization:
    """Rebuild every list of ``ref``'s dataset from ``inventory``; keep them if every hash matches.

    For a recipe dataset whose pack ships no key list. The inventory's rows of a compression the
    card's ``compressions`` does not list are left out: they are no part of the release. The
    videos are the other rows as
    :class:`~dfwb.core.records.VideoRecord`, without the attributes named in ``local_attrs``
    (facts about the local copy, which the builder declares and a pack never publishes); every
    scheme of the card is recomputed from its rule
    and parameters, as :func:`materialize` recomputes one; ``pairs`` are the dataset's pairs,
    drawn from the same inventory by the pairing rule the card declares (``pairing_rule``), which
    only the dataset's inventory builder can run. The video list, every split and the pair list
    are hashed exactly as the pack builder hashed them, and compared with the card's
    ``videos_sha256``, scheme ``sha256`` and ``pairs_sha256``.

    Only when every hash matches is anything written:
    ``<work_root>/<dataset>/materialized/{videos.jsonl.gz, pairs.jsonl.gz (when there are pairs),
    splits/<scheme>.tsv.gz, hashes.json}``, byte for byte the files a list pack would ship, plus
    the card hashes they matched, which :func:`~dfwb.protocols.protocol.load` checks against the
    installed card. The folder replaces any earlier one as a whole. A scheme or pin in ``ref`` is
    checked; every scheme is materialised either way.

    ``official`` is the publisher's split (record key -> split), needed when any scheme's rule
    reads it; ``None`` otherwise. Nothing is ever written inside a datasets root
    (``datasets_roots``, default the resolved ones).

    Raises:
        UnknownKeyError: the dataset, pack or scheme is unknown.
        ConfigError: there is no inventory at ``inventory``, a rule needs ``official`` and it is
            ``None``, the materialized folder would be inside a datasets root, or it cannot be
            written (no permission, no space left).
        ContractError: the card lacks ``videos_sha256`` or ``pairs_sha256``, a scheme's rule
            cannot be recomputed, a pin does not match, ``pairs`` were drawn by another rule than
            the card's, the inventory repeats a video, or any rebuilt list does not hash to its
            published value. The message then names every list that differs, with the rebuilt
            and published split counts of each scheme, the videos and compressions the inventory
            gives, and the inventory builder versions when they are not the pack's, and the hint
            names the likeliest cause.
    """
    scheme = _resolve(ref)
    card = scheme.dataset_card
    dataset = scheme.dataset
    target = materialized_dir(work_root, dataset)
    check_outside_datasets_roots(target, datasets_roots, what="the materialised lists")
    check_recipe_card(card)
    if not inventory.is_file():
        raise ConfigError(
            f"{dataset}: no inventory at {inventory}",
            hint=f"run: dfwb inventory build {dataset}",
        )
    for name, scheme_card in sorted(card.schemes.items()):
        if rule_needs_official(scheme_card.rule, scheme_card.params) and official is None:
            raise ConfigError(
                f"{dataset}/{name}: the {scheme_card.rule} rule needs the publisher's official "
                "split, and none was given",
                hint="run dfwb protocols materialize, which reads it with the dataset's "
                "inventory builder",
            )
    _check_pairing_rule(pairs, card)

    rows = release_rows(read_jsonl(inventory, InventoryRecord), card)
    records = _video_records(rows, inventory, dataset, local_attrs)
    is_real = _is_real_from(
        scheme.labels,
        dataset,
        hint="the inventory was made from another release, or by another version of the "
        f"{dataset} inventory builder than the pack's: rebuild it with the dfwb version the "
        "pack's PROVENANCE.json records, or a later one with the same builder version: "
        f"dfwb inventory build {dataset}",
    )
    schemes: dict[str, list[SplitRow]] = {}
    for name, scheme_card in sorted(card.schemes.items()):
        needed = official if rule_needs_official(scheme_card.rule, scheme_card.params) else None
        assignment = assign_rule(
            scheme_card.rule, scheme_card.params, records, official=needed, is_real=is_real
        )
        schemes[name] = rows_from_assignment(assignment)
    unique_pairs = sorted(set(pairs), key=lambda p: (p.real_key, p.fake_key, p.rule))

    videos_sha256 = records_sha256(records)
    pairs_sha256 = records_sha256(unique_pairs)
    scheme_sha256 = {name: split_sha256(rows) for name, rows in schemes.items()}
    checks = [
        ("videos", videos_sha256, card.videos_sha256),
        *((name, sha256, card.schemes[name].sha256) for name, sha256 in scheme_sha256.items()),
        ("pairs", pairs_sha256, card.pairs_sha256),
    ]
    failed = [what for what, sha256, published in checks if sha256 != published]
    if failed:
        raise _mismatch(scheme, rows, schemes, failed, len(checks))

    hashes = {
        "videos_sha256": videos_sha256,
        "pairs_sha256": pairs_sha256,
        "schemes": scheme_sha256,
    }
    _write_materialized(target, videos=records, pairs=unique_pairs, schemes=schemes, hashes=hashes)
    return DatasetMaterialization(
        dataset=dataset,
        path=target,
        videos_sha256=videos_sha256,
        pairs_sha256=pairs_sha256,
        schemes=scheme_sha256,
        n_videos=len(records),
        n_pairs=len(unique_pairs),
    )
