"""Generic split, benchmark and pairing rules (contract C3a): pure, deterministic, stdlib only.

Every published scheme hash depends on the exact behaviour of these functions, so the details
below are part of the contract, not implementation choices: the md5 carve formula and its
thresholds, the benchmark's pool order, group order and single shared random draw, and the pairing
rule's resolution order. Changing any of them changes a scheme's membership and therefore needs a
new major pack version.

The rules work on any record with ``key``, ``compression``, ``identity``, ``target_id`` and
``source_id`` -- a :class:`~dfwb.core.records.VideoRecord` or, by duck typing, an
:class:`~dfwb.core.records.InventoryRecord`. Keys are ``<task>/<local key>``: the task is the part
before the first ``/`` and the local key is the rest.

An :data:`Assignment` maps ``(key, compression)`` to a split. Records a rule does not assign are
simply absent; ``exclude`` is reserved for explicit exclusions.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from dfwb.core.errors import ContractError
from dfwb.core.records import PairRecord

__all__ = [
    "BENCHMARK_REALS",
    "Assignment",
    "BenchmarkSpec",
    "RuleRecord",
    "Split",
    "assign_72_14_14",
    "assign_80_20",
    "assign_all_test",
    "assign_benchmark",
    "assign_official",
    "assign_official_plus_80_20",
    "carve_key",
    "local_key",
    "md5_mod_100",
    "resolve_pairs",
    "task_of",
]

Split = Literal["train", "val", "test", "exclude"]
Assignment = dict[tuple[str, str | None], Split]

_SPLITS: frozenset[str] = frozenset({"train", "val", "test", "exclude"})

# ident-72-14-14: h < 14 -> val, 14 <= h < 28 -> test, else train. 80-20: h < 20 -> val.
_VAL_72_14_14 = 14
_TEST_72_14_14 = 28
_VAL_80_20 = 20

# The record fields a benchmark may stratify on; "task" is the key's task prefix.
_STRATA_FIELDS: tuple[str, ...] = ("identity", "target_id", "source_id", "task")


class RuleRecord(Protocol):
    """What the rules read from a record."""

    @property
    def key(self) -> str: ...
    @property
    def compression(self) -> str | None: ...
    @property
    def identity(self) -> str | None: ...
    @property
    def target_id(self) -> str | None: ...
    @property
    def source_id(self) -> str | None: ...


# ---------------------------------------------------------------------------------------------
# Keys and hashes
# ---------------------------------------------------------------------------------------------


def md5_mod_100(text: str) -> int:
    """The carve hash: ``int(md5(text as UTF-8), 16) % 100``."""
    return int(hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest(), 16) % 100


def local_key(key: str) -> str:
    """The part of ``key`` after the first ``/`` (the whole key if it has none)."""
    _, sep, rest = key.partition("/")
    return rest if sep else key


def task_of(key: str) -> str:
    """The part of ``key`` before the first ``/`` (``""`` if it has none)."""
    task, sep, _ = key.partition("/")
    return task if sep else ""


def carve_key(record: RuleRecord) -> str:
    """What the carves hash: the record's identity if it has one, else its local key.

    Hashing the identity keeps every video of one person on the same side of a carve; the local
    key (not the full key) keeps the same video under two tasks on the same side too.
    """
    return str(record.identity) if record.identity else local_key(record.key)


# ---------------------------------------------------------------------------------------------
# Split rules
# ---------------------------------------------------------------------------------------------


def assign_all_test(records: Iterable[RuleRecord]) -> Assignment:
    """Every record is ``test``."""
    return {(r.key, r.compression): "test" for r in records}


def _split_72_14_14(record: RuleRecord) -> Split:
    h = md5_mod_100(carve_key(record))
    if h < _VAL_72_14_14:
        return "val"
    if h < _TEST_72_14_14:
        return "test"
    return "train"


def _split_80_20(record: RuleRecord) -> Split:
    return "val" if md5_mod_100(carve_key(record)) < _VAL_80_20 else "train"


def assign_72_14_14(records: Iterable[RuleRecord]) -> Assignment:
    """Identity-disjoint 72/14/14 carve: hash < 14 is ``val``, < 28 is ``test``, else ``train``."""
    return {(r.key, r.compression): _split_72_14_14(r) for r in records}


def assign_80_20(records: Iterable[RuleRecord]) -> Assignment:
    """Identity-disjoint 80/20 carve: hash < 20 is ``val``, else ``train``."""
    return {(r.key, r.compression): _split_80_20(r) for r in records}


def _check_official(official: Mapping[str, str]) -> None:
    for key, split in official.items():
        if split not in _SPLITS:
            raise ContractError(
                f"official split for {key!r} is {split!r}, not one of {sorted(_SPLITS)}",
                hint="the builder's official split reader must return train, val, test or exclude",
            )


def assign_official(records: Iterable[RuleRecord], official: Mapping[str, Split]) -> Assignment:
    """The publisher's split: ``official`` maps a record key to its split.

    The split applies to every compression of that key. Records whose key is not in ``official``
    are left unassigned, and keys in ``official`` with no record are ignored.

    Raises:
        ContractError: a value of ``official`` is not a split.
    """
    _check_official(official)
    return {(r.key, r.compression): official[r.key] for r in records if r.key in official}


def assign_official_plus_80_20(
    records: Iterable[RuleRecord],
    official: Mapping[str, Split],
    *,
    policy: Literal["test-only-official", "official-train-test"],
) -> Assignment:
    """The publisher's test split, plus train/val carved 80/20 by identity.

    ``test-only-official``
        ``test`` is the official test; every other record is carved into train/val.
    ``official-train-test``
        ``test`` is the official test; the official train records are carved into train/val;
        records outside the official split stay unassigned.

    Under either policy an official ``exclude`` stays ``exclude`` and is never carved.

    Raises:
        ContractError: ``official`` holds a ``val`` row (this rule carves val itself), a value that
            is not a split, or ``policy`` is unknown.
    """
    if policy not in ("test-only-official", "official-train-test"):
        raise ContractError(
            f"unknown official+80-20 policy {policy!r}",
            hint="use 'test-only-official' or 'official-train-test'",
        )
    _check_official(official)
    official_val = sorted(key for key, split in official.items() if split == "val")
    if official_val:
        raise ContractError(
            f"official split has {len(official_val)} val row(s) (first: {official_val[0]!r}), "
            f"but policy {policy!r} carves val from train",
            hint="use the 'official' rule for a dataset whose publisher defines val",
        )
    assignment: Assignment = {}
    for r in records:
        split = official.get(r.key)
        if split in ("test", "exclude"):
            assignment[(r.key, r.compression)] = split
        elif policy == "test-only-official" or split == "train":
            assignment[(r.key, r.compression)] = _split_80_20(r)
    return assignment


# ---------------------------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------------------------


#: How a benchmark draws its reals (step 5 of :func:`assign_benchmark`, without ``k_real_cap``), in
#: the words a benchmark scheme's rationale uses. Reals come from the same pool as the fakes; a
#: pool with fewer reals than fakes drawn keeps every one, so the subset is not class-balanced.
BENCHMARK_REALS: Final = "with as many reals as fakes, or every real when there are fewer"


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """A benchmark subset: ``k_fake`` fakes (per stratum, or in total), then up to as many reals.

    Attributes:
        k_fake: Fakes drawn per stratum when ``strata`` is set, otherwise in total.
        strata: Fields to stratify fakes on: ``identity``, ``target_id``, ``source_id``,
            ``task`` (the key's task prefix).
        k_real_cap: Optional ceiling on the number of reals, applied after balancing.
        exclude_tasks: Tasks dropped before anything is drawn.
        seed: Seed of the single random generator every draw shares.
        compressions: The compressions the benchmark is defined at: records of any other
            compression (or of none) are dropped before anything is drawn, so the subset does not
            depend on which other compressions a local copy holds. ``None``: every record.

    Raises:
        ContractError: a stratum is not one of those four fields, or ``compressions`` is empty.
    """

    k_fake: int
    strata: tuple[str, ...] = ()
    k_real_cap: int | None = None
    exclude_tasks: tuple[str, ...] = ()
    seed: int = 0
    compressions: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        unknown = [name for name in self.strata if name not in _STRATA_FIELDS]
        if unknown:
            raise ContractError(
                f"unknown benchmark stratum {unknown[0]!r}",
                hint=f"strata are drawn from {', '.join(_STRATA_FIELDS)}",
            )
        if self.compressions is not None and not self.compressions:
            raise ContractError(
                "a benchmark's compressions must name at least one compression",
                hint="use None to draw from every record",
            )


def _rank(task_rank: Mapping[str, int], key: str) -> int:
    task = task_of(key)
    try:
        return task_rank[task]
    except KeyError:
        raise ContractError(
            f"task {task!r} of {key!r} has no rank in the task table",
            hint="task_rank must rank every task of the records (its position in the builder)",
        ) from None


def _stratum(record: RuleRecord, name: str) -> object:
    value = task_of(record.key) if name == "task" else getattr(record, name)
    return value or ""


def _draw_fakes[R: RuleRecord](pool: list[R], spec: BenchmarkSpec, rng: random.Random) -> list[R]:
    if spec.k_fake <= 0 or not pool:
        return []
    if not spec.strata:
        return list(pool) if len(pool) <= spec.k_fake else rng.sample(pool, spec.k_fake)
    groups: dict[tuple[object, ...], list[R]] = {}
    for record in pool:
        groups.setdefault(tuple(_stratum(record, name) for name in spec.strata), []).append(record)
    selected: list[R] = []
    for group in sorted(groups, key=lambda g: tuple(str(x) for x in g)):
        members = groups[group]
        selected.extend(
            members if len(members) <= spec.k_fake else rng.sample(members, spec.k_fake)
        )
    return selected


def _draw_reals[R: RuleRecord](
    pool: list[R], n_fakes: int, cap: int | None, rng: random.Random
) -> list[R]:
    n_fakes = max(n_fakes, 0)
    selected = list(pool) if len(pool) <= n_fakes else rng.sample(pool, n_fakes)
    if cap is not None and cap >= 0 and len(selected) > cap:
        selected = rng.sample(selected, cap)  # samples the draw order, never a re-sorted list
    return selected


def assign_benchmark[R: RuleRecord](
    records: Iterable[R],
    *,
    spec: BenchmarkSpec,
    is_real: Callable[[R], bool],
    task_rank: Mapping[str, int],
    pool_keys: Collection[str] | None,
) -> Assignment:
    """A seeded test subset: stratified fakes, then as many reals, or every real if fewer.

    The steps, in this exact order, share one ``random.Random(spec.seed)``:

    1. The pool is the records whose key is in ``pool_keys`` (every compression of it) -- the
       dataset's official test. ``None``, or an empty collection (an official scheme that
       publishes no test), means every record.
    2. Records of ``spec.exclude_tasks`` are dropped, and so, when ``spec.compressions`` is set,
       are records of any other compression.
    3. ``is_real`` splits the pool into fakes and reals; each is sorted by
       ``(local key, task_rank[task], compression or "")``.
    4. Fakes: with strata, they are grouped by the tuple of their strata values (a missing value is
       ``""``); groups are visited in ``tuple(str(v) for v in group)`` order, and a group larger
       than ``k_fake`` is drawn with ``rng.sample``. Without strata, all fakes are kept if there
       are at most ``k_fake``, else ``rng.sample(fakes, k_fake)``. ``k_fake <= 0`` draws none.
    5. Reals: all are kept if there are at most as many as the fakes drawn, else
       ``rng.sample(reals, n_fakes)``; then, if more than ``k_real_cap`` remain, a second
       ``rng.sample`` keeps ``k_real_cap`` of them.

    Every drawn record is ``test``.

    Raises:
        ContractError: a record's task has no entry in ``task_rank``.
    """
    pool = list(records)
    if pool_keys:
        keys = frozenset(pool_keys)
        pool = [r for r in pool if r.key in keys]
    if spec.exclude_tasks:
        excluded = frozenset(spec.exclude_tasks)
        pool = [r for r in pool if task_of(r.key) not in excluded]
    if spec.compressions is not None:
        kept = frozenset(spec.compressions)
        pool = [r for r in pool if r.compression in kept]

    def order(record: R) -> tuple[str, int, str]:
        return (local_key(record.key), _rank(task_rank, record.key), record.compression or "")

    fakes: list[R] = []
    reals: list[R] = []
    for record in pool:
        (reals if is_real(record) else fakes).append(record)
    fakes.sort(key=order)
    reals.sort(key=order)

    rng = random.Random(spec.seed)
    chosen_fakes = _draw_fakes(fakes, spec, rng)
    chosen_reals = _draw_reals(reals, len(chosen_fakes), spec.k_real_cap, rng)
    return {(r.key, r.compression): "test" for r in (*chosen_fakes, *chosen_reals)}


# ---------------------------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------------------------


def _values(found: str | Sequence[str] | None) -> list[str]:
    if found is None:
        return []
    if isinstance(found, str):
        return [found]
    return list(found)


def resolve_pairs[R: RuleRecord](
    records: Iterable[R],
    *,
    is_real: Callable[[R], bool],
    candidates: Callable[[R], str | Sequence[str] | None],
    fanout_cap: int | None,
    rule: str,
    task_rank: Mapping[str, int] | None = None,
) -> list[PairRecord]:
    """Pair each fake with its real(s).

    ``candidates(fake)`` returns a value, several values, or ``None``. Each non-empty value is
    resolved on its own:

    - if it is the local key of one or more reals, those reals are its pairs;
    - otherwise every real whose ``identity`` equals it is a candidate: they are ordered by
      ``(local key, task rank)`` and cut to the first ``fanout_cap`` (``None``: no cap).

    ``task_rank`` orders reals that share a local key; without it, they are ordered by task name.
    A fake with no resolvable value gets no pair. Pairs are between keys (every compression of a
    record is the same record here), deduplicated, and sorted by ``(real_key, fake_key)``.

    Raises:
        ContractError: ``fanout_cap`` is negative, or ``task_rank`` is given and misses a real's
            task.
    """
    if fanout_cap is not None and fanout_cap < 0:
        raise ContractError(
            f"fan-out cap must be at least 0, got {fanout_cap}",
            hint="use None for no cap",
        )

    def order(key: str) -> tuple[str, int, str]:
        rank = 0 if task_rank is None else _rank(task_rank, key)
        return (local_key(key), rank, task_of(key))

    fakes: list[R] = []
    by_local: dict[str, set[str]] = {}
    by_identity: dict[str, set[str]] = {}
    for record in records:
        if not is_real(record):
            fakes.append(record)
            continue
        by_local.setdefault(local_key(record.key), set()).add(record.key)
        if record.identity:
            by_identity.setdefault(str(record.identity), set()).add(record.key)
    ordered_local = {value: sorted(keys, key=order) for value, keys in by_local.items()}
    ordered_identity = {value: sorted(keys, key=order) for value, keys in by_identity.items()}

    pairs: set[PairRecord] = set()
    for fake in fakes:
        for value in _values(candidates(fake)):
            if not value:
                continue
            if value in ordered_local:
                reals = ordered_local[value]
            else:
                reals = ordered_identity.get(value, [])
                if fanout_cap is not None:
                    reals = reals[:fanout_cap]
            pairs.update(PairRecord(fake.key, real_key, rule) for real_key in reals)
    return sorted(pairs, key=lambda p: (p.real_key, p.fake_key, p.rule))
