"""Recompute a recipe scheme from a local inventory and check it against its published hash (C3a).

A pack can publish a scheme as a *recipe*: its rule, the rule's parameters and the ``sha256`` of
the split rows the rule produces, but no key list. That is how a dataset whose terms forbid
redistributing even file names can still have a comparable split. :func:`materialize` rebuilds the
rows from the user's own inventory and keeps them only when they hash to the published value, so a
local split is either exactly the one the pack describes or it is refused.

:func:`assign_rule` runs a scheme's rule from its card's ``rule`` and ``params``. Pack building
(``dfwb protocols build``) produces every scheme through this same function, so a published scheme
and its materialized copy can never be computed two different ways.

The publisher's own split (the ``official`` rules, and a benchmark drawn from the official test)
comes from files only the dataset's inventory builder knows how to read, and this layer never
imports builders: the caller passes that split in as ``official``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError, did_you_mean
from dfwb.core.records import (
    InventoryRecord,
    LabelVocab,
    SchemeCard,
    VideoRecord,
    read_jsonl,
    split_sha256,
    to_video_record,
    write_jsonl,
    write_split_tsv,
)
from dfwb.protocols._yaml import read_card, read_labels
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
    "MaterializeResult",
    "assign_rule",
    "benchmark_params",
    "materialize",
    "needs_official",
    "rule_needs_official",
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


@dataclass(frozen=True)
class MaterializeResult:
    """A materialized scheme: its reference, the split file written, and its hash."""

    ref: str
    path: Path
    sha256: str
    matched: bool


# ---------------------------------------------------------------------------------------------
# Rules from a scheme card
# ---------------------------------------------------------------------------------------------


def benchmark_params(
    spec: BenchmarkSpec, *, task_order: Sequence[str], pool: Literal["official-test", "all"]
) -> dict[str, Any]:
    """The benchmark rule's parameters, as a scheme card records them.

    ``task_order`` lists the dataset's tasks in rank order (it breaks ties between records that
    share a local key); ``pool`` says whether the subset is drawn from the dataset's official test
    (``official-test``) or from every record (``all``). Together with the spec's own fields that is
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
    spec = BenchmarkSpec(
        k_fake=int(params["k_fake"]),
        strata=tuple(params.get("strata") or ()),
        k_real_cap=params.get("k_real_cap"),
        exclude_tasks=tuple(params.get("exclude_tasks") or ()),
        seed=int(params.get("seed", 0)),
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
            hint=f"run `dfwb protocols info {parsed.dataset}` to see its schemes",
        )
    scheme_card = card.schemes[name]
    canonical = f"{parsed.dataset}/{name}"
    if parsed.pin is not None:
        _check_pin(canonical, parsed.pin, pack.version, scheme_card.sha256)
    return _Scheme(canonical, parsed.dataset, name, scheme_card, lambda: read_labels(dataset_dir))


def needs_official(ref: str | ProtocolRef) -> bool:
    """Whether materializing ``ref`` needs the publisher's official split (see :func:`materialize`).

    Raises:
        UnknownKeyError, ContractError: as :func:`materialize` does when resolving ``ref``.
    """
    scheme = _resolve(ref)
    return rule_needs_official(scheme.card.rule, scheme.card.params)


def _is_real_from(labels: Callable[[], LabelVocab], ref: str) -> Callable[[VideoRecord], bool]:
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
                hint=_VERIFY_HINT,
            )
        return entry.get("binary") == 0

    return is_real


def _read_records(inventory: Path, dataset: str) -> list[VideoRecord]:
    records = [to_video_record(r) for r in read_jsonl(inventory, InventoryRecord)]
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


def materialize(
    ref: str | ProtocolRef,
    *,
    inventory: Path,
    official: Mapping[str, Split] | None,
    work_root: Path,
) -> MaterializeResult:
    """Recompute ``ref``'s split from ``inventory`` and keep it only if it matches the pack.

    The rule and its parameters come from the pack's scheme card (the scheme defaults to the
    card's default scheme); the records are ``inventory``'s rows as
    :class:`~dfwb.core.records.VideoRecord`. The rows are hashed exactly as a split file is, and
    only a match with the card's ``sha256`` writes
    ``<work_root>/<dataset>/materialized/{videos.jsonl.gz,splits/<scheme>.tsv.gz}``, where
    :func:`~dfwb.protocols.protocol.load` then finds them. A mismatch writes nothing, so the work
    root is left exactly as it was.

    ``official`` is the publisher's split (record key -> split) for the rules that need it (see
    :func:`needs_official`); ``None`` otherwise.

    Raises:
        UnknownKeyError: the dataset, pack or scheme is unknown.
        ConfigError: there is no inventory at ``inventory``, or the rule needs ``official`` and it
            is ``None``.
        ContractError: the scheme's rule cannot be recomputed, a pin does not match, the
            inventory repeats a video, or the recomputed rows do not hash to the published value.
    """
    scheme = _resolve(ref)
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
    records = _read_records(inventory, scheme.dataset)
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

    materialized = work_root / scheme.dataset / "materialized"
    (materialized / "splits").mkdir(parents=True, exist_ok=True)
    write_jsonl(materialized / "videos.jsonl.gz", records)
    path = materialized / "splits" / f"{scheme.name}.tsv.gz"
    write_split_tsv(path, rows)
    return MaterializeResult(scheme.ref, path, sha256, True)
