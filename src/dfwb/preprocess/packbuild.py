"""Build one dataset of a protocol pack from a local inventory (contract C3a).

This is where a dataset's inventory builder -- the dataset knowledge: its official split reader,
pairing rule, benchmark spec, label table and card -- meets the generic rules and the pack writer
of :mod:`dfwb.protocols`. :func:`build_dataset` reads the inventory, computes every scheme with
:func:`~dfwb.protocols.materialization.assign_rule` (the very function
``dfwb protocols materialize`` recomputes a recipe scheme with, so a published scheme and its
materialized copy cannot drift apart), resolves the fake/real pairs and writes the dataset's
files.

The output depends only on the inventory, the builder and the publisher's split files: nothing
holds a timestamp or a local path, so rebuilding from the same inventory gives the same bytes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from dfwb._version import __version__
from dfwb.core.errors import (
    ConfigError,
    ContractError,
    UnknownKeyError,
    did_you_mean,
    validation_messages,
)
from dfwb.core.hashing import sha256_text
from dfwb.core.paths import (
    ResolvedRoot,
    RootName,
    absolute,
    dataset_overrides,
    locate_dataset,
    require_root,
    resolve_roots,
)
from dfwb.core.records import (
    DatasetCard,
    InventoryRecord,
    PackCard,
    PackProvenance,
    PairRecord,
    SchemeCard,
    SplitRow,
    read_jsonl,
    records_sha256,
    to_video_record,
)
from dfwb.preprocess.inventory.base import BaseBuilder, SchemeSpec
from dfwb.preprocess.inventory.runner import (
    dataset_copies,
    get_builder,
    inventory_path,
    metadata_copy,
    read_inventory,
)
from dfwb.protocols._yaml import read_card
from dfwb.protocols.materialization import (
    OFFICIAL_RULES,
    DatasetMaterialization,
    assign_rule,
    benchmark_params,
    materialize_dataset,
    rule_needs_official,
)
from dfwb.protocols.packs import find_dataset
from dfwb.protocols.refs import ProtocolRef, parse_ref
from dfwb.protocols.rules import Assignment, Split, resolve_pairs
from dfwb.protocols.writer import rows_from_assignment, scheme_card_for, write_dataset_files

__all__ = [
    "BuildResult",
    "add_to_pack_yaml",
    "assign_scheme",
    "build_dataset",
    "locate_metadata_root",
    "materialize_recipe",
]

_log = logging.getLogger(__name__)

PACK_YAML: Final = "pack.yaml"

# What a decided distribution lets the pack publish, as the NOTICE words it.
_DISTRIBUTION_MEANING: Final = {
    "list": "these lists may be redistributed",
    "recipe": "the published pack carries no key list, only each split's rule and parameters "
    "and the hashes of the video, pair and split lists; dfwb protocols materialize rebuilds the "
    "lists from a local copy of the dataset and checks every hash",
}

# What a dataset folder holds, as the NOTICE words it: the lists themselves, or (a recipe, once
# the release build has taken its key lists out) only what rebuilds and checks them.
_HOLDS_LISTS: Final = (
    "Never media: this folder lists video keys, labels, split assignments and fake/real pairs, "
    "derived from the release's file names and metadata. It holds no videos, frames, crops, face "
    "boxes, landmarks or anything else derived from pixels."
)
_HOLDS_RECIPE: Final = (
    "Never media, and in the published pack no key list either: this folder holds the dataset "
    "card (each split's rule and parameters, and the hashes of the video, pair and split lists), "
    "the label vocabulary and this notice. dfwb protocols materialize rebuilds the lists from a "
    "local copy of the dataset. Nothing here is derived from pixels: no videos, frames, crops, "
    "face boxes or landmarks."
)


def _locate_hint(dataset_id: str) -> str:
    return f"locate the dataset folder (see dfwb datasets info {dataset_id})"


@dataclass(frozen=True)
class BuildResult:
    """What :func:`build_dataset` wrote: the folder, each scheme's card, and the row counts."""

    out: Path
    schemes: dict[str, SchemeCard]
    n_videos: int
    n_pairs: int


# ---------------------------------------------------------------------------------------------
# The dataset folder and the publisher's split
# ---------------------------------------------------------------------------------------------


def locate_metadata_root(
    builder: BaseBuilder, roots: Mapping[RootName, ResolvedRoot] | None = None
) -> Path:
    """The copy of the dataset folder the builder's official split files are read from.

    It is found exactly as ``dfwb inventory build`` finds it: the dataset override, or every copy
    of the folder across the datasets roots, of which the first holding all of the builder's
    ``metadata_files`` is the one read.

    Raises:
        ConfigError: the dataset folder is not found (the message names the roots searched).
    """
    resolved = resolve_roots() if roots is None else roots
    try:
        location = locate_dataset(
            builder.dataset_id, builder.expected_folder, resolved, overrides=dataset_overrides()
        )
    except ConfigError as exc:
        raise ConfigError(exc.message, hint=_locate_hint(builder.dataset_id)) from None
    return metadata_copy(builder, dataset_copies(builder, location, resolved)).path


def _official_reader(
    builder: BaseBuilder,
    records: Sequence[InventoryRecord],
    dataset_dir: Callable[[], Path | None],
) -> Callable[[], dict[str, Split]]:
    """Read the publisher's split once, on first use, from the folder ``dataset_dir`` gives."""
    cache: list[dict[str, Split]] = []

    def read() -> dict[str, Split]:
        if not cache:
            folder = dataset_dir()
            if folder is None:
                raise ConfigError(
                    f"{builder.dataset_id}: the publisher's split is read from the dataset "
                    "folder, and no folder was given",
                    hint=_locate_hint(builder.dataset_id),
                )
            cache.append(builder.official_splits(folder, records))
        return cache[0]

    return read


# ---------------------------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------------------------


def _scheme_spec(builder: BaseBuilder, scheme: str) -> SchemeSpec:
    try:
        return builder.schemes[scheme]
    except KeyError:
        raise UnknownKeyError(
            f"{builder.dataset_id}: unknown scheme {scheme!r}"
            f"{did_you_mean(scheme, builder.schemes)}",
            hint="schemes: " + ", ".join(builder.schemes),
        ) from None


def _official_test_exists(
    records: Sequence[InventoryRecord], official: Callable[[], Mapping[str, Split]]
) -> bool:
    """Whether the publisher's split puts at least one of ``records`` in ``test``."""
    split = official()
    return any(split.get(record.key) == "test" for record in records)


def _scheme_params(
    builder: BaseBuilder,
    spec: SchemeSpec,
    records: Sequence[InventoryRecord],
    official: Callable[[], Mapping[str, Split]],
) -> dict[str, Any]:
    """The rule parameters a scheme card records: the builder's, plus a benchmark's full spec.

    A benchmark is drawn from the official test when the dataset has an official scheme whose
    split puts records in ``test``. It is drawn from every record when there is no official
    scheme, or when the publisher labels no test (a release that publishes val only), and the
    card then says so, so recomputing it never needs the official split.
    """
    if spec.rule != "benchmark":
        return dict(spec.params)
    if builder.benchmark is None:
        raise ContractError(
            f"{builder.dataset_id}: a benchmark scheme needs the builder's benchmark spec",
            hint="set the builder's benchmark attribute (a BenchmarkSpec)",
        )
    has_official = any(other.rule in OFFICIAL_RULES for other in builder.schemes.values())
    from_official = has_official and _official_test_exists(records, official)
    return {
        **spec.params,
        **benchmark_params(
            builder.benchmark,
            task_order=[task.abbr for task in builder.tasks],
            pool="official-test" if from_official else "all",
        ),
    }


def _assign(
    builder: BaseBuilder,
    scheme: str,
    records: Sequence[InventoryRecord],
    official: Callable[[], Mapping[str, Split]],
) -> tuple[Assignment, dict[str, Any]]:
    spec = _scheme_spec(builder, scheme)
    params = _scheme_params(builder, spec, records, official)
    needed = official() if rule_needs_official(spec.rule, params) else None
    assignment = assign_rule(spec.rule, params, records, official=needed, is_real=builder.is_real)
    return assignment, params


def assign_scheme(
    builder: BaseBuilder,
    scheme: str,
    records: Sequence[InventoryRecord],
    *,
    dataset_dir: Path | None,
) -> Assignment:
    """The assignment of ``builder``'s scheme ``scheme`` over ``records``.

    It dispatches on the scheme's rule: ``official`` and ``official+ident-80-20`` read the
    publisher's split with ``builder.official_splits(dataset_dir, records)``; ``ident-72-14-14``
    and ``all-test`` need nothing else; ``benchmark`` uses the builder's benchmark spec, its
    visual real/fake label and its task order, and draws from the records the official split puts
    in ``test`` when the dataset has an official scheme (from every record when it has none, or
    when that split has no test).

    Raises:
        UnknownKeyError: ``scheme`` is not one of the builder's schemes.
        ConfigError: the rule reads the publisher's split and ``dataset_dir`` is ``None``.
    """
    records = list(records)
    return _assign(
        builder, scheme, records, _official_reader(builder, records, lambda: dataset_dir)
    )[0]


def _selected(builder: BaseBuilder, schemes: Sequence[str] | None) -> list[str]:
    if schemes is None:
        return list(builder.schemes)
    names: list[str] = []
    for name in schemes:
        _scheme_spec(builder, name)
        if name not in names:
            names.append(name)
    if builder.default_scheme not in names:
        raise ConfigError(
            f"{builder.dataset_id}: the selected schemes ({', '.join(names)}) leave out the "
            f"default scheme {builder.default_scheme!r}",
            hint=f"add --scheme {builder.default_scheme}: a dataset card always holds its "
            "default scheme",
        )
    return names


# ---------------------------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------------------------


def _read_records(
    builder: BaseBuilder, inventory: Path | None, roots: Mapping[RootName, ResolvedRoot]
) -> list[InventoryRecord]:
    dataset_id = builder.dataset_id
    if inventory is None:
        records = read_inventory(dataset_id, require_root("work", roots))
    else:
        if not inventory.is_file():
            raise ConfigError(
                f"{dataset_id}: no inventory at {inventory}",
                hint=f"run: dfwb inventory build {dataset_id}",
            )
        records = read_jsonl(inventory, InventoryRecord)
    stale = 0
    for record in records:
        if record.builder.id != dataset_id:
            raise ContractError(
                f"{dataset_id}: inventory row {record.key!r} was built by "
                f"{record.builder.id!r}, not {dataset_id!r}",
                hint=f"build this dataset's inventory: dfwb inventory build {dataset_id}",
            )
        stale += record.builder.version != builder.version
    if stale:
        _log.warning(
            "%s: %d inventory row(s) were built by another version of the builder (now %s); "
            "rebuild the inventory so the pack describes what the builder finds today",
            dataset_id,
            stale,
            builder.version,
        )
    return sorted(records, key=lambda r: (r.key, r.compression or ""))


def _is_inside(path: Path, parent: Path) -> bool:
    return path.is_relative_to(parent) or path.resolve().is_relative_to(parent.resolve())


def _check_outside_raw_data(
    out: Path, dataset_id: str, roots: Mapping[RootName, ResolvedRoot]
) -> None:
    """Raw data is never written to: refuse an output inside a datasets root or dataset folder."""
    datasets = roots.get("datasets")
    places = [("the datasets root", root) for root in (datasets.paths if datasets else ())]
    override = dataset_overrides().get(dataset_id)
    if override is not None:
        places.append(("the dataset folder", override[0]))
    for label, place in places:
        if _is_inside(out, place):
            raise ConfigError(
                f"refusing to write {out}: it is inside {label} {place}",
                hint="raw data is read-only; pass an --out outside every datasets root",
            )


def _source_listing_sha256(records: Sequence[InventoryRecord]) -> str:
    """sha256 of the sorted ``key<TAB>compression<TAB>relpath`` lines of the inventory."""
    lines = sorted(f"{r.key}\t{r.compression or ''}\t{r.relpath}\n" for r in records)
    return sha256_text("".join(lines))


def _pairs(builder: BaseBuilder, records: Sequence[InventoryRecord]) -> list[PairRecord]:
    if builder.pairing_rule is None:
        return []
    return resolve_pairs(
        records,
        is_real=builder.is_real,
        candidates=builder.pair_candidates,
        fanout_cap=builder.pairing_fanout,
        rule=builder.pairing_rule,
        task_rank=builder.task_rank(),
    )


def _notice(card: DatasetCard) -> str:
    """The dataset's NOTICE: its owner, how to get it, and what the pack holds (never media)."""
    homepage = card.homepage or "no homepage recorded"
    owner = (
        f"{card.name} ({card.id}) belongs to its publishers, and its videos to them and to the "
        f"people who appear in them. Homepage: {homepage}."
    )
    if card.paper is not None:
        venue = ", ".join(str(part) for part in (card.paper.venue, card.paper.year) if part)
        owner += f' Paper: "{card.paper.title}"' + (f" ({venue})." if venue else ".")
    licence = f"Licence: {card.license.summary}" + (
        f" ({card.license.url})" if card.license.url else ""
    )
    lines = [
        f"# {card.name}: notice",
        "",
        "## Owner",
        "",
        owner,
        "",
        "## Access",
        "",
        card.access,
        "",
        licence,
        "",
        "## What these files hold",
        "",
        _HOLDS_RECIPE if card.distribution == "recipe" else _HOLDS_LISTS,
        "",
        "## Terms",
        "",
        _terms(card),
    ]
    return "\n".join(lines) + "\n"


def _terms(card: DatasetCard) -> str:
    """The NOTICE's terms paragraph, which always agrees with the card's ``distribution``.

    Any note already recorded on ``card.terms`` (e.g. a maintainer's proposal awaiting review)
    is appended whether or not the distribution itself has been decided yet, so it is never
    silently dropped from the published notice.
    """
    if card.distribution == "undecided":
        text = (
            "Terms review pending: whether these lists may be redistributed has not been decided "
            "yet, so dataset.yaml records distribution: undecided. This notice is completed once "
            "the dataset's terms have been reviewed."
        )
    else:
        text = (
            f"Terms reviewed: dataset.yaml records distribution: {card.distribution}, so "
            f"{_DISTRIBUTION_MEANING[card.distribution]}."
        )
    return f"{text} {card.terms.notes}" if card.terms.notes else text


# ---------------------------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------------------------


def _keep_terms_review(card: DatasetCard, out: Path) -> DatasetCard:
    """``card`` with the ``distribution`` and ``terms`` already recorded in ``out/dataset.yaml``.

    Those two fields are decided after a build, by reviewing the dataset's terms; rebuilding
    (after a builder fix, say) keeps them instead of resetting the dataset to undecided. An
    existing card that cannot be read is refused, never overwritten.
    """
    if not (out / "dataset.yaml").is_file():
        return card
    try:
        existing = read_card(out)
    except ContractError as exc:
        raise ContractError(
            exc.message,
            hint="fix or remove that dataset.yaml: a rebuild keeps the terms review it records",
        ) from None
    return card.model_copy(update={"distribution": existing.distribution, "terms": existing.terms})


def build_dataset(
    dataset_id: str,
    *,
    out: Path,
    inventory: Path | None = None,
    schemes: Sequence[str] | None = None,
    roots: Mapping[RootName, ResolvedRoot] | None = None,
) -> BuildResult:
    """Write ``out/`` as dataset ``dataset_id`` of a protocol pack, from its local inventory.

    Args:
        dataset_id: The inventory builder's registry key.
        out: The dataset's folder in the pack, normally ``<pack root>/<dataset_id>``.
        inventory: The inventory to build from (default: the work root's).
        schemes: The schemes to build (default: every scheme of the builder); the default scheme
            must be among them.
        roots: Resolved roots (default: :func:`~dfwb.core.paths.resolve_roots`).

    The videos are the inventory's rows; each scheme's rows come from its rule (see
    :func:`assign_scheme`) and its card records the rule, its parameters, the split counts and
    the rows' hash; the pairs come from the builder's pairing rule; the dataset card also records
    the hashes of the video and pair lists (``videos_sha256``, ``pairs_sha256``) and the pairing
    rule, so the dataset can later be published as a recipe, without its key lists, and still be
    rebuilt and checked by ``dfwb protocols materialize``; the labels, the rest of the dataset
    card and the notice come from the builder, except that a rebuild keeps the ``distribution`` and
    ``terms`` already recorded in ``out/dataset.yaml`` (the maintainer's terms review, not the
    builder's). ``PROVENANCE.json`` records the builder, the dfwb version, each scheme's rule and
    parameters, and the hash of the inventory's ``key``/``compression``/``relpath`` listing. The
    dataset folder is located (as ``dfwb inventory build`` locates it) only when a scheme reads
    the publisher's split.

    Raises:
        UnknownKeyError: the builder or a scheme is unknown.
        ConfigError: there is no inventory, the selection leaves out the default scheme, the
            dataset folder is needed and not found, or ``out`` is inside raw data.
        ContractError: an inventory row comes from another builder, ``out`` already holds a
            ``dataset.yaml`` that cannot be read, or a written file would break the pack contract.
    """
    builder = get_builder(dataset_id)
    resolved = resolve_roots() if roots is None else roots
    names = _selected(builder, schemes)
    out = absolute(out)
    _check_outside_raw_data(out, dataset_id, resolved)
    records = _read_records(builder, inventory, resolved)
    official = _official_reader(builder, records, lambda: locate_metadata_root(builder, resolved))

    rows: dict[str, list[SplitRow]] = {}
    cards: dict[str, SchemeCard] = {}
    rules: dict[str, dict[str, Any]] = {}
    for name in names:
        spec = builder.schemes[name]
        assignment, params = _assign(builder, name, records, official)
        rows[name] = rows_from_assignment(assignment)
        cards[name] = scheme_card_for(
            rows[name],
            kind=spec.kind,
            rule=spec.rule,
            source=spec.source,
            params=params,
            rationale=spec.rationale,
        )
        rules[name] = {"rule": spec.rule, "params": params}

    videos = [to_video_record(record) for record in records]
    pairs = _pairs(builder, records)
    card = _keep_terms_review(builder.dataset_card(cards), out).model_copy(
        update={
            "videos_sha256": records_sha256(videos),
            "pairs_sha256": records_sha256(pairs),
            "pairing_rule": builder.pairing_rule,
        }
    )
    provenance = PackProvenance(
        builder={"id": builder.dataset_id, "version": builder.version},
        dfwb=__version__,
        source_listing_sha256=_source_listing_sha256(records),
        rules=rules,
    )
    write_dataset_files(
        out,
        videos=videos,
        schemes=rows,
        pairs=pairs,
        card=card,
        labels=builder.label_vocab(),
        provenance=provenance,
        notice=_notice(card),
    )
    return BuildResult(out, cards, len(records), len(pairs))


def materialize_recipe(
    ref: str | ProtocolRef,
    *,
    inventory: Path | None = None,
    roots: Mapping[RootName, ResolvedRoot] | None = None,
) -> DatasetMaterialization:
    """Rebuild every list of a recipe dataset shipped without them, with its inventory builder.

    What only the dataset's builder knows is supplied here: the publisher's split, read from the
    dataset folder (located as ``dfwb inventory build`` locates it) only when a scheme's rule reads
    it, and the pairs, drawn by the builder's pairing rule, which must be the rule the pack's card
    declares. :func:`~dfwb.protocols.materialization.materialize_dataset` then rebuilds and checks
    every list and writes them under the work root.

    Args:
        ref: The dataset (a scheme or pin in it is checked; every scheme is materialised).
        inventory: The inventory to rebuild from (default: the work root's).
        roots: Resolved roots (default: :func:`~dfwb.core.paths.resolve_roots`).

    Raises:
        UnknownKeyError: the dataset, pack, scheme or builder is unknown.
        ConfigError: there is no inventory, or the dataset folder is needed and not found.
        ContractError: the builder pairs by another rule than the card declares, or as
            :func:`~dfwb.protocols.materialization.materialize_dataset` raises, notably when a
            rebuilt list does not hash to its published value.
    """
    parsed = parse_ref(ref) if isinstance(ref, str) else ref
    resolved = resolve_roots() if roots is None else roots
    work_root = require_root("work", resolved)
    dataset_id = parsed.dataset
    card = read_card(find_dataset(dataset_id, pack=parsed.pack).dataset_dir(dataset_id))
    builder = get_builder(dataset_id)
    if builder.pairing_rule != card.pairing_rule:
        raise ContractError(
            f"{dataset_id}: the pack pairs fakes with reals by the rule {card.pairing_rule!r}, "
            f"and this dfwb's {dataset_id} inventory builder pairs by {builder.pairing_rule!r}",
            hint="install the dfwb version the pack was built with (its PROVENANCE.json "
            "records it)",
        )
    source = inventory if inventory is not None else inventory_path(dataset_id, work_root)
    records = read_jsonl(source, InventoryRecord) if source.is_file() else []
    official: dict[str, Split] | None = None
    if source.is_file() and any(
        rule_needs_official(scheme.rule, scheme.params) for scheme in card.schemes.values()
    ):
        official = builder.official_splits(locate_metadata_root(builder, resolved), records)
    datasets = resolved.get("datasets")
    return materialize_dataset(
        parsed,
        inventory=source,
        official=official,
        pairs=_pairs(builder, records),
        work_root=work_root,
        datasets_roots=datasets.paths if datasets else (),
    )


def add_to_pack_yaml(pack_root: Path, dataset_id: str) -> bool:
    """List ``dataset_id`` in ``pack_root/pack.yaml`` if it is not listed yet.

    The ``datasets`` list is kept sorted and every other key is kept as it was. Returns whether
    the file changed.

    Raises:
        ConfigError: ``pack_root`` has no ``pack.yaml``.
        ContractError: ``pack.yaml`` is not a valid pack card.
    """
    path = pack_root / PACK_YAML
    if not path.is_file():
        raise ConfigError(
            f"no {PACK_YAML} at {path}",
            hint="build into <pack root>/<dataset id> of a pack made with dfwb protocols new-pack",
        )
    try:
        data = yaml.safe_load(path.read_text("utf-8"))
        card = PackCard.model_validate(data)
    except yaml.YAMLError as exc:
        raise ContractError(f"{path}: invalid YAML ({exc})", hint="fix the YAML syntax") from None
    except ValidationError as exc:
        raise ContractError(
            f"{path}: " + "; ".join(validation_messages(exc)), hint="not a valid PackCard"
        ) from None
    if dataset_id in card.datasets:
        return False
    data["datasets"] = sorted({*card.datasets, dataset_id})
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), "utf-8", newline="\n"
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return True
