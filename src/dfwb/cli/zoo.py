"""``dfwb zoo``: list, inspect, fetch, verify and check parity of registered zoo adapters
(contract C4 adapters over published third-party detectors).

A thin layer over :mod:`dfwb.zoo`: every command resolves an adapter through the ``detectors``
registry and calls straight into the already-tested zoo machinery (the card schema, the weight
manager, the licence gate, the code strategies, the parity harness); no adapter-resolution or
verification logic is duplicated here. Every command imports its heavy dependencies lazily inside
its own body, so this module -- and every subcommand that never actually scores anything --
imports and runs without torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from dfwb.cli._output import emit_json, json_option, table
from dfwb.core.errors import ConfigError

if TYPE_CHECKING:
    from dfwb.core.registry import Entry
    from dfwb.zoo.card import AdapterCard


@click.group()
def zoo() -> None:
    """List, inspect, fetch and check parity of registered zoo adapters."""


def _load_adapter(name: str) -> tuple[Entry, Any]:
    """The registry entry and adapter instance for ``name`` (did-you-mean on an unknown name)."""
    from dfwb.core.plugins import get_registry

    registry = get_registry("detectors")
    entry = registry.entry(name)
    adapter_cls = registry.load(entry.qualified_key)
    return entry, adapter_cls()


def _input_summary(card: AdapterCard) -> str:
    width, height = card.input.size
    return f"{card.input.crop} {width}x{height}x{card.input.frames}f"


def _license_summary(card: AdapterCard) -> str:
    text = card.license.code
    if card.license.weights and card.license.weights != card.license.code:
        text += f"/{card.license.weights}"
    if card.license.requires_ack:
        text += " (ack)"
    return text


# ============================================================================================
# list
# ============================================================================================


def _list_row(entry: Entry, card: AdapterCard) -> dict[str, Any]:
    return {
        "name": card.name,
        "display_name": card.display_name,
        "provider": entry.provider,
        "code_strategy": card.code_strategy,
        "license": card.license.model_dump(mode="json"),
        "input": card.input.model_dump(mode="json"),
        "n_weights": len(card.weights),
        "reported": len(card.reported),
        "parity": len(card.parity),
    }


@zoo.command("list")
@json_option
def list_(as_json: bool) -> None:
    """List every registered zoo adapter: licence, input and reported/parity counts."""
    from dfwb.core.plugins import get_registry

    registry = get_registry("detectors")
    rows = [(entry, registry.load(entry.qualified_key)().card) for entry in registry.entries()]

    if as_json:
        emit_json([_list_row(entry, card) for entry, card in rows])
        return
    if not rows:
        click.echo("no adapters registered")
        return
    table_rows = [
        [
            card.name,
            _license_summary(card),
            _input_summary(card),
            len(card.reported),
            len(card.parity),
        ]
        for _entry, card in rows
    ]
    click.echo(table(["NAME", "LICENSE", "INPUT", "REPORTED", "PARITY"], table_rows))


# ============================================================================================
# info
# ============================================================================================


@zoo.command("info")
@click.argument("name")
@json_option
def info(name: str, as_json: bool) -> None:
    """Show NAME's full adapter card."""
    from dfwb.core.licenses import is_accepted

    _entry, adapter = _load_adapter(name)
    card = adapter.card
    accepted = is_accepted(card.name) if card.license.requires_ack else None

    if as_json:
        emit_json({"card": card.model_dump(mode="json"), "license_accepted": accepted})
        return

    click.echo(f"{card.name}  ({card.display_name})")
    click.echo(f"contract_version: {'.'.join(map(str, card.contract_version))}")
    click.echo(f"code_strategy: {card.code_strategy}")
    if card.upstream is not None:
        click.echo(f"upstream: {card.upstream.repo} @ {card.upstream.commit}")
    license_line = f"license: code={card.license.code}"
    if card.license.weights:
        license_line += f" weights={card.license.weights}"
    license_line += f" requires_ack={card.license.requires_ack}"
    click.echo(license_line)
    if accepted is not None:
        click.echo(f"licence acknowledged: {accepted}")
    click.echo(f"input: {_input_summary(card)}")
    click.echo(f"polarity: {card.polarity}")
    if card.weights:
        click.echo("weights:")
        for w in card.weights:
            click.echo(f"  {w.id}: {w.format} sha256={w.sha256[:12]}.. bytes={w.bytes}")
    if card.reported:
        click.echo("reported:")
        for r in card.reported:
            click.echo(f"  {r.protocol}/{r.split} {r.metric}={r.value} (source: {r.source})")
    if card.parity:
        click.echo("parity:")
        for p in card.parity:
            click.echo(
                f"  {p.protocol}/{p.split} {p.metric}={p.value} "
                f"(tolerance={p.tolerance}, dfwb {p.dfwb_version}, {p.date})"
            )


# ============================================================================================
# fetch
# ============================================================================================


@zoo.command("fetch")
@click.argument("name")
@click.option(
    "--weights",
    "weights_id",
    default=None,
    metavar="ID",
    help="Weight variant id (only needed when the adapter has more than one).",
)
@click.option(
    "--accept-license",
    "accept_license",
    is_flag=True,
    help="Record the licence acknowledgement this adapter needs, if any.",
)
@json_option
def fetch(name: str, weights_id: str | None, accept_license: bool, as_json: bool) -> None:
    """Download and verify NAME's code and weights, ready for zoo:NAME to use."""
    from dfwb.zoo.adapter import accept_license as record_license_acceptance
    from dfwb.zoo.adapter import require_license_accepted
    from dfwb.zoo.source import select_weight
    from dfwb.zoo.strategies import ensure_clone
    from dfwb.zoo.weights import ensure_weights

    _entry, adapter = _load_adapter(name)
    card = adapter.card

    if accept_license:
        record_license_acceptance(card)

    require_license_accepted(card)

    code_root = None
    if card.code_strategy == "pinned-clone":
        code_root = ensure_clone(card)

    weight_info: dict[str, Any] | None = None
    if card.weights:
        spec = select_weight(card, weights_id)
        path = ensure_weights(card.name, spec)
        weight_info = {"id": spec.id, "path": str(path), "sha256": spec.sha256, "bytes": spec.bytes}
    elif weights_id is not None:
        raise ConfigError(
            f"zoo:{card.name}: has no weights, but --weights {weights_id!r} was given",
            hint=f"omit --weights for {card.name}",
        )

    if as_json:
        emit_json(
            {
                "name": card.name,
                "code": str(code_root) if code_root is not None else None,
                "weights": weight_info,
            }
        )
        return

    if code_root is None and weight_info is None:
        click.echo(f"{card.name}: nothing to fetch (no pinned-clone code, no weights)")
        return
    if code_root is not None:
        click.echo(f"code: {code_root}")
    if weight_info is not None:
        click.echo(f"weights {weight_info['id']}: {weight_info['path']}")


# ============================================================================================
# verify
# ============================================================================================


@zoo.command("verify")
@click.argument("name")
@json_option
def verify(name: str, as_json: bool) -> int:
    """Re-hash NAME's cached weights and check its code pin, without downloading anything."""
    from dfwb.core.errors import ContractError
    from dfwb.zoo.strategies import clone_cache_dir, head_commit, is_clean_worktree
    from dfwb.zoo.weights import cache_dir, verify_weights, weights_filename

    _entry, adapter = _load_adapter(name)
    card = adapter.card

    ok = True
    weight_rows: list[dict[str, Any]] = []
    for spec in card.weights:
        path = cache_dir(card.name, spec.sha256) / weights_filename(spec)
        if not path.is_file():
            weight_rows.append({"id": spec.id, "path": str(path), "status": "not-cached"})
            continue
        try:
            verify_weights(path, spec)
        except ContractError as exc:
            ok = False
            weight_rows.append(
                {"id": spec.id, "path": str(path), "status": "mismatch", "detail": exc.message}
            )
        else:
            weight_rows.append({"id": spec.id, "path": str(path), "status": "ok"})

    code_row: dict[str, Any] | None = None
    if card.code_strategy == "pinned-clone" and card.upstream is not None:
        dest = clone_cache_dir(card.name, card.upstream.commit)
        if not dest.is_dir():
            code_row = {"path": str(dest), "status": "not-cloned"}
        else:
            head = head_commit(dest)
            if head != card.upstream.commit:
                ok = False
                code_row = {"path": str(dest), "status": "wrong-commit", "head": head}
            elif not is_clean_worktree(dest):
                ok = False
                code_row = {"path": str(dest), "status": "dirty"}
            else:
                code_row = {"path": str(dest), "status": "ok"}

    if as_json:
        emit_json({"name": card.name, "ok": ok, "weights": weight_rows, "code": code_row})
        return 0 if ok else 4

    click.echo(f"{card.name}: {'ok' if ok else 'problems found'}")
    for row in weight_rows:
        detail = f" ({row['detail']})" if "detail" in row else ""
        click.echo(f"  weights {row['id']}: {row['status']}{detail}")
    if code_row is not None:
        click.echo(f"  code: {code_row['status']} ({code_row['path']})")
    if not weight_rows and code_row is None:
        click.echo("  nothing to verify (no weights, no pinned-clone code)")
    return 0 if ok else 4


# ============================================================================================
# licenses
# ============================================================================================


@zoo.command("licenses")
@json_option
def licenses_(as_json: bool) -> None:
    """List licence acknowledgements recorded on this machine."""
    from dfwb.core.licenses import all_accepted

    entries = sorted(all_accepted().items())

    if as_json:
        emit_json(
            [
                {"name": name, "license": acc.license, "accepted_at": acc.accepted_at}
                for name, acc in entries
            ]
        )
        return
    if not entries:
        click.echo("no licences acknowledged")
        return
    rows = [[name, acc.license, acc.accepted_at] for name, acc in entries]
    click.echo(table(["NAME", "LICENSE", "ACCEPTED_AT"], rows))


# ============================================================================================
# parity
# ============================================================================================


@zoo.command("parity")
@click.argument("name")
@click.option(
    "--protocol",
    "protocol_ref",
    default=None,
    help="Only check reported metrics for this protocol reference.",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.01,
    show_default=True,
    help="Allowed |measured - reported| before a metric is reported as failing.",
)
@json_option
def parity(name: str, protocol_ref: str | None, tolerance: float, as_json: bool) -> int:
    """Score NAME's parity set and compare it against its card's reported numbers.

    Writes the result into a local card overlay (``dfwb zoo info`` does not show it; the overlay
    is meant to be pasted into the adapter's own card in a PR).
    """
    from dfwb.zoo.parity import compare_parity, parity_path, write_parity_overlay

    _entry, adapter = _load_adapter(name)
    card = adapter.card

    claims = [c for c in card.reported if protocol_ref is None or c.protocol == protocol_ref]
    if protocol_ref is not None and not claims:
        raise ConfigError(
            f"zoo:{card.name}: no reported metrics for protocol {protocol_ref!r}",
            hint="omit --protocol to check every reported metric",
        )

    measured: dict[tuple[str, str, str], float] = {}
    pairs = sorted({(c.protocol, c.split) for c in claims})
    if pairs:
        from dfwb.core.records import read_scores
        from dfwb.eval.coverage import labels_and_scores
        from dfwb.eval.metrics import compute as compute_metric
        from dfwb.score.harness import score as run_score

        for protocol, split in pairs:
            result = run_score(f"zoo:{card.name}", protocol=protocol, split=split)
            scored = read_scores(result.csv_path)
            y, p, _kept = labels_and_scores(scored.rows, missing="exclude")
            metrics_needed = sorted(
                {c.metric for c in claims if c.protocol == protocol and c.split == split}
            )
            for metric in metrics_needed:
                measured[(protocol, split, metric)] = compute_metric(metric, y, p)

    filtered_card = card.model_copy(update={"reported": tuple(claims)})
    checks = compare_parity(filtered_card, measured, tolerance=tolerance)
    if checks:
        write_parity_overlay(card.name, [c.as_metric() for c in checks])
    ok = all(c.passed for c in checks)

    if as_json:
        emit_json(
            {
                "name": card.name,
                "ok": ok,
                "checks": [
                    {
                        "protocol": c.protocol,
                        "split": c.split,
                        "metric": c.metric,
                        "reported": c.reported,
                        "measured": c.measured,
                        "tolerance": c.tolerance,
                        "passed": c.passed,
                    }
                    for c in checks
                ],
                "overlay": str(parity_path(card.name)) if checks else None,
            }
        )
        return 0 if ok else 4

    if not checks:
        what = f" for protocol {protocol_ref!r}" if protocol_ref else ""
        click.echo(f"{card.name}: no reported metrics{what} to check")
        return 0
    rows = [
        [
            c.protocol,
            c.split,
            c.metric,
            c.reported,
            c.measured,
            c.tolerance,
            "pass" if c.passed else "FAIL",
        ]
        for c in checks
    ]
    click.echo(
        table(["PROTOCOL", "SPLIT", "METRIC", "REPORTED", "MEASURED", "TOLERANCE", "STATUS"], rows)
    )
    click.echo(f"wrote {parity_path(card.name)}")
    return 0 if ok else 4
