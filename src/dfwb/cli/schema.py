"""``dfwb schema export``: JSON Schemas for the five contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from dfwb.cli._output import json_option

TITLES = {
    "c1": "Plugin API: registry entries and plugin load records",
    "c2": "Config (schema dfwb.train/1)",
    "c3": "Dataset and protocol records",
    "c4": "Detector and adapter metadata",
    "c5": "Score file (schema dfwb.scores/1)",
}


def _types(contract: str) -> tuple[str, list[Any]]:
    if contract == "c1":
        from dfwb.core.plugins import PLUGIN_API_VERSION, PluginRecord
        from dfwb.core.registry import Entry

        return ".".join(map(str, PLUGIN_API_VERSION)), [Entry, PluginRecord]
    if contract == "c2":
        from dfwb.core.config.schema import TrainConfig

        return "dfwb.train/1", [TrainConfig]
    if contract == "c3":
        from dfwb.core import records

        types = [
            records.PackCard,
            records.DatasetCard,
            records.LabelVocab,
            records.VideoRecord,
            records.SplitRow,
            records.InventoryRecord,
            records.ProcessingProfile,
            records.ProcessedRecord,
        ]
        return str(records.protocol.PACK_SCHEMA_VERSION), types
    if contract == "c4":
        from dfwb.core.detector import DETECTOR_CONTRACT_VERSION, DetectorMeta, InputSpec

        return ".".join(map(str, DETECTOR_CONTRACT_VERSION)), [InputSpec, DetectorMeta]
    from dfwb.core.records.scores import SCORE_SCHEMA, ScoreMeta, ScoreRow

    return SCORE_SCHEMA, [ScoreRow, ScoreMeta]


def build_schema(contract: str) -> dict[str, Any]:
    """One JSON Schema document holding every model of a contract under ``$defs``."""
    from pydantic import TypeAdapter

    from dfwb.core.records.scores import ScoreRow, score_row_json_schema

    version, types = _types(contract)
    definitions: dict[str, Any] = {}
    for kind in types:
        if kind is ScoreRow:  # a CSV row: columns and x_* extensions, not dataclass fields
            definitions["ScoreRow"] = score_row_json_schema()
            continue
        schema = TypeAdapter(kind).json_schema(by_alias=True, ref_template="#/$defs/{model}")
        definitions.update(schema.pop("$defs", {}))
        definitions[kind.__name__] = schema
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"dfwb {contract.upper()}: {TITLES[contract]}",
        "x-dfwb-contract": {"id": contract, "version": version},
        "$defs": dict(sorted(definitions.items())),
    }


@click.group()
def schema() -> None:
    """Export the contract JSON Schemas."""


@schema.command("export")
@click.argument("contract", type=click.Choice(sorted(TITLES), case_sensitive=False))
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write to a file instead of stdout.",
)
@json_option
def export(contract: str, out: Path | None, as_json: bool) -> None:
    """Print the JSON Schema of contract c1..c5 (output is always JSON)."""
    text = json.dumps(build_schema(contract.lower()), indent=2, sort_keys=True) + "\n"
    if out is None:
        click.echo(text, nl=False)
    else:
        out.write_text(text, encoding="utf-8")
        click.echo(f"wrote {out}", err=True)
