"""The exported JSON Schemas are part of the contracts: they change only on purpose.

Regenerate after an intended change with: DFWB_UPDATE_GOLDEN=1 uv run pytest tests/contract
"""

import json
import os
from pathlib import Path

import pytest

from dfwb.cli.main import main

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.parametrize("contract", ["c1", "c2", "c3", "c4", "c5"])
def test_schema_export_matches_golden(contract, capsys):
    assert main(["schema", "export", contract]) == 0
    text = capsys.readouterr().out
    golden = GOLDEN / f"{contract}.schema.json"
    if os.environ.get("DFWB_UPDATE_GOLDEN") == "1":
        golden.parent.mkdir(exist_ok=True)
        golden.write_text(text)
    assert text == golden.read_text(), f"{contract} schema changed; see the module docstring"
    document = json.loads(text)
    assert document["x-dfwb-contract"]["id"] == contract


def test_c3_and_c5_list_every_record(capsys):
    main(["schema", "export", "c3"])
    names = set(json.loads(capsys.readouterr().out)["$defs"])
    assert {
        "PackCard",
        "DatasetCard",
        "LabelVocab",
        "VideoRecord",
        "SplitRow",
        "InventoryRecord",
        "ProcessingProfile",
        "ProcessedRecord",
    } <= names
    main(["schema", "export", "C5"])
    c5 = json.loads(capsys.readouterr().out)["$defs"]
    assert c5["ScoreRow"]["additionalProperties"] is False
    assert "schema" in c5["ScoreMeta"]["properties"]


def test_export_to_file(tmp_path, capsys):
    assert main(["schema", "export", "c4", "--out", str(tmp_path / "c4.json")]) == 0
    assert json.loads((tmp_path / "c4.json").read_text())["x-dfwb-contract"]["version"] == "1.0"


def test_group_without_a_subcommand_shows_its_help(capsys):
    assert main(["schema"]) == 0
    assert "Commands:" in capsys.readouterr().out
