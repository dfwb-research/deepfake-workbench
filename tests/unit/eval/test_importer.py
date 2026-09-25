"""``dfwb eval import``: key templates, polarity, label checks, missing rows, and the key-fix
suggestions a foreign score CSV's mismatched keys need (never a silent 0% join).
"""

from __future__ import annotations

import pytest

from dfwb.core.errors import ConfigError, ContractError
from dfwb.eval.importer import ImportResult, import_scores, parse_map, suggest_key_fixes

# `import_pack` is a fixture from tests/unit/conftest.py (shared with the CLI's import tests);
# pytest finds it by name, so it does not need to be imported here.


def _csv(tmp_path, name, header, rows, delimiter=","):
    path = tmp_path / name
    lines = [delimiter.join(header)]
    lines += [delimiter.join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- parse_map


def test_parse_map_basic():
    assert parse_map("key=video_id,score=prediction") == {
        "key": "video_id",
        "score": "prediction",
    }


def test_parse_map_with_label_and_compression():
    result = parse_map("key={task}/{video_id},score=prediction,label=label,compression=comp")
    assert result["key"] == "{task}/{video_id}"
    assert result["label"] == "label"
    assert result["compression"] == "comp"


def test_parse_map_missing_key_raises():
    with pytest.raises(ConfigError, match="missing 'key'"):
        parse_map("score=prediction")


def test_parse_map_unknown_field_raises():
    with pytest.raises(ConfigError, match="unknown field"):
        parse_map("key=video_id,score=prediction,bogus=x")


def test_parse_map_duplicate_field_raises():
    with pytest.raises(ConfigError, match="given twice"):
        parse_map("key=video_id,key=other,score=prediction")


# --------------------------------------------------------------------------- import_scores


def test_import_with_key_template_and_matching_scores(tmp_path, import_pack):
    path = _csv(
        tmp_path,
        "videos.csv",
        ["video_id", "task", "label", "prediction"],
        [
            ("00001", "CDF", 0, 0.1),
            ("00002", "CDF", 0, 0.2),
            ("00003", "CDF", 1, 0.9),
            ("00004", "CDF", 1, 0.8),
            ("00005", "CDF", 1, 0.7),
        ],
    )
    result = import_scores(
        path,
        protocol="cdf/official",
        split="test",
        key="{task}/{video_id}",
        score="prediction",
        label="label",
    )
    assert isinstance(result, ImportResult)
    by_key = {r.key: r for r in result.rows}
    assert by_key["CDF/00001"].score == pytest.approx(0.1)
    assert by_key["CDF/00001"].label == 0
    assert by_key["CDF/00003"].score == pytest.approx(0.9)
    assert by_key["CDF/00003"].label == 1
    assert all(r.status == "ok" for r in result.rows)
    assert result.meta.coverage.ok == 5
    assert result.meta.coverage.expected == 5
    assert result.meta.labels == "binary"
    assert result.meta.protocol.id == "cdf/official"
    assert result.meta.protocol.split == "test"


def test_import_polarity_flip(tmp_path, import_pack):
    # "prediction" here is P(real): 00001 (real) scores high, 00003 (fake) scores low.
    path = _csv(
        tmp_path,
        "videos.csv",
        ["video_id", "task", "prediction"],
        [
            ("00001", "CDF", 0.9),
            ("00002", "CDF", 0.8),
            ("00003", "CDF", 0.1),
            ("00004", "CDF", 0.2),
            ("00005", "CDF", 0.3),
        ],
    )
    result = import_scores(
        path,
        protocol="cdf/official",
        split="test",
        key="{task}/{video_id}",
        score="prediction",
        polarity="real-high",
    )
    by_key = {r.key: r for r in result.rows}
    assert by_key["CDF/00001"].score == pytest.approx(0.1)
    assert by_key["CDF/00003"].score == pytest.approx(0.9)


def test_import_bare_column_key_without_template(tmp_path, import_pack):
    # a single-column key works too, when it already matches the pack's key exactly.
    path = _csv(
        tmp_path,
        "videos.csv",
        ["key", "prediction"],
        [
            ("CDF/00001", 0.1),
            ("CDF/00002", 0.2),
            ("CDF/00003", 0.9),
            ("CDF/00004", 0.8),
            ("CDF/00005", 0.7),
        ],
    )
    result = import_scores(
        path, protocol="cdf/official", split="test", key="key", score="prediction"
    )
    assert {r.key for r in result.rows} == {f"CDF/0000{i}" for i in range(1, 6)}


def test_import_label_mismatch_raises_contract_error(tmp_path, import_pack):
    path = _csv(
        tmp_path,
        "videos.csv",
        ["video_id", "task", "label", "prediction"],
        [
            ("00001", "CDF", 1, 0.1),  # wrong: the pack says 00001 is real (0)
            ("00002", "CDF", 0, 0.2),
            ("00003", "CDF", 1, 0.9),
            ("00004", "CDF", 1, 0.8),
            ("00005", "CDF", 1, 0.7),
        ],
    )
    with pytest.raises(ContractError, match="label"):
        import_scores(
            path,
            protocol="cdf/official",
            split="test",
            key="{task}/{video_id}",
            score="prediction",
            label="label",
        )


def test_import_missing_rows_become_status_missing(tmp_path, import_pack):
    # 00005 is a test-split pack video the file simply lacks.
    path = _csv(
        tmp_path,
        "videos.csv",
        ["video_id", "task", "prediction"],
        [
            ("00001", "CDF", 0.1),
            ("00002", "CDF", 0.2),
            ("00003", "CDF", 0.9),
            ("00004", "CDF", 0.8),
        ],
    )
    result = import_scores(
        path, protocol="cdf/official", split="test", key="{task}/{video_id}", score="prediction"
    )
    by_key = {r.key: r for r in result.rows}
    assert by_key["CDF/00005"].status == "missing"
    assert by_key["CDF/00005"].score is None
    assert result.meta.coverage.ok == 4
    assert result.meta.coverage.missing == 1
    assert result.meta.coverage.expected == 5


def test_import_delimiter_option(tmp_path, import_pack):
    path = _csv(
        tmp_path,
        "videos.tsv",
        ["video_id", "task", "prediction"],
        [
            ("00001", "CDF", 0.1),
            ("00002", "CDF", 0.2),
            ("00003", "CDF", 0.9),
            ("00004", "CDF", 0.8),
            ("00005", "CDF", 0.7),
        ],
        delimiter="\t",
    )
    result = import_scores(
        path,
        protocol="cdf/official",
        split="test",
        key="{task}/{video_id}",
        score="prediction",
        delimiter="\t",
    )
    assert len(result.rows) == 5


def test_import_status_col(tmp_path, import_pack):
    path = _csv(
        tmp_path,
        "videos.csv",
        ["video_id", "task", "prediction", "state"],
        [
            ("00001", "CDF", 0.1, "ok"),
            ("00002", "CDF", "", "error"),
            ("00003", "CDF", 0.9, "ok"),
            ("00004", "CDF", 0.8, "ok"),
            ("00005", "CDF", 0.7, "ok"),
        ],
    )
    result = import_scores(
        path,
        protocol="cdf/official",
        split="test",
        key="{task}/{video_id}",
        score="prediction",
        status_col="state",
    )
    by_key = {r.key: r for r in result.rows}
    assert by_key["CDF/00002"].status == "error"
    assert by_key["CDF/00002"].score is None


# --------------------------------------------------------------------------- key-fix suggestions


def test_import_suggests_key_fixes_for_extensions(tmp_path, import_pack):
    path = _csv(
        tmp_path,
        "videos.csv",
        ["key", "prediction"],
        [("CDF/00001.mp4", 0.1), ("CDF/00002.mp4", 0.2), ("CDF/00003.mp4", 0.9)],
    )
    with pytest.raises(ContractError) as excinfo:
        import_scores(path, protocol="cdf/official", split="test", key="key", score="prediction")
    assert "not in" in excinfo.value.message
    assert "extension" in excinfo.value.hint


def test_import_suggests_key_fixes_for_directory_prefix(tmp_path, import_pack):
    path = _csv(
        tmp_path,
        "videos.csv",
        ["key", "prediction"],
        [("real/00001", 0.1), ("real/00002", 0.2), ("fake/00003", 0.9)],
    )
    with pytest.raises(ContractError) as excinfo:
        import_scores(path, protocol="cdf/official", split="test", key="key", score="prediction")
    assert "directory prefix" in excinfo.value.hint


def test_import_suggests_key_fixes_for_missing_task_prefix(tmp_path, import_pack):
    path = _csv(
        tmp_path,
        "videos.csv",
        ["video_id", "prediction"],
        [("00001", 0.1), ("00002", 0.2), ("00003", 0.9)],
    )
    with pytest.raises(ContractError) as excinfo:
        import_scores(
            path, protocol="cdf/official", split="test", key="video_id", score="prediction"
        )
    assert "task prefix" in excinfo.value.hint
    assert "{task}" in excinfo.value.hint


def test_import_unknown_key_never_silently_partial(tmp_path, import_pack):
    """Even if most keys match, any unmatched key fails loudly rather than a partial join."""
    path = _csv(
        tmp_path,
        "videos.csv",
        ["video_id", "task", "prediction"],
        [
            ("00001", "CDF", 0.1),
            ("00002", "CDF", 0.2),
            ("00003", "CDF", 0.9),
            ("00004", "CDF", 0.8),
            ("00999", "CDF", 0.5),  # not a pack video at all
        ],
    )
    with pytest.raises(ContractError, match="CDF/00999"):
        import_scores(
            path, protocol="cdf/official", split="test", key="{task}/{video_id}", score="prediction"
        )


def test_suggest_key_fixes_unit():
    expected = ["CDF/00001", "CDF/00002"]
    assert any("extension" in s for s in suggest_key_fixes(["CDF/00001.mp4"], expected))
    assert any("directory prefix" in s for s in suggest_key_fixes(["real/00001"], expected))
    assert any("task prefix" in s for s in suggest_key_fixes(["00001"], expected))
    assert suggest_key_fixes(["totally/unrelated"], expected) == []
