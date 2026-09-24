import datetime
import json

import pytest

from dfwb.core.errors import ContractError
from dfwb.core.records import ScoreMeta, ScoreRow, read_scores, write_scores

ROWS = [
    ScoreRow(
        "celebdf-v2",
        "id0_0000",
        None,
        0,
        0.12,
        "ok",
        logit=-1.99,
        label_key="CDFv2-CR",
        method="original",
        n_clips=4,
        n_frames=32,
    ),
    ScoreRow("celebdf-v2", "id0_id1_0000", None, 1, 0.9, "ok", extras={"x_note": "a,b"}),
    ScoreRow("celebdf-v2", "id9_0001", None, 1, None, "missing"),
]


def _meta(**change):
    data = {
        "schema": "dfwb.scores/1",
        "detector": {
            "name": "tiny",
            "version": "0.1.0",
            "source": "run:abc",
            "checkpoint_sha256": "d" * 64,
            "contract_version": [1, 0],
        },
        "protocol": {
            "id": "celebdf-v2/official",
            "split": "test",
            "where": {},
            "pack": "dfwb-protocols",
            "pack_version": "1.0.0",
            "scheme_sha256": "e" * 64,
        },
        "labels": "binary",
        "processing_profile": {"id": "face-256-1.3x-32f-ab12cd34", "sha256": "f" * 64},
        "input_adaptation": {"derived_crop": False, "mismatch_override": False},
        "aggregation": {"clip_to_video": "mean-prob", "clips_per_video": 4},
        "coverage": {"expected": 3, "ok": 2, "missing": 1, "error": 0},
        "seed": 42,
        "env": {"dfwb": "0.1.0a1", "python": "3.12.14", "torch": None},
        "git": {"commit": "0" * 40, "dirty": False},
        "command": "dfwb score --detector run:runs:vit/2026 --split test",
        "created": "2026-09-24T10:00:00Z",
    }
    data.update(change)
    return ScoreMeta.model_validate(data)


def test_round_trip(tmp_path):
    csv_path, meta_path = write_scores(tmp_path / "tiny.scores.csv", ROWS, _meta())
    assert meta_path.name == "tiny.scores.meta.json"
    loaded = read_scores(csv_path)
    assert loaded.rows == ROWS
    assert loaded.meta == _meta()
    assert loaded.meta.created == datetime.datetime(2026, 9, 24, 10, tzinfo=datetime.UTC)
    header = csv_path.read_text().splitlines()[0]
    assert header.split(",") == [
        *("dataset", "key", "compression", "label", "score", "status"),
        *("logit", "label_key", "method", "n_clips", "n_frames", "x_note"),
    ]
    assert json.loads(meta_path.read_text())["schema"] == "dfwb.scores/1"


def test_row_invariants():
    with pytest.raises(ContractError, match=r"needs a score in \[0, 1\]"):
        ScoreRow("d", "k", None, 1, 1.5, "ok")
    with pytest.raises(ContractError, match="needs a score"):
        ScoreRow("d", "k", None, 1, None, "ok")
    with pytest.raises(ContractError, match="must have an empty score"):
        ScoreRow("d", "k", None, 1, 0.5, "missing")
    with pytest.raises(ContractError, match="status 'skipped'"):
        ScoreRow("d", "k", None, 1, None, "skipped")  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="must start with 'x_'"):
        ScoreRow("d", "k", None, 1, 0.5, "ok", extras={"note": "x"})


def test_writer_checks(tmp_path):
    with pytest.raises(ContractError, match=r"named <name>\.scores\.csv"):
        write_scores(tmp_path / "tiny.csv", ROWS, _meta())
    with pytest.raises(ContractError, match="duplicate row"):
        write_scores(
            tmp_path / "t.scores.csv",
            [ROWS[0], ROWS[0]],
            _meta(coverage={"expected": 2, "ok": 2, "missing": 0, "error": 0}),
        )
    with pytest.raises(ContractError, match="does not match the rows"):
        write_scores(tmp_path / "t.scores.csv", ROWS[:2], _meta())
    with pytest.raises(ContractError, match="absolute path"):
        write_scores(
            tmp_path / "t.scores.csv", ROWS, _meta(command="dfwb score --detector run:/data/runs/x")
        )
    assert not list(tmp_path.iterdir())


def test_meta_validation():
    with pytest.raises(ValueError, match="ok \\+ missing \\+ error must equal expected"):
        _meta(coverage={"expected": 4, "ok": 2, "missing": 1, "error": 0})
    with pytest.raises(ValueError, match="timezone"):
        _meta(created="2026-09-24T10:00:00")


def test_reader_accepts_minimal_foreign_file_without_meta(tmp_path):
    path = tmp_path / "mine.scores.csv"
    path.write_text(
        "dataset,key,compression,label,score,status\nd,k1,,0,0.25,ok\nd,k2,c23,1,,error\n"
    )
    loaded = read_scores(path)
    assert loaded.meta is None
    assert loaded.rows == [
        ScoreRow("d", "k1", None, 0, 0.25, "ok"),
        ScoreRow("d", "k2", "c23", 1, None, "error"),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("dataset,key,label,score,status\n", r"missing required column\(s\) \['compression'\]"),
        ("dataset,key,compression,label,score,status,notes\n", r"unknown column\(s\) \['notes'\]"),
        (
            "dataset,key,compression,label,score,status\nd,k,,zero,0.1,ok\n",
            r"x.scores.csv:2: column 'label' must be int",
        ),
        (
            "dataset,key,compression,label,score,status\nd,k,,1,2.0,ok\n",
            r"x.scores.csv:2: d/k: status ok needs a score",
        ),
        ("dataset,key,compression,label,score,status\nd,k,,,0.1,ok\n", "label is empty"),
    ],
)
def test_reader_errors(tmp_path, content, message):
    path = tmp_path / "x.scores.csv"
    path.write_text(content)
    with pytest.raises(ContractError, match=message):
        read_scores(path)


def test_reader_checks_meta_against_rows(tmp_path):
    write_scores(tmp_path / "t.scores.csv", ROWS, _meta())
    (tmp_path / "t.scores.csv").write_text(
        (tmp_path / "t.scores.csv").read_text().rsplit("\n", 2)[0] + "\n"
    )
    with pytest.raises(ContractError, match="does not match the rows"):
        read_scores(tmp_path / "t.scores.csv")
    (tmp_path / "t.scores.meta.json").write_text('{"schema": "dfwb.scores/2"}')
    with pytest.raises(ContractError, match=r"schema: 'dfwb\.scores/2' is not one of"):
        read_scores(tmp_path / "t.scores.csv")


def test_numpy_values_are_written_as_plain_numbers(tmp_path):
    import numpy as np

    row = ScoreRow("d", "k", None, np.int64(1), np.float64(0.25), "ok", logit=np.float32(-1.5))
    meta = _meta(coverage={"expected": 1, "ok": 1, "missing": 0, "error": 0})
    csv_path, _ = write_scores(tmp_path / "np.scores.csv", [row], meta)
    assert csv_path.read_text().splitlines()[1] == "d,k,,1,0.25,ok,-1.5,,,,"
    assert read_scores(csv_path).rows == [ScoreRow("d", "k", None, 1, 0.25, "ok", logit=-1.5)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"label": True}, "label must be an integer"),
        ({"label": 1.0}, "label must be an integer"),
        ({"score": "0.5"}, "score must be a number"),
    ],
)
def test_row_types_are_checked(kwargs, message):
    values = {
        "dataset": "d",
        "key": "k",
        "compression": None,
        "label": 1,
        "score": 0.5,
        "status": "ok",
    }
    with pytest.raises(ContractError, match=message):
        ScoreRow(**(values | kwargs))


def test_rows_with_absolute_paths_are_refused(tmp_path):
    row = ScoreRow("d", "k", None, 1, 0.5, "ok", extras={"x_src": "/data/raw/k.mp4"})
    meta = _meta(coverage={"expected": 1, "ok": 1, "missing": 0, "error": 0})
    with pytest.raises(ContractError, match=r"t.scores.csv\[0\].extras.x_src"):
        write_scores(tmp_path / "t.scores.csv", [row], meta)


def test_reader_handles_bom_and_rejects_extra_cells(tmp_path):
    path = tmp_path / "excel.scores.csv"
    path.write_bytes(b"\xef\xbb\xbfdataset,key,compression,label,score,status\nd,k,,1,0.5,ok\n")
    assert read_scores(path).rows == [ScoreRow("d", "k", None, 1, 0.5, "ok")]
    path.write_text("dataset,key,compression,label,score,status\nd,k,c23,1,0.5,ok,EXTRA\n")
    with pytest.raises(ContractError, match=r"excel.scores.csv:2: the row has more cells"):
        read_scores(path)


def test_unreadable_score_files_are_contract_errors(tmp_path):
    latin = tmp_path / "latin.scores.csv"
    latin.write_bytes(
        "dataset,key,compression,label,score,status\nd,café,,1,0.5,ok\n".encode("latin-1")
    )
    with pytest.raises(ContractError, match=r"latin.scores.csv: cannot read \(UnicodeDecodeError"):
        read_scores(latin)
    (tmp_path / "dir.scores.csv").mkdir()
    with pytest.raises(ContractError, match=r"dir.scores.csv: cannot read \(IsADirectoryError"):
        read_scores(tmp_path / "dir.scores.csv")


@pytest.mark.parametrize(
    "content",
    [
        # a blank line before the bad row
        "dataset,key,compression,label,score,status\nd,k1,,0,0.5,ok\n\nd,k2,,1,bad,ok\n",
        # a quoted cell spanning two lines before the bad row
        'dataset,key,compression,label,score,status,method\nd,k1,,0,0.5,ok,"two\nlines"\nd,k2,,1,bad,ok,m\n',
    ],
)
def test_reader_reports_the_physical_line(tmp_path, content):
    path = tmp_path / "lines.scores.csv"
    path.write_text(content)
    with pytest.raises(ContractError, match=r"lines.scores.csv:4: column 'score' must be float"):
        read_scores(path)
