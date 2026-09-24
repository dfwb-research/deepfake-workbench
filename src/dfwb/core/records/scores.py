"""Score files (contract C5): ``<name>.scores.csv`` plus ``<name>.scores.meta.json``.

The interchange format between detection and evaluation. One row per ``(dataset, key,
compression)`` in the split; videos that could not be scored are rows with ``status`` ``missing``
or ``error`` and an empty score, never dropped.
"""

from __future__ import annotations

import csv
import math
import numbers
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import AwareDatetime, ConfigDict, Field, ValidationError, model_validator, with_config

from dfwb.core.errors import ContractError, validation_messages
from dfwb.core.records._base import RecordModel, Sha256, assert_no_absolute_paths

__all__ = [
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SCORE_SCHEMA",
    "ScoreFile",
    "ScoreMeta",
    "ScoreRow",
    "meta_path_for",
    "read_scores",
    "score_row_json_schema",
    "write_scores",
]

SCORE_SCHEMA: Final = "dfwb.scores/1"
REQUIRED_COLUMNS = ("dataset", "key", "compression", "label", "score", "status")
OPTIONAL_COLUMNS = ("logit", "label_key", "method", "n_clips", "n_frames")
_STATUSES = ("ok", "missing", "error")
_CSV_SUFFIX = ".scores.csv"


@with_config(ConfigDict(extra="forbid"))
@dataclass(slots=True, frozen=True)
class ScoreRow:
    """One row of a score file. ``score`` is P(fake): higher means more fake, always."""

    dataset: str
    key: str
    compression: str | None
    label: int
    score: float | None
    status: Literal["ok", "missing", "error"]
    logit: float | None = None
    label_key: str | None = None
    method: str | None = None
    n_clips: int | None = None
    n_frames: int | None = None
    extras: dict[str, str] = field(default_factory=dict)  # x_* extension columns

    def __post_init__(self) -> None:
        where = f"{self.dataset}/{self.key}"
        label: object = self.label  # numpy integers are allowed; bools and floats are not
        if isinstance(label, bool) or not isinstance(label, numbers.Integral):
            raise ContractError(f"{where}: label must be an integer, got {label!r}", hint="use 0/1")
        for name in ("score", "logit"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, numbers.Real)
            ):
                raise ContractError(
                    f"{where}: {name} must be a number, got {value!r}", hint="use a float"
                )
        if self.status not in _STATUSES:
            raise ContractError(
                f"{where}: status {self.status!r} is not one of {list(_STATUSES)}",
                hint="use ok, missing or error",
            )
        if self.status == "ok":
            if self.score is None or not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
                raise ContractError(
                    f"{where}: status ok needs a score in [0, 1], got {self.score!r}",
                    hint="scores are P(fake)",
                )
        elif self.score is not None:
            raise ContractError(
                f"{where}: status {self.status} must have an empty score", hint="leave score empty"
            )
        for name in self.extras:
            if not name.startswith("x_"):
                raise ContractError(
                    f"{where}: extension column {name!r} must start with 'x_'",
                    hint="rename it to x_...",
                )


class DetectorInfo(RecordModel):
    name: str
    version: str
    source: str
    checkpoint_sha256: Sha256 | None = None
    contract_version: tuple[int, int]


class ProtocolInfo(RecordModel):
    id: str
    split: str
    where: dict[str, Any] = Field(default_factory=dict)
    pack: str
    pack_version: str
    scheme_sha256: Sha256


class ProfileRef(RecordModel):
    id: str
    sha256: Sha256


class InputAdaptation(RecordModel):
    derived_crop: bool = False
    mismatch_override: bool = False


class Aggregation(RecordModel):
    clip_to_video: str
    clips_per_video: int = Field(ge=1)


class Coverage(RecordModel):
    expected: int = Field(ge=0)
    ok: int = Field(ge=0)
    missing: int = Field(ge=0)
    error: int = Field(ge=0)

    @model_validator(mode="after")
    def _adds_up(self) -> Coverage:
        if self.ok + self.missing + self.error != self.expected:
            raise ValueError("ok + missing + error must equal expected")
        return self


class GitState(RecordModel):
    commit: str | None = None
    dirty: bool | None = None


class ScoreMeta(RecordModel):
    """``<name>.scores.meta.json``: everything the scores depend on."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["dfwb.scores/1"] = Field(default=SCORE_SCHEMA, alias="schema")
    detector: DetectorInfo
    protocol: ProtocolInfo
    labels: str
    processing_profile: ProfileRef | None = None
    input_adaptation: InputAdaptation = Field(default_factory=InputAdaptation)
    aggregation: Aggregation | None = None
    coverage: Coverage
    seed: int | None = None
    env: dict[str, str | None] = Field(default_factory=dict)
    git: GitState | None = None
    command: str | None = None
    created: AwareDatetime


@dataclass(frozen=True)
class ScoreFile:
    """A score file as read from disk."""

    path: Path
    rows: list[ScoreRow]
    meta: ScoreMeta | None


def meta_path_for(path: Path) -> Path:
    """``x.scores.csv`` -> ``x.scores.meta.json``."""
    return path.with_name(path.name[: -len(".csv")] + ".meta.json")


def _check_name(path: Path) -> None:
    if not path.name.endswith(_CSV_SUFFIX):
        raise ContractError(
            f"{path.name}: score files are named <name>{_CSV_SUFFIX}",
            hint=f"rename it to end with {_CSV_SUFFIX}",
        )


def _coverage(rows: list[ScoreRow]) -> dict[str, int]:
    counts = Counter(row.status for row in rows)
    return {
        "expected": len(rows),
        "ok": counts["ok"],
        "missing": counts["missing"],
        "error": counts["error"],
    }


def _check_rows(rows: list[ScoreRow], where: str) -> None:
    seen: set[tuple[str, str, str | None]] = set()
    for row in rows:
        ident = (row.dataset, row.key, row.compression)
        if ident in seen:
            raise ContractError(
                f"{where}: duplicate row for {row.dataset}/{row.key} "
                f"(compression {row.compression!r})",
                hint="a score file has one row per (dataset, key, compression)",
            )
        seen.add(ident)


def _check_coverage(rows: list[ScoreRow], meta: ScoreMeta, where: str) -> None:
    actual = _coverage(rows)
    recorded = meta.coverage.model_dump()
    if actual != recorded:
        raise ContractError(
            f"{where}: meta coverage {recorded} does not match the rows {actual}",
            hint="rewrite the meta file",
        )


def _cell(value: Any) -> str:
    # numpy scalars are Integral/Real but repr() as "np.float64(0.25)": convert first.
    if value is None:
        return ""
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return repr(float(value))
    return str(value)


def write_scores(
    path: str | os.PathLike[str], rows: Iterable[ScoreRow], meta: ScoreMeta
) -> tuple[Path, Path]:
    """Write ``<name>.scores.csv`` and its ``.meta.json``; returns both paths."""
    target = Path(path)
    _check_name(target)
    materialised = list(rows)
    _check_rows(materialised, target.name)
    _check_coverage(materialised, meta, target.name)
    assert_no_absolute_paths(meta, where="meta")
    for index, row in enumerate(materialised):
        assert_no_absolute_paths(row, where=f"{target.name}[{index}]")
    extra_columns = sorted({name for row in materialised for name in row.extras})
    header = [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, *extra_columns]
    meta_target = meta_path_for(target)
    tmp_csv = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    tmp_meta = meta_target.with_name(f".{meta_target.name}.tmp-{os.getpid()}")
    try:
        with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            for row in materialised:
                values = [getattr(row, name) for name in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)]
                writer.writerow(
                    [*map(_cell, values), *(row.extras.get(name, "") for name in extra_columns)]
                )
        tmp_meta.write_text(meta.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8")
        tmp_csv.replace(target)
        tmp_meta.replace(meta_target)
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
    return target, meta_target


def _parse(raw: Mapping[str, str | None], lineno: int, name: str) -> ScoreRow:
    def text(column: str) -> str | None:
        value = raw.get(column)
        return value if value not in (None, "") else None

    def number(column: str, kind: type[int] | type[float]) -> Any:
        value = text(column)
        if value is None:
            return None
        try:
            return kind(value)
        except ValueError:
            raise ContractError(
                f"{name}:{lineno}: column {column!r} must be {kind.__name__}, got {value!r}",
                hint="fix the value in the CSV",
            ) from None

    if None in raw:  # csv puts cells beyond the header under the key None
        raise ContractError(
            f"{name}:{lineno}: the row has more cells than the header",
            hint="quote cells that contain commas",
        )
    label = number("label", int)
    if label is None:
        raise ContractError(
            f"{name}:{lineno}: label is empty", hint="every row needs an integer label"
        )
    return ScoreRow(
        dataset=text("dataset") or "",
        key=text("key") or "",
        compression=text("compression"),
        label=label,
        score=number("score", float),
        status=raw.get("status") or "",  # type: ignore[arg-type]  # validated in __post_init__
        logit=number("logit", float),
        label_key=text("label_key"),
        method=text("method"),
        n_clips=number("n_clips", int),
        n_frames=number("n_frames", int),
        extras={k: v for k, v in raw.items() if k.startswith("x_") and v},
    )


def read_scores(path: str | os.PathLike[str]) -> ScoreFile:
    """Read a score file and, if present, its meta; both are checked against C5."""
    source = Path(path)
    _check_name(source)
    try:
        rows = _read_rows(source)
        meta_file = meta_path_for(source)
        meta_text = meta_file.read_text("utf-8") if meta_file.is_file() else None
    except FileNotFoundError:
        raise ContractError(f"score file not found: {source}", hint="check the path") from None
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ContractError(
            f"{source.name}: cannot read ({type(exc).__name__}: {exc})",
            hint="score files are UTF-8 CSV; re-save the file as UTF-8",
        ) from None
    _check_rows(rows, source.name)
    meta: ScoreMeta | None = None
    if meta_text is not None:
        try:
            meta = ScoreMeta.model_validate_json(meta_text)
        except ValidationError as exc:
            raise ContractError(
                f"{meta_file.name}: " + "; ".join(validation_messages(exc)),
                hint="see `dfwb schema export c5`",
            ) from None
        _check_coverage(rows, meta, source.name)
    return ScoreFile(source, rows, meta)


def _read_rows(source: Path) -> list[ScoreRow]:
    with source.open(encoding="utf-8-sig", newline="") as handle:  # tolerate a BOM (Excel, pandas)
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ContractError(
                f"{source.name}: missing required column(s) {missing}",
                hint="see `dfwb schema export c5`",
            )
        known = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
        unknown = [c for c in columns if c not in known and not c.startswith("x_")]
        if unknown:
            raise ContractError(
                f"{source.name}: unknown column(s) {unknown}",
                hint="extension columns must start with x_",
            )
        rows: list[ScoreRow] = []
        for raw in reader:
            lineno = reader.line_num  # physical line; blank lines and multi-line cells count
            try:
                rows.append(_parse(raw, lineno, source.name))
            except ContractError as exc:
                prefix = "" if exc.message.startswith(source.name) else f"{source.name}:{lineno}: "
                raise exc.with_prefix(prefix) from None
    return rows


def score_row_json_schema() -> dict[str, Any]:
    """JSON Schema of one score-file row as a parsed CSV record (columns, not dataclass fields)."""
    nullable_string = {"type": ["string", "null"]}
    return {
        "title": "ScoreRow",
        "description": "One row of <name>.scores.csv (empty cells are null); score is P(fake).",
        "type": "object",
        "properties": {
            "dataset": {"type": "string"},
            "key": {"type": "string"},
            "compression": nullable_string,
            "label": {"type": "integer"},
            "score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "status": {"enum": list(_STATUSES)},
            "logit": {"type": ["number", "null"]},
            "label_key": nullable_string,
            "method": nullable_string,
            "n_clips": {"type": ["integer", "null"]},
            "n_frames": {"type": ["integer", "null"]},
        },
        "patternProperties": {"^x_": {"type": ["string", "null"]}},
        "additionalProperties": False,
        "required": list(REQUIRED_COLUMNS),
    }
