"""Import a foreign score CSV against a protocol split, turning it into C5 rows.

Any codebase that can write ``video_id,label,prediction,...`` can be scored by ``dfwb eval``
through this: a ``--map`` tells it which columns hold the key, the score and (optionally) the
label, the pack supplies every label and fills in a ``missing`` row for a split video the file
lacks, and, whenever a key in the file cannot be found in the pack, this fails outright rather than
silently evaluating on whatever subset happened to match -- with suggestions for the three ways a
key commonly fails to line up (a file extension, a directory prefix the pack does not use, or a
missing task prefix), so the fix is usually obvious from the error alone.
"""

from __future__ import annotations

import csv
import datetime
import posixpath
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dfwb.core.errors import ConfigError, ContractError, did_you_mean
from dfwb.core.records import ScoreMeta, ScoreRow
from dfwb.core.records.protocol import SplitRow, VideoRecord
from dfwb.core.records.scores import Coverage, DetectorInfo, ProtocolInfo
from dfwb.core.records.scores import GitState as _MetaGitState
from dfwb.core.runmeta import RunInfo, collect_run_info
from dfwb.protocols.protocol import load as load_protocol

__all__ = ["ImportResult", "import_scores", "parse_map", "suggest_key_fixes"]

#: ``--map`` fields ``dfwb eval import`` understands.
_MAP_FIELDS = ("key", "score", "label", "compression")
_POLARITIES = ("fake-high", "real-high")
_STATUSES = ("ok", "missing", "error")
_MAX_SUGGESTION_SAMPLE = 5


@dataclass(frozen=True)
class ImportResult:
    """The result of :func:`import_scores`: rows ready for
    :func:`~dfwb.core.records.write_scores`.
    """

    rows: list[ScoreRow]
    meta: ScoreMeta


def parse_map(spec: str) -> dict[str, str]:
    """Parse ``--map``'s ``key=...,score=...[,label=...][,compression=...]`` into a plain dict.

    ``key``'s value may itself contain ``{...}`` placeholders (a template composed from several
    columns, e.g. ``key={task}/{video_id}``): this only splits on top-level commas and ``=``, it
    does not interpret the template.

    Raises:
        ConfigError: a piece is not ``name=value``, a field is given twice, an unknown field is
            given, or ``key``/``score`` is missing.
    """
    result: dict[str, str] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            raise ConfigError(f"--map {spec!r}: empty entry", hint="use key=...,score=...")
        name, eq, value = piece.partition("=")
        name = name.strip()
        if not eq or not name:
            raise ConfigError(
                f"--map {spec!r}: {piece!r} is not name=value", hint="use key=...,score=..."
            )
        if name in result:
            raise ConfigError(f"--map: {name!r} given twice", hint="pass each mapping once")
        result[name] = value.strip()
    unknown = sorted(set(result) - set(_MAP_FIELDS))
    if unknown:
        raise ConfigError(
            f"--map: unknown field(s) {unknown}", hint="fields: " + ", ".join(_MAP_FIELDS)
        )
    for required in ("key", "score"):
        if required not in result:
            raise ConfigError(f"--map: missing {required!r}", hint="use key=...,score=...")
    return result


def suggest_key_fixes(bad_keys: Sequence[str], expected_keys: Sequence[str]) -> list[str]:
    """Suggestions for why none of ``bad_keys`` matched any of ``expected_keys``.

    Tries, on a small sample of ``bad_keys``: stripping a file extension (``00011.mp4`` when the
    pack key is ``00011``), and, via the pack keys' final path segment, a directory prefix the pack
    does not use (``real/00011`` when the pack key is ``CDF/00011``) or a missing task prefix
    (``00011`` when the pack key is ``CDF/00011``). Returns one line per distinct match found, in
    no particular order beyond that; an empty list means none of these common mistakes explain it.
    """
    expected_set = set(expected_keys)
    tail_to_key: dict[str, str] = {}
    for full_key in expected_keys:
        tail_to_key.setdefault(posixpath.basename(full_key), full_key)

    suggestions: list[str] = []
    sample = list(dict.fromkeys(bad_keys))[:_MAX_SUGGESTION_SAMPLE]
    for bad in sample:
        stem, ext = posixpath.splitext(bad)
        if ext and stem in expected_set:
            suggestions.append(
                f"{bad!r} carries a file extension the pack key does not: use {stem!r} instead "
                "(strip the extension from the key column, or the key template)"
            )
            continue
        tail = posixpath.basename(bad)
        matched = tail_to_key.get(tail)
        if matched is None or matched == bad:
            continue
        if "/" in bad:
            suggestions.append(
                f"{bad!r} has a directory prefix the pack does not use: the pack key is "
                f"{matched!r} -- use just {tail!r}, or compose key={{task}}/{{...}}"
            )
        else:
            suggestions.append(
                f"{bad!r} is missing the task prefix the pack key has: the pack key is "
                f"{matched!r} -- compose key={{task}}/{{...}}"
            )
    seen: set[str] = set()
    unique: list[str] = []
    for suggestion in suggestions:
        if suggestion not in seen:
            unique.append(suggestion)
            seen.add(suggestion)
    return unique


def _other_split_hints(
    bad_keys: Sequence[str], split_rows: Sequence[SplitRow], split: str
) -> list[str]:
    """One line per other split holding some of ``bad_keys`` exactly: a file that covers more than
    the one split being imported (train and test together, say) needs filtering, not a key fix."""
    splits_of: dict[str, set[str]] = {}
    for row in split_rows:
        splits_of.setdefault(row.key, set()).add(row.split)
    found: dict[str, int] = {}
    for key in dict.fromkeys(bad_keys):
        for other in sorted(splits_of.get(key, set()) - {split}):
            found[other] = found.get(other, 0) + 1
    return [
        f"{count} of these keys are in the pack's {other!r} split, not {split!r}: keep only the "
        f"file's {split!r} rows, or import it once per split"
        for other, count in sorted(found.items())
    ]


def _read_csv(path: Path, delimiter: str) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    """Every data row of ``path``, paired with its physical line number (the header is line 1)."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:  # tolerate a BOM
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = list(reader.fieldnames or [])
            rows = [(reader.line_num, dict(raw)) for raw in reader]
    except OSError as exc:
        raise ContractError(
            f"{path}: cannot read ({type(exc).__name__}: {exc})", hint="check the path"
        ) from None
    return columns, rows


def _compose(template: str, row: Mapping[str, str], columns: Sequence[str]) -> str:
    if "{" not in template:
        if template not in columns:
            raise ConfigError(
                f"column {template!r} is not in the file{did_you_mean(template, columns)}",
                hint="columns: " + ", ".join(columns),
            )
        return row[template]
    try:
        return template.format(**row)
    except KeyError as exc:
        raise ConfigError(
            f"key template {template!r} needs column {exc.args[0]!r}, which is not in the file",
            hint="columns: " + ", ".join(columns),
        ) from None


def _coverage(rows: Sequence[ScoreRow]) -> dict[str, int]:
    counts = Counter(row.status for row in rows)
    return {
        "expected": len(rows),
        "ok": counts["ok"],
        "missing": counts["missing"],
        "error": counts["error"],
    }


def _parse_created(text: str) -> datetime.datetime:
    """``dfwb.core.runmeta.utc_now()``'s string, as the aware ``datetime`` ``ScoreMeta.created``
    is typed as (the constructor validates a plain string too, but not statically)."""
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC)


def _match_video(
    composed_key: str,
    comp_value: str | None,
    by_key: Mapping[str, list[VideoRecord]],
    *,
    require_compression: bool,
) -> VideoRecord | None:
    candidates = by_key.get(composed_key)
    if not candidates:
        return None
    if require_compression or len(candidates) > 1:
        return next((v for v in candidates if v.compression == comp_value), None)
    return candidates[0]


def import_scores(
    path: str | Path,
    *,
    protocol: str,
    split: str,
    key: str,
    score: str,
    label: str | None = None,
    compression: str | None = None,
    polarity: str = "fake-high",
    status_col: str | None = None,
    delimiter: str = ",",
    labels: str = "binary",
    run_info: RunInfo | None = None,
) -> ImportResult:
    """Import ``path`` against ``protocol``'s ``split``, filling every video from the pack.

    ``key`` is a column name or a ``{col}`` template (:func:`parse_map` parses ``--map`` into the
    keyword arguments here). Labels always come from the pack's ``labels`` mapping; a supplied
    ``label`` column is cross-checked against it. A pack video the file lacks becomes a
    ``status="missing"`` row; a video whose mapped label is ``"exclude"`` (a pack label mapping may
    legitimately say so) is skipped entirely -- neither a row nor counted as missing, the same as
    it would be for any other consumer of that mapping. A file row whose key is not one of the
    split's pack videos fails the whole import (see :func:`suggest_key_fixes`) rather than silently
    narrowing the join; two file rows that resolve to the *same* pack video also fail the whole
    import (naming the key and both input line numbers) rather than the second silently winning.

    Raises:
        ConfigError: ``polarity`` is invalid, the file has no data rows, the split is empty, or a
            column named by ``key``/``score``/``label``/``compression`` does not exist.
        ContractError: the file cannot be read, a key in the file does not match any pack video,
            two rows resolve to the same pack video, a supplied label disagrees with the pack, or
            a score/label value is not a number.
    """
    if polarity not in _POLARITIES:
        raise ConfigError(
            f"unknown --polarity {polarity!r}", hint="polarity: " + ", ".join(_POLARITIES)
        )
    source = Path(path)
    columns, raw_rows = _read_csv(source, delimiter)
    if not raw_rows:
        raise ContractError(f"{source.name}: no data rows", hint="check the file and --delimiter")

    proto = load_protocol(protocol)
    label_mapping = proto.labels(labels)
    videos = proto.records(split=split)
    if not videos:
        raise ConfigError(
            f"{protocol}/{split}: no videos in this split", hint="check --protocol/--split"
        )

    by_key: dict[str, list[VideoRecord]] = {}
    for record in videos:
        by_key.setdefault(record.key, []).append(record)

    matched: dict[tuple[str, str | None], tuple[int, dict[str, str]]] = {}
    unmatched_keys: list[str] = []
    for lineno, source_row in raw_rows:
        composed = _compose(key, source_row, columns)
        comp_value = source_row.get(compression) if compression else None
        matched_video = _match_video(
            composed, comp_value, by_key, require_compression=bool(compression)
        )
        if matched_video is None:
            unmatched_keys.append(composed)
            continue
        ident = (matched_video.key, matched_video.compression)
        if ident in matched:
            prev_lineno, _prev_row = matched[ident]
            raise ContractError(
                f"{source.name}: lines {prev_lineno} and {lineno} both resolve to key "
                f"{matched_video.key!r}",
                hint="deduplicate the file, or pass a key template that distinguishes the rows",
            )
        matched[ident] = (lineno, source_row)

    if unmatched_keys:
        expected_keys = [v.key for v in videos]
        suggestions = [
            *_other_split_hints(unmatched_keys, proto.split_rows(), split),
            *suggest_key_fixes(unmatched_keys, expected_keys),
        ]
        hint = (
            "; ".join(suggestions) if suggestions else "check --map key=... against the pack's keys"
        )
        sample = ", ".join(repr(k) for k in unmatched_keys[:_MAX_SUGGESTION_SAMPLE])
        raise ContractError(
            f"{source.name}: {len(unmatched_keys)} row(s) have keys not in {protocol}/{split} "
            f"(e.g. {sample})",
            hint=hint,
        )

    rows: list[ScoreRow] = []
    for video in videos:
        mapped = label_mapping(video.label_key)
        if mapped == "exclude":
            continue
        if not isinstance(mapped, int):
            raise ContractError(
                f"label mapping {labels!r} gives {video.label_key!r} the non-integer value "
                f"{mapped!r}",
                hint="use an integer label mapping, e.g. 'binary'",
            )
        matched_entry = matched.get((video.key, video.compression))
        if matched_entry is None:
            rows.append(
                ScoreRow(
                    proto.dataset,
                    video.key,
                    video.compression,
                    mapped,
                    None,
                    "missing",
                    label_key=video.label_key,
                    method=video.method,
                )
            )
            continue
        _lineno, matched_row = matched_entry
        if label is not None:
            _check_label(source, video, matched_row, label, mapped)
        status = _row_status(source, video, matched_row, status_col)
        if status != "ok":
            rows.append(
                ScoreRow(
                    proto.dataset,
                    video.key,
                    video.compression,
                    mapped,
                    None,
                    status,
                    label_key=video.label_key,
                    method=video.method,
                )
            )
            continue
        final_score = _row_score(source, video, matched_row, score, polarity)
        rows.append(
            ScoreRow(
                proto.dataset,
                video.key,
                video.compression,
                mapped,
                final_score,
                "ok",
                label_key=video.label_key,
                method=video.method,
            )
        )

    info = run_info or collect_run_info()
    git = _MetaGitState(commit=info.git.commit, dirty=info.git.dirty) if info.git else None
    meta = ScoreMeta(
        detector=DetectorInfo(
            name="import", version="1", source=f"import:{source.name}", contract_version=(1, 0)
        ),
        protocol=ProtocolInfo(
            id=proto.ref,
            split=split,
            where={},
            pack=proto.pack.name,
            pack_version=proto.pack_version,
            scheme_sha256=proto.sha256,
        ),
        labels=labels,
        coverage=Coverage(**_coverage(rows)),
        env=info.env,
        git=git,
        command=info.command,
        created=_parse_created(info.created),
    )
    return ImportResult(rows=rows, meta=meta)


def _check_label(
    source: Path, video: VideoRecord, raw: Mapping[str, str], label_col: str, mapped: int
) -> None:
    raw_label = raw.get(label_col)
    if raw_label is None or raw_label == "":
        raise ContractError(
            f"{source.name}: row for {video.key!r} has no value in label column {label_col!r}",
            hint="check --map label=...",
        )
    try:
        supplied = int(float(raw_label))
    except ValueError:
        raise ContractError(
            f"{source.name}: label {raw_label!r} for {video.key!r} is not a number",
            hint="check --map label=...",
        ) from None
    if supplied != mapped:
        raise ContractError(
            f"{source.name}: row for {video.key!r} has label {supplied} but the pack's labels "
            f"say {mapped} for {video.label_key!r}",
            hint="the file's label column disagrees with the pack; fix the file or drop label= "
            "from --map",
        )


def _row_status(
    source: Path, video: VideoRecord, raw: Mapping[str, str], status_col: str | None
) -> Literal["ok", "missing", "error"]:
    if status_col is None:
        return "ok"
    raw_status = (raw.get(status_col) or "ok").strip().lower()
    if raw_status not in _STATUSES:
        raise ConfigError(
            f"{source.name}: status {raw_status!r} for {video.key!r} is not one of {_STATUSES}",
            hint="fix --status-col values",
        )
    return raw_status  # type: ignore[return-value]  # checked against _STATUSES just above


def _row_score(
    source: Path, video: VideoRecord, raw: Mapping[str, str], score_col: str, polarity: str
) -> float:
    raw_value = raw.get(score_col)
    if raw_value is None or raw_value == "":
        raise ContractError(
            f"{source.name}: row for {video.key!r} has no value in score column {score_col!r}",
            hint="check --map score=...",
        )
    try:
        raw_score = float(raw_value)
    except ValueError:
        raise ContractError(
            f"{source.name}: score {raw_value!r} for {video.key!r} is not a number",
            hint="check --map score=...",
        ) from None
    final_score = raw_score if polarity == "fake-high" else 1.0 - raw_score
    if not 0.0 <= final_score <= 1.0:
        raise ContractError(
            f"{source.name}: score {final_score!r} for {video.key!r} is out of [0, 1]",
            hint="scores must be probabilities; check --polarity",
        )
    return final_score
