"""Fast, deterministic readers and writers for JSONL(.gz) and split TSV(.gz) records.

Readers use plain ``json`` and slotted dataclasses (250k rows in well under a second). Pass
``strict=True`` to also validate every row's types with pydantic, as ``dfwb protocols lint`` and
the tests do.
"""

from __future__ import annotations

import dataclasses
import gzip
import io
import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import TypeAdapter, ValidationError

from dfwb.core.errors import ContractError, validation_messages
from dfwb.core.hashing import sha256_text
from dfwb.core.records._base import assert_no_absolute_paths
from dfwb.core.records.local import BuilderRef, InventoryRecord, Probe, ProcessedRecord, TrackStats
from dfwb.core.records.protocol import SplitRow

__all__ = [
    "iter_jsonl_dicts",
    "read_jsonl",
    "read_split_tsv",
    "split_sha256",
    "write_jsonl",
    "write_split_tsv",
]

# Nested dataclass fields that the fast path must build from dicts.
_NESTED: dict[type[Any], dict[str, type[Any]]] = {
    InventoryRecord: {"builder": BuilderRef, "probe": Probe},
    ProcessedRecord: {"track": TrackStats},
}
_SPLITS = frozenset({"train", "val", "test", "exclude"})


@contextmanager
def _open_text(path: Path, mode: Literal["r", "w"]) -> Iterator[IO[str]]:
    """Open text for reading or writing; ``.gz`` paths are gzip, written with mtime 0."""
    if path.suffix != ".gz":
        with path.open(mode, encoding="utf-8", newline="\n") as handle:
            yield handle
    elif mode == "w":
        with (
            path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed,
            io.TextIOWrapper(packed, encoding="utf-8", newline="\n") as handle,
        ):
            yield handle
    else:
        with gzip.open(path, "rt", encoding="utf-8", newline="\n") as handle:
            yield handle


def _tmp_for(target: Path) -> Path:
    """A hidden sibling with the same final suffix, so ``.gz`` still means gzip."""
    return target.with_name(f".{target.stem}.tmp-{os.getpid()}{target.suffix}")


def _build[R](record_type: type[R], data: dict[str, Any]) -> R:
    for name, nested in _NESTED.get(record_type, {}).items():
        value = data.get(name)
        if isinstance(value, dict):
            data[name] = nested(**value)
    return record_type(**data)


def iter_jsonl_dicts(path: str | os.PathLike[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(lineno, dict)`` for each non-blank line of a ``.jsonl``/``.jsonl.gz`` file.

    Owns the open/gzip/decode/JSON-error mapping that every JSONL reader needs, so a missing,
    truncated, mis-encoded or corrupt-JSON file always raises the same :class:`ContractError`
    (naming ``file:lineno`` for a bad line), no matter what record type is eventually built from
    each dict. :func:`iter_jsonl` builds on this; callers that need a raw dict before -- or
    instead of -- building a record (e.g. to filter cheaply, as ``Protocol.records`` does) use it
    directly.
    """
    p = Path(path)
    try:
        yield from _iter_jsonl_dicts(p)
    except FileNotFoundError:
        raise ContractError(f"file not found: {p}", hint="check the path") from None
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise ContractError(
            f"{p.name}: cannot read ({type(exc).__name__}: {exc})",
            hint="the file is truncated, corrupt or misnamed (.gz means gzip); rebuild it",
        ) from None


def _iter_jsonl_dicts(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    # ``f"{path.name}:{lineno}"`` is built only on the (rare) error paths below, not per line: it
    # is pure string formatting on the hot path otherwise, and this loop must stay comfortably
    # linear for 250k-row files (the read-performance budget).
    with _open_text(path, "r") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"{path.name}:{lineno}: invalid JSON ({exc.msg})",
                    hint="the file is corrupt; rebuild it",
                ) from None
            if not isinstance(data, dict):
                raise ContractError(
                    f"{path.name}:{lineno}: expected a JSON object",
                    hint="the file is corrupt; rebuild it",
                )
            yield lineno, data


def iter_jsonl[R](path: Path, record_type: type[R], *, strict: bool = False) -> Iterator[R]:
    """Yield records from a ``.jsonl`` or ``.jsonl.gz`` file."""
    adapter = TypeAdapter(record_type) if strict else None
    for lineno, data in iter_jsonl_dicts(path):
        if adapter is not None:
            try:
                yield adapter.validate_python(data)
            except ValidationError as exc:
                raise ContractError(
                    f"{path.name}:{lineno}: " + "; ".join(validation_messages(exc)),
                    hint=f"not a valid {record_type.__name__}",
                ) from None
            continue
        try:
            yield _build(record_type, data)
        except TypeError as exc:
            known = {f.name for f in dataclasses.fields(record_type)}  # type: ignore[arg-type]
            unknown = sorted(set(data) - known)
            detail = f"unexpected field(s) {unknown}" if unknown else str(exc).split(") ", 1)[-1]
            raise ContractError(
                f"{path.name}:{lineno}: not a valid {record_type.__name__}: {detail}",
                hint="a file written by a newer dfwb needs a newer dfwb; otherwise rebuild it",
            ) from None


def read_jsonl[R](
    path: str | os.PathLike[str], record_type: type[R], *, strict: bool = False
) -> list[R]:
    """Read every record of a ``.jsonl`` or ``.jsonl.gz`` file."""
    return list(iter_jsonl(Path(path), record_type, strict=strict))


def write_jsonl(path: str | os.PathLike[str], records: Iterable[Any]) -> None:
    """Write records as compact, key-sorted JSON lines; ``.gz`` output is byte-reproducible.

    The file is written atomically. Raises :class:`ContractError` (and leaves no file) if any
    record holds an absolute path.
    """
    target = Path(path)
    tmp = _tmp_for(target)
    try:
        with _open_text(tmp, "w") as handle:
            for index, record in enumerate(records):
                assert_no_absolute_paths(record, where=f"{target.name}[{index}]")
                data = dataclasses.asdict(record)
                handle.write(
                    json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
                handle.write("\n")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def _canonical_split_lines(rows: Iterable[SplitRow]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        compression = row.compression or ""
        for value in (row.key, compression):
            if "\t" in value or "\n" in value:
                raise ContractError(
                    f"{row.key!r}: keys and compressions cannot contain tabs or newlines",
                    hint="fix the key",
                )
        if row.split not in _SPLITS:
            raise ContractError(
                f"{row.key!r}: split {row.split!r} is not one of {sorted(_SPLITS)}",
                hint="fix the split",
            )
        lines.append(f"{row.key}\t{compression}\t{row.split}\n")
    return sorted(lines)


def split_sha256(rows: Iterable[SplitRow]) -> str:
    """The scheme hash: sha256 of the sorted, uncompressed ``key\\tcompression\\tsplit`` lines."""
    return sha256_text("".join(_canonical_split_lines(rows)))


def write_split_tsv(path: str | os.PathLike[str], rows: Iterable[SplitRow]) -> str:
    """Write a canonical (sorted) split file and return its scheme hash."""
    lines = _canonical_split_lines(rows)
    target = Path(path)
    tmp = _tmp_for(target)
    try:
        with _open_text(tmp, "w") as handle:
            handle.writelines(lines)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return sha256_text("".join(lines))


def read_split_tsv(path: str | os.PathLike[str]) -> list[SplitRow]:
    """Read a split file (``key\\tcompression\\tsplit`` per line, no header)."""
    source = Path(path)
    rows: list[SplitRow] = []
    try:
        with _open_text(source, "r") as handle:
            for lineno, line in enumerate(handle, start=1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3 or parts[2] not in _SPLITS:
                    raise ContractError(
                        f"{source.name}:{lineno}: expected key<TAB>compression<TAB>split",
                        hint="split is one of train, val, test, exclude",
                    )
                rows.append(SplitRow(parts[0], parts[1] or None, parts[2]))  # type: ignore[arg-type]
    except FileNotFoundError:
        raise ContractError(f"file not found: {source}", hint="check the path") from None
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise ContractError(
            f"{source.name}: cannot read ({type(exc).__name__}: {exc})",
            hint="the file is truncated, corrupt or misnamed (.gz means gzip); rebuild it",
        ) from None
    return rows
