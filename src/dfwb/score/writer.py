"""Assembles a score file's C5 meta (``<name>.scores.meta.json``) from what the harness knows
about a scoring run: the resolved detector, the protocol split, the chosen processing profile and
how it was adapted, the aggregation mode, and the rows that were actually written. Also writes the
optional per-clip/per-frame dump (``--frames``, ``<name>.frames.parquet``).

Kept separate from :mod:`dfwb.score.harness` so the shape of a C5 meta payload -- and of the
frame-level dump -- lives in one place, independent of how the clips were actually scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dfwb.core.errors import InstallationError
from dfwb.core.records import ScoreMeta, ScoreRow
from dfwb.core.records.scores import coverage_counts
from dfwb.core.registry import install_hint

if TYPE_CHECKING:
    from dfwb.core.detector import DetectorMeta
    from dfwb.core.records.local import ProcessingProfile
    from dfwb.data.adapt import AdaptResult
    from dfwb.protocols.protocol import Protocol

__all__ = ["FrameRecord", "assemble_meta", "frames_path_for", "require_pyarrow", "write_frames"]


def _git_payload(cwd: Path) -> dict[str, Any] | None:
    from dfwb.core.runmeta import capture_git

    state = capture_git(cwd)
    return None if state is None else {"commit": state.commit, "dirty": state.dirty}


def assemble_meta(
    *,
    detector_meta: DetectorMeta,
    checkpoint_sha256: str | None,
    protocol: Protocol,
    split: str,
    where: Mapping[str, Any] | None,
    labels: str,
    processing_profile: ProcessingProfile,
    adaptation: AdaptResult,
    aggregate_mode: str,
    clips_per_video: int,
    rows: Sequence[ScoreRow],
    seed: int,
    device: str,
    precision: str,
    store_index_sha256: str | None,
    command: str | None = None,
) -> ScoreMeta:
    """Build the C5 meta for one scoring run.

    ``coverage`` is always recomputed from ``rows`` (never taken on faith), so it can never drift
    from what :func:`~dfwb.core.records.scores.write_scores` will itself check. ``seed`` is
    whatever the caller decides is the scoring run's seed (the harness records the detector's own
    ``training_seed`` when it has one, else the ``seed`` argument it was called with); ``device``
    is the device the run actually scored on, recorded verbatim rather than
    :func:`~dfwb.core.runmeta.capture_env`'s own guess (which reports a CUDA device whenever one
    happens to be available on the machine, whether or not this run used it). ``precision`` (the
    resolved autocast precision, ``"fp32"`` for none) and ``store_index_sha256`` (the hash of the
    processed store's ``index.jsonl`` the run read, ``None`` when it had none) change the scores
    but have no field of their own in the C5 meta, so both are recorded in its free-form ``env``.
    """
    from dfwb.core.runmeta import capture_env, utc_now

    env = capture_env()
    env["device"] = device
    env["precision"] = precision
    env["store_index_sha256"] = store_index_sha256
    return ScoreMeta.model_validate(
        {
            "detector": {
                "name": detector_meta.name,
                "version": detector_meta.version,
                "source": detector_meta.source or "unknown",
                "checkpoint_sha256": checkpoint_sha256,
                "contract_version": list(detector_meta.contract_version),
            },
            "protocol": {
                "id": protocol.ref,
                "split": split,
                "where": dict(where or {}),
                "pack": protocol.pack.name,
                "pack_version": protocol.pack_version,
                "scheme_sha256": protocol.sha256,
            },
            "labels": labels,
            "processing_profile": {
                "id": processing_profile.profile_id(),
                "sha256": processing_profile.sha256(),
            },
            "input_adaptation": {
                "derived_crop": adaptation.derived_crop,
                "mismatch_override": adaptation.mismatch,
            },
            "aggregation": {
                "clip_to_video": aggregate_mode,
                "clips_per_video": clips_per_video,
            },
            "coverage": coverage_counts(list(rows)),
            "seed": seed,
            "env": env,
            "git": _git_payload(Path.cwd()),
            "command": command,
            "created": utc_now(),
        }
    )


# ---------------------------------------------------------------------------------- frame dump


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One row of ``--frames``'s ``<name>.frames.parquet``: one clip's one frame.

    ``frame_score`` is the detector's own per-frame score when it returned one
    (``DetectorOutput.frame_scores``); detectors that score a clip as a whole (most of them) leave
    ``frame_scores`` unset, so the clip's own ``clip_score`` is repeated for each of its frames --
    still useful for later analysis (which frames a clip's score drew on), even though every frame
    of that clip then reads the same value.
    """

    dataset: str
    key: str
    compression: str | None
    clip_index: int
    frame_index: int
    frame_score: float
    clip_score: float


def frames_path_for(csv_path: Path) -> Path:
    """``<name>.scores.csv`` -> ``<name>.frames.parquet``."""
    return csv_path.with_name(csv_path.name.removesuffix(".scores.csv") + ".frames.parquet")


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow
        import pyarrow.parquet
    except ModuleNotFoundError as exc:
        raise InstallationError(
            "the frame-level dump (--frames) needs pyarrow, which is not installed",
            hint=install_hint("pyarrow"),
        ) from exc
    return pyarrow, pyarrow.parquet


def require_pyarrow() -> None:
    """Raise now if pyarrow (the ``[eval]`` extra) is not installed -- a fail-fast pre-flight
    check for ``--frames``, called before any clip is scored, rather than discovering the same
    thing only after a whole run's worth of work.

    Raises:
        InstallationError: pyarrow is not installed.
    """
    _pyarrow()


def write_frames(path: Path, records: Sequence[FrameRecord]) -> Path:
    """Write ``records`` as ``path`` (a parquet file): one row per clip per frame.

    Raises:
        InstallationError: pyarrow (the ``[eval]`` extra) is not installed.
    """
    pa, pq = _pyarrow()
    columns = {
        "dataset": [r.dataset for r in records],
        "key": [r.key for r in records],
        "compression": [r.compression for r in records],
        "clip_index": [r.clip_index for r in records],
        "frame_index": [r.frame_index for r in records],
        "frame_score": [r.frame_score for r in records],
        "clip_score": [r.clip_score for r in records],
    }
    table = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path
