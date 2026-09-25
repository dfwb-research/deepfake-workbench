"""Assembles a score file's C5 meta (``<name>.scores.meta.json``) from what the harness knows
about a scoring run: the resolved detector, the protocol split, the chosen processing profile and
how it was adapted, the aggregation mode, and the rows that were actually written.

Kept separate from :mod:`dfwb.score.harness` so the shape of a C5 meta payload lives in one place,
independent of how the clips were actually scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dfwb.core.records import ScoreMeta, ScoreRow
from dfwb.core.records.scores import coverage_counts

if TYPE_CHECKING:
    from dfwb.core.detector import DetectorMeta
    from dfwb.core.records.local import ProcessingProfile
    from dfwb.data.adapt import AdaptResult
    from dfwb.protocols.protocol import Protocol

__all__ = ["assemble_meta"]


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
    command: str | None = None,
) -> ScoreMeta:
    """Build the C5 meta for one scoring run.

    ``coverage`` is always recomputed from ``rows`` (never taken on faith), so it can never drift
    from what :func:`~dfwb.core.records.scores.write_scores` will itself check. ``seed`` is
    whatever the caller decides is the scoring run's seed (the harness records the detector's own
    ``training_seed`` when it has one, else the ``seed`` argument it was called with); ``device``
    is the device the run actually scored on, recorded verbatim rather than
    :func:`~dfwb.core.runmeta.capture_env`'s own guess (which reports a CUDA device whenever one
    happens to be available on the machine, whether or not this run used it).
    """
    from dfwb.core.runmeta import capture_env, utc_now

    env = capture_env()
    env["device"] = device
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
