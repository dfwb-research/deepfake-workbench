"""Caches C5 score files: whether a file already sitting at a request's own output path can be
trusted, so re-running :func:`dfwb.score.harness.score` with an identical configuration reuses it
instead of rescoring every clip.

The cache key is a hash of everything a scoring run depends on -- the detector's exact identity,
the protocol split (including its scheme hash, the pack's version and any ``where`` filter), the
processing profile, what the processed store actually serves (the hash of its ``index.jsonl``),
the aggregation mode and clip count, the label mapping and the precision -- so the *path* a score
file is written to already encodes its own configuration (``dfwb.score.harness``'s output naming
appends the first 8 hex characters of this key). A file already at that path is then read back
and its meta is checked field by field against what this request expects, rather than trusted on
the path alone: a hash collision, or a stale/foreign file left there by hand, is recomputed rather
than served. The precision and the store's index hash have no field of their own in a C5 meta, so
they are recorded in its free-form ``env`` (as ``precision`` and ``store_index_sha256``) and
checked there.

A detector source may set plain attributes on the ``Detector`` it returns, beyond contract C4
(``meta``, ``to()``, ``predict()``): ``checkpoint_sha256`` (the sha256 of the exact weights file
scored), ``training_seed`` (the seed it was trained with), ``fingerprint_extra`` (a source-owned
string that stands in for ``meta.source`` in the cache key, for a source whose ``meta.source`` is
not a reliable identity on its own) and ``cacheable`` (``bool``, default ``True`` when unset --
``False`` means an existing file at this detector's cache path is never trusted, no matter how
recently it was written; the harness always recomputes and overwrites it instead). None is
required -- read with ``getattr(detector, "checkpoint_sha256", None)`` -- but when present they
sharpen the C5 meta and the cache key beyond what ``meta.source`` alone can
(:mod:`dfwb.models.source`'s ``run:`` sets the first two; :func:`dfwb.score.sources.load_py`'s
``py:`` sets the third always, and the fourth when it cannot tell whether its own source has
changed since the last load).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dfwb.core.errors import ContractError
from dfwb.core.hashing import fingerprint
from dfwb.core.records.scores import ScoreMeta, canonical_where, meta_path_for, read_scores

__all__ = [
    "CacheHit",
    "DetectorIdentity",
    "cache_key",
    "cache_matches",
    "look_up",
    "score_path",
    "slug",
]

_log = logging.getLogger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """Lower-kebab-case a name for use inside a filesystem path."""
    return _SLUG.sub("-", text.strip().lower()).strip("-") or "x"


# ------------------------------------------------------------------------------- detector identity


@dataclass(frozen=True)
class DetectorIdentity:
    """What names a detector's exact weights, read duck-typed off whatever
    :func:`~dfwb.score.sources.resolve_detector` returned (see the module docstring)."""

    source: str | None
    checkpoint_sha256: str | None
    training_seed: int | None
    fingerprint_extra: str | None
    cacheable: bool

    @classmethod
    def of(cls, detector: Any) -> DetectorIdentity:
        return cls(
            source=detector.meta.source,
            checkpoint_sha256=getattr(detector, "checkpoint_sha256", None),
            training_seed=getattr(detector, "training_seed", None),
            fingerprint_extra=getattr(detector, "fingerprint_extra", None),
            cacheable=getattr(detector, "cacheable", True),
        )

    def effective_seed(self, requested_seed: int) -> int:
        """The seed C5 records: the detector's own training seed when it has one, else whatever
        :func:`~dfwb.score.harness.score` was called with."""
        return self.training_seed if self.training_seed is not None else requested_seed


def _fingerprint_payload(detector: Any, identity: DetectorIdentity) -> dict[str, Any]:
    meta = detector.meta
    return {
        "name": meta.name,
        "version": meta.version,
        "source": identity.source,
        "contract_version": list(meta.contract_version),
        "checkpoint_sha256": identity.checkpoint_sha256,
        "fingerprint_extra": identity.fingerprint_extra,
    }


# ------------------------------------------------------------------------------------ cache key


def cache_key(
    *,
    detector: Any,
    identity: DetectorIdentity,
    effective_seed: int,
    scheme_sha256: str,
    split: str,
    where: Mapping[str, Any] | None,
    profile_sha256: str,
    aggregate_mode: str,
    clips_per_video: int,
    labels: str,
    precision: str,
    store_index_sha256: str | None,
    pack_version: str,
) -> str:
    """sha256 of the canonical JSON of everything a scoring run depends on: the detector's
    fingerprint, the protocol scheme hash, the split, any ``where`` filter, the processing
    profile's hash, the aggregation mode and clip count, the label mapping, the resolved
    ``precision`` (``"fp32"`` when no autocast was asked for), the hash of the processed store's
    ``index.jsonl`` (``None`` for a store with no index yet) and the pack's version (a release
    can fix labels without touching the split file, so the scheme hash alone cannot see it).
    Changing any one of these changes the key, and so the output path (see :func:`score_path`)."""
    payload = {
        "detector": _fingerprint_payload(detector, identity),
        "seed": effective_seed,
        "scheme_sha256": scheme_sha256,
        "split": split,
        "where": canonical_where(where),
        "profile_sha256": profile_sha256,
        "aggregation": aggregate_mode,
        "clips_per_video": clips_per_video,
        "labels": labels,
        "precision": precision,
        "store_index_sha256": store_index_sha256,
        "pack_version": pack_version,
    }
    return fingerprint(payload)


def score_path(
    out_root: Path, *, detector_name: str, protocol_ref: str, split: str, key: str
) -> Path:
    """``<out>/<detector-slug>/<protocol-slug>/<split>-<key[:8]>.scores.csv``."""
    return out_root / slug(detector_name) / slug(protocol_ref) / f"{split}-{key[:8]}.scores.csv"


# ------------------------------------------------------------------------------------ cache lookup


@dataclass(frozen=True)
class CacheHit:
    """A cached score file whose meta matches a request exactly."""

    csv_path: Path
    meta_path: Path
    coverage: dict[str, int]


def cache_matches(
    meta: ScoreMeta,
    *,
    identity: DetectorIdentity,
    scheme_sha256: str,
    split: str,
    where: Mapping[str, Any] | None,
    profile_sha256: str,
    aggregate_mode: str,
    clips_per_video: int,
    labels: str,
    precision: str,
    store_index_sha256: str | None,
    pack_version: str,
) -> bool:
    """Whether a cached file's meta actually matches this request -- the cache path is already
    the request's own hash, so a mismatch would mean a hash collision or a stale/foreign file left
    at that path by hand; either way, the honest thing is to recompute rather than trust it. Checks
    the fields of :func:`cache_key` that a C5 meta records, not just the ones most likely to
    collide; a meta written before ``precision`` and ``store_index_sha256`` were recorded in its
    ``env`` never matches, so such a file is recomputed once rather than trusted."""
    return (
        meta.detector.source == (identity.source or "unknown")
        and meta.detector.checkpoint_sha256 == identity.checkpoint_sha256
        and meta.protocol.scheme_sha256 == scheme_sha256
        and meta.protocol.pack_version == pack_version
        and meta.protocol.split == split
        and canonical_where(meta.protocol.where) == canonical_where(where)
        and meta.processing_profile is not None
        and meta.processing_profile.sha256 == profile_sha256
        and meta.aggregation is not None
        and meta.aggregation.clip_to_video == aggregate_mode
        and meta.aggregation.clips_per_video == clips_per_video
        and meta.labels == labels
        and meta.env.get("precision") == precision
        and "store_index_sha256" in meta.env
        and meta.env["store_index_sha256"] == store_index_sha256
    )


def look_up(target: Path, **expected: Any) -> CacheHit | None:
    """A :class:`CacheHit` for ``target`` when it exists, is a readable C5 file,
    :func:`cache_matches` (given ``expected``) confirms it, and its coverage has no ``error``
    rows; ``None`` otherwise -- an unreadable file, one with no meta, one whose meta does not
    match, or one where the detector failed on some of its videos last time is never trusted.
    A detector error is typically transient (an OOM, a flaky device fault); the reuse rule is
    about an identical *successful* result, so an errored cache entry is retried instead
    (logged at info level: how many videos), not served as if it were complete."""
    try:
        existing = read_scores(target)
    except (ContractError, OSError):
        return None
    if existing.meta is None or not cache_matches(existing.meta, **expected):
        return None
    coverage = dict(existing.meta.coverage.model_dump())
    if coverage["error"] > 0:
        _log.info(
            "%s: %d video(s) errored in the cached file; retrying them rather than reusing it",
            target,
            coverage["error"],
        )
        return None
    return CacheHit(target, meta_path_for(target), coverage)
