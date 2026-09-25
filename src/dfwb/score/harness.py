"""Runs any C4 :class:`~dfwb.core.detector.Detector` over a protocol split and writes a C5 score
file: resolve the detector, choose a processing profile, adapt stored clips to its input, score
every clip, aggregate clip -> video, and write ``<name>.scores.{csv,meta.json}``.

Every video the split names gets a row: ``ok`` (scored), ``missing`` (no usable processed clip) or
``error`` (the detector raised while scoring it, or returned an output that fails validation).
Nothing is silently dropped.

A detector source may set two plain attributes on the ``Detector`` it returns, beyond contract C4
(``meta``, ``to()``, ``predict()``): ``checkpoint_sha256`` (the sha256 of the exact weights file
scored) and ``training_seed`` (the seed it was trained with). Neither is required -- read with
``getattr(detector, "checkpoint_sha256", None)`` -- but when present they sharpen the C5 meta and
the cache key beyond what ``meta.source`` alone can (:mod:`dfwb.models.source`'s ``run:`` sets
both).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError, ContractError, did_you_mean
from dfwb.core.hashing import fingerprint
from dfwb.core.paths import require_root, resolve_roots
from dfwb.core.records import ScoreRow, read_scores, write_scores
from dfwb.core.records.scores import ScoreMeta, meta_path_for
from dfwb.data.index import SourceSpec, VideoIndex
from dfwb.eval.aggregate import aggregate as aggregate_scores
from dfwb.protocols.protocol import Protocol
from dfwb.protocols.protocol import load as load_protocol
from dfwb.score.sources import resolve_detector
from dfwb.score.writer import assemble_meta

if TYPE_CHECKING:
    from torch import Tensor

    from dfwb.core.records.local import ProcessingProfile
    from dfwb.data.adapt import AdaptResult

__all__ = ["ScoreResult", "score"]

_log = logging.getLogger(__name__)

VideoKey = tuple[str, str, str | None]  # (dataset, key, compression)

_SLUG = re.compile(r"[^a-z0-9]+")
_SCORES_SUBDIR = "scores"
_PRECISIONS = ("fp16", "bf16", "fp32")
_AGGREGATE_MODES = ("mean-prob", "mean-logit", "max", "median")


@dataclass(frozen=True)
class ScoreResult:
    """What one call to :func:`score` produced."""

    csv_path: Path
    meta_path: Path
    coverage: dict[str, int]
    cached: bool


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.strip().lower()).strip("-") or "x"


def _check_choice(value: str, allowed: Sequence[str], *, name: str) -> None:
    if value not in allowed:
        raise ConfigError(
            f"{name}: {value!r} is not one of {list(allowed)}{did_you_mean(value, allowed)}",
            hint=f"{name} accepts: " + ", ".join(allowed),
        )


# --------------------------------------------------------------------------------- profile choice


def _profile_matches(profile: ProcessingProfile, token: str) -> bool:
    return token in (profile.id, profile.profile_id())


def _candidate_text(candidates: Sequence[ProcessingProfile]) -> str:
    return ", ".join(p.profile_id() for p in candidates) if candidates else "(none processed)"


def _choose_profile(
    *,
    dataset: str,
    requested: str | None,
    spec: InputSpec,
    candidates: Sequence[ProcessingProfile],
) -> ProcessingProfile:
    """Pick one local processing profile: ``requested``, else ``spec.preferred_profile`` if it is
    processed locally, else the only local profile compatible with ``spec``
    (:func:`~dfwb.data.adapt.compatible_profiles`), else the only local profile at all
    (deferring the actual pass/refuse decision to :func:`~dfwb.data.adapt.adapt`), else an error.

    Raises:
        ConfigError: ``requested`` is not a locally processed profile, no profile has been
            processed locally at all, or more than one candidate is left after every rule above
            (the message lists only the tied candidates: every compatible one when there is more
            than one, otherwise every local one).
    """
    from dfwb.data.adapt import compatible_profiles

    if requested is not None:
        matches = [p for p in candidates if _profile_matches(p, requested)]
        if len(matches) == 1:
            return matches[0]
        raise ConfigError(
            f"profile {requested!r} is not a locally processed profile for {dataset!r} "
            f"(available: {_candidate_text(candidates)})",
            hint="run `dfwb preprocess profiles` to see what is processed locally",
        )
    if not candidates:
        raise ConfigError(
            f"{dataset!r} has no processed store locally", hint="run `dfwb preprocess` first"
        )
    if spec.preferred_profile is not None:
        preferred = [p for p in candidates if _profile_matches(p, spec.preferred_profile)]
        if len(preferred) == 1:
            return preferred[0]
    compatible = compatible_profiles(spec, candidates)
    if len(compatible) == 1:
        return compatible[0]
    if compatible:
        raise ConfigError(
            f"{len(compatible)} local processing profiles are compatible with this detector's "
            f"input for {dataset!r}: {_candidate_text(compatible)}",
            hint="pass profile=<id> to choose one explicitly",
        )
    if len(candidates) == 1:
        return candidates[0]
    raise ConfigError(
        f"no local processing profile can serve this detector's input for {dataset!r} "
        f"(available: {_candidate_text(candidates)})",
        hint="pass profile=<id> to choose one explicitly, or process this dataset with a "
        "compatible profile",
    )


# ------------------------------------------------------------------------------------- clip spec


def _clip_spec(spec: InputSpec, clips_per_video: int) -> Any:
    from dfwb.data.clips import ClipSpec, ClipsPerVideo

    sampling: Literal["uniform", "consecutive"] = (
        "consecutive" if spec.sampling == "consecutive" else "uniform"
    )
    return ClipSpec(
        frames=spec.frames,
        sampling=sampling,
        clips_per_video=ClipsPerVideo(train=clips_per_video, eval=clips_per_video),
        stride=1,
    )


# -------------------------------------------------------------------------------- scoring the run


def _validated_scores(output: Any, batch_size: int) -> Tensor:
    """``output.score``, checked against what C4 promises: one finite value in ``[0, 1]`` per
    clip. A ``[B, 1]`` score is squeezed (a common, harmless shape slip); anything else that is
    not already ``[B]`` is rejected, as is a non-finite or out-of-range value.

    Raises:
        ValueError: the score is not a tensor, is not (after squeezing) exactly ``[batch_size]``,
            or holds a non-finite value or one outside ``[0, 1]``.
    """
    import torch

    score = output.score
    if not isinstance(score, torch.Tensor):
        raise ValueError(f"predict() returned score of type {type(score).__name__}, not a Tensor")
    if score.ndim == 2 and score.shape[1] == 1:
        score = score.squeeze(-1)
    if score.ndim != 1 or score.shape[0] != batch_size:
        raise ValueError(
            f"predict() returned score of shape {tuple(score.shape)}, expected ({batch_size},)"
        )
    if not bool(torch.isfinite(score).all()):
        raise ValueError("predict() returned a non-finite score")
    if bool((score < 0.0).any()) or bool((score > 1.0).any()):
        raise ValueError("predict() returned a score outside [0, 1]")
    return score


def _score_videos(
    detector: Any,
    dataset: Any,
    *,
    batch_size: int,
    device: str,
    precision: str | None,
) -> tuple[dict[VideoKey, list[float]], dict[VideoKey, str]]:
    """Score every clip of ``dataset`` and group it by video.

    Returns ``(clip_scores, errored)``: ``clip_scores`` maps a video key to its clip scores
    (``ok``); ``errored`` maps a video key to why its batch failed. A video's clips never split
    across two batches (:class:`~dfwb.data.samplers.VideoGrouped`), so a key lands in exactly one
    of the two. ``batch.labels`` is cleared before ``predict()``: contract C4 gives labels to
    training and validation batches only, never a scoring one.
    """
    import torch
    from torch.utils.data import DataLoader

    from dfwb.data.collate import collate_clips
    from dfwb.data.samplers import VideoGrouped

    torch_device = torch.device(device)
    detector = detector.to(torch_device)
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_sampler=VideoGrouped(dataset, batch_size),
        collate_fn=collate_clips,
        num_workers=0,
    )
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}  # fp32 (or None): no autocast
    autocast_dtype = dtypes.get(precision or "")

    clip_scores: dict[VideoKey, list[float]] = {}
    errored: dict[VideoKey, str] = {}
    for batch in loader:
        keys: list[VideoKey] = [
            (batch.dataset_ids[i], batch.keys[i], batch.compressions[i])
            for i in range(len(batch.keys))
        ]
        try:
            batch.labels = None
            batch.clips = batch.clips.to(torch_device)
            with torch.inference_mode():
                if autocast_dtype is not None:
                    with torch.autocast(device_type=torch_device.type, dtype=autocast_dtype):
                        output = detector.predict(batch)
                else:
                    output = detector.predict(batch)
                validated = _validated_scores(output, len(keys))
            scores = validated.detach().float().cpu().tolist()
        except Exception as exc:  # a detector may fail for any reason; scoring continues
            reason = f"{type(exc).__name__}: {exc}"
            unique = sorted(set(keys))
            _log.warning("detector failed on %d video(s) (%s): %s", len(unique), unique, reason)
            for key in unique:
                errored[key] = reason
            continue
        for key, value in zip(keys, scores, strict=True):
            clip_scores.setdefault(key, []).append(float(value))
    return clip_scores, errored


def _rows_from_results(
    index: VideoIndex,
    clip_scores: Mapping[VideoKey, list[float]],
    errored: Mapping[VideoKey, str],
    *,
    aggregate_mode: str,
    frames_per_clip: int,
) -> list[ScoreRow]:
    items = {(item.dataset, item.key, item.compression): item for item in index.items}
    video_scores: dict[VideoKey, float] = {}
    if clip_scores:
        flat = ((key, value) for key, values in clip_scores.items() for value in values)
        video_scores = aggregate_scores(flat, aggregate_mode)

    rows: list[ScoreRow] = []
    for key, values in clip_scores.items():
        item = items[key]
        rows.append(
            ScoreRow(
                dataset=item.dataset,
                key=item.key,
                compression=item.compression,
                label=item.label,
                score=video_scores[key],
                status="ok",
                label_key=item.label_key,
                method=item.method,
                n_clips=len(values),
                n_frames=len(values) * frames_per_clip,
            )
        )
    for key in errored:
        item = items[key]
        rows.append(
            ScoreRow(
                dataset=item.dataset,
                key=item.key,
                compression=item.compression,
                label=item.label,
                score=None,
                status="error",
                label_key=item.label_key,
                method=item.method,
            )
        )
    return rows


def _missing_rows(
    protocol: Protocol,
    *,
    split: str,
    where: Mapping[str, Any] | None,
    excluded: Sequence[tuple[int, str, str | None, str]],
    labels: str,
) -> list[ScoreRow]:
    """A ``missing`` row for every split video the store could not serve. A video the label
    mapping excludes is not part of the split under that mapping, so it gets no row."""
    unscored = {
        (key, compression) for _, key, compression, reason in excluded if reason != "label-excluded"
    }
    if not unscored:
        return []
    mapping = protocol.labels(labels)
    rows: list[ScoreRow] = []
    for record in protocol.records(split=split, where=where):
        if (record.key, record.compression) not in unscored:
            continue
        label = mapping(record.label_key)
        if not isinstance(label, int):
            continue
        rows.append(
            ScoreRow(
                dataset=protocol.dataset,
                key=record.key,
                compression=record.compression,
                label=label,
                score=None,
                status="missing",
                label_key=record.label_key,
                method=record.method,
            )
        )
    return rows


# ------------------------------------------------------------------------------- detector identity


@dataclass(frozen=True)
class _DetectorIdentity:
    """What names a detector's exact weights, read duck-typed off whatever
    :func:`~dfwb.score.sources.resolve_detector` returned (see the module docstring)."""

    source: str | None
    checkpoint_sha256: str | None
    training_seed: int | None

    @classmethod
    def of(cls, detector: Any) -> _DetectorIdentity:
        return cls(
            source=detector.meta.source,
            checkpoint_sha256=getattr(detector, "checkpoint_sha256", None),
            training_seed=getattr(detector, "training_seed", None),
        )

    def effective_seed(self, requested_seed: int) -> int:
        """The seed C5 records: the detector's own training seed when it has one, else whatever
        :func:`score` was called with."""
        return self.training_seed if self.training_seed is not None else requested_seed


def _fingerprint_payload(detector: Any, identity: _DetectorIdentity) -> dict[str, Any]:
    meta = detector.meta
    return {
        "name": meta.name,
        "version": meta.version,
        "source": identity.source,
        "contract_version": list(meta.contract_version),
        "checkpoint_sha256": identity.checkpoint_sha256,
    }


# -------------------------------------------------------------------------------------- output path


def _cache_key(
    *,
    detector: Any,
    identity: _DetectorIdentity,
    effective_seed: int,
    scheme_sha256: str,
    split: str,
    where: Mapping[str, Any] | None,
    profile_sha256: str,
    aggregate_mode: str,
    clips_per_video: int,
    labels: str,
) -> str:
    payload = {
        "detector": _fingerprint_payload(detector, identity),
        "seed": effective_seed,
        "scheme_sha256": scheme_sha256,
        "split": split,
        "where": dict(where or {}),
        "profile_sha256": profile_sha256,
        "aggregation": aggregate_mode,
        "clips_per_video": clips_per_video,
        "labels": labels,
    }
    return fingerprint(payload)


def _score_path(
    out_root: Path, *, detector_name: str, protocol_ref: str, split: str, key: str
) -> Path:
    return out_root / _slug(detector_name) / _slug(protocol_ref) / f"{split}-{key[:8]}.scores.csv"


def _cache_matches(
    meta: ScoreMeta,
    *,
    identity: _DetectorIdentity,
    scheme_sha256: str,
    split: str,
    profile_sha256: str,
    aggregate_mode: str,
) -> bool:
    """Whether a cached file's meta actually matches this request -- the cache path is already
    the request's own hash, so a mismatch would mean a hash collision or a stale/foreign file left
    at that path by hand; either way, the honest thing is to recompute rather than trust it."""
    return (
        meta.detector.source == (identity.source or "unknown")
        and meta.detector.checkpoint_sha256 == identity.checkpoint_sha256
        and meta.protocol.scheme_sha256 == scheme_sha256
        and meta.protocol.split == split
        and meta.processing_profile is not None
        and meta.processing_profile.sha256 == profile_sha256
        and meta.aggregation is not None
        and meta.aggregation.clip_to_video == aggregate_mode
    )


def _cached_result(target: Path, **expected: Any) -> ScoreResult | None:
    try:
        existing = read_scores(target)
    except (ContractError, OSError):
        return None
    if existing.meta is None or not _cache_matches(existing.meta, **expected):
        return None
    coverage = dict(existing.meta.coverage.model_dump())
    return ScoreResult(target, meta_path_for(target), coverage, True)


# -------------------------------------------------------------------------------------------- API


def score(
    detector_uri: str,
    *,
    protocol: str,
    split: str,
    where: Mapping[str, Any] | None = None,
    profile: str | None = None,
    clips_per_video: int = 4,
    aggregate: str = "mean-prob",
    batch_size: int = 32,
    device: str = "cpu",
    precision: str | None = None,
    allow_input_mismatch: bool = False,
    labels: str = "binary",
    out: Path | None = None,
    force: bool = False,
    seed: int = 0,
) -> ScoreResult:
    """Score every video of ``protocol``'s ``split`` with the detector named by ``detector_uri``.

    Resolves the detector (:func:`~dfwb.score.sources.resolve_detector`), chooses a processing
    profile and adapts stored clips to the detector's input, then -- unless an identical, still
    valid score file already exists at the cache path and ``force`` is ``False`` -- scores every
    clip under ``torch.inference_mode()``, aggregates clip scores to one score per video, and
    writes a C5 score file. Every video of the split gets a row: ``ok``, ``missing`` (no usable
    processed clip), or ``error`` (the detector raised while scoring it, or returned an output
    that fails validation).

    Raises:
        ConfigError: ``precision``/``aggregate`` is not one of the values below, or no local
            processing profile can serve the detector's input (see :func:`_choose_profile`).
        ContractError: the detector's input cannot be adapted from the chosen profile and
            ``allow_input_mismatch`` is ``False`` (see :func:`~dfwb.data.adapt.adapt`).
    """
    if precision is not None:
        _check_choice(precision, _PRECISIONS, name="precision")
    _check_choice(aggregate, _AGGREGATE_MODES, name="aggregate")

    from dfwb.data.adapt import adapt as adapt_input
    from dfwb.data.adapt import available_profiles
    from dfwb.data.dataset import ClipDataset

    detector = resolve_detector(detector_uri)
    spec: InputSpec = detector.meta.input
    identity = _DetectorIdentity.of(detector)
    effective_seed = identity.effective_seed(seed)

    roots = resolve_roots()
    work_root = require_root("work", roots)

    loaded_protocol = load_protocol(protocol, work_root=work_root)
    candidates = available_profiles(work_root, loaded_protocol.dataset)
    chosen_profile = _choose_profile(
        dataset=loaded_protocol.dataset, requested=profile, spec=spec, candidates=candidates
    )
    adaptation: AdaptResult = adapt_input(
        spec, chosen_profile, allow_mismatch=allow_input_mismatch, candidates=candidates
    )
    if adaptation.mismatch:
        _log.warning("%s: input mismatch allowed: %s", loaded_protocol.ref, adaptation.reason)

    # Everything the cache key needs is known now; check before doing any of the actual scoring
    # work (joining the split with the store, building a dataset, running the detector).
    out_root = Path(out) if out is not None else require_root("runs", roots) / _SCORES_SUBDIR
    profile_sha256 = chosen_profile.sha256()
    key = _cache_key(
        detector=detector,
        identity=identity,
        effective_seed=effective_seed,
        scheme_sha256=loaded_protocol.sha256,
        split=split,
        where=where,
        profile_sha256=profile_sha256,
        aggregate_mode=aggregate,
        clips_per_video=clips_per_video,
        labels=labels,
    )
    target = _score_path(
        out_root,
        detector_name=detector.meta.name,
        protocol_ref=loaded_protocol.ref,
        split=split,
        key=key,
    )
    if not force and target.is_file():
        cached = _cached_result(
            target,
            identity=identity,
            scheme_sha256=loaded_protocol.sha256,
            split=split,
            profile_sha256=profile_sha256,
            aggregate_mode=aggregate,
        )
        if cached is not None:
            return cached

    index = VideoIndex.build(
        [SourceSpec(protocol, split, where)],
        profile=chosen_profile.profile_id(),
        labels=labels,
        work_root=work_root,
    )

    clip_spec = _clip_spec(spec, clips_per_video)
    dataset = ClipDataset(
        index, clip_spec, train=False, transform=None, adapt_chain=adaptation.chain, seed=seed
    )

    clip_scores, errored = _score_videos(
        detector, dataset, batch_size=batch_size, device=device, precision=precision
    )
    rows = _rows_from_results(
        index, clip_scores, errored, aggregate_mode=aggregate, frames_per_clip=spec.frames
    )
    rows.extend(
        _missing_rows(
            loaded_protocol, split=split, where=where, excluded=index.excluded, labels=labels
        )
    )
    rows.sort(key=lambda row: (row.dataset, row.key, row.compression or ""))

    meta = assemble_meta(
        detector_meta=detector.meta,
        checkpoint_sha256=identity.checkpoint_sha256,
        protocol=loaded_protocol,
        split=split,
        where=where,
        labels=labels,
        processing_profile=chosen_profile,
        adaptation=adaptation,
        aggregate_mode=aggregate,
        clips_per_video=clips_per_video,
        rows=rows,
        seed=effective_seed,
        device=device,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = write_scores(target, rows, meta)
    return ScoreResult(csv_path, meta_path, dict(meta.coverage.model_dump()), False)
