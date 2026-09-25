"""Runs any C4 :class:`~dfwb.core.detector.Detector` over a protocol split and writes a C5 score
file: resolve the detector, choose a processing profile, adapt stored clips to its input, score
every clip, aggregate clip -> video, and write ``<name>.scores.{csv,meta.json}``.

Every video the split names gets a row: ``ok`` (scored), ``missing`` (no usable processed clip) or
``error`` (the detector raised while scoring it, or returned an output that fails validation).
Nothing is silently dropped.

A detector source may set plain attributes on the ``Detector`` it returns, beyond contract C4
(``meta``, ``to()``, ``predict()``) -- see :mod:`dfwb.score.cache` for what they are and how the
cache uses them; :func:`score` reuses a cached file already sitting at its own output path unless
``force`` is given or the detector itself says it is not cacheable.

``frames=True`` also writes ``<name>.frames.parquet`` (the ``[eval]`` extra: pyarrow; skipped,
with a log message, when it is not installed): one row per clip per frame actually scored, so a
cache hit -- which scores nothing -- never produces one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError, InstallationError, did_you_mean
from dfwb.core.paths import require_root, resolve_roots
from dfwb.core.records import ScoreRow, write_scores
from dfwb.data.index import SourceSpec, VideoIndex
from dfwb.eval.aggregate import aggregate as aggregate_scores
from dfwb.protocols.protocol import Protocol
from dfwb.protocols.protocol import load as load_protocol
from dfwb.score.cache import DetectorIdentity, cache_key, look_up, score_path
from dfwb.score.sources import resolve_detector
from dfwb.score.writer import FrameRecord, assemble_meta, frames_path_for, write_frames

if TYPE_CHECKING:
    from torch import Tensor

    from dfwb.core.records.local import ProcessingProfile
    from dfwb.data.adapt import AdaptResult

__all__ = ["ScoreResult", "score"]

_log = logging.getLogger(__name__)

VideoKey = tuple[str, str, str | None]  # (dataset, key, compression)

_SCORES_SUBDIR = "scores"
_PRECISIONS = ("fp16", "bf16", "fp32")
_AGGREGATE_MODES = ("mean-prob", "mean-logit", "max", "median")


@dataclass(frozen=True)
class ScoreResult:
    """What one call to :func:`score` produced.

    ``frames_path`` is set only when ``frames=True`` was given, the run actually scored some
    clips (not a cache hit) and pyarrow is installed; otherwise it is ``None``.
    """

    csv_path: Path
    meta_path: Path
    coverage: dict[str, int]
    cached: bool
    frames_path: Path | None = None


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


def _frame_records(
    batch: Any, keys: Sequence[VideoKey], scores: Sequence[float], output: Any
) -> list[FrameRecord]:
    """One :class:`~dfwb.score.writer.FrameRecord` per clip per frame of a successfully scored
    batch. ``output.frame_scores`` (``[B, T]``) is used when the detector set it; a detector that
    scores a clip as a whole (``frame_scores`` left ``None``) has its clip score repeated across
    the clip's own frames instead, so every clip still contributes a row per frame it drew on."""
    frame_scores = getattr(output, "frame_scores", None)
    per_clip_frame_scores = (
        frame_scores.detach().float().cpu().tolist() if frame_scores is not None else None
    )
    records: list[FrameRecord] = []
    for i, key in enumerate(keys):
        dataset, video_key, compression = key
        clip_index = int(batch.clip_index[i].item())
        frame_indices = batch.frame_indices[i].tolist()
        per_frame = (
            per_clip_frame_scores[i]
            if per_clip_frame_scores is not None
            else [scores[i]] * len(frame_indices)
        )
        for frame_index, frame_score in zip(frame_indices, per_frame, strict=True):
            records.append(
                FrameRecord(
                    dataset=dataset,
                    key=video_key,
                    compression=compression,
                    clip_index=clip_index,
                    frame_index=int(frame_index),
                    frame_score=float(frame_score),
                    clip_score=scores[i],
                )
            )
    return records


def _score_videos(
    detector: Any,
    dataset: Any,
    *,
    batch_size: int,
    device: str,
    precision: str | None,
    collect_frames: bool = False,
) -> tuple[dict[VideoKey, list[float]], dict[VideoKey, str], list[FrameRecord]]:
    """Score every clip of ``dataset`` and group it by video.

    Returns ``(clip_scores, errored, frame_records)``: ``clip_scores`` maps a video key to its
    clip scores (``ok``); ``errored`` maps a video key to why its batch failed; ``frame_records``
    is every scored clip's per-frame record (see :func:`_frame_records`), collected only when
    ``collect_frames`` is true. A video's clips never split across two batches
    (:class:`~dfwb.data.samplers.VideoGrouped`), so a key lands in exactly one of the two.
    ``batch.labels`` is cleared before ``predict()``: contract C4 gives labels to training and
    validation batches only, never a scoring one.
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
    frame_records: list[FrameRecord] = []
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
        if collect_frames:
            frame_records.extend(_frame_records(batch, keys, scores, output))
    return clip_scores, errored, frame_records


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
    frames: bool = False,
) -> ScoreResult:
    """Score every video of ``protocol``'s ``split`` with the detector named by ``detector_uri``.

    Resolves the detector (:func:`~dfwb.score.sources.resolve_detector`), chooses a processing
    profile and adapts stored clips to the detector's input, then -- unless an identical, still
    valid score file already exists at the cache path (:mod:`dfwb.score.cache`) and ``force`` is
    ``False``, and the detector itself is cacheable (``getattr(detector, "cacheable", True)``) --
    scores every clip under ``torch.inference_mode()``, aggregates clip scores to one score per
    video, and writes a C5 score file. Every video of the split gets a row: ``ok``, ``missing``
    (no usable processed clip), or ``error`` (the detector raised while scoring it, or returned an
    output that fails validation).

    ``frames=True`` additionally writes ``<name>.frames.parquet`` when clips were actually scored
    (never on a cache hit, which scores nothing); see :attr:`ScoreResult.frames_path`.

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
    identity = DetectorIdentity.of(detector)
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
    key = cache_key(
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
    target = score_path(
        out_root,
        detector_name=detector.meta.name,
        protocol_ref=loaded_protocol.ref,
        split=split,
        key=key,
    )
    if not force and identity.cacheable and target.is_file():
        hit = look_up(
            target,
            identity=identity,
            scheme_sha256=loaded_protocol.sha256,
            split=split,
            profile_sha256=profile_sha256,
            aggregate_mode=aggregate,
        )
        if hit is not None:
            return ScoreResult(hit.csv_path, hit.meta_path, hit.coverage, True)

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

    clip_scores, errored, frame_records = _score_videos(
        detector,
        dataset,
        batch_size=batch_size,
        device=device,
        precision=precision,
        collect_frames=frames,
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

    frames_result_path: Path | None = None
    if frames and frame_records:
        try:
            frames_result_path = write_frames(frames_path_for(csv_path), frame_records)
        except InstallationError as exc:
            _log.warning("--frames: %s (%s); skipping the frame-level dump", exc.message, exc.hint)

    return ScoreResult(
        csv_path, meta_path, dict(meta.coverage.model_dump()), False, frames_result_path
    )
