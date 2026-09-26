"""Runs any C4 :class:`~dfwb.core.detector.Detector` over a protocol split and writes a C5 score
file: resolve the detector, choose a processing profile, adapt stored clips to its input, score
every clip, aggregate clip -> video, and write ``<name>.scores.{csv,meta.json}``.

Every video the split names gets a row: ``ok`` (scored), ``missing`` (no usable processed clip) or
``error`` (the detector raised while scoring it, it returned an output that fails validation, or
one of its stored frames could not be read -- corrupt, or not the size its processing profile
declares). Nothing is silently dropped.

A detector source may set plain attributes on the ``Detector`` it returns, beyond contract C4
(``meta``, ``to()``, ``predict()``) -- see :mod:`dfwb.score.cache` for what they are and how the
cache uses them; :func:`score` reuses a cached file already sitting at its own output path unless
``force`` is given, the detector itself says it is not cacheable, or the cached file's own coverage
recorded any ``error`` row (a detector failure is usually transient, so the reuse rule is about an
identical *successful* result -- an errored cache entry is retried, not served).

``frames=True`` also writes ``<name>.frames.parquet`` (the ``[eval]`` extra: pyarrow, checked for
before any clip is scored, not after) with one row per clip per frame actually scored. A cache hit
returns its own ``<name>.frames.parquet`` when one already sits next to the cached CSV; when it
does not (the file was cached before ``--frames`` was first asked for, say), the hit is treated as
a miss and the run is recomputed, so ``frames=True`` always means "one exists" on return, never
"one might, depending on history". Whenever a fresh CSV is written without a matching frames dump
(``frames=False``, or every clip errored/was missing), any stale frames file already at that path
is removed, so a frames file's mere presence is always trustworthy on its own -- a caller never has
to also check the CSV's own coverage before believing it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError, did_you_mean
from dfwb.core.paths import require_root, resolve_roots
from dfwb.core.records import ScoreRow, canonical_where, write_scores
from dfwb.data.index import SourceSpec, VideoIndex, store_index_sha256
from dfwb.eval.aggregate import aggregate as aggregate_scores
from dfwb.protocols.protocol import Protocol
from dfwb.protocols.protocol import load as load_protocol
from dfwb.score.cache import DetectorIdentity, cache_key, look_up, score_path
from dfwb.score.sources import resolve_detector
from dfwb.score.writer import (
    FrameRecord,
    assemble_meta,
    frames_path_for,
    require_pyarrow,
    write_frames,
)

if TYPE_CHECKING:
    from torch import Tensor

    from dfwb.core.records.local import ProcessingProfile
    from dfwb.data.adapt import AdaptResult

__all__ = ["ScoreResult", "check_options", "score"]

_log = logging.getLogger(__name__)

VideoKey = tuple[str, str, str | None]  # (dataset, key, compression)

_SCORES_SUBDIR = "scores"
_PRECISIONS = ("fp16", "bf16", "fp32")
_NO_AUTOCAST = "fp32"
_AGGREGATE_MODES = ("mean-prob", "mean-logit", "max", "median")


@dataclass(frozen=True)
class ScoreResult:
    """What one call to :func:`score` produced.

    ``frames_path`` is set whenever ``frames=True`` was given and there was anything to write it
    from: a fresh run with at least one ``ok`` clip, or a cache hit whose own frames file already
    exists. It is ``None`` only when ``frames=False``, or when ``frames=True`` but nothing was
    ever actually scored for this call (every video ``missing``/``error``, so there is no
    frame-level data at all).
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


# ------------------------------------------------------------------------------------- device


_DEVICE_NAMES = ("cpu", "cuda", "gpu")


def _normalize_device(device: str) -> str:
    """Validate ``--device`` before any work (resolving the detector, reading the store, ...),
    and return a string :func:`torch.device` itself accepts.

    Accepts the same grammar :func:`dfwb.train.run.device_options` does: ``cpu``, ``cuda`` (or
    the common alias ``gpu``), or ``cuda:<index>``/``gpu:<index>`` for one particular GPU --
    normalised to ``cuda``/``cuda:<index>``, since raw :func:`torch.device` does not know ``gpu``
    as a device type the way Lightning's accelerator names do.

    Raises:
        ConfigError: ``device`` names no known scheme, or asks for CUDA when
            ``torch.cuda.is_available()`` is ``False``.
    """
    import torch

    name, sep, index = device.partition(":")
    if name == "cpu" and not sep:
        return device
    if name in ("cuda", "gpu") and (not sep or index.isdigit()):
        if not torch.cuda.is_available():
            raise ConfigError(
                f"--device: {device!r} needs a CUDA device, but none is available on this machine",
                hint="use --device cpu",
            )
        return f"cuda:{index}" if sep else "cuda"
    raise ConfigError(
        f"--device: {device!r} is not a device{did_you_mean(name, _DEVICE_NAMES)}",
        hint="use cpu, cuda, or cuda:<index> for one particular GPU",
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


def _validated_frame_scores(output: Any, batch: Any) -> list[list[float]] | None:
    """``output.frame_scores`` as plain per-clip lists, checked against what C4 promises: a
    tensor of shape ``[B, T]`` (one per frame of each clip, the shape of ``batch.frame_indices``),
    every value finite and in ``[0, 1]``. ``None`` when the detector set none.

    Raises:
        ValueError: ``frame_scores`` is not a tensor, has another shape, or holds a non-finite
            value or one outside ``[0, 1]``.
    """
    import torch

    frame_scores = getattr(output, "frame_scores", None)
    if frame_scores is None:
        return None
    if not isinstance(frame_scores, torch.Tensor):
        raise ValueError(
            f"predict() returned frame_scores of type {type(frame_scores).__name__}, not a Tensor"
        )
    expected = tuple(batch.frame_indices.shape)
    if tuple(frame_scores.shape) != expected:
        raise ValueError(
            f"predict() returned frame_scores of shape {tuple(frame_scores.shape)}, "
            f"expected {expected}"
        )
    values = frame_scores.detach().float().cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("predict() returned a non-finite frame score")
    if bool((values < 0.0).any()) or bool((values > 1.0).any()):
        raise ValueError("predict() returned a frame score outside [0, 1]")
    per_clip: list[list[float]] = values.tolist()
    return per_clip


def _frame_records(
    batch: Any, keys: Sequence[VideoKey], scores: Sequence[float], output: Any
) -> list[FrameRecord]:
    """One :class:`~dfwb.score.writer.FrameRecord` per clip per frame of a successfully scored
    batch. ``output.frame_scores`` (``[B, T]``, checked by :func:`_validated_frame_scores`) is
    used when the detector set it; a detector that scores a clip as a whole (``frame_scores``
    left ``None``) has its clip score repeated across the clip's own frames instead, so every
    clip still contributes a row per frame it drew on.

    Raises:
        ValueError: ``output.frame_scores`` fails validation.
    """
    per_clip_frame_scores = _validated_frame_scores(output, batch)
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
    clip scores (``ok``); ``errored`` maps a video key to why it failed; ``frame_records`` is
    every scored clip's per-frame record (see :func:`_frame_records`), collected only when
    ``collect_frames`` is true.

    A video's clips never split across two batches (:class:`~dfwb.data.samplers.VideoGrouped`),
    but several videos' clips do share one (``batch_size`` clips a batch, by default 32): a batch
    whose stored frames fail to fetch -- one video's corrupt or mis-sized frame -- is retried one
    video at a time instead, so only the video actually at fault is marked ``error``; every other
    video sharing that batch is scored normally, exactly as if it had had a batch of its own. A
    detector failure (raised, or an invalid output) is not retried this way, since it happened
    after the batch was already fetched intact and cannot be attributed to one clip in it: it
    still marks every video of that batch ``error``, as before. ``batch.labels`` is cleared before
    ``predict()``: contract C4 gives labels to training and validation batches only, never a
    scoring one. A detector with an ``eval()`` method (a torch module, typically) is switched to
    inference behaviour first -- no dropout, batch norm using its stored statistics -- whatever
    state its source left it in.
    """
    import itertools

    import torch

    from dfwb.data.collate import collate_clips
    from dfwb.data.samplers import VideoGrouped

    torch_device = torch.device(device)
    detector = detector.to(torch_device)
    set_eval = getattr(detector, "eval", None)
    if callable(set_eval):
        set_eval()
    batches = list(VideoGrouped(dataset, batch_size))
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}  # fp32 (or None): no autocast
    autocast_dtype = dtypes.get(precision or "")

    clip_scores: dict[VideoKey, list[float]] = {}
    errored: dict[VideoKey, str] = {}
    frame_records: list[FrameRecord] = []

    def _predict(batch: Any, keys: Sequence[VideoKey]) -> tuple[list[float], list[FrameRecord]]:
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
        records = _frame_records(batch, keys, scores, output) if collect_frames else []
        return scores, records

    for indices in batches:
        keys: list[VideoKey] = [dataset.video_key(i) for i in indices]
        try:
            samples = [dataset[i] for i in indices]
        except Exception as exc:
            unique = sorted(set(keys))
            _log.warning(
                "%d video(s) share a batch whose stored frames failed to load (%s); scoring "
                "them one at a time to isolate the failure: %s",
                len(unique),
                unique,
                f"{type(exc).__name__}: {exc}",
            )
            groups = (list(g) for _, g in itertools.groupby(indices, key=dataset.video_key))
            for video_indices in groups:
                video_key = dataset.video_key(video_indices[0])
                try:
                    video_samples = [dataset[i] for i in video_indices]
                    scores, records = _predict(
                        collate_clips(video_samples), [video_key] * len(video_indices)
                    )
                except Exception as video_exc:
                    reason = f"{type(video_exc).__name__}: {video_exc}"
                    _log.warning("scoring failed on video %s: %s", video_key, reason)
                    errored[video_key] = reason
                    continue
                clip_scores.setdefault(video_key, []).extend(float(value) for value in scores)
                frame_records.extend(records)
            continue

        try:
            scores, records = _predict(collate_clips(samples), keys)
        except Exception as exc:  # a detector may fail for any reason; scoring continues
            reason = f"{type(exc).__name__}: {exc}"
            unique = sorted(set(keys))
            _log.warning("detector failed on %d video(s) (%s): %s", len(unique), unique, reason)
            for key in unique:
                errored[key] = reason
            continue
        for key, value in zip(keys, scores, strict=True):
            clip_scores.setdefault(key, []).append(float(value))
        frame_records.extend(records)
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


def check_options(*, precision: str | None, aggregate: str, device: str, frames: bool) -> str:
    """Check the options of a scoring request before any work -- resolving the detector, reading
    the store -- and return ``device`` normalised to what :func:`torch.device` accepts.
    :func:`score` calls this itself; a caller that resolves the detector once for several
    requests (``dfwb score --suite``) calls it first, so a bad option is still refused before
    that.

    Raises:
        ConfigError: ``precision``/``aggregate`` is not one of the accepted values, or ``device``
            names no known scheme or asks for CUDA when none is available.
        InstallationError: ``frames`` is true and pyarrow (the ``[eval]`` extra) is not
            installed.
    """
    if precision is not None:
        _check_choice(precision, _PRECISIONS, name="precision")
    _check_choice(aggregate, _AGGREGATE_MODES, name="aggregate")
    normalized = _normalize_device(device)
    if frames:
        require_pyarrow()
    return normalized


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
    shipped_profiles: Sequence[ProcessingProfile] = (),
    command: str | None = None,
    detector: Any = None,
) -> ScoreResult:
    """Score every video of ``protocol``'s ``split`` with the detector named by ``detector_uri``.

    Resolves the detector (:func:`~dfwb.score.sources.resolve_detector`), chooses a processing
    profile and adapts stored clips to the detector's input, then -- unless an identical, still
    valid score file already exists at the cache path (:mod:`dfwb.score.cache`) and ``force`` is
    ``False``, and the detector itself is cacheable (``getattr(detector, "cacheable", True)``) --
    scores every clip under ``torch.inference_mode()``, aggregates clip scores to one score per
    video, and writes a C5 score file. Every video of the split gets a row: ``ok``, ``missing``
    (no usable processed clip), or ``error`` (the detector raised while scoring it, it returned an
    output that fails validation, or one of its stored frames could not be read).

    ``frames=True`` additionally writes ``<name>.frames.parquet``; see the module docstring for
    exactly when, and :attr:`ScoreResult.frames_path`.

    ``shipped_profiles`` is the processing profiles the installed framework ships (``dfwb.score``
    never imports the face pipeline that owns them, so a caller that may -- the CLI -- passes them
    in, the same way ``dfwb train`` does): when no local, compatible processing profile exists, the
    ``ContractError`` below also names the shipped profiles that would serve the detector once data
    is processed with one of them.

    ``detector``, when given, is what ``resolve_detector(detector_uri, seed=seed)`` already
    returned for this very URI and seed, and is used as it is instead of resolving it again: a
    caller scoring several splits with one detector (``dfwb score --suite``) loads its checkpoint,
    or hashes its weights, once rather than once per split.

    ``command`` is the command line recorded in the C5 meta (``dfwb score`` passes its own,
    sanitised by :func:`~dfwb.core.runmeta.sanitize_command`, so it names no absolute path); a
    cache hit keeps the command of the run that wrote the file.

    Raises:
        ConfigError: ``precision``/``aggregate`` is not one of the values below, ``device`` names
            no known scheme or asks for CUDA when none is available, or no local processing
            profile can serve the detector's input (see :func:`_choose_profile`).
        ContractError: the detector's input cannot be adapted from the chosen profile and
            ``allow_input_mismatch`` is ``False`` (see :func:`~dfwb.data.adapt.adapt`).
        InstallationError: ``frames=True`` and pyarrow (the ``[eval]`` extra) is not installed --
            checked before any clip is scored, not discovered afterwards.
    """
    device = check_options(precision=precision, aggregate=aggregate, device=device, frames=frames)
    # Canonicalised once, here, and threaded through everything below that hashes, stores or
    # compares it (the cache key, the C5 meta a run is recorded under, a cache hit's own match
    # against that meta, and the split join) -- never canonicalised again at each of those points
    # separately, which would let a freshly canonicalised incoming `where` fail to match a stored
    # one that was never canonicalised in the first place.
    where = canonical_where(where)

    from dfwb.data.adapt import adapt as adapt_input
    from dfwb.data.adapt import available_profiles
    from dfwb.data.dataset import ClipDataset

    if detector is None:
        detector = resolve_detector(detector_uri, seed=seed)
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
        spec,
        chosen_profile,
        allow_mismatch=allow_input_mismatch,
        candidates=candidates,
        shipped=shipped_profiles,
    )
    if adaptation.mismatch:
        _log.warning("%s: input mismatch allowed: %s", loaded_protocol.ref, adaptation.reason)

    # Everything the cache key needs is known now; check before doing any of the actual scoring
    # work (joining the split with the store, building a dataset, running the detector).
    out_root = Path(out) if out is not None else require_root("runs", roots) / _SCORES_SUBDIR
    profile_sha256 = chosen_profile.sha256()
    resolved_precision = precision or _NO_AUTOCAST
    store_sha256 = store_index_sha256(
        work_root, loaded_protocol.dataset, chosen_profile.profile_id()
    )
    key_parts: dict[str, Any] = {
        "scheme_sha256": loaded_protocol.sha256,
        "split": split,
        "where": where,
        "profile_sha256": profile_sha256,
        "aggregate_mode": aggregate,
        "clips_per_video": clips_per_video,
        "labels": labels,
        "precision": resolved_precision,
        "store_index_sha256": store_sha256,
        "pack_version": loaded_protocol.pack_version,
    }
    key = cache_key(
        detector=detector, identity=identity, effective_seed=effective_seed, **key_parts
    )
    target = score_path(
        out_root,
        detector_name=detector.meta.name,
        protocol_ref=loaded_protocol.ref,
        split=split,
        key=key,
    )
    if not force and identity.cacheable and target.is_file():
        hit = look_up(target, identity=identity, **key_parts)
        if hit is not None:
            cached_frames_path = frames_path_for(hit.csv_path)
            if not frames:
                return ScoreResult(hit.csv_path, hit.meta_path, hit.coverage, True)
            if cached_frames_path.is_file():
                return ScoreResult(
                    hit.csv_path, hit.meta_path, hit.coverage, True, cached_frames_path
                )
            _log.info(
                "%s: cached, but --frames was asked for and %s does not exist; recomputing",
                target,
                cached_frames_path,
            )

    index = VideoIndex.build(
        [SourceSpec(protocol, split, where)],
        profile=chosen_profile.profile_id(),
        labels=labels,
        work_root=work_root,
    )

    clip_spec = _clip_spec(spec, clips_per_video)
    dataset = ClipDataset(
        index,
        clip_spec,
        train=False,
        transform=None,
        adapt_chain=adaptation.chain,
        seed=seed,
        expected_frame_size=chosen_profile.crop.size,
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
    if not rows:
        _log.warning(
            "%s: split %r matches no videos (where=%r); writing an empty score file",
            loaded_protocol.ref,
            split,
            where,
        )

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
        precision=resolved_precision,
        store_index_sha256=store_sha256,
        command=command,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = write_scores(target, rows, meta)

    # A frames file's mere presence must always be trustworthy on its own (no need to also check
    # the CSV it sits next to): write a fresh one when there is anything to write, otherwise clear
    # out whatever (now stale) one a previous, --frames run left at this same path.
    frames_target = frames_path_for(csv_path)
    frames_result_path: Path | None = None
    if frames and frame_records:
        frames_result_path = write_frames(frames_target, frame_records)
    else:
        frames_target.unlink(missing_ok=True)

    return ScoreResult(
        csv_path, meta_path, dict(meta.coverage.model_dump()), False, frames_result_path
    )
