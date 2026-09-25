"""Runs any C4 :class:`~dfwb.core.detector.Detector` over a protocol split and writes a C5 score
file: resolve the detector, choose a processing profile, adapt stored clips to its input, score
every clip, aggregate clip -> video, and write ``<name>.scores.{csv,meta.json}``.

Every video the split names gets a row: ``ok`` (scored), ``missing`` (no usable processed clip) or
``error`` (the detector raised while scoring it). Nothing is silently dropped.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.hashing import fingerprint
from dfwb.core.paths import require_root, resolve_roots
from dfwb.core.records import ScoreRow, read_scores, write_scores
from dfwb.core.records.scores import meta_path_for
from dfwb.data.index import SourceSpec, VideoIndex
from dfwb.eval.aggregate import aggregate as aggregate_scores
from dfwb.protocols.protocol import Protocol
from dfwb.protocols.protocol import load as load_protocol
from dfwb.score.sources import resolve_detector
from dfwb.score.writer import assemble_meta

if TYPE_CHECKING:
    from dfwb.core.records.local import ProcessingProfile
    from dfwb.data.adapt import AdaptResult

__all__ = ["ScoreResult", "score"]

_log = logging.getLogger(__name__)

VideoKey = tuple[str, str, str | None]  # (dataset, key, compression)

_SLUG = re.compile(r"[^a-z0-9]+")
_SCORES_SUBDIR = "scores"


@dataclass(frozen=True)
class ScoreResult:
    """What one call to :func:`score` produced."""

    csv_path: Path
    meta_path: Path
    coverage: dict[str, int]
    cached: bool


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.strip().lower()).strip("-") or "x"


# --------------------------------------------------------------------------------- profile choice


def _crop_kind(profile: ProcessingProfile) -> str:
    return "full-frame" if profile.backend.name == "center" else "face"


def _profile_matches(profile: ProcessingProfile, token: str) -> bool:
    return token in (profile.id, profile.profile_id())


def _is_compatible(spec: InputSpec, profile: ProcessingProfile) -> bool:
    if _crop_kind(profile) != spec.crop:
        return False
    return spec.crop_scale is None or profile.crop.scale >= spec.crop_scale


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
    processed locally, else the only local profile compatible with ``spec``, else the only local
    profile at all (deferring the actual pass/refuse decision to :func:`~dfwb.data.adapt.adapt`),
    else an error listing every local candidate.

    Raises:
        ConfigError: ``requested`` is not a locally processed profile, no profile has been
            processed locally at all, or more than one candidate is left after every rule above.
    """
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
    compatible = [p for p in candidates if _is_compatible(spec, p)]
    if len(compatible) == 1:
        return compatible[0]
    if not compatible and len(candidates) == 1:
        return candidates[0]
    raise ConfigError(
        f"no single local processing profile can serve this detector's input for {dataset!r} "
        f"(available: {_candidate_text(candidates)})",
        hint="pass profile=<id> to choose one explicitly",
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
    of the two.
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
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    autocast_dtype = dtypes.get(precision) if precision else None

    clip_scores: dict[VideoKey, list[float]] = {}
    errored: dict[VideoKey, str] = {}
    for batch in loader:
        keys: list[VideoKey] = [
            (batch.dataset_ids[i], batch.keys[i], batch.compressions[i])
            for i in range(len(batch.keys))
        ]
        try:
            batch.clips = batch.clips.to(torch_device)
            with torch.inference_mode():
                if autocast_dtype is not None:
                    with torch.autocast(device_type=torch_device.type, dtype=autocast_dtype):
                        output = detector.predict(batch)
                else:
                    output = detector.predict(batch)
            scores = output.score.detach().float().cpu().tolist()
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


# -------------------------------------------------------------------------------------- output path


def _detector_fingerprint(detector: Any, checkpoint_sha256: str | None) -> dict[str, Any]:
    meta = detector.meta
    return {
        "name": meta.name,
        "version": meta.version,
        "source": meta.source,
        "contract_version": list(meta.contract_version),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _cache_key(
    *,
    detector: Any,
    checkpoint_sha256: str | None,
    scheme_sha256: str,
    split: str,
    where: Mapping[str, Any] | None,
    profile_sha256: str,
    aggregate_mode: str,
    clips_per_video: int,
    labels: str,
) -> str:
    payload = {
        "detector": _detector_fingerprint(detector, checkpoint_sha256),
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


def _cached_result(target: Path) -> ScoreResult | None:
    try:
        existing = read_scores(target)
    except (ContractError, OSError):
        return None
    if existing.meta is None:
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
    profile and adapts stored clips to the detector's input, scores every clip under
    ``torch.inference_mode()``, aggregates clip scores to one score per video, and writes a C5
    score file. Every video of the split gets a row: ``ok``, ``missing`` (no usable processed
    clip), or ``error`` (the detector raised while scoring it).

    Raises:
        ConfigError: no local processing profile can serve the detector's input (see
            :func:`_choose_profile`).
        ContractError: the detector's input cannot be adapted from the chosen profile and
            ``allow_input_mismatch`` is ``False`` (see :func:`~dfwb.data.adapt.adapt`).
    """
    from dfwb.data.adapt import adapt as adapt_input
    from dfwb.data.adapt import available_profiles
    from dfwb.data.dataset import ClipDataset

    detector = resolve_detector(detector_uri)
    spec: InputSpec = detector.meta.input
    checkpoint_sha256: str | None = None

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
        checkpoint_sha256=checkpoint_sha256,
        protocol=loaded_protocol,
        split=split,
        where=where,
        labels=labels,
        processing_profile=chosen_profile,
        adaptation=adaptation,
        aggregate_mode=aggregate,
        clips_per_video=clips_per_video,
        rows=rows,
        seed=seed,
    )

    out_root = Path(out) if out is not None else require_root("runs", roots) / _SCORES_SUBDIR
    key = _cache_key(
        detector=detector,
        checkpoint_sha256=checkpoint_sha256,
        scheme_sha256=loaded_protocol.sha256,
        split=split,
        where=where,
        profile_sha256=chosen_profile.sha256(),
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
        cached = _cached_result(target)
        if cached is not None:
            return cached

    target.parent.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = write_scores(target, rows, meta)
    return ScoreResult(csv_path, meta_path, dict(meta.coverage.model_dump()), False)
