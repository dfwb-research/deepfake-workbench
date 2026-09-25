"""Input adaptation (contract C4): the deterministic chain that turns stored face clips into
exactly what a detector's :class:`~dfwb.core.detector.InputSpec` expects.

The chain is built once, from the spec and the store's :class:`~dfwb.core.records.local.
ProcessingProfile`, never from a clip itself: derived crop, resize, colour order, value range,
mean/std, in that order. Two things cannot be adapted honestly, only refused or (with
``allow_mismatch=True``) recorded and worked around: a face crop asked of a full-frame store (or
the reverse), and a crop scale wider than what the store actually kept. Everything else -- size,
colour order, value range, normalisation -- is always derivable from what a store has, so it never
raises.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError
from torch import Tensor

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ContractError
from dfwb.core.records.local import ProcessingProfile
from dfwb.data.dataset import ClipTransform
from dfwb.data.transforms import CenterCrop, Normalize, Resize

__all__ = ["AdaptResult", "adapt", "available_profiles"]

_log = logging.getLogger(__name__)

_PROFILE_FILE = "profile.json"


@dataclass(frozen=True)
class AdaptResult:
    """The built chain, plus what it had to do to get there.

    ``mismatch`` is ``True`` only when ``allow_mismatch`` let :func:`adapt` proceed past a
    refusal it would otherwise have raised; ``reason`` then says which one, in the same words the
    ``ContractError`` would have used.
    """

    chain: Callable[[Tensor], Tensor]
    derived_crop: bool
    mismatch: bool
    reason: str | None = None


def _crop_kind(profile: ProcessingProfile) -> Literal["face", "full-frame"]:
    """A profile is ``full-frame`` when its backend is ``center`` (no face detection at all),
    ``face`` otherwise."""
    return "full-frame" if profile.backend.name == "center" else "face"


def _requirement_text(spec: InputSpec) -> str:
    scale_part = f" at scale >= {spec.crop_scale}" if spec.crop_scale is not None else ""
    return f"a {spec.crop} crop{scale_part}, size {spec.size}"


def _find_compatible(
    spec: InputSpec, candidates: Sequence[ProcessingProfile]
) -> ProcessingProfile | None:
    """The best of ``candidates`` that could actually serve ``spec``: same crop kind, and (when
    ``spec.crop_scale`` is set) at least that scale. Ties broken by the smallest sufficient scale,
    then by profile id, so the choice is deterministic."""
    matches = [
        candidate
        for candidate in candidates
        if _crop_kind(candidate) == spec.crop
        and (spec.crop_scale is None or candidate.crop.scale >= spec.crop_scale)
    ]
    if not matches:
        return None
    return min(matches, key=lambda candidate: (candidate.crop.scale, candidate.profile_id()))


def _refuse(
    spec: InputSpec,
    profile: ProcessingProfile,
    problem: str,
    candidates: Sequence[ProcessingProfile],
) -> ContractError:
    compatible = _find_compatible(spec, candidates)
    if compatible is not None:
        message = (
            f"{problem}; profile {compatible.profile_id()!r} (id {compatible.id!r}) is compatible"
        )
        hint = f"process this data with profile {compatible.id!r}, or point at its store instead"
    else:
        message = f"{problem}; no available profile provides {_requirement_text(spec)}"
        hint = "run `dfwb preprocess profiles` to see what is available"
    return ContractError(message, hint=hint)


# Every step of a chain is a module-level class or function, never a closure: a DataLoader whose
# workers are started by spawn or forkserver (the default start method on Python 3.14) pickles
# the dataset, this chain included, into every worker, and a local function cannot be pickled.


class _Step:
    """A registered clip transform, called without a generator (the adaptation chain is never
    random), narrowed to the plain ``Tensor -> Tensor`` shape :class:`AdaptResult.chain` needs."""

    def __init__(self, transform: ClipTransform) -> None:
        self.transform = transform

    def __call__(self, clip: Tensor) -> Tensor:
        return self.transform(clip)


def _channel_flip(clip: Tensor) -> Tensor:
    """Reverses the channel axis: RGB -> BGR (and back), since stored frames are always RGB."""
    return clip.flip(dims=(1,))


class _ValueRange:
    """``x * (hi - lo) + lo``, mapping a clip already in ``[0, 1]`` into ``[lo, hi]``."""

    def __init__(self, lo: float, hi: float) -> None:
        self.lo = lo
        self.scale = hi - lo

    def __call__(self, clip: Tensor) -> Tensor:
        return clip * self.scale + self.lo


class _Chain:
    """Composes adaptation steps, applied in order to the same clip. No randomness anywhere in
    here -- unlike :mod:`dfwb.data.transforms`, nothing in this chain ever needs a generator."""

    def __init__(self, steps: Sequence[Callable[[Tensor], Tensor]]) -> None:
        self._steps = tuple(steps)

    def __call__(self, clip: Tensor) -> Tensor:
        for step in self._steps:
            clip = step(clip)
        return clip


def adapt(
    spec: InputSpec,
    profile: ProcessingProfile,
    *,
    allow_mismatch: bool = False,
    candidates: Iterable[ProcessingProfile] = (),
) -> AdaptResult:
    """Build the deterministic chain that turns a store's clips into ``spec``'s own shape.

    Stored frames are always a square crop, ``profile.crop.size`` pixels a side, in RGB, values in
    ``[0, 1]``. From there the chain does, in order: a derived centre crop (only when ``spec``
    asks for a narrower one than the store kept), a resize to ``spec.size`` (only when that
    differs from what the crop step leaves), a channel flip to BGR (only when ``spec.color ==
    "bgr"``), a value-range remap (only when ``spec.value_range`` is not already ``(0, 1)``), and
    finally mean/std normalisation (only when both are set) -- applied exactly once, last.

    Two things cannot be derived from what a store has, and are refused outright unless
    ``allow_mismatch=True``:

    - a crop kind mismatch (``spec.crop`` asks for ``face`` against a ``full-frame`` store, or the
      reverse) -- there is no honest crop to derive in either direction;
    - ``spec.crop_scale`` wider than ``profile.crop.scale`` -- the store never kept the extra
      margin a wider crop would need.

    A refusal's ``ContractError`` names the best of ``candidates`` that could actually serve
    ``spec`` (its :meth:`~dfwb.core.records.local.ProcessingProfile.profile_id`, plus its shipped
    ``id`` slug), or, when none of ``candidates`` is compatible, states the requirement itself
    (crop kind, minimum scale, size) and points at ``dfwb preprocess profiles``.

    With ``allow_mismatch=True``, both refusals instead proceed: ``AdaptResult.mismatch`` is
    ``True``, ``reason`` explains which one (the same wording the ``ContractError`` would have
    used), no derived crop is applied, and the chain resizes the store's full stored crop instead.

    Raises:
        ContractError: a crop-kind mismatch or an over-wide ``spec.crop_scale``, and
            ``allow_mismatch`` is ``False``.
    """
    candidate_list = list(candidates)
    store_kind = _crop_kind(profile)

    mismatch = False
    reason: str | None = None
    derived_crop = False
    steps: list[Callable[[Tensor], Tensor]] = []
    current_size = profile.crop.size

    if store_kind != spec.crop:
        problem = (
            f"detector needs {_requirement_text(spec)}, but profile {profile.profile_id()!r} "
            f"is a {store_kind} crop"
        )
        if not allow_mismatch:
            raise _refuse(spec, profile, problem, candidate_list)
        mismatch = True
        reason = problem
    elif spec.crop_scale is not None and spec.crop_scale != profile.crop.scale:
        if spec.crop_scale > profile.crop.scale:
            problem = (
                f"detector needs {_requirement_text(spec)}, but profile "
                f"{profile.profile_id()!r} only has scale {profile.crop.scale}"
            )
            if not allow_mismatch:
                raise _refuse(spec, profile, problem, candidate_list)
            mismatch = True
            reason = problem
        else:
            derived_px = round(profile.crop.size * spec.crop_scale / profile.crop.scale)
            steps.append(_Step(CenterCrop(size=derived_px)))
            derived_crop = True
            current_size = derived_px

    if spec.size != (current_size, current_size):
        steps.append(_Step(Resize(size=spec.size)))

    if spec.color == "bgr":
        steps.append(_channel_flip)

    lo, hi = spec.value_range
    if (lo, hi) != (0.0, 1.0):
        steps.append(_ValueRange(lo, hi))

    if spec.mean is not None and spec.std is not None:
        steps.append(_Step(Normalize(mean=spec.mean, std=spec.std)))

    return AdaptResult(
        chain=_Chain(steps), derived_crop=derived_crop, mismatch=mismatch, reason=reason
    )


def available_profiles(work_root: Path, dataset: str) -> list[ProcessingProfile]:
    """Every processing profile with a store under ``<work_root>/<dataset>/processed/``.

    Reads each ``processed/*/profile.json`` and validates its ``"profile"`` object with the core
    :class:`~dfwb.core.records.local.ProcessingProfile` model -- exactly the payload
    :meth:`dfwb.preprocess.face.store.Store.write_profile` writes, read here without depending on
    that module (``dfwb.data`` never imports ``dfwb.preprocess``). A file that cannot be read as
    one -- missing, not JSON, no ``"profile"`` key, or a profile that no longer validates -- is
    skipped with a warning rather than failing the whole listing, since one stale or half-written
    store should not hide every other one.
    """
    processed_dir = work_root / dataset / "processed"
    if not processed_dir.is_dir():
        return []

    profiles: list[ProcessingProfile] = []
    for store_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        path = store_dir / _PROFILE_FILE
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text("utf-8"))
            profile = ProcessingProfile.model_validate(payload["profile"])
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValidationError,
        ) as exc:
            _log.warning("%s: not a readable processing profile (%s); skipping", path, exc)
            continue
        profiles.append(profile)
    return profiles
