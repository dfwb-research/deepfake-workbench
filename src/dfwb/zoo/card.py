"""The adapter card: the one schema every zoo adapter declares itself with.

An adapter card is plain YAML, validated against :class:`AdapterCard` (pydantic, unknown keys
rejected): upstream provenance, licensing, how its code is obtained, its weight variants, the
input it expects, its score polarity, and the numbers it claims (``reported``) versus the numbers
DFWB has reproduced (``parity``). Nothing here needs torch: a card is data, read and validated the
same way whether or not the adapter it describes can currently be loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError, field_validator

from dfwb.core.detector import InputSpec
from dfwb.core.errors import ContractError, DFWBError, validation_messages
from dfwb.core.metric_spec import check_metric_spec

__all__ = [
    "AdapterCard",
    "AdapterInputSpec",
    "AdapterLicense",
    "InstallSpec",
    "ParityMetric",
    "ReportedMetric",
    "Upstream",
    "UpstreamPaper",
    "WeightSpec",
    "parse_card",
    "read_card",
]

# A name safe to use as a path component (cache dirs, module names): lower-kebab, the same
# convention the plugin registries use for their own keys -- so it can never contain a "/" or a
# ".." segment.
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class _Strict(BaseModel):
    """Shared strictness for every piece of the card: unknown keys are always a mistake."""

    model_config = ConfigDict(extra="forbid")


class UpstreamPaper(_Strict):
    """The paper an adapter's upstream code and weights came from."""

    title: str
    venue: str
    year: int
    doi: str | None = None


class Upstream(_Strict):
    """Where the adapter's upstream code lives, and at which exact commit."""

    repo: str
    commit: CommitSha
    paper: UpstreamPaper | None = None
    bibtex: str | None = None


class AdapterLicense(_Strict):
    """Licensing terms: the code's own licence, the weights' (if different), and whether using
    the weights needs an explicit, once-per-machine acknowledgement first."""

    code: str
    weights: str | None = None
    requires_ack: bool = False


class InstallSpec(_Strict):
    """How the adapter's own extra pulls in its upstream code (``code_strategy: pip``)."""

    extra: str | None = None
    pip: tuple[str, ...] = ()


class WeightSpec(_Strict):
    """One downloadable weight variant."""

    id: str
    url: str
    sha256: Sha256Hex
    bytes: int
    format: Literal["safetensors", "pytorch"]
    trained_on: tuple[str, ...] = ()
    redistribution: Literal["undecided", "allowed", "forbidden"] = "undecided"


class AdapterInputSpec(_Strict):
    """What the adapter consumes -- the card's own subset of contract C4's ``InputSpec``, which
    :meth:`to_input_spec` completes with that contract's remaining defaults."""

    crop: Literal["face", "full-frame"] = "face"
    crop_scale: float | None = 1.3
    size: tuple[int, int] = (224, 224)
    frames: int = 1
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    def to_input_spec(self) -> InputSpec:
        return InputSpec(
            crop=self.crop,
            crop_scale=self.crop_scale,
            size=self.size,
            frames=self.frames,
            mean=self.mean,
            std=self.std,
        )


# The contract's own adapter-card example names its metric ``video_auc``: a video-level AUC. Every
# row of a score file is one video, so that is exactly ``auc`` over a score file; the example's
# spelling is accepted, and read as the metric it names.
_METRIC_ALIASES = {"video_auc": "auc"}


def _known_metric(value: str) -> str:
    """``value`` as a metric spec the ``metrics`` registry knows (a documented alias resolved to
    its metric), checked the way ``dfwb eval --metrics`` checks one, with a did-you-mean."""
    metric = _METRIC_ALIASES.get(value, value)
    try:
        check_metric_spec(metric)
    except DFWBError as exc:  # the registry's own message, did-you-mean included
        raise ValueError(exc.message) from None
    return metric


class ReportedMetric(_Strict):
    """One number the upstream paper claims. ``metric`` is a metric spec ``dfwb eval`` knows
    (``auc``, ``eer``, ``tpr@fpr=0.01``, ...), checked when the card is read."""

    protocol: str
    split: str
    metric: str
    value: float
    source: str

    @field_validator("metric")
    @classmethod
    def _check_metric(cls, value: str) -> str:
        return _known_metric(value)


class ParityMetric(_Strict):
    """One number DFWB has reproduced, checked against :attr:`ReportedMetric.value`: how many
    videos it was computed over (``n``), what fraction of the split was scored (``coverage``),
    and which weight variant was measured (``weights``, for a card with any), so a reader can
    tell a number measured on the whole split from one measured on part of it."""

    protocol: str
    split: str
    metric: str
    value: float
    tolerance: float
    dfwb_version: str
    date: str
    n: int | None = None
    coverage: float | None = None
    weights: str | None = None

    @field_validator("metric")
    @classmethod
    def _check_metric(cls, value: str) -> str:
        return _known_metric(value)


class AdapterCard(_Strict):
    """The full adapter card: contract C4's schema, exactly, with unknown keys rejected."""

    name: Slug
    display_name: str
    contract_version: tuple[int, int]
    upstream: Upstream | None = None
    license: AdapterLicense
    code_strategy: Literal["pip", "vendored", "pinned-clone"]
    install: InstallSpec | None = None
    weights: tuple[WeightSpec, ...] = ()
    input: AdapterInputSpec
    polarity: Literal["fake-high", "real-high"] = "fake-high"
    reported: tuple[ReportedMetric, ...] = ()
    parity: tuple[ParityMetric, ...] = ()


def parse_card(text: str, *, source: str = "<card>") -> AdapterCard:
    """Parse and validate an adapter card's YAML text.

    Raises:
        ContractError: ``text`` is not valid YAML, or does not validate as :class:`AdapterCard`.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(f"{source}: invalid YAML ({exc})", hint="fix the YAML syntax") from None
    try:
        return AdapterCard.model_validate(data)
    except ValidationError as exc:
        raise ContractError(
            f"{source}: " + "; ".join(validation_messages(exc)),
            hint="not a valid adapter card",
        ) from None


def read_card(path: Path) -> AdapterCard:
    """Read and validate an adapter card from a local file.

    Raises:
        ContractError: ``path`` cannot be read, or its contents do not validate.
    """
    try:
        text = path.read_text("utf-8")
    except OSError as exc:
        raise ContractError(
            f"{path}: cannot read ({type(exc).__name__}: {exc})",
            hint="check that the adapter's card file exists",
        ) from None
    return parse_card(text, source=str(path))
