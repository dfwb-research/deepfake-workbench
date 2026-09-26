"""A tiny fixture pack, processed store(s) and fake detector for ``dfwb.score`` tests.

``scoretoy``: 4 real (``REAL/r00..r03``) and 4 fake (``FAKE/f00..f03``) videos, all in the
``test`` split of the ``official`` scheme. The fake ``fake:`` detector source
(:func:`load_fake`) never touches torch weights: it scores a clip by the mean of its (already
adapted) pixel values, which are always in ``[0, 1]``, and can be told to raise for a chosen set
of keys, so a batch's failure is deterministic and reproducible.

Only the ``dfwb.plugins`` entry-point group is intercepted (to add the pack and the ``fake``
detector source), so the real ``dfwb.builtins`` registrations (the ``run`` source, metrics, ...)
stay available.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.unit.data.conftest import processed_record, write_store_frames, write_store_index
from tests.unit.protocols.conftest import make_pack

from dfwb.core import plugins
from dfwb.core.detector import DETECTOR_CONTRACT_VERSION, DetectorMeta, DetectorOutput, InputSpec
from dfwb.core.records import (
    DatasetCard,
    LabelVocab,
    SchemeCard,
    SplitRow,
    VideoRecord,
    write_jsonl,
    write_split_tsv,
)
from dfwb.core.records.local import (
    BackendSpec,
    CropSpec,
    DecodeSpec,
    ExtrasSpec,
    ProcessingProfile,
    SamplingSpec,
    TrackSpec,
)
from dfwb.core.records.protocol import LabelMappingSpec, LicenseInfo

__all__ = [
    "DATASET",
    "PACK",
    "PROTOCOL",
    "SUITE",
    "SUITE_ANY_OF",
    "FakeDetector",
    "install_scoretoy_pack",
    "load_fake",
    "toy_profile",
    "write_toy_store",
]

DATASET = "scoretoy"
PACK = "scoretoy-pack"
PROTOCOL = f"{PACK}:{DATASET}/official"
SUITE = "scoretoy"
SUITE_ANY_OF = "scoretoy-any-of"
N_VIDEOS = 4  # per class

# Two entries, each a single video (by identity) of PROTOCOL's test split, in different groups --
# enough to exercise "one C5 file per entry" without scoring the whole fixture pack twice over.
_SUITE_YAML = f"""\
name: {SUITE}
entries:
  - {{protocol: {PROTOCOL}, split: test, where: {{identity: r00}}, group: real}}
  - {{protocol: {PROTOCOL}, split: test, where: {{identity: f00}}, group: fake}}
"""

# Entries whose ``where`` holds a list, written the way a real pack's suite writes one: the list
# is in the order a person typed it, not the sorted order a score file's meta stores it in, and
# the protocol is the plain ``<dataset>/<scheme>`` a score file's meta records. The second entry
# has the shape of an in-domain-per-method entry (``method: [original, <fake method>]``): every
# real video plus one manipulation's fakes.
_SUITE_ANY_OF_YAML = f"""\
name: {SUITE_ANY_OF}
entries:
  - {{"protocol": "{DATASET}/official", "split": "test",
     "where": {{"identity": ["r00", "f00"]}}, "group": "pair"}}
  - {{"protocol": "{DATASET}/official", "split": "test",
     "where": {{"method": ["swap", "original"]}}, "group": "in-domain"}}
aggregates:
  - {{"group": "pair", "metric": "auc", "how": "mean"}}
  - {{"group": "in-domain", "metric": "auc", "how": "mean"}}
"""


def _dump(model: Any) -> str:
    import yaml

    return yaml.safe_dump(model.model_dump(mode="json", by_alias=True), sort_keys=False)


def _keys() -> list[tuple[str, int]]:
    return [(f"REAL/r{i:02d}", 0) for i in range(N_VIDEOS)] + [
        (f"FAKE/f{i:02d}", 1) for i in range(N_VIDEOS)
    ]


def write_scoretoy_dataset(dataset_dir: Path, dataset_id: str) -> None:
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "splits").mkdir()
    videos: list[VideoRecord] = []
    rows: list[SplitRow] = []
    for key, label in _keys():
        videos.append(
            VideoRecord(
                key,
                None,
                "SCORETOY-FAKE" if label else "SCORETOY-REAL",
                "swap" if label else "original",
                identity=key.rsplit("/", 1)[1],
            )
        )
        rows.append(SplitRow(key, None, "test"))
    write_jsonl(dataset_dir / "videos.jsonl.gz", videos)
    sha256 = write_split_tsv(dataset_dir / "splits" / "official.tsv.gz", rows)
    card = DatasetCard(
        id=dataset_id,
        name="Score Toy",
        release="1",
        license=LicenseInfo(summary="Synthetic fixture pack for tests"),
        access="tests only",
        modalities=["video"],
        key_rule="fixture",
        schemes={"official": SchemeCard(kind="official", source="fixture", sha256=sha256)},
        default_scheme="official",
    )
    (dataset_dir / "dataset.yaml").write_text(_dump(card))
    vocab = {"SCORETOY-REAL": {"binary": 0}, "SCORETOY-FAKE": {"binary": 1}}
    mappings = {
        "binary": LabelMappingSpec(from_="binary"),
        "binary-exclude-fake": LabelMappingSpec(
            from_="binary", override={"SCORETOY-FAKE": "exclude"}
        ),
    }
    labels = LabelVocab(vocab=vocab, mappings=mappings)
    (dataset_dir / "labels.yaml").write_text(_dump(labels))


class _FakeEntryPoint:
    def __init__(self, name: str, register: Callable[[Any], None]) -> None:
        self.name = name
        self.value = f"fake_module_{name}:register"
        self.dist = SimpleNamespace(name="fixture-packs", version="0.0.0")
        self._register = register

    def load(self) -> Callable[[Any], None]:
        return self._register


def install_scoretoy_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install ``scoretoy``, the ``fake`` detector source, and the ``scoretoy`` eval suite (two
    single-video entries of ``PROTOCOL``'s test split); returns its dataset directory."""
    root = make_pack(tmp_path, PACK, {DATASET: {}}, builders={DATASET: write_scoretoy_dataset})
    suites_dir = root.parent / "suites"
    suites_dir.mkdir()
    (suites_dir / "scoretoy.yaml").write_text(_SUITE_YAML)
    (suites_dir / "scoretoy-any-of.yaml").write_text(_SUITE_ANY_OF_YAML)
    monkeypatch.syspath_prepend(str(root.parent.parent))

    def _register(api: Any) -> None:
        api.protocol_packs.add(
            PACK, target=f"{root.parent.name}:{root.name}", summary="fixture pack"
        )
        api.detector_sources.add(
            "fake", target="tests.unit.score._toy:load_fake", summary="fake in-memory detector"
        )
        api.eval_suites.add(
            SUITE, target=f"{root.parent.name}:suites/scoretoy.yaml", summary="fixture suite"
        )
        api.eval_suites.add(
            SUITE_ANY_OF,
            target=f"{root.parent.name}:suites/scoretoy-any-of.yaml",
            summary="fixture suite with list-valued where filters",
        )

    real_entry_points = plugins._entry_points

    def _entry_points(group: str) -> list[Any]:
        if group == plugins.ENTRY_POINT_GROUP:
            return [*real_entry_points(group), _FakeEntryPoint(PACK, _register)]
        return real_entry_points(group)

    monkeypatch.setattr(plugins, "_entry_points", _entry_points)
    return root / DATASET


def toy_profile(
    profile_id: str, *, backend: str = "insightface", scale: float = 1.3, size: int = 32
) -> ProcessingProfile:
    return ProcessingProfile(
        id=profile_id,
        backend=BackendSpec(name=backend),
        track=TrackSpec(iou=0.5, strategy="greedy"),
        crop=CropSpec(scale=scale, size=size, square=True, align="none"),
        sampling=SamplingSpec(mode="uniform", frames=4),
        decode=DecodeSpec(library="opencv", color="rgb"),
        extras=ExtrasSpec(landmarks=False, mesh=False, masks=False),
    )


def write_toy_store(
    work_root: Path, profile: ProcessingProfile, *, skip: Iterable[str] = (), n_frames: int = 4
) -> Path:
    """Write a processed store for ``profile`` under ``work_root``: every ``scoretoy`` video
    except ``skip`` (left unprocessed, so it becomes a ``missing`` row). Returns the store dir."""
    store_dir = work_root / DATASET / "processed" / profile.profile_id()
    store_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile.model_dump(mode="json"),
        "sha256": profile.sha256(),
        "profile_id": profile.profile_id(),
        "backend": {"name": profile.backend.name, "version": None, "license": None, "meta": {}},
    }
    (store_dir / "profile.json").write_text(json.dumps(payload), encoding="utf-8")
    skipped = set(skip)
    records = []
    for key, _ in _keys():
        if key in skipped:
            continue
        record = processed_record(key, n_frames=n_frames)
        records.append(record)
        write_store_frames(store_dir, record, size=profile.crop.size)
    write_store_index(store_dir, records)
    return store_dir


#: ``{spy_id: predict() call count}`` and ``{spy_id: [torch.is_inference_mode_enabled(), ...]}``,
#: so a test can watch a ``fake:spy=<id>`` detector's ``predict()`` calls across two separate
#: ``score()`` calls (each resolves a fresh ``FakeDetector`` instance, so a plain instance
#: attribute could not be read back afterwards). Tests own their ``spy_id`` (a fresh one each,
#: e.g. ``str(uuid.uuid4())``) rather than clearing these dicts.
SPY_CALLS: dict[str, int] = {}
SPY_INFERENCE_MODE: dict[str, list[bool]] = {}


class FakeDetector:
    """An in-memory C4 detector: scores a clip by its mean pixel value (already in ``[0, 1]``
    once adapted), and can be told to raise for a chosen set of video keys, or to return a bad
    output (``bad_output``: ``"nan"``, ``"length"`` or ``"shape"``) to exercise the harness's own
    output validation.

    ``scripted``, when given, replaces the pixel-mean score with a fixed, known-in-advance score
    per clip: clip ``i`` of every video gets ``scripted[i % len(scripted)]``, read off
    ``batch.clip_index`` (eval-mode clips are sampled deterministically, so this is exactly the
    same clip every time) -- the toy store's frames are all identical, so a hand-computed
    aggregation fixture needs scores that do not depend on pixel content at all.

    Asserts ``batch.labels is None`` on every call: contract C4 gives labels to training and
    validation batches only, never a scoring one, so the harness must clear them first. Also
    records, in :attr:`saw_inference_mode`, whether ``torch.is_inference_mode_enabled()`` was true
    on each call.
    """

    def __init__(
        self,
        spec: InputSpec,
        *,
        raise_for: frozenset[str] = frozenset(),
        bad_output: str | None = None,
        spy_id: str | None = None,
        scripted: tuple[float, ...] | None = None,
    ) -> None:
        self.meta = DetectorMeta(
            name="fake-detector",
            version="0",
            contract_version=DETECTOR_CONTRACT_VERSION,
            input=spec,
            license="MIT",
            weights_license=None,
            citation=None,
            source="fake:test",
        )
        self._raise_for = raise_for
        self._bad_output = bad_output
        self._spy_id = spy_id
        self._scripted = scripted
        self.saw_inference_mode: list[bool] = []

    def to(self, device: Any) -> FakeDetector:
        return self

    def predict(self, batch: Any) -> DetectorOutput:
        import torch

        assert batch.labels is None, "predict() must never see labels while scoring (C4)"
        self.saw_inference_mode.append(torch.is_inference_mode_enabled())
        if self._spy_id is not None:
            SPY_CALLS[self._spy_id] = SPY_CALLS.get(self._spy_id, 0) + 1
            SPY_INFERENCE_MODE.setdefault(self._spy_id, []).append(
                torch.is_inference_mode_enabled()
            )
        if self._raise_for.intersection(batch.keys):
            raise RuntimeError("fake detector: configured to fail on this batch")
        n = len(batch.keys)
        if self._scripted is not None:
            table = self._scripted
            values = [table[int(i) % len(table)] for i in batch.clip_index.tolist()]
            return DetectorOutput(score=torch.tensor(values, dtype=torch.float32))
        mean = batch.clips.mean(dim=tuple(range(1, batch.clips.ndim))).clamp(0.0, 1.0)
        if self._bad_output == "nan":
            score: Any = torch.full((n,), float("nan"))
        elif self._bad_output == "length":
            score = torch.zeros(max(n - 1, 0))
        elif self._bad_output == "shape":
            score = torch.zeros(n, 2)
        elif self._bad_output == "nontensor":
            score = mean.tolist()  # a plain list, not a Tensor
        elif self._bad_output == "squeeze":
            score = mean.unsqueeze(-1)  # [B, 1]: a harmless shape slip the harness squeezes
        elif self._bad_output == "range":
            score = mean + 1.5  # pushes every value past 1.0
        else:
            score = mean
        return DetectorOutput(score=score)


def load_fake(ref: str) -> FakeDetector:
    """The ``fake:`` detector source: ``fake:key=value&key=value...`` configures the returned
    :class:`FakeDetector` -- ``crop`` (default ``face``), ``scale`` (default ``1.3``), ``size``
    (default ``32``), ``frames`` (``InputSpec.frames``, default ``1``), ``preferred``
    (``InputSpec.preferred_profile``), ``raise`` (a comma-separated list of video keys
    :meth:`FakeDetector.predict` raises for), ``bad`` (see ``bad_output`` above), ``spy`` (see
    :data:`SPY_CALLS` above) and ``scripted`` (a comma-separated list of floats, see
    :attr:`FakeDetector._scripted` above)."""
    options: dict[str, str] = {}
    for part in ref.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        options[key] = value
    crop = options.get("crop", "face")
    scale = float(options.get("scale", "1.3"))
    size = int(options.get("size", "32"))
    frames = int(options.get("frames", "1"))
    preferred = options.get("preferred")
    raise_for = frozenset(options["raise"].split(",")) if options.get("raise") else frozenset()
    scripted = (
        tuple(float(v) for v in options["scripted"].split(",")) if options.get("scripted") else None
    )
    spec = InputSpec(
        crop=crop,  # type: ignore[arg-type]  # test-only: trusted fixture input
        crop_scale=scale,
        size=(size, size),
        frames=frames,
        preferred_profile=preferred,
    )
    return FakeDetector(
        spec,
        raise_for=raise_for,
        bad_output=options.get("bad"),
        spy_id=options.get("spy"),
        scripted=scripted,
    )


def toy_run_profile() -> ProcessingProfile:
    """A processing profile compatible with ``tiny-cnn``'s native input (face, scale 1.3, 64px),
    for the ``run:`` end-to-end tests (which score a real, trained-shape ``AssembledDetector``,
    not the pixel-mean :class:`FakeDetector`)."""
    return toy_profile("toy-run-face", scale=1.3, size=64)


def write_toy_run(
    runs_root: Path,
    *,
    name: str,
    seed: int | None,
    fingerprint: str,
    tags: Iterable[str] = ("best",),
) -> Path:
    """Build a real run directory by hand, the same way ``tests/unit/models/test_source.py``
    does: ``<runs_root>/<name>/<stamp>-s<seed or 0>/checkpoints/<tag>/`` for each of ``tags``,
    each holding its own freshly (and separately) initialised ``tiny-cnn`` detector -- so two
    tags' ``model.safetensors`` never hash the same, exactly like two real checkpoints -- sharing
    ``source=f"run:{fingerprint}"``. An ``env.json`` records ``seed`` (omitted when ``seed`` is
    ``None``, so :func:`dfwb.models.source.load_run`'s ``training_seed`` falls back to ``None``).
    Returns the run directory.
    """
    from dfwb.core.config.schema import ComponentSpec, ModelSection
    from dfwb.models import checkpoint as checkpoint_module
    from dfwb.models.detector import build_detector

    model_cfg = ModelSection(
        backbone=ComponentSpec(name="tiny-cnn"),
        temporal_pool=ComponentSpec(name="mean"),
        head=ComponentSpec(name="linear"),
    )
    run_dir = runs_root / name / f"20260101-000000-s{seed if seed is not None else 0}"
    for tag in tags:
        detector = build_detector(model_cfg, source=f"run:{fingerprint}")
        checkpoint_module.save(run_dir / "checkpoints" / tag, detector, model_cfg)
    if seed is not None:
        (run_dir / "env.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    return run_dir
