# Detectors

A **detector** is anything that can score clips: a model `dfwb train` just produced, a run trained
earlier and reloaded, or (later) a pretrained adapter. Every one of them implements the same small
contract, `dfwb.core.detector.Detector`, so any code that scores clips — training's own validation
loop, and future evaluation code — works the same way whatever detector it is holding.

## The `Detector` contract

```python
class Detector(Protocol):
    meta: DetectorMeta

    def to(self, device: torch.device) -> "Detector": ...

    def predict(self, batch: ClipBatch) -> DetectorOutput:
        """Score a batch; called under torch.inference_mode()."""
```

`to` moves the detector to a device and returns it (`nn.Module.to` already has this shape, so
`AssembledDetector`, the detector `dfwb train` builds, needs no extra code for it).
`predict` takes a batch already matched to the detector's own `InputSpec` and returns scores.
`dfwb.core.detector` imports without torch — `Tensor` and `torch.device` only appear under
`TYPE_CHECKING` — so importing it costs nothing on a machine without the `train` extra.

### `InputSpec`

What a detector consumes: crop kind and scale, frame size, how many frames per clip, colour order,
value range, and normalisation.

```python
@dataclass(frozen=True)
class InputSpec:
    modality: Literal["frames", "audio", "audiovisual"] = "frames"
    crop: Literal["face", "full-frame"] = "face"
    crop_scale: float | None = 1.3
    size: tuple[int, int] = (224, 224)
    frames: int = 1
    sampling: Literal["uniform", "consecutive", "any"] = "any"
    color: Literal["rgb", "bgr"] = "rgb"
    value_range: tuple[float, float] = (0.0, 1.0)
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    preferred_profile: str | None = None
```

A trained backbone sets its own `InputSpec` (its `native_input`, e.g. what a `timm` model's
pretrained weights expect); the harness — not the detector — turns whatever a processed store
holds into exactly that, or refuses. See [input adaptation](#input-adaptation) below.

### `DetectorMeta`

Identity, licensing and provenance, carried alongside every detector:

```python
@dataclass(frozen=True)
class DetectorMeta:
    name: str
    version: str
    contract_version: tuple[int, int]
    input: InputSpec
    license: str  # SPDX identifier for the code
    weights_license: str | None  # SPDX or LicenseRef-*
    citation: str | None  # BibTeX
    source: str | None  # e.g. "run:<fingerprint>"
    training_data: tuple[str, ...] = ()  # protocol refs the weights were trained on
```

For a detector `dfwb train` assembles, `name` is `<backbone>-<pool>-<head>`, `version` is the
installed `dfwb` version, `license` comes from the `deepfake-workbench` distribution's own
metadata, and `weights_license`/`citation` are left unset (nothing to declare for a run trained
locally). `source` is filled in once the detector is loaded back from a saved run (see below).

### `ClipBatch` and `DetectorOutput`

```python
@dataclass
class ClipBatch:
    clips: Tensor  # [B, T, C, H, W], float, already matched to InputSpec
    keys: list[str]
    dataset_ids: list[str]
    compressions: list[str | None]
    clip_index: Tensor  # [B] which clip of the video this is
    frame_indices: Tensor  # [B, T] source frame numbers
    audio: Tensor | None = None
    labels: Tensor | None = None  # [B], training and validation batches only
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorOutput:
    score: Tensor  # [B] P(fake) in [0, 1] -- higher = more fake, always
    logit: Tensor | None = None  # [B]
    frame_scores: Tensor | None = None  # [B, T]
    features: Tensor | None = None  # [B, D]
```

`extras` is plugin-owned, per-sample data the framework never interprets — keys are namespaced
`"<provider>/<name>"` so two plugins never collide.

**Score polarity is fixed:** `score` is always P(fake), higher meaning more fake. There is no
detector-specific flag to check; a detector whose underlying model reports P(real) has to flip it
itself before returning `DetectorOutput`.

## `AssembledDetector`

`dfwb.models.detector.AssembledDetector` is the concrete `Detector` that `dfwb train` builds:
`backbone -> (stem ->) (temporal pool ->) head`. It has two entry points, not one:

- `forward(batch)` — the training path. Returns logits (`DetectorOutput.logit` set, `score` its
  sigmoid); losses read `logit`, not `score`.
- `predict(batch)` — the `Detector` contract's inference path. Switches to `eval()` for the
  duration (restoring whatever mode the module was in before, even on error), runs under
  `torch.inference_mode()`, and also fills `frame_scores` when the backbone is an image backbone
  (a `kind="video"` backbone never exposes individual frame features, so `frame_scores` stays
  `None`).

`predict()` only supports binary heads (`num_classes == 1`); calling it on a multi-class head
raises `ContractError` — train that kind of head through `forward()` instead.

## Loading a trained run back: the `run:` detector source

Training writes a self-describing run directory; the `run:` detector source rebuilds a `Detector`
from one, registered under the `detector_sources` registry as `run`:

```
run:<dir>[#best|#last]
```

`<dir>` may be a run directory itself (holding `checkpoints/<tag>/`), a `latest` symlink, or a
run-name directory with a `latest` symlink inside it; `#best` is the default tag. For example,
after `dfwb train -c my-config.yaml`, `run:runs/my-experiment/latest#best` and
`run:runs/my-experiment/latest#last` both resolve, as does `run:runs/my-experiment` (its own
`latest` symlink) or the exact timestamped directory. A dangling `latest` symlink, or a tag with no
matching checkpoint, raises `ConfigError` naming what was tried.

Resolving the reference rebuilds the detector purely from registries and JSON — `dfwb.models.source`
never imports `dfwb.train`, so loading a run needs only the `train` extra's runtime pieces
(torch, timm/transformers as applicable), not Lightning.

### What a run directory holds

```
runs/<run.name>/<YYYYmmdd-HHMMSS>-s<seed>/
  config.resolved.yaml          # the resolved config
  fingerprint.txt                # the config's fingerprint (same for every run of the experiment)
  env.json                       # seed, versions, device, git state, command, plugin providers
  data.json                      # per source: protocol, pack version, split hash, profile, counts
  checkpoints/best/model.safetensors
  checkpoints/best/detector.json
  checkpoints/last/model.safetensors
  checkpoints/last/detector.json
  logs/                          # metrics.csv, TensorBoard, heartbeat.json
  scores/val/<source>.scores.{csv,meta.json}
  report.md
  metrics.json
runs/<run.name>/latest -> <newest run>       # relative symlink
```

`checkpoints/<tag>/` is the only place weights live, and it is always two files:

- **`model.safetensors`** — weights only, via `safetensors.torch.save_file`/`load_file`.
- **`detector.json`** — `DetectorMeta` (as a plain dict), the resolved `model:` config, and, for
  every component the config uses (`backbones`, `temporal_pools`, `heads`, and `layers` when a
  stem is set), which registry key built it, which provider registered that key, and the
  provider's installed version at save time.

**No pickled modules, ever.** Loading a checkpoint never calls `torch.load`; it reads
`model.safetensors` (a data-only tensor format with no code execution) and `detector.json` (plain
JSON), then rebuilds the module from scratch through the same registries the original config used.
If a recorded provider is not installed, or is installed at an incompatible major version,
`InstallationError` names it (registry, key, provider and version) rather than failing on an
`AttributeError` deep inside a class that no longer exists. Resuming an interrupted run (`resume/`
inside the run directory, removed once the run finishes) follows the same rule: weights and
optimiser state are safetensors, everything else — epoch counters, RNG state, callback state — is
JSON.

## Input adaptation

A detector's `InputSpec` and a processed store's `ProcessingProfile` rarely match exactly (a face
model trained at scale 1.3 against a store cropped at scale 1.5, say). `dfwb.data.adapt(spec,
profile, allow_mismatch=False)` builds the transform chain that closes that gap: a derived centre
crop (only when `spec` asks for a narrower crop than the store kept), a resize to `spec.size`, a
colour-order flip, a value-range remap, and finally mean/std normalisation — in that order, and
only the steps actually needed.

Two things cannot be derived honestly, and are refused with `ContractError` unless
`allow_mismatch=True`:

- **a crop-kind mismatch** — `spec.crop="face"` against a `full-frame` store (backend `center`), or
  the reverse;
- **`spec.crop_scale` wider than the store's own** — the store never kept the extra margin a wider
  crop would need.

The error names a compatible profile when one of the caller's candidate profiles would actually
serve the spec (same crop kind, at least the requested scale), and otherwise states the
requirement plainly and points at `dfwb preprocess profiles`. With `allow_mismatch=True`, both
refusals instead proceed: `AdaptResult.mismatch` is `True` and `AdaptResult.reason` records which
mismatch was allowed, so the caller (training, and later scoring) can note it rather than silently
pretend the data lines up.

Normalisation always lives in this adaptation step, never hard-coded on a backbone or baked into a
transform pipeline — a backbone declares the mean/std its pretrained weights expect through its own
`native_input`, and that is what ends up applied.

## See also

- `docs/concepts/processing-profiles.md` — what a processed store's `ProcessingProfile` records,
  which `adapt()` reads.
- `docs/guides/write-a-plugin.md` — registering a `backbones`, `temporal_pools`, `heads` or `layers`
  (stem) component that assembles into a `Detector`.
