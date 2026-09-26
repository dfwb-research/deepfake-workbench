# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A documentation site (MkDocs Material): the six user journeys (a five-minute toyfake walkthrough,
  the same shape with real datasets, evaluating score files from your own code with no `torch`
  install, adding a dataset, adding a detector, and adding a plugin), plus concept pages for the
  pipeline, protocols, processing profiles, detectors, score files, training and plugins.
- A generated reference, rebuilt from the code at every docs build rather than hand-maintained: the
  full `dfwb` CLI tree with its help and options, the experiment config schema, the C1-C5 contract
  JSON Schemas, the public API of every layer, and one page per dataset dfwb has a built-in
  inventory builder for.
- `examples/`: runnable scripts mirroring the toyfake quickstart, importing foreign score files,
  and scoring a detector of your own -- each exercised by its own test, in a temporary directory.

## [0.1.0b2] - 2026-09-26

### Added

- The `dfwb.score` layer and `dfwb score`: runs any C4 `Detector` — a trained run, a zoo adapter,
  or your own code — over a protocol split (or every entry of a registered suite, `--suite`) and
  writes a C5 score file, with a row for every video in the split (`ok`, `missing` or `error`,
  never dropped). The processing profile is chosen automatically (`--profile` overrides it: the
  detector's own `preferred_profile` if processed locally, else the only locally compatible one,
  else an error listing the candidates, including — for a caller that may import the face
  pipeline, such as the CLI — the shipped profiles that would serve the detector once data is
  processed with one of them). Clip scores are aggregated to one score per video
  (`--aggregate mean-prob|mean-logit|max|median`); `--frames` additionally writes a
  `<name>.frames.parquet` per-clip/per-frame dump (the `[eval]` extra: pyarrow), its frame scores
  checked like the clip scores. A batch whose output fails validation marks its videos `error` and
  scoring continues; a detector with an `eval()` method is put in inference mode first. The meta
  records the command line, with no absolute path in it. `--min-coverage` (default `0.99`, as
  `dfwb eval`) exits `3`, after every file is written and reported, when a file's `ok` fraction is
  lower; `--suite` resolves the detector once for every entry.
- **Caching.** An identical scoring request (the detector's exact identity, the protocol split,
  pack version and any `--where`, the processing profile and the processed store's own index, the
  aggregation, the label mapping and the precision) reuses a score file already sitting at its own
  output path instead of rescoring, unless `--force` is given; a cached file with any `error` row is
  always retried rather than reused, since a detector failure is usually transient. A store that
  has processed more videos since, or re-processed some, is scored again.
- Detector sources (registry `detector_sources`, resolved from a `<scheme>:<rest>` URI):
  `run:<dir>[#best|#last]` (a trained run, `dfwb.models`), `zoo:<name>[@<weights id>]` (a
  registered zoo adapter) and `py:<module>:<factory>` (your own code: the named factory returns a
  C4 `Detector`, recorded as `py:<module>:<factory>` when it sets no source of its own; the cache
  key folds in the sha256 of the module's own source file, so an edit to it is never served a stale
  cached file, and a source with no readable file at all is never cached and always recomputed). A
  detector source may set a few optional attributes beyond contract C4
  (`checkpoint_sha256`, `training_seed`, `fingerprint_extra`, `cacheable`) that sharpen a score
  file's meta and its cache key.
- The `dfwb.zoo` layer: a uniform, licence-aware way to run published third-party detectors. This
  release ships the machinery and two dummy adapters only — `zoo:chance` (constant 0.5) and
  `zoo:random` (seeded uniform scores, `AUC ≈ 0.5` on toyfake) — no real third-party adapters. The
  `AdapterCard` schema (pydantic, unknown keys rejected) records upstream provenance, licensing,
  weight variants (each pinned to a sha256), the detector's input spec, its score polarity, and its
  reported versus reproduced (parity) numbers. Weights are downloaded once, sha256-and-size
  verified, cached under `$DFWB_CACHE_ROOT/zoo/<name>/<sha256>/`, and re-verified on every use
  rather than trusted by their path alone; a licence needing acknowledgement
  (`license.requires_ack`) gates use until `dfwb zoo fetch --accept-license` records it once, per
  machine. Upstream model code is obtained by one of three strategies the card declares: `pip` (the
  adapter's own extra pins it), `vendored` (MIT/BSD/Apache-compatible code copied verbatim, with its
  licence, a notice and a per-file hash list) or `pinned-clone` (an incompatible-licence or large
  upstream, cloned at an exact commit into the cache and imported under a private module name,
  never added to `sys.path`, so it can never collide with anything else importable). `dfwb zoo`:
  `list`, `info`, `fetch`, `verify` (re-hashes cached weights and checks the code pin, downloading
  nothing), `parity` (scores an adapter's parity set against its card's reported numbers, for one
  weight variant, `--weights`; refuses with exit code `3` below `--min-coverage`, and records each
  number's `n`, coverage and weight variant) and `licenses`. A card's reported metrics are checked
  against the `metrics` registry when it is read; `video_auc` is read as `auc`.
- `dfwb eval`'s `--bootstrap 0` now reports every metric's point value with no confidence interval
  (`ci_lo`/`ci_hi` both `null`) instead of raising, for `dfwb eval` and `dfwb eval compare` alike; a
  suite (registry `eval_suites`) may also carry a free-text `description` of what it measures.
- `[eval]` now also pulls in pyarrow, for `dfwb score --frames`'s per-clip/per-frame dump.
- User docs: score files (`docs/concepts/score-files.md`: the C5 schema, statuses, the cache, and
  the optional detector attributes), cross-dataset evaluation (`docs/guides/cross-dataset-eval.md`:
  scoring and evaluating a suite, coverage policy, `compare` and DeLong), adding a detector
  (`docs/guides/add-a-detector.md`: the `py:` source and adapter cards) and reproducing a run
  (`docs/guides/reproduce-a-run.md`: from a run directory to a comparable score file); the README
  restyled in the organisation's shared format, with its own hero, and two added journeys: scoring
  and evaluating score files produced by your own code with no training or torch install, and
  scoring a detector of your own through the `py:` source with no adapter card.

### Changed

- `dfwb eval`: a metric undefined for one file (a two-class metric on a single-class file, say) is
  an `undefined` cell (in JSON, a `null` value with the reason) instead of aborting the whole run,
  and with `--suite` a group whose files all have the metric undefined gets an `undefined` row;
  the command exits `4` only when no requested metric is defined for any file. The seeds table
  also requires the label mapping, aggregation and processing profile to agree, and warns (keeping
  the first) when two files share a seed as well.
- `dfwb eval compare` reports the `ok` rows unique to each file (`only_a`, `only_b`), refuses two
  files that give a shared row different labels, and gives DeLong's exact answer at zero paired
  variance (equal AUCs: `z = 0`, `p = 1`; unequal: an infinite `z`, `p = 0`) or `null` with a
  reason when a class has fewer than two rows; `--json` is always valid JSON (an infinity is
  written as `"inf"`/`"-inf"`).
- `dfwb eval import` says when unmatched keys belong to another split of the pack.

### Fixed

- A score file whose `where` holds a list (repeated `--where key=v`) no longer crashes `dfwb
  eval`, and a suite entry with a list filter matches its file whatever the list's order.
- The command line recorded in a run's or a score file's metadata never holds an absolute path,
  wherever it appears inside an argument.

## [0.1.0b1] - 2026-09-26

### Added

- The `dfwb.data` layer: clip sampling (`ClipSpec`: frames, `uniform`/`consecutive`/`random-window`
  sampling; training clips drawn afresh every epoch, seeded by `(seed, epoch, index)` and never by
  the loader worker, with `uniform` drawing one frame inside each of `c·T` equal segments of the
  video, so `c·T` equal to the stored frame count trains on every frame; deterministic
  evenly-spaced eval windows that repeat a video's last frame and record `padded=true` when it is
  shorter than the clip), clip-consistent torchvision v2 transforms (`resize`, `center-crop`,
  `random-resized-crop`, `hflip`, `color-jitter`, `grayscale`, `gaussian-blur`, `gaussian-noise`,
  `jpeg` (encoded with Pillow), `normalize` — parameters drawn once per clip and applied to every
  frame), `MultiSource` and paired (`data.pairs: true`) datasets (pair ids unique across sources),
  label- and source-balanced (by each source's `weight`) and pair-grouped samplers, and
  `VideoIndex`: the join of a protocol split against a processed face store, which records every
  video's outcome, including one missing from the store (`not-processed`), rather than dropping it
  silently. Stored frames are decoded with Pillow, and every dataset, transform and adaptation
  step pickles, so loader workers start under `spawn` and `forkserver` (Python 3.14's default)
  as well as `fork`.
- `dfwb.data.adapt`: the deterministic chain (derived crop, resize, colour order, value range,
  mean/std normalisation) that turns a processed store's clips into exactly what a detector's
  `InputSpec` asks for. A crop-kind mismatch or a store narrower than the detector needs raises
  `ContractError` naming a compatible processed store, if there is one, and the built-in profiles
  that would serve the detector; `allow_input_mismatch: true` proceeds instead and records the
  gap. Normalisation lives here, never hard-coded on a backbone — the old fixed
  ImageNet-normalisation-on-GPU bug is not carried over.
- The `dfwb.models` layer: `Backbone`/`TemporalPool`/`Head` interfaces assembled into
  `AssembledDetector`, the concrete `Detector` (contract C4) that training builds. Built-in
  backbones `tiny-cnn` (CPU tests and toy runs), `timm` (any timm image model) and `hf-vision`
  (`[hf]`: CLIP, DINOv2, SigLIP 2 and others via `AutoModel`); temporal pools `mean`, `max`,
  `attention`; heads `linear` and `mlp`. Freeze modes `none`, `full`, `norm-only`, `partial`
  (by a backbone's own declared block groups, never a name pattern over parameters — the fix for
  the old partial-freeze bug that froze everything) and `lora` (`[peft]`, on the `timm` and
  `hf-vision` backbones). An optional `model.stem` hook puts any registered `layers` component (a
  forensic front-end such as an SRM filter bank, published separately) in front of a backbone,
  adapting its channel count back to 3 automatically when it isn't already (image backbones only;
  a stem in front of a video backbone is a config error). The head and the sigmoid always run in
  float32, so mixed-precision training never rounds scores.
- Checkpoints: `model.safetensors` (weights only) plus `detector.json` (`DetectorMeta`, the
  resolved `model:` config, and every component's registry key, provider and installed version) —
  no pickled modules anywhere, and loading a checkpoint never calls `torch.load`. A missing or
  version-incompatible provider (another major version, or below 1.0 another minor one) raises
  `InstallationError` naming it. The `run:<dir>[#best|#last]` detector source (registry
  `detector_sources`) rebuilds a checkpoint's detector purely from registries, JSON and its
  weights: loading never downloads anything (backbones are rebuilt with `pretrained` off, and
  `hf-vision` from the Hugging Face config saved with it), and the saved `DetectorMeta` comes back
  as saved. A trained detector records the clip regime it trained on (`meta.input.frames` and
  `sampling`) and its training protocols (`meta.training_data`, pinned
  `<pack>:<dataset>/<scheme>@<version>` references).
- The `dfwb.train` layer, wrapping Lightning 2: losses `bce`, `ce`, `focal` and
  `label-smoothing-bce` (registry `losses`, dispatched through the registry, fixing the old
  if-chain); optimisers `adamw`/`sgd`, with per-group `lr_scale`/`weight_decay` and geometric
  layer-wise LR decay (`layer_decay: γ` gives block `i` the multiplier `γ^(K−i)`) over a backbone's
  own declared parameter groups; schedules `constant`/`cosine`/`step`, each with linear warmup;
  and `train.precision`: `auto` (the default: `bf16-mixed` or `16-mixed` on a CUDA GPU, `32-true`
  elsewhere, recorded in `env.json`) or one of Lightning's `32-true`, `bf16-mixed` and
  `16-mixed`. A plugin's callbacks join a run through `train.callbacks`, their state saved with the
  resume state. Optimisers and schedules are validated against the already-assembled
  detector, not a plugin registry, since group names come from the detector itself.
- The `dfwb.eval` layer and `dfwb eval`, torch-free and in the base install: coverage-aware metrics
  (registry `metrics`: `auc`, `ap`, `eer`, `acc@thr`, `tpr@fpr=x`, `fpr@tpr=x`, `ece`, `brier`,
  `nll`, `aurc`) with a stratified bootstrap confidence interval per file (`--bootstrap N`,
  `--seed`), a `--missing` coverage policy (`exclude`/`as-real`/`as-fake`/`as-chance`) and
  `--min-coverage` (exit code `3` without raising), a breakdown by
  method/family/compression/label_key (`--by`), suites (registry `eval_suites`: named collections
  of protocol/split/where entries with aggregate rows, `--suite`) and, for several files that agree
  on everything but their seed, a per-seed mean ± sd table. `dfwb eval compare` pairs two or more
  C5 files on the intersection of their `ok` rows, with a paired-bootstrap delta and (for AUC,
  needing scipy) the DeLong test, Holm-corrected across more than two files. `dfwb eval calibrate`
  fits a post-hoc calibration (`temperature`, `platt` or `isotonic`) on one file and applies it to
  another, writing a new C5 file whose meta records the calibration's provenance. `dfwb eval
  import` turns a foreign score CSV into a C5 file against a protocol split (`--map
  key=...,score=...`), with suggestions for the common ways a key fails to line up. `--out DIR`
  writes `metrics.json` and a `report.{md,csv,tex}` (`--format`), plus ROC/DET/reliability/
  risk-coverage plots with the `[eval]` extra (matplotlib).
- Validation reuses the same `dfwb.eval` metric code as final evaluation, so there is no metric
  drift between the two: a single-class validation split reports its metric as not defined, with a
  warning, instead of a silent 0.5, and the checkpoint monitor falls back to `val/loss` (with a
  warning) once every validation source is undefined — several defined sources are averaged.
- Training callbacks: best/last safetensors checkpointing (`best` by the monitor, `best` mirrors
  `last` when there is nothing to monitor), a per-validation-source score dump
  (`scores/val/<source>.scores.{csv,meta.json}`, exactly the rows the logged metrics were computed
  from), early stopping, a non-finite-loss guard (stops after `nan_tolerance` bad steps in a row,
  naming the step), a heartbeat file, and Lightning's learning-rate monitor; loggers CSV (always,
  so a run's numbers never depend on an optional extra), TensorBoard (when installed) and Weights
  & Biases (`[wandb]`, only when listed in `train.loggers`).
- Self-describing run directories: `runs/<run.name>/<timestamp>-s<seed>/` holding the resolved
  config, its fingerprint, `env.json` (versions, device, git state, command, plugin providers),
  `data.json` (per source: protocol ref, pack version, split hash, profile id and hash, join
  counts), `checkpoints/{best,last}/`, `logs/`, the last epoch's validation score dumps and a
  final `report.md`/`metrics.json` (the report says when the monitor fell back and which metrics
  were undefined); `runs/<run.name>/latest` is a relative symlink to the newest run.
- `dfwb train`: trains a config, one run per seed. Every problem the config alone can show —
  including a typo in a nested component parameter, an `eval.metrics` spec, `eval.aggregate` or
  `train.precision` — is reported with its exact dotted path and a did-you-mean before any data
  loads and before a run directory is created. `data.test` is not run by `dfwb train` in this
  version, and it says so. `dfwb train --resume <run dir>` continues an interrupted run from its
  safetensors weights and optimiser state and its JSON-encoded epoch/step counters, schedule,
  callback and RNG state (never a pickle), giving the same final metrics as an uninterrupted run
  (checked on a tiny, deterministic CPU model); it checks for the Lightning loop internals it
  sets when fitting starts, naming the tested Lightning series if one is missing.
- `dfwb runs`: `list` (every run under the runs root, or `--root DIR`, the newest of each name
  marked) and `show`
  (one run's full detail — data sources, validation metrics, environment, and the config
  fingerprint that identifies the experiment); both read plain files and JSON, needing no torch
  installed.
- Templates: `binary-frame.yaml` (a filled-in starter for face-crop binary training) and
  `toy-cpu.yaml` (`tiny-cnn` on the toyfake tree, two epochs, CPU, batch 16, no workers — used by
  the README quickstart and by the toyfake training tests).
- The torch-free guarantee holds through this release: `core`, `protocols`, `eval` and
  `preprocess` still import and run with torch blocked; only `train`, `hf`, `peft` and `zoo` pull
  torch in, each its own extra, and `pip install deepfake-workbench` alone never does.
- User docs: the `Detector` contract, what a run directory holds, the `run:` detector source and
  input adaptation (`docs/concepts/detectors.md`); clip sampling, data loading, precision,
  freezing (BatchNorm statistics keep updating), the optimiser (weight decay applies to every
  parameter) and plugins in training (`docs/concepts/training.md`); registering a stem layer,
  backbone, temporal pool, head, loss, callback or detector source as a plugin
  (`docs/guides/write-a-plugin.md`); the README
  quickstart extended through training a toy run and reading it back with `dfwb runs`.

## [0.1.0a3] - 2026-09-25

### Added

- The face pipeline (contract C3c): processing profiles (a named, hashed recipe for detection,
  tracking, cropping, sampling and decoding), five shipped profiles including the setting that
  reproduces the author's earlier FaceForensics++ stores, and a lossless PNG frame store with one
  `clip.json` per video (per-frame boxes, scores and landmarks; dropped frames and why; the
  tracking strategy and whether it may have switched identity partway through a clip).
- Three face-detection backends: `center` (no detector, a centred crop, for pre-cropped data and
  smoke tests), `insightface` (a dependency-free, line-for-line port of insightface 0.7.3's SCRFD
  detection and ArcFace embedding, run with onnxruntime against the original `buffalo_l` ONNX
  weights) and `mediapipe` (Google's Apache-2.0 BlazeFace detector).
- Exact crop and tracking, ported unchanged from the earlier pipeline: replicate-padded, scaled
  square crops; largest-face-then-IoU tracking with optional EMA smoothing of the reported box;
  and identity-guided subject selection for clips whose backend estimates face embeddings.
- `dfwb preprocess`: `run` (process a dataset with a profile, scoped to a protocol split or a
  `where` filter, with `--workers`, `--device`, `--shard`, `--redo` and `--limit`), `status`
  (outcome counts by task and, with a protocol, split), `merge` (combine a sharded run's index
  files) and `profiles` (list the shipped profiles). A run resumes automatically: a video with any
  row in the index is skipped unless its outcome is named in `--redo`, and no half-written output
  ever survives an interrupted or crashed run.
- Sharded runs (`--shard I/N`) for splitting one dataset's processing across several machines,
  each writing its own shard index file and a `.running` marker for as long as it may still append
  to it; `preprocess merge` folds every shard back into one index, and `preprocess status` reports
  a still-sharded store accurately even before it is merged.
- A licence-acknowledgement gate (`dfwb.core.licenses`) for model weights under stricter terms
  than dfwb's own code, recorded once per machine and checked again before every use; insightface's
  `buffalo_l` weights (non-commercial research use only) are gated this way, acknowledged with
  `dfwb preprocess run --accept-license`. `dfwb doctor` lists every acknowledgement recorded.
- Verified asset fetching (`dfwb.core.fetch`): an atomic, sha256-checked download for model
  weights, refused under `DFWB_OFFLINE`; used by the `insightface` and `mediapipe` backends to
  find or fetch their models under the cache root.
- User docs: processing profiles (the shipped profiles, the store layout and `clip.json`, resuming
  and redoing a run, sharding and merge, and the licence gate) and an extras-installation guide
  covering the face backends, the mediapipe/OpenCV package conflict and its fix, and GPU
  onnxruntime.

## [0.1.0a2] - 2026-09-25

### Added

- Protocols (contract C3a): protocol references (`[<pack>:]<dataset>[/<scheme>][@<version or
  hash>]`), installed-pack discovery with ambiguous-key detection across packs, scheme cards with
  per-split counts, and the split rules `official`, `official+ident-80-20`, `ident-72-14-14`,
  `all-test` and `benchmark`. A benchmark can be defined at given compressions, so the subset is
  the same whichever other compressions a local copy holds; the FaceForensics++ and
  DeepFakeDetection benchmarks are defined at c23.
- `dfwb protocols`: `list`, `info`, `verify` (coverage buckets, partial-coverage and mismatch
  exit codes, and a release-mismatch heuristic), `build` (deterministic pack files from a local
  inventory; a rebuild keeps the terms review already recorded in the card, and its notice
  always carries any note recorded there, decided or not), `materialize` (recompute and
  hash-check a recipe scheme), `lint`, `diff` (SemVer bump checking) and `new-pack` (scaffold a
  dependency-free protocol pack distribution).
- `dfwb inventory`: `build` and `show`, backed by a table-driven `BaseBuilder` and 21 built-in
  dataset inventory builders; raw folders are scanned deterministically, skipping hidden files and
  non-video clutter, matching video suffixes without regard to case, and following symlinks.
- Multi-location datasets: `DFWB_DATASETS_ROOT` takes several roots, searched in order and merged
  per task and compression when one dataset's copy is split across more than one of them;
  `DFWB_DATASET_<ID>` environment variables and `dfwb.toml` `[hosts.<host>]`/`[datasets]` tables
  override any one dataset's folder.
- `toyfake`: a synthetic dataset generated entirely from a seed (`dfwb datasets synth toyfake`,
  no download, no licence to agree to), its inventory builder, and a matching built-in protocol
  pack, so the whole raw-dataset-to-protocol pipeline can be exercised on a bare install with
  `dfwb datasets synth toyfake --no-media` (writing the videos themselves needs the `preprocess`
  extra).
- Optional media probing (`dfwb inventory build --probe`, the `preprocess` extra): frame count,
  fps, size, duration, audio and codec, with PyAV imported lazily and never needed unless
  `--probe` or toyfake video encoding is actually used.
- User docs: protocol concepts (references, schemes, verify buckets and exit codes) and a guide
  to adding a new dataset's inventory builder and protocol pack.
- Package scaffold: `deepfake-workbench` distribution, `dfwb` import package, `dfwb` command.
- `dfwb.core`: registries with lazy import-path targets, aliases, provider-qualified keys and
  install hints; plugin discovery through the `dfwb.plugins` entry point with failure isolation
  and plugin API version checks (contract C1).
- Config composition: `extends` chains (files, `dfwb://templates/…`, `<package>://…`), deep merge
  with `+key` appends, `${env:…}`/`${ref:…}` interpolation, dotted overrides, strict validation
  with path-exact errors, and config fingerprints (contract C2). Template `binary-frame`.
- Record contracts C3 (packs, dataset cards, labels, videos, splits, inventories, processing
  profiles, processed indexes) and C5 (score files) with fast, deterministic readers and writers.
- Detector contract C4 (`InputSpec`, `DetectorMeta`, `ClipBatch`, `DetectorOutput`, `Detector`).
- Root resolution (`DFWB_*_ROOT`, `dfwb.toml`, user config), root-relative paths, seeding,
  logging and run metadata.
- `dfwb` commands: `doctor`, `plugins list|info`, `config templates|init|show|validate`,
  `schema export c1…c5`, `completion`.
- Architecture tests: import-linter layer contracts, torch-free imports and commands, no
  import-time side effects, no machine-specific paths.
