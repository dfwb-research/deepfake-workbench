# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  inventory; a rebuild keeps the terms review already recorded in the card), `materialize` (recompute and hash-check a recipe scheme), `lint`, `diff` (SemVer
  bump checking) and `new-pack` (scaffold a dependency-free protocol pack distribution).
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
