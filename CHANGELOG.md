# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
