# Deepfake Workbench

![Deepfake Workbench: a four-stage pipeline, each stage ticking green in turn: datasets (verified inventories), protocols (versioned splits), training (seed-locked runs) and evaluation (AUC with confidence intervals and coverage).](assets/hero-light.svg#only-light)
![Deepfake Workbench: a four-stage pipeline, each stage ticking green in turn: datasets (verified inventories), protocols (versioned splits), training (seed-locked runs) and evaluation (AUC with confidence intervals and coverage).](assets/hero-dark.svg#only-dark)

`dfwb` is a uniform, plugin-driven workbench for deepfake detection research: raw dataset →
verified inventory → face clips → trained detector → score file → evaluation report, with every
stage pluggable and every score file recording the protocol version it covers and how it was
produced -- for a trained run, the processing profile and the config fingerprint too.

!!! note "Pre-release"
    `dfwb` is at `0.1.0b2`. Nothing is published on PyPI yet -- install from a clone (see
    [Install](install.md)). Linux is the only supported and tested OS, on Python 3.12 or newer.

## Principles

- **Never media.** dfwb works with identifiers, labels and splits only; every dataset still comes
  from its own owner, under the owner's own terms.
- **Reproducible and traceable.** Every score file records the protocol version and scheme hash it
  covers, and how it was produced: for a trained run, the processing profile, the config
  fingerprint, the checkpoint's hash and the training seed. A result can always be traced back to
  exactly what produced it.
- **Progressive installs.** The base install pulls no PyTorch at all: browsing protocols, building
  inventories and evaluating score files all work from it. Extras (`preprocess`, `train`, `zoo`)
  add exactly the heavier dependencies each stage needs, never more.

## Six ways in

- **[Try it in five minutes](quickstart.md)** -- install the `train` and `preprocess` extras and
  run dfwb's own synthetic dataset, `toyfake`, end to end: generate it, build its inventory, verify
  it against a protocol, process it, train a tiny model on it, and read the run back.
- **[Bring your own datasets](quickstart.md#with-your-own-datasets)** -- point
  `DFWB_DATASETS_ROOT` at real data (FaceForensics++ and Celeb-DF, say), build an inventory, verify
  it, process it, train from a shipped template, and score across datasets.
- **[Evaluate score files from your own code](guides/cross-dataset-eval.md#calibration-and-importing-foreign-scores)**
  -- `dfwb eval import` turns a CSV your own code already produced into a score file dfwb can
  report coverage-aware metrics and confidence intervals over, with no `torch` install at all.
- **[Add a dataset](guides/add-a-dataset.md)** -- write an inventory builder for a dataset's raw
  layout, plus a dataset card, and publish a protocol pack describing how it splits.
- **[Add a detector](guides/add-a-detector.md)** -- score any Python callable that returns a
  `Detector` through a `py:` source, or package one as a shareable, licence-aware zoo adapter card.
- **[Add a layer, backbone or loss](guides/write-a-plugin.md)** -- register a model component
  through `register(api)`, in a separately-installable plugin distribution.

## Map of the docs

| Section | What is in it |
|---|---|
| [Install](install.md) | Extras, and installing a PyTorch build first. |
| [Quickstart](quickstart.md) | The toyfake walkthrough, and the same shape with real datasets. |
| **Concepts** | [The pipeline](concepts/pipeline.md), [protocols](concepts/protocols.md), [processing profiles](concepts/processing-profiles.md), [detectors](concepts/detectors.md), [score files](concepts/score-files.md), [training](concepts/training.md), [plugins](concepts/plugins.md). |
| **Guides** | Task-oriented walkthroughs: adding a [dataset](guides/add-a-dataset.md) or [detector](guides/add-a-detector.md), [writing a plugin](guides/write-a-plugin.md), [cross-dataset evaluation](guides/cross-dataset-eval.md), [reproducing a run](guides/reproduce-a-run.md). |
| **Reference** | Generated from the code itself: the [CLI](reference/cli.md), the [config schema](reference/config.md), the [C1-C5 contracts](reference/contracts/index.md) and the [public API](reference/api/index.md). |
| [Datasets](datasets/index.md) | Every dataset dfwb has a built-in inventory builder for. |
| [Ecosystem](ecosystem.md) | Related packages: protocol packs, PyTorch extras, plugins. |
| [Citing](citing.md) | How to cite dfwb. |
| [Changelog](changelog.md) | What changed, release by release. |
