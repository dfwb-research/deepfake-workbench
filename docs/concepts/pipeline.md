# The pipeline

dfwb turns a raw dataset release into an evaluation report through a fixed sequence of stages,
each one reading and writing a small, versioned file format -- a **contract** -- so any stage can
be swapped, re-run or replaced by a plugin without the others knowing or caring.

```text
raw dataset  ->  inventory  ->  protocol split  ->  processed store  ->  detector  ->  score file  ->  report
  (owner's)     (C3b builder)     (C3a pack)       (processing        (C4)          (C5)           (dfwb eval)
                                                     profile)
```

| Stage | Command | Reads | Writes |
|---|---|---|---|
| Inventory | `dfwb inventory build` | A dataset folder, via a registered `InventoryBuilder` | `<work root>/<dataset>/inventory.jsonl` |
| Protocol | `dfwb protocols verify` | The local inventory, a protocol pack's split | A coverage report |
| Preprocessing | `dfwb preprocess run` | Raw videos, a `ProcessingProfile` | A processed store (cropped, tracked clips) |
| Training | `dfwb train` | A processed store, an experiment config | A run directory with a checkpoint |
| Scoring | `dfwb score` | Any `Detector` (a run, a zoo adapter, your own code), a protocol split | A C5 score file |
| Evaluation | `dfwb eval` | One or more C5 score files | Metrics, confidence intervals, comparisons |

Every stage after inventory-building works from what the stage before it wrote, never by
re-deriving it, so a step can be re-run in isolation and its output inspected on its own terms.

## The five contracts

A **contract** is a JSON Schema, versioned independently of dfwb's own release version, that a
file format or API boundary commits to; `dfwb schema export <id>` prints any of them, and
[Contracts](../reference/contracts/index.md) publishes all five, generated from the same pydantic
models dfwb validates against at runtime -- there is exactly one definition of each, never a
hand-copied second one.

- **C1 -- Plugin API.** Registry entries (`Entry`: a key, an import-path target, a provider, what
  it requires) and plugin load records (`PluginRecord`: which plugin, whether it loaded, why not).
  What a `register(api)` function in a plugin distribution produces; see
  [Plugins](plugins.md).
- **C2 -- Config.** `schema: dfwb.train/1`: one training experiment -- data sources, the model's
  components, the loss, optimiser, schedule and training loop settings. `dfwb config
  validate`/`dfwb config show` check and resolve it; see [Config reference](../reference/config.md).
- **C3 -- Dataset and protocol records.** Two related shapes: **C3a**, what a protocol pack
  publishes (`PackCard`, `DatasetCard`, `LabelVocab`, the split files) and **C3b**, the inventory
  format an `InventoryBuilder` produces locally (`InventoryRecord`) before it is checked against a
  pack. See [Protocols](./protocols.md) and [Adding a dataset](../guides/add-a-dataset.md).
- **C4 -- Detector and adapter metadata.** The `Detector` protocol itself (`meta`, `to()`,
  `predict()`), its `InputSpec` and `DetectorMeta`, and the zoo `AdapterCard` schema that wraps a
  published third-party detector. See [Detectors](detectors.md).
- **C5 -- Score file.** `schema: dfwb.scores/1`: a CSV of per-video scores plus a meta JSON file
  recording exactly what produced them. What `dfwb score` writes and `dfwb eval` reads; see
  [Score files](score-files.md).

## Processing profiles: the stage between raw video and a detector

A **processing profile** (`ProcessingProfile`, part of C3b) names one way of turning raw video into
a fixed-shape clip store: which face-detection backend, what crop scale and size, how many frames
and how they are sampled. `dfwb preprocess run <dataset> --profile <name>` runs it once per
dataset; the resulting store is then shared by every training run, scoring pass and detector that
asks for clips shaped that way, so the (often expensive) face-detection step happens once per
dataset and profile, not once per experiment. See [Processing profiles](processing-profiles.md).

## Everything is pluggable

Every stage after the raw dataset is a plugin boundary: an `InventoryBuilder` and a protocol pack
know one dataset each; a `backbones`/`temporal_pools`/`heads`/`losses` component assembles into a
trained detector; a `detector_sources` entry (`run:`, `zoo:`, `py:`) turns a string into a live
`Detector` for `dfwb score`; a `metrics` entry adds a metric spec `dfwb eval` understands. Nothing
here is hard-coded to a specific dataset, model architecture or metric -- see
[Plugins](plugins.md) for how a plugin registers into any of these.

## See also

- [Protocols](./protocols.md) -- protocol references, split schemes, `verify`'s buckets and exit
  codes.
- [Detectors](detectors.md) -- the `Detector` contract and `AssembledDetector`.
- [Score files](score-files.md) -- the C5 schema, statuses and the cache.
- [Plugins](plugins.md) -- entry points, registries and `register(api)`.
