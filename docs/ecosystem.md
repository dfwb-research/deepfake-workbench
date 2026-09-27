# Ecosystem

`deepfake-workbench` (`dfwb`) is the framework: verified dataset inventories, face preprocessing,
training, scoring any detector, and evaluation with uncertainty. It never distributes media, and
its base install pulls in no PyTorch at all. Two related, separately-installable projects add
datasets and PyTorch building blocks; a plugin can add anything else.

!!! note "In preparation"
    Neither is published on PyPI yet. `dfwb-torch` is public
    ([github.com/dfwb-research/dfwb-torch](https://github.com/dfwb-research/dfwb-torch)) and
    installs from its repository; `dfwb-protocols` is in preparation and not public yet.

## `dfwb-protocols`

Versioned train/validation/test splits for public deepfake datasets, released under CC BY 4.0.
Installing it registers its protocol packs with dfwb through the `dfwb.plugins` entry point; it
has no runtime dependencies of its own, and dfwb has none on it either -- `dfwb protocols verify`
and `dfwb score --suite` work against whatever packs happen to be installed, `dfwb-protocols` or
otherwise. It never contains media: only identifiers, labels, split assignments and fake/real
pairs, plus per-video metadata already public in a release's own files. See
[Protocols](concepts/protocols.md) and
[Adding a dataset](guides/add-a-dataset.md) for the pack format it publishes.

## `dfwb-torch`

Small, standalone PyTorch building blocks for media forensics -- forensic front-ends, backbones,
losses -- each its own distribution (`dfwb-torch-<name>`), versioned and released independently,
whose only runtime dependency is `torch` (plus `numpy` where genuinely needed). None of them
depends on `deepfake-workbench`; a package may optionally self-register as a dfwb plugin (a
`layers`, `backbones` or `losses` entry) through the same `dfwb.plugins` entry point any plugin
uses, so it works standalone in someone else's code and inside dfwb equally well. The first planned
package is `dfwb-torch-srm`, a fixed high-pass SRM filter-bank stem. See
[Writing a plugin](guides/write-a-plugin.md) for the `layers` (stem) contract such a package
registers into.

## Plugins

Anyone can publish a plugin: an inventory builder and protocol pack for a dataset not listed in
[Datasets](datasets/index.md) ([Adding a dataset](guides/add-a-dataset.md)), or a model/training
component -- a stem, backbone, temporal pool, head, loss, callback or `detector_sources` entry
([Writing a plugin](guides/write-a-plugin.md)) -- or a zoo adapter card wrapping a published
third-party detector ([Adding a detector](guides/add-a-detector.md)). `dfwb plugins list --all`
shows every plugin installed alongside dfwb, whether it loaded, and why one that failed did.
There is no separate directory of known third-party plugins yet; check the organisation's
repositories for what exists.

## See also

- [The pipeline](concepts/pipeline.md) -- where a pack, a PyTorch package or a plugin fits in.
- [Plugins](concepts/plugins.md) -- entry points, registries, and `register(api)`.
