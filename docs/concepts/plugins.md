# Plugins

A **plugin** is a small, separately-installable Python distribution that registers components into
dfwb's registries -- a backbone, a metric, a dataset's inventory builder, a protocol pack -- without
patching the framework itself. dfwb's own built-in components (the metrics, the two dummy zoo
adapters, every shipped inventory builder, `toyfake`'s protocol pack) register exactly the same
way, through the same mechanism, as a third-party plugin would.

## The entry point

A plugin declares a `dfwb.plugins` entry point pointing at a module-level `register(api)` function:

```toml
# pyproject.toml
[project.entry-points."dfwb.plugins"]
my-plugin = "my_plugin:register"
```

```python
# my_plugin/__init__.py
def register(api): ...
```

Discovery is **lazy**: nothing is imported until the first time any registry is actually read (for
example, by `dfwb plugins list`, or by resolving a config's `model.backbone`). dfwb's own built-ins
register through the same shape, a `dfwb.builtins` entry point pointing at
`dfwb._builtins:register`, so they show up in `dfwb plugins list` with provider `dfwb` rather than
being special-cased anywhere.

## Registries

`api` is one `PluginAPI` object, holding one `Registry` per registry name. A registry maps
lower-kebab-case keys to *import-path targets* (`"package.module:Attr"`) plus metadata -- nothing
is imported at registration time, so listing components never needs an optional dependency
installed, and the fourteen registries below cost nothing until one of their keys is actually
resolved:

`layers`, `transforms`, `backbones`, `temporal_pools`, `heads`, `losses`, `metrics`,
`eval_suites`, `face_backends`, `inventory_builders`, `protocol_packs`, `detectors`,
`detector_sources`, `callbacks`.

Two plugins may register the same key under different providers; the plain key then becomes
ambiguous and must be qualified as `<provider>:<key>` (`dfwb plugins list` shows the provider of
every entry). A registry's `.add(key, target, *, summary, aliases=(), requires=(), params=None,
**meta)` records the registration; `requires=("torch", "timm")` is checked only when the key is
actually built, so a plugin that needs an optional dependency fails with a clear
`InstallationError` and an install hint at the point of use, not at discovery time.

## `register(api)` failing is isolated

If `register()` raises, every registration it made is rolled back and that plugin shows up as
`failed` in `dfwb plugins list --all`, with the exception's message as its reason; every other
plugin still loads. A plugin can also pin the plugin API version it was written against
(`DFWB_PLUGIN_API = ">=1.0,<2"`, read before `register()` runs), so an incompatible future dfwb
skips it cleanly -- `skipped`, not `failed`.

## Looking at what is registered

```bash
dfwb plugins list                 # every registered component: registry/key, provider, summary
dfwb plugins list --all           # also plugins that failed or were skipped, and why
dfwb plugins info backbones/timm  # one component's target, provider, requirements, params model
```

`dfwb plugins list`'s component rows are contract C1's `Entry` records; `dfwb plugins list --all`'s
plugin rows are `PluginRecord`s (name, provider, version, status, reason) -- both are the shapes
[C1 in Contracts](../reference/contracts/c1.md) publishes.

## Writing one

Two kinds of plugin cover everything dfwb's own layers do not hard-code:

- A **dataset plugin** -- an `InventoryBuilder` and, optionally, a protocol pack -- teaches dfwb a
  dataset's raw layout and how it splits. See [Adding a dataset](../guides/add-a-dataset.md).
- A **model or training plugin** -- a stem layer, a backbone, a temporal pool, a head, a loss, a
  training callback, or a `detector_sources` entry -- turns published research into something a
  config can name. See [Writing a plugin](../guides/write-a-plugin.md), which works through a full
  worked example end to end.

## See also

- [The pipeline](pipeline.md) -- the five contracts, including C1.
- [Adding a detector](../guides/add-a-detector.md) -- the `detector_sources` registry specifically
  (`run:`, `zoo:`, `py:`), and packaging a detector as a zoo adapter card.
