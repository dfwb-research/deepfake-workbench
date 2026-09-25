# Writing a plugin

A **plugin** is a small, separately-installable Python distribution that registers components
into dfwb's registries: `layers`, `transforms`, `backbones`, `temporal_pools`, `heads`, `losses`,
`metrics`, `eval_suites`, `face_backends`, `inventory_builders`, `protocol_packs`, `detectors`,
`detector_sources` and `callbacks`. Inventory builders and protocol packs are covered in
`docs/guides/add-a-dataset.md`; this guide covers the model and training side — a stem layer, a
backbone, a temporal pool, a head, a loss, or a `detector_sources` entry — since those are what
turn published research (a forensic front-end, a backbone architecture, a loss function) into
something a `model:`/`loss:` config can name, without patching the framework itself.

## The entry point

A plugin declares a `dfwb.plugins` entry point pointing at a `register(api)` function:

```toml
# pyproject.toml
[project.entry-points."dfwb.plugins"]
my-plugin = "my_plugin:register"
```

```python
# my_plugin/__init__.py
def register(api): ...
```

`register(api)` is called once, the first time any registry is read (plugin discovery is lazy).
`api` is the one `PluginAPI` object every plugin gets: a `Registry` per registry name, each with an
`.add(key, target, *, summary, aliases=(), requires=(), params=None, **meta)` method. `target` is
an import path, `"module.path:AttrName"` — nothing is imported at registration time, so listing
components (`dfwb plugins list`) never needs torch installed. If `register()` raises, every
registration it made is rolled back and the plugin shows up as `failed` in `dfwb plugins list
--all`; every other plugin still loads.

Pin the plugin API version your plugin was written against, so an incompatible future dfwb skips
it cleanly instead of breaking:

```python
DFWB_PLUGIN_API = ">=1.0,<2"  # module-level, read before register() is called
```

## A worked example

A small plugin, `my-dfwb-plugin`, registering a stem layer and a loss:

```
my-dfwb-plugin/
  pyproject.toml
  src/my_dfwb_plugin/
    __init__.py
    stem.py
    loss.py
```

```toml
# pyproject.toml
[project]
name = "my-dfwb-plugin"
version = "0.1.0"
dependencies = ["deepfake-workbench", "torch"]

[project.entry-points."dfwb.plugins"]
my-dfwb-plugin = "my_dfwb_plugin:register"
```

```python
# src/my_dfwb_plugin/__init__.py
DFWB_PLUGIN_API = ">=1.0,<2"


def register(api):
    api.layers.add(
        "channel-mean",
        target="my_dfwb_plugin.stem:ChannelMean",
        summary="Collapses RGB to its per-pixel mean, one channel",
        requires=("torch",),
    )
    api.losses.add(
        "weighted-bce",
        target="my_dfwb_plugin.loss:WeightedBCE",
        summary="BCE on the logit with a fixed positive-class weight",
        requires=("torch",),
    )
```

```python
# src/my_dfwb_plugin/stem.py
from torch import nn


class ChannelMean(nn.Module):
    """A toy stem layer: RGB -> its per-pixel mean, one channel."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.out_channels = 1  # every layers/ target the stem hook builds must set this

    def forward(self, x):
        return x.mean(dim=1, keepdim=True)
```

```python
# src/my_dfwb_plugin/loss.py
import torch.nn.functional as F
from torch import nn

from dfwb.core.detector import ClipBatch, DetectorOutput
from dfwb.train.losses import LossOutput


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, out: DetectorOutput, batch: ClipBatch) -> LossOutput:
        weight = out.logit.new_tensor(self.pos_weight)
        loss = F.binary_cross_entropy_with_logits(
            out.logit, batch.labels.float(), pos_weight=weight
        )
        return LossOutput(total=loss, parts={"weighted-bce": loss})
```

Once installed in the same environment as dfwb (`uv pip install -e my-dfwb-plugin`), a config can
use both:

```yaml
model:
  stem: {name: channel-mean}
loss: {name: weighted-bce, pos_weight: 2.0}
```

`model.stem` puts the built layer in front of the backbone; since `ChannelMean.out_channels == 1`,
the stem hook (`dfwb.models.stem.build_stem`) automatically appends a `1x1` convolution back to 3
channels, so any backbone can still follow it. A layer that already outputs 3 channels is used as
is, with no adapter appended. `dfwb plugins list` shows `layers/channel-mean` and
`losses/weighted-bce` once the plugin is installed; `dfwb plugins info layers/channel-mean` shows
its target, its provider, what it requires and the params model it names, if any.

## The component contracts

Every registry key builds one class or factory function; what it must look like depends on the
registry.

### `layers` (stems)

Any `nn.Module` with an `out_channels: int` attribute. `model.stem: {name: ..., **params}` builds
it with `in_channels` passed automatically (unless the config sets it), and the framework — never
the layer itself — appends a `1x1` conv when `out_channels != 3`. This is the generic hook that
makes forensic front-ends (an SRM residual filter bank, a frequency-domain transform, ...) usable
in front of any backbone without either backbone or framework code knowing about them.

### `backbones`

```python
class Backbone(nn.Module):
    kind: Literal["image", "video"]  # image: sees [B*T,C,H,W]; video: sees [B,T,C,H,W]
    out_dim: int
    native_input: InputSpec  # size/mean/std the pretrained weights expect

    def forward(self, x: Tensor) -> BackboneOutput: ...  # pooled [N,D], tokens [N,L,D] | None
    def param_groups(self) -> dict[str, list[nn.Parameter]]: ...  # e.g. {"blocks.0": [...], ...}
    def apply_freeze(
        self, freeze: FreezeSpec
    ) -> None: ...  # base class handles every mode but lora
```

`param_groups()` names this backbone's own parameter groups (by convention `"stem"`, `"blocks.<i>"`,
`"norm"`, ...); `partial` freezing and layer-wise LR decay both go only by this ordering, never a
name pattern over individual parameters, so a backbone that reports its groups honestly always
freezes (or decays) exactly what it says it has. Subclass `Backbone` (from `dfwb.models.backbone`)
and the base class's `apply_freeze` already implements `none`, `full`, `norm-only` and `partial`;
only `lora` needs backbone-specific wiring (`lora_module()`), and only when your backbone supports
it.

### `temporal_pools`

```python
class TemporalPool(nn.Module):
    def __init__(self, dim: int) -> None: ...  # dim is always passed by the assembler
    def forward(self, x: Tensor) -> Tensor: ...  # [B,T,D] -> [B,D]
```

Only wired in for `kind="image"` backbones; a `kind="video"` backbone already returns one feature
per clip, so no pool applies.

### `heads`

```python
class Head(nn.Module):
    num_classes: int  # set in __init__; scoring needs num_classes == 1

    def forward(self, x: Tensor) -> Tensor: ...  # [B,D] -> [B] (num_classes=1) or [B,K]
```

`dim` (the backbone's `out_dim`) is always passed by the assembler, the same as for a pool.

### `losses`

```python
def forward(self, out: DetectorOutput, batch: ClipBatch) -> LossOutput: ...
```

`LossOutput(total, parts)` — `total` is what gets backpropagated, `parts` is named components for
logging (a loss with no sub-parts still puts `total` under one key, as the worked example above
does). Read `out.logit`, never `out.score`, for a training loss; `out.logit` is only set on the
`forward()` (training) path.

### `detector_sources`

A `detector_sources` entry is a callable, `str -> Detector` (the reference minus the registry key
itself, e.g. `<dir>[#best|#last]` for the built-in `run` source strips the leading `run:`). It
resolves a URI into a live `Detector`. `dfwb.models.source:load_run` — registered as `run` — is the
one built-in source; it rebuilds a checkpoint's `AssembledDetector` from its `detector.json`
through the same registries the training config used (see `docs/concepts/detectors.md`). A
`zoo:<name>` source, or a `py:<module:factory>` source for arbitrary user code, would register the
same way.

### Optimisers and schedules are not plugin registries

`optim:` (`adamw`, `sgd`) and `schedule:` (`constant`, `cosine`, `step`) are **not** part of the
plugin system — there is no `optimizers` or `schedules` registry. They are built directly from the
already-assembled `AssembledDetector`, because `optim.groups` keys on that detector's own parameter
group names (`param_groups()` plus `head`, `pool` and `stem`), and a typo in a group name can only
be caught once the detector actually exists. Both are validated by fixed pydantic models in
`dfwb.train.optim`/`dfwb.train.schedules`; adding a new one means changing that module, not
installing a plugin.

## How config parameters are validated

`api.<registry>.build(key, **params)` (what a config's `{name: ..., **params}` ultimately calls)
validates `params` before the target is ever called:

- **By default**, against the target's own `__init__`/call signature, which means importing the
  target's module to read it — a keyword the target doesn't accept is a `ConfigError` naming it
  (with a did-you-mean suggestion), and a missing required one or a wrong type is reported the
  same way, all without invoking the target.
- **With `params="module:Model"`** (a pydantic `BaseModel`, a dataclass, or a `TypedDict`),
  against that model instead — useful when the target's own signature is untyped or takes
  `**kwargs`, since a params model validates without importing the (possibly heavy) target class
  at all. `WeightedBCE` above didn't need one, since its plain typed `__init__` is already enough.

Either way, an unknown key, a type mismatch, or a missing required parameter is reported with the
dotted path into the config (`loss.pos_weight: ...`), before any data loads — never a bare
`TypeError` from deep inside your class. `requires=("torch", "timm")` on `.add()` is checked before
the target is even imported: a registered component whose dependency isn't installed fails with
`InstallationError` and the exact install command, rather than a stack trace through someone else's
import.

## Checking it

```bash
dfwb plugins list                       # every registered component, once your plugin is installed
dfwb plugins info layers/channel-mean   # target, provider, requirements, params model
dfwb plugins list --all                 # also shows plugins that failed or were skipped, and why
```

## See also

- `docs/concepts/detectors.md` — the `Detector` contract these components assemble into, and how a
  trained run is loaded back through the `run:` detector source.
- `docs/guides/add-a-dataset.md` — the other half of the plugin system: inventory builders and
  protocol packs.
