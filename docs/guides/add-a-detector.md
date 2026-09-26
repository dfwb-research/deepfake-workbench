# Adding a detector

Anything `dfwb score` can point `--detector` at is resolved through the `detector_sources`
registry from a `<scheme>:<rest>` URI: `run:` for a trained run (`docs/concepts/detectors.md`),
`zoo:` for a registered zoo adapter, and `py:` for your own code. This guide covers the two ways
to bring a detector of your own into `dfwb score`: a quick `py:` factory function, and a full
zoo adapter card for a detector meant to be shared. **No real third-party adapters ship with this
version** — the zoo machinery below is exercised by two dummies, `chance` and `random`, which are
sanity floors, not real detectors.

## The quick way: `py:<module>:<factory>`

`py:<module>:<factory>` imports `<module>` (which must be importable on the interpreter's own
path — on `PYTHONPATH`, or installed) and calls its module-level `<factory>()` with no arguments,
which must return a C4 `Detector`: an object with `meta` (a `DetectorMeta`), `to(device)` and
`predict(batch)`. This runs your own code exactly as if you had imported and called it yourself --
there is no sandboxing.

```python
# my_detector.py
from __future__ import annotations

import torch

from dfwb.core.detector import DetectorMeta, DetectorOutput, InputSpec

_SPEC = InputSpec(crop="full-frame", crop_scale=None, size=(64, 64), frames=8)


class BrightnessDetector:
    """P(fake) = the clip's own mean pixel brightness -- a toy, not a real detector."""

    def __init__(self) -> None:
        self.meta = DetectorMeta(
            name="brightness",
            version="0.1",
            contract_version=(1, 0),
            input=_SPEC,
            license="MIT",
            weights_license=None,
            citation=None,
            source="py:my_detector:make_detector",
        )

    def to(self, device: torch.device) -> "BrightnessDetector":
        return self

    def predict(self, batch) -> DetectorOutput:
        score = batch.clips.mean(dim=tuple(range(1, batch.clips.ndim))).clamp(0.0, 1.0)
        return DetectorOutput(score=score)


def make_detector() -> BrightnessDetector:
    return BrightnessDetector()
```

```bash
dfwb score --detector py:my_detector:make_detector \
           --protocol toyfake/official --split test --profile toy-64-center-8f
```

This factory's `InputSpec` (`crop="full-frame"`) matches the toyfake quickstart's
`toy-64-center-8f` store exactly, so no `--allow-input-mismatch` is needed; a detector expecting a
face crop against a full-frame store (or the reverse) needs it, the same as any other detector
source (`docs/concepts/detectors.md#input-adaptation`).

**Caching.** `py:`'s cache key folds in the sha256 of `my_detector.py` itself (via
`fingerprint_extra`, not `meta.source`, since your `meta.source` need not change from one edit to
the next): editing the file and scoring again writes a fresh score file rather than reusing a
stale one. When the module has no readable source file at all (a compiled extension, a namespace
package), `dfwb score` cannot tell whether the code changed between two loads, so it never trusts
a cached file for that detector: every scoring run recomputes, logged when it happens. See
`docs/concepts/score-files.md` for the optional attributes (`checkpoint_sha256`, `training_seed`,
`fingerprint_extra`, `cacheable`) a source may set on the detector it returns.

A factory that raises, or returns something missing `meta`, `predict` or `to`, is reported by name
rather than as a bare traceback — try, for instance, a factory that returns `None`.

## The full way: an adapter card

An **adapter card** (contract C4's schema, `dfwb.zoo.card.AdapterCard`, plain YAML, unknown keys
rejected) is how a detector becomes a shareable, licence-aware `zoo:<name>` source: upstream
provenance, licensing, how its code is obtained, its weight variants (each with a sha256), the
input it expects, its score polarity, and the numbers it claims (`reported`) versus what has
actually been reproduced (`parity`). A fictional example, as if adding a small published
classifier:

```yaml
# cards/tinynet.yaml
name: tinynet
display_name: TinyNet
contract_version: [1, 0]
upstream:
  repo: https://github.com/example/tinynet
  commit: 0123456789abcdef0123456789abcdef01234567
  paper: {title: "TinyNet: a small deepfake classifier", venue: "Example Workshop", year: 2024}
license:
  code: MIT
  weights: CC-BY-NC-4.0
  requires_ack: true
code_strategy: pip
install: {extra: zoo-tinynet, pip: ["tinynet==1.0"]}
weights:
  - id: default
    url: https://example.org/tinynet-weights.safetensors
    sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    bytes: 12345678
    format: safetensors
polarity: fake-high
input: {crop: face, crop_scale: 1.3, size: [224, 224], frames: 1}
reported:
  - {protocol: ffpp/official, split: test, metric: auc, value: 0.97, source: "paper, Table 2"}
```

- **`code_strategy`** says how the adapter's upstream model code is obtained:
  - **`pip`** — the card's `install.extra` pins the upstream package (`pip install
    "deepfake-workbench[<extra>]"`), and the adapter imports it normally. This is the default, and
    what both dummy adapters use (with no real upstream package to install).
  - **`vendored`** — upstream code under an MIT/BSD/Apache-compatible licence, small enough to
    copy verbatim into `dfwb/zoo/_vendor/<name>/`, alongside its own `LICENSE`, a `NOTICE`
    (upstream repo, commit, files, any modifications) and a `HASHES.sha256` list proving the copy
    is still unmodified.
  - **`pinned-clone`** — upstream code under an incompatible licence, or too large to vendor:
    `dfwb zoo fetch <name>` clones it at the card's pinned commit into
    `$DFWB_CACHE_ROOT/zoo/<name>/code/<commit>/` (gated on the licence acknowledgement first),
    never committed to a dfwb-research repository, and imported under a private module name
    (`dfwb_zoo_ext_<name>`) rather than added to `sys.path` — so a clone that happens to contain
    its own top-level `src/` package never collides with anything else importable.
- **`weights`** lists every downloadable variant with its exact sha256 and byte count. `dfwb zoo
  fetch <name>` downloads (atomically, sha256-checked) into
  `$DFWB_CACHE_ROOT/zoo/<name>/<sha256>/`; a file already there is re-verified against the card on
  every use, not trusted by its path alone, so local tampering or corruption is always caught. A
  weights format is always `safetensors` or a `torch.load(weights_only=True)` checkpoint — never
  an arbitrary pickle.
- **The licence gate.** `license.requires_ack: true` (weights under stricter terms than the code,
  as `tinynet`'s fictional `CC-BY-NC-4.0` weights are here) means `zoo:tinynet` refuses with a hint
  until the licence is acknowledged once, on this machine:

  ```bash
  dfwb zoo fetch tinynet --accept-license
  dfwb zoo licenses   # lists every acknowledgement recorded
  ```

  The acknowledgement is stored with a timestamp and is not asked for again. Both dummy adapters
  set `requires_ack: false` — there is nothing to acknowledge to use them.

## Checking an adapter card

```bash
dfwb zoo list                 # every registered adapter: licence, input, reported/parity counts
dfwb zoo info random          # a card's full detail
dfwb zoo verify random        # re-hash cached weights and check the code pin, downloading nothing
dfwb zoo parity random --tolerance 0.01
```

`dfwb zoo parity <name>` scores the adapter's own parity set and compares the measured numbers
against `card.reported`, writing the result into a local overlay file (`dfwb zoo info` does not
show it) meant to be pasted into the card itself once it passes. `chance` and `random` report no
`reported` metrics, so there is nothing for `parity` to check against yet — exactly what "the
machinery, not real adapters" means in this version. Scoring either of them still runs the whole
path:

```bash
dfwb score --detector zoo:random --protocol toyfake/official --split test --allow-input-mismatch
```

## See also

- `docs/concepts/score-files.md` — the C5 schema and the cache, including the optional attributes
  a detector source can set.
- `docs/guides/cross-dataset-eval.md` — scoring a suite and turning score files into metrics.
- `docs/concepts/detectors.md` — the `Detector` contract (C4) itself, and the `run:` source.
