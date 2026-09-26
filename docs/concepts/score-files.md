# Score files

A **score file** (contract C5) is dfwb's interchange format between anything that produces a
per-video score — a trained run, a zoo adapter, your own code — and `dfwb eval`. It is always
two files together, read and written as a pair:

- **`<name>.scores.csv`** — one row per video.
- **`<name>.scores.meta.json`** — everything the scores depend on: the detector, the protocol
  split, the processing profile, the aggregation, and how much of the split was actually scored.

`dfwb score` writes both from a live detector (`docs/guides/cross-dataset-eval.md`); `dfwb eval
import` writes both from a CSV your own code already produced
(`docs/guides/reproduce-a-run.md`).

## The CSV

Required columns: `dataset, key, compression, label, score, status`. Optional columns, present
only when a writer sets them: `logit, label_key, method, n_clips, n_frames`. A column named
`x_...` is an extension column a writer added on top of the contract; `dfwb eval` ignores columns
it does not know.

- **`score`** is always P(fake) in `[0, 1]`, higher meaning more fake — there is no
  detector-specific polarity flag to check. A detector or an imported file that reports P(real)
  has to flip it before it ever reaches a score file.
- **`label`** is the ground-truth integer label under whichever label mapping produced the file
  (recorded as `labels` in the meta, below) — `0`/`1` for the usual binary mapping.
- **`compression`** is empty for datasets with no compression variants.

### Statuses: nothing is ever dropped

Every video the split names gets exactly one row, whatever happened to it:

- **`ok`** — scored; `score` is a number in `[0, 1]`.
- **`missing`** — no usable processed clip for this video (nothing was there to score); `score`
  is empty.
- **`error`** — scoring raised, or the detector's output failed validation; `score` is empty.

A row is never silently left out of the file, so a score file's own row count next to its meta's
`coverage.expected` is always the true size of the split under the label mapping used — and a
`dfwb eval` run over it reports exactly how much of that was actually scored (coverage and the
`--missing` policy: `docs/guides/cross-dataset-eval.md`).

## The meta file

`<name>.scores.meta.json` records what produced these numbers: the detector's identity, the
protocol split and pack version, the label mapping, the processing profile and any input
adaptation, the clip -> video aggregation, the seed, the environment (versions, the device, the
precision, and the hash of the processed store's index the run read), the git state of the
working directory, and the command line. `dfwb score` records its own command with no absolute
path in it (a path under a DFWB root is written as `$DFWB_*_ROOT/...`, any other keeps only its
final component); a file written by `dfwb zoo parity`, or through the Python API without
`command=`, has `"command": null`, and a cache hit keeps the command of the run that wrote the
file:

```json
{
  "schema": "dfwb.scores/1",
  "detector": {
    "name": "tiny-cnn-mean-linear",
    "version": "0.1.0b2",
    "source": "run:0b4c5f19880c55a91710f38ccbe3074788fa0ca95e291559df6cb1ff6a7a0330",
    "checkpoint_sha256": "4a4d9c4e57113a1b5fcf7c57da43f076fbddd58320833a6b4edf27d4df3aca1c",
    "contract_version": [1, 0]
  },
  "protocol": {
    "id": "toyfake/official",
    "split": "test",
    "where": {},
    "pack": "toyfake",
    "pack_version": "0.1.0b2",
    "scheme_sha256": "340a0e8650fe661bf9e318d3bbf9b88cb8564f0c74d3228f69d03342dffd110d"
  },
  "labels": "binary",
  "processing_profile": {"id": "toy-64-center-8f-28b2d3ca", "sha256": "28b2d3ca22a0e99e284e33a5e77671fb201e5ddeabcc5e94f4565161d20a1667"},
  "input_adaptation": {"derived_crop": false, "mismatch_override": false},
  "aggregation": {"clip_to_video": "mean-prob", "clips_per_video": 4},
  "coverage": {"expected": 41, "ok": 41, "missing": 0, "error": 0},
  "seed": 0,
  "env": {
    "dfwb": "0.1.0b2", "python": "3.12.14", "platform": "Linux-x86_64", "torch": "2.14.0+cpu",
    "cuda": null, "device": "cpu", "precision": "fp32",
    "store_index_sha256": "a41ed4149e8c61c7915a679cb69713578023b29bd634872e7665df58f8166370"
  },
  "git": null,
  "command": "dfwb score --detector 'run:runs/toy-cpu/latest#best' --protocol toyfake/official --split test",
  "created": "2026-09-26T06:21:33Z",
  "calibration": null
}
```

(Taken from a real run of the toyfake quickstart.) `source` is `run:<config fingerprint>` for a
run detector, `zoo:<name>` or `zoo:<name>@<weights id>` for a zoo adapter, and
`py:<module>:<factory>` for a `py:` detector whose factory leaves `meta.source` unset (one that
sets it keeps its own).

`seed` is the seed the weights were trained with when the detector has one (a `run:` detector's
own seed), otherwise the scoring command's `--seed`. `dfwb eval` folds files that agree on the
detector's `source`, the protocol split and `where`, the label mapping, the aggregation and the
processing profile, but not on `seed`, into a seeds table (mean and standard deviation across
seeds) — but a `run:` detector's
`source` is its config's fingerprint, and the config's `run.seeds` is part of that fingerprint.
So the seeds table groups the seeds of one experiment config: list every seed in one config's
`run.seeds` to get it. Seeds trained as separate jobs, each from a config naming only its own
seed, have different `source` values and are never folded together
(`docs/guides/reproduce-a-run.md`).

`coverage` is always recomputed from the rows when the file is written or read, never taken on
faith, so it can never drift from what the CSV actually says. `calibration` is set only on a file
`dfwb eval calibrate` produced (method, what it was fit on, and its parameters).

## The cache

`dfwb score` reuses a file already sitting at its own output path instead of rescoring, unless
you pass `--force`.

- **The output path already encodes the request.** `dfwb score` writes to
  `<out>/<detector-slug>/<protocol-slug>/<split>-<key[:8]>.scores.csv`, where `<key>` is a sha256
  of everything the run depends on: the detector's exact identity, the protocol's scheme hash and
  the pack's version, the split, any `--where` filter, the processing profile's hash, the hash of
  the processed store's `index.jsonl`, the aggregation mode and clip count, the label mapping, and
  the precision (`--precision`; none means `fp32`). Two requests that agree on all of that always
  land at the same path; changing any one of them changes the path.
- **The store's contents are part of the request.** Processing more videos, or re-processing some,
  appends to the store's `index.jsonl`, so its hash changes and the next `dfwb score` scores the
  store as it is now, instead of serving a file whose `missing` rows are no longer missing. A pack
  release that fixes labels without touching the split file changes the pack's version, and so
  the key, the same way.
- **A file found there is still checked, not just trusted by its path.** Its meta is read back and
  compared field by field against what the new request expects. A mismatch — a hash collision, or
  a stale or hand-placed file — is recomputed rather than served. The precision and the store's
  index hash have no field of their own in the meta, so they are recorded in `env` (as
  `precision` and `store_index_sha256`) and checked there.
- **A cached file with any `error` row is never reused.** A detector failure is usually transient
  (an out-of-memory error, a flaky device fault), so the reuse rule is about an identical
  *successful* result: an errored cache entry is retried, not served as if it were complete.
- **`--force`** recomputes regardless of what is already there, and overwrites it.
- **`--frames` has its own reuse rule.** A cache hit is only returned as satisfying `--frames`
  when its own `<name>.frames.parquet` already exists next to it; a file that was cached before
  `--frames` was first asked for is recomputed instead, so `--frames` always means "one exists" on
  return, never "one might, depending on history".

The detector's own identity matters as much as the protocol side: see the next section for what a
detector source can set to sharpen it.

## Optional detector attributes

A `Detector` (contract C4) needs only `meta`, `to()` and `predict()`. A detector source may set a
few plain attributes beyond that, read duck-typed (`getattr(detector, "...", None)`), that
sharpen the score file's meta and its cache key beyond what `meta.source` alone can say:

| Attribute | Type | Meaning |
|---|---|---|
| `checkpoint_sha256` | `str \| None` | The sha256 of the exact weights file scored. |
| `training_seed` | `int \| None` | The seed the detector was trained (or otherwise seeded) with; recorded as the score file's own `seed` in preference to whatever `--seed` the scoring command was given. |
| `fingerprint_extra` | `str \| None` | A source-owned string folded into the cache key in place of `meta.source`, for a source whose `meta.source` is not on its own a reliable identity. |
| `cacheable` | `bool` (default `True`) | `False` means a file already sitting at this detector's cache path is never trusted, however recently it was written — `dfwb score` always recomputes and overwrites it. |

None is required. The built-in sources set them as follows:

- **`run:`** sets `checkpoint_sha256` and `training_seed` from the checkpoint it loads.
- **`py:`** sets `fingerprint_extra` to `<module>:<factory>:<sha256 of the module's source file>`,
  so editing your factory's file changes its cache key even when its `meta.source` does not; when
  the module has no readable source file (a compiled extension, a namespace package), it falls
  back to the stable `<module>:<factory>` and sets `cacheable=False`, since it can no longer tell
  whether the code has changed between two loads.
- **`zoo:`** sets `checkpoint_sha256` from the resolved weight variant's own sha256 (already
  verified by the weight manager), and rewrites `meta.source` to `zoo:<name>` or
  `zoo:<name>@<weights id>` — whichever weight id a bare `zoo:<name>` actually resolved to.

## See also

- `docs/concepts/detectors.md` — the `Detector` contract (C4), and what a run directory holds.
- `docs/guides/cross-dataset-eval.md` — scoring a suite and turning score files into metrics.
- `docs/guides/add-a-detector.md` — the `py:` source and adapter cards.
- `docs/guides/reproduce-a-run.md` — from a run directory to a comparable score file.
