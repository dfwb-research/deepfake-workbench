# Reproducing a run

A run directory is self-describing (`docs/concepts/detectors.md#what-a-run-directory-holds`):
everything needed to load its detector back and score it again lives inside it. This guide walks
from a run directory someone hands you — or one you trained yourself — to a score file, and says
what makes two score files from two different runs actually comparable.

## What a run directory holds, and why it is enough

```
runs/<run.name>/<YYYYmmdd-HHMMSS>-s<seed>/
  config.resolved.yaml   # the exact, fully-resolved config this run trained
  fingerprint.txt        # that config's fingerprint (shared by every seed in its run.seeds)
  env.json                # seed, dfwb/python/torch versions, device, git state, the command run
  data.json               # per source: protocol ref, pack version, split hash, profile, counts
  checkpoints/best/{model.safetensors, detector.json}
  checkpoints/last/{model.safetensors, detector.json}
  scores/val/<source>.scores.{csv,meta.json}   # the last epoch's validation scores
  report.md
  metrics.json
```

`config.resolved.yaml` is what actually trained — every `extends:` chain already followed, every
default filled in — so it needs no other file to make sense of it. `env.json` records the exact
`dfwb`/`torch`/`python` versions and the git commit the run was made with, plus the command line
that started it; `data.json` records, per data source, exactly which protocol split, pack version
and processing profile were joined, and how many videos each contributed. None of this needs to be
re-derived or guessed: reproducing a run's numbers is reading these files back, not re-training
and hoping for the same answer.

## Loading the detector back: the `run:` source

```bash
dfwb score --detector "run:runs/toy-cpu/latest#best" --protocol toyfake/official --split test
```

`run:<dir>[#best|#last]` (`#best` is the default) resolves `<dir>` — a run directory itself, a
`latest` symlink, or a run-name directory holding one — and rebuilds the detector purely from
`checkpoints/<tag>/detector.json` and `model.safetensors` through the same registries the original
config used. **This never downloads anything**: a backbone is rebuilt with `pretrained` switched
off, since its actual weights come from `model.safetensors`. If a component's registered provider
is not installed, or is installed at an incompatible version, this fails by naming exactly which
registry, key, provider and version — never a bare `AttributeError` deep inside a class that no
longer exists. See `docs/concepts/detectors.md` for the full contract.

The resulting score file's meta records the checkpoint's exact identity:

```json
"detector": {
  "name": "tiny-cnn-mean-linear",
  "source": "run:0b4c5f19880c55a91710f38ccbe3074788fa0ca95e291559df6cb1ff6a7a0330",
  "checkpoint_sha256": "4a4d9c4e57113a1b5fcf7c57da43f076fbddd58320833a6b4edf27d4df3aca1c",
  ...
},
"seed": 0
```

`source` is `run:<config fingerprint>` — the same fingerprint `fingerprint.txt` and `dfwb runs
list` show, identifying the *experiment config* (every seed listed in its `run.seeds`), not one
particular checkpoint file. `run.seeds` is itself part of the fingerprint: a seed trained as a
separate job, from a config or override that names only that seed, gets a fingerprint of its own
(see the seeds table below). `checkpoint_sha256` is the sha256 of the exact `model.safetensors`
that was scored, so two score files that share it were unquestionably produced from the same
weights, whatever their `source` fingerprint says.

## `training_seed`: the score file records the run's seed, not the scoring command's

The `run:` source reads the run's own seed straight out of its `env.json` and sets it as the
returned detector's `training_seed`; `dfwb score`'s C5 meta then records *that* seed rather than
whatever `--seed` the scoring command itself was given (the default, `0`, used above, matters only
for a detector that has no seed of its own, such as `zoo:random`). Scoring the same run twice --
even with two different `--seed` values on the command line — always records the same `seed` in
the resulting score files' meta, because that number describes the weights, not the scoring
invocation.

## What makes two score files comparable

- **`dfwb eval compare`** needs no protocol match at all: it works from the intersection of two
  files' `ok` rows, keyed by `(dataset, key, compression)`, and reports how big that intersection
  is (`n`) — comparing across different splits or datasets is possible, but only ever as
  meaningful as the videos the two files actually share.
- **The "same run, another seed" table** (`dfwb eval`'s automatic `seeds` breakdown) is stricter:
  two files fold together only when their `detector.source`, `protocol.id`, `protocol.split`,
  `where`, `labels`, `aggregation` and `processing_profile` all agree and their `seed` differs —
  two runs of one experiment config, scored the same way. The table groups the seeds of **one
  experiment config**: `run:`'s `source` is the config fingerprint, and `run.seeds` is part of
  that fingerprint, so **list every seed in one config's `run.seeds`** (one `dfwb train` trains
  them all) to get it. Seeds trained as separate jobs, each from a config or override naming only
  its own seed, have different fingerprints and never fold together; compare their files with
  `dfwb eval compare` instead.
- **Two files with the same identity and the same seed** — a run's `#best` and `#last`
  checkpoints, say — are not two seeds. The seeds table keeps the first of them given on the
  command line, and a warning names the one it ignored.
- **A suite entry** matches a file by `protocol.id`, `protocol.split` and `where` alone
  (`docs/guides/cross-dataset-eval.md`), regardless of which detector produced it — so a suite
  aggregate genuinely compares different detectors on identical splits.

In short: two score files are safely comparable exactly when their metas agree on everything that
should not have changed between them, and disagree only on the one thing you are actually
comparing (the detector, or the seed).

## See also

- `docs/concepts/detectors.md` — the run directory in full, checkpoints, and the `run:` source.
- `docs/concepts/score-files.md` — the C5 schema and what a detector source can set on it.
- `docs/guides/cross-dataset-eval.md` — turning one or more score files into metrics and
  comparisons.
