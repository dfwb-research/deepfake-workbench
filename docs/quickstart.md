# Quickstart

## Try it in five minutes: toyfake

dfwb ships a synthetic dataset, **toyfake**, generated entirely from a seed, so the whole raw
dataset → inventory → protocol → training pipeline can be tried with nothing to download. From a
clone (see [Install](install.md)), install the `train` and `preprocess` extras:

```bash
uv sync --extra train --extra preprocess
```

Then run the pipeline end to end:

```bash
export DFWB_DATASETS_ROOT=~/datasets
export DFWB_WORK_ROOT=~/dfwb-work

uv run dfwb datasets synth toyfake --out "$DFWB_DATASETS_ROOT"
uv run dfwb inventory build toyfake
uv run dfwb protocols verify toyfake

uv run dfwb preprocess run toyfake --profile toy-64-center-8f
uv run dfwb preprocess status toyfake --profile toy-64-center-8f

cat > toy-cpu.yaml <<'YAML'
schema: dfwb.train/1
extends: [dfwb://templates/toy-cpu.yaml]
YAML
uv run dfwb train -c toy-cpu.yaml --device cpu

uv run dfwb runs list
uv run dfwb runs show toy-cpu
```

- **`datasets synth`** writes a small set of synthetic real and blended-fake videos to
  `$DFWB_DATASETS_ROOT/toyfake` (80 reals, 60 fakes of each of two blend methods, deterministic
  from `--seed`, default `0`).
- **`inventory build`** scans that folder and writes `$DFWB_WORK_ROOT/toyfake/inventory.jsonl`.
- **`protocols verify`** joins that inventory against dfwb's built-in toyfake protocol pack and
  writes a coverage report; with the defaults above it reports full coverage (`have: 200`,
  `missing: 0`).
- **`preprocess run`** samples eight frames from every video and keeps each frame's centred
  64-pixel square, with the `toy-64-center-8f` profile -- which has no face detector and no model
  to download, only the `preprocess` extra -- writing a lossless frame store under
  `$DFWB_WORK_ROOT/toyfake/processed/`; `preprocess status` then counts it by outcome.
- **`train`** needs the `train` extra (PyTorch, Lightning, timm); the `toy-cpu.yaml` file extends
  the shipped `toy-cpu` template, training a tiny CNN on the toyfake tree just built, for two
  epochs, writing its run under `./runs/toy-cpu/`. `runs list` shows it; `runs show` shows its full
  detail: data sources, validation metrics, and the config fingerprint that identifies the
  experiment.

From here, score the trained run and turn its scores into metrics with confidence intervals:

```bash
uv run dfwb score --detector "run:runs/toy-cpu/latest#best" --protocol toyfake/official --split test
uv run dfwb eval runs/scores/tiny-cnn-mean-linear/toyfake-official/*.scores.csv
```

See [The pipeline](concepts/pipeline.md) for how these stages and contracts fit together,
[Score files](concepts/score-files.md) for what `dfwb score` writes, and
[Cross-dataset evaluation](guides/cross-dataset-eval.md) for scoring a suite and comparing
detectors.

## With your own datasets

!!! warning "Needs your own data"
    This section is written from the code, the same way as the toyfake walkthrough above, but it
    is **not** run as part of building these docs: FaceForensics++ and Celeb-DF are licensed
    datasets you obtain from their own publishers, not from dfwb. Adjust the paths, protocol refs
    and template below to whatever datasets and split schemes you actually have.

The shape is the same as the toyfake walkthrough, just pointed at real data and a real
face-detection backend. Say you have FaceForensics++ under `~/datasets/FaceForensics++` and
Celeb-DF v2 under `~/datasets/Celeb-DF-v2` (dfwb's own folder names for each, shown by
`dfwb datasets info ffpp` / `dfwb datasets info celebdf-v2`, and in [Datasets](datasets/index.md)):

Split schemes for real datasets are not shipped with dfwb itself; `dfwb inventory build` needs
only the dataset's own folder, but `dfwb protocols verify` needs a separately installed protocol
pack that publishes it, such as `dfwb-protocols` (in preparation -- see
[Ecosystem](ecosystem.md)), installed in the same environment as dfwb:

```bash
export DFWB_DATASETS_ROOT=~/datasets
export DFWB_WORK_ROOT=~/dfwb-work

uv run dfwb inventory build ffpp
uv run dfwb inventory build celebdf-v2

uv run dfwb protocols verify ffpp
uv run dfwb protocols verify celebdf-v2
```

`verify` reports coverage against whichever scheme each dataset defaults to (`ffpp`'s is
`official`; `celebdf-v2`'s is `official+ident-80-20` -- the official test split plus an
identity-disjoint train/val carve).

Processing real video needs a real face-detection backend. dfwb's own `binary-frame` template
(used below) and this walkthrough both use the shipped `face-256-1.3x-32f` profile (the
`insightface` backend, whose `buffalo_l` weights are for non-commercial research use only) or its
permissive twin, `face-256-1.3x-32f-mp` (Apache-2.0 `mediapipe`); see [Install](install.md) for
both extras.

```bash
# --inexact: a plain `uv sync` removes anything not in the lockfile, including the
# already-installed dfwb-protocols pack this section's `protocols verify` step above needed.
uv sync --extra train --extra face-mediapipe --inexact

uv run dfwb preprocess run ffpp --profile face-256-1.3x-32f-mp
uv run dfwb preprocess run celebdf-v2 --profile face-256-1.3x-32f-mp
```

Train from the shipped `binary-frame` template, extending it with your own run name, data sources
and backbone (`dfwb config init --template binary-frame` writes a starter file that does this):

```yaml
# ffpp-vit.yaml
schema: dfwb.train/1
extends: [dfwb://templates/binary-frame.yaml]

run:
  name: ffpp-vit
  seeds: [42]

data:
  processing: face-256-1.3x-32f-mp
  train: [{protocol: ffpp/official, split: train, where: {compression: c23}}]
  val:   [{protocol: ffpp/official, split: val,   where: {compression: c23}}]

model:
  backbone: {name: timm, model: vit_base_patch16_224.augreg_in21k, pretrained: true}
```

```bash
uv run dfwb config validate -c ffpp-vit.yaml
uv run dfwb train -c ffpp-vit.yaml
```

`config validate` needs no data at all -- it checks the config's shape and every component's
parameters against the installed plugins -- so it is worth running as soon as you have written a
config, well before pointing it at real data.

Then score the trained run **cross-dataset**, against Celeb-DF v2 rather than the FaceForensics++
split it trained on, and evaluate the result:

```bash
uv run dfwb score --detector "run:runs/ffpp-vit/latest#best" --protocol celebdf-v2 --split test
uv run dfwb eval runs/scores/*/celebdf-v2-*/*.scores.csv --bootstrap 2000
```

See [Cross-dataset evaluation](guides/cross-dataset-eval.md) for scoring several splits in one
command (`--suite`) once a pack such as `dfwb-protocols` registers a real cross-dataset panel, and
[Adding a dataset](guides/add-a-dataset.md) if your dataset has no inventory builder yet.
