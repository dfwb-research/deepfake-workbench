# Reproduce a benchmark

This guide walks from a fresh clone all the way to a scored, evaluated benchmark run, using the
runnable configs in [`configs/`](https://github.com/dfwb-research/deepfake-workbench/tree/main/configs)
at the repository root. It covers getting the data, building and verifying its inventory,
processing it, training, and scoring and evaluating the result. For the other direction --
turning a run directory someone already handed you (or one you trained yourself) back into a
score file -- see [Reproduce a run](reproduce-a-run.md) instead; that page starts from a run
directory, this one starts from a clone.

## The toyfake benchmark: nothing to download

`configs/toyfake-cpu.yaml` extends the shipped `toy-cpu` template and runs entirely on dfwb's
synthetic toyfake dataset, generated from a seed -- there is nothing to obtain and no licence to
accept. From a fresh clone:

```bash
./scripts/setup.sh

uv run dfwb datasets synth toyfake --out data/datasets
uv run dfwb inventory build toyfake
uv run dfwb protocols verify toyfake

uv run dfwb preprocess run toyfake --profile toy-64-center-8f

uv run dfwb train -c configs/toyfake-cpu.yaml --device cpu

uv run dfwb score --detector "run:runs/toy-cpu/latest#best" --suite toyfake
uv run dfwb eval runs/scores/tiny-cnn-mean-linear/toyfake-*/*.scores.csv --suite toyfake
```

`scripts/setup.sh` (see the [README](https://github.com/dfwb-research/deepfake-workbench#run-it-from-a-clone))
installs the `train` and `preprocess` extras, copies `.env.example` to `.env` so
`DFWB_DATASETS_ROOT`, `DFWB_WORK_ROOT` and `DFWB_CACHE_ROOT` default to
`./data/{datasets,work,cache}` inside the clone and `DFWB_RUNS_ROOT` to `./runs`, and creates
those directories -- `dfwb` itself loads `.env` and reads them from there, so every command above
needs nothing exported by hand, **except** `datasets synth --out`: a plain file-write flag with no
environment default, so it names the same path (`data/datasets`) directly, run from the clone's
own root the way the whole block assumes. The run lands in `runs/toy-cpu/`, and `dfwb score`
writes its score files under `runs/scores/`. `dfwb score --suite toyfake` scores the trained run
against the framework's own two-entry suite: the in-domain test split (group `in-domain`) and the
identity-disjoint split (group `cross`). toyfake is a single dataset, so that second entry only
stands in for a cross-dataset one: it tests videos of identities the training split never saw,
not a different dataset. `dfwb eval --suite` prints the suite's aggregate rows (the mean AUC of
each group) beneath the usual per-file table, in every output form including the plain terminal
one. See
[Cross-dataset evaluation](cross-dataset-eval.md) for what a suite is and how the coverage policy
and comparisons work, and the [quickstart](../quickstart.md) for a slower walk through each step.

## A real benchmark: FaceForensics++ c23 with ViT-B/16

The other three configs need real, licensed data of your own; dfwb never distributes media, and
none of the steps below run as part of building these docs.

### 1. Get the data, under the owner's own terms

- **FaceForensics++**: request the download script through the form linked from the
  [FaceForensics repository](https://github.com/ondyari/FaceForensics) (`ondyari/FaceForensics`
  on GitHub). Its terms are non-commercial research use only.
- **Celeb-DF v2**: request the download from the authors through the form linked from the
  [Celeb-DF repository](https://github.com/yuezunli/celeb-deepfakeforensics)
  (`yuezunli/celeb-deepfakeforensics` on GitHub). Its terms are also non-commercial research use
  only.

Place each dataset in its own folder named the way dfwb expects (`dfwb datasets info ffpp` /
`dfwb datasets info celebdf-v2` print the exact layout) under a datasets root, then point
`DFWB_DATASETS_ROOT` at that root -- either in `.env`, or exported, same as any other setting (see
[Data in several places, several machines](https://github.com/dfwb-research/deepfake-workbench#data-in-several-places-several-machines)
in the README for a root that spans more than one location or machine).

Split schemes for real datasets are not shipped with dfwb itself: install
dfwb-protocols (in preparation, not public yet) into the
clone's environment, the same way [With dfwb-protocols and dfwb-torch](https://github.com/dfwb-research/deepfake-workbench#with-dfwb-protocols-and-dfwb-torch)
in the README shows, before `protocols verify` below.

### 2. Inventory: scan the folder you placed

```bash
uv run dfwb inventory build ffpp
uv run dfwb inventory build celebdf-v2
```

`inventory build` reads only the dataset's own folder -- it needs no protocol pack -- and writes
`$DFWB_WORK_ROOT/<dataset>/inventory.jsonl`.

### 3. Verify: check your copy against the published splits

```bash
uv run dfwb protocols verify ffpp
uv run dfwb protocols verify celebdf-v2
```

This joins the inventory just built against dfwb-protocols' split scheme for each dataset
(`ffpp/official`, `celebdf-v2/official+ident-80-20` -- each dataset's own default) and reports
coverage: `have`, `missing_requested` and `label_mismatch`. Fix any gap it reports (a missing
folder, a compression dfwb-protocols expects that your copy lacks) before preprocessing --
preprocessing an incomplete copy just reproduces the same gap later, as a training or scoring
failure instead of a clear coverage report.

### 4. Preprocess: run the face pipeline

`configs/ffpp-c23-vit-b16.yaml` and `configs/celebdf-v2-vit-b16.yaml` both use the
`face-256-1.3x-64f` profile, the `insightface` backend (`buffalo_l` weights, non-commercial
research use only -- see [Install](../install.md#face-insightface-cpu-by-default-gpu-by-hand) for
the extra and [Processing profiles](../concepts/processing-profiles.md#the-licence-gate) for
the acknowledgement `--accept-license` records once per machine).

A ViT-B/16 wants a GPU. Set the environment up in this order, the one
[Install](../install.md#training-a-cuda-build-with-uv) gives and `./scripts/setup.sh --gpu` prints:
first `uv sync` with every extra you need, then reinstall a CUDA build of `torch` over the CPU
build the lock pins, then run every later command with `uv run --no-sync`. Any later `uv sync`, or
a `uv run` without `--no-sync`, puts the CPU build back.

```bash
uv sync --locked --extra train --extra face-insightface

# Pick the index for your driver from the official selector at
# https://pytorch.org/get-started/locally/ -- cu121 here is an example.
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121

uv run --no-sync dfwb preprocess run ffpp --profile face-256-1.3x-64f --accept-license --device cuda:0
uv run --no-sync dfwb preprocess run celebdf-v2 --profile face-256-1.3x-64f --device cuda:0
uv run --no-sync dfwb preprocess status ffpp --profile face-256-1.3x-64f
uv run --no-sync dfwb preprocess status celebdf-v2 --profile face-256-1.3x-64f
```

The face detector itself runs on the GPU only with `onnxruntime-gpu` installed in place of
`onnxruntime` ([Install](../install.md#face-insightface-cpu-by-default-gpu-by-hand) says how);
without it, `--device cuda:0` runs the detector on the CPU, with a warning.

### 5. Train with `configs/…`

```bash
uv run --no-sync dfwb config validate -c configs/ffpp-c23-vit-b16.yaml
uv run --no-sync dfwb train -c configs/ffpp-c23-vit-b16.yaml --device cuda:0

uv run --no-sync dfwb config validate -c configs/celebdf-v2-vit-b16.yaml
uv run --no-sync dfwb train -c configs/celebdf-v2-vit-b16.yaml --device cuda:0
```

`config validate` needs no data at all: it checks the config's shape and every component
(backbone, head, loss, schedule, ...) against the installed plugins, so it is worth running as
soon as the `train` extra is installed, before pointing anything at the data prepared above. Both
configs' headers name exactly the steps above, so a config file is self-contained: everything it
needs to run is either in the file or in its own header comment.

`configs/cross-dataset-ffpp-celebdf.yaml` trains on FaceForensics++ c23, validating both
in-domain (FaceForensics++'s own val split) and cross-dataset (Celeb-DF v2's val split) every
epoch, once both datasets above are processed:

```bash
uv run --no-sync dfwb config validate -c configs/cross-dataset-ffpp-celebdf.yaml
uv run --no-sync dfwb train -c configs/cross-dataset-ffpp-celebdf.yaml --device cuda:0
```

That training-time validation is for watching generalisation as training goes; it is not the
final cross-dataset numbers -- those come from scoring and evaluating a suite, step 6 below.

### 6. Score the suite

dfwb-protocols (in preparation, not public yet) registers a cross-dataset suite,
`cross-dataset-v1` (see [Ecosystem](../ecosystem.md)): seventeen test splits, FaceForensics++'s
own held-out FaceShifter subset and sixteen other datasets, all in one group, `cross-dataset`.
Score a trained run against it in one command, exactly as the toyfake walkthrough above scores
against the framework's own `toyfake` suite:

```bash
uv run --no-sync dfwb score --detector "run:runs/ffpp-c23-vit-b16/latest#best" --suite cross-dataset-v1 --out runs/scores/ffpp-c23-vit-b16 --device cuda:0
```

`dfwb score` writes one file per entry of the suite, at
`<out>/<detector>/<protocol>/<split>-<key>.scores.csv`: here
`runs/scores/ffpp-c23-vit-b16/timm-mean-linear/ffpp-official/test-….scores.csv`,
`…/celebdf-v2-official-ident-80-20/test-….scores.csv`, and so on, one folder per protocol of the
suite. `<detector>` is named after the model's parts (`timm-mean-linear` for every ViT-B/16 run
above), so `--out` keeps this run's files apart from any other run's. Scoring the suite needs
every dataset it names inventoried and processed locally; to score one dataset at a time
instead:

```bash
uv run --no-sync dfwb score --detector "run:runs/ffpp-c23-vit-b16/latest#best" --protocol celebdf-v2 --split test --out runs/scores/ffpp-c23-vit-b16 --device cuda:0
```

### 7. Eval

```bash
uv run --no-sync dfwb eval runs/scores/ffpp-c23-vit-b16/*/*/*.scores.csv --suite cross-dataset-v1 --bootstrap 2000
```

The glob matches every file step 6 wrote, one per protocol folder. `dfwb eval --suite` reports
the usual per-file, coverage-aware metrics with confidence intervals, plus the suite's aggregate
row beneath them: `cross-dataset-v1` has one, the mean AUC over the entries of its one group,
`cross-dataset`. It is printed in every output form, including the plain terminal one. To
evaluate the one-split file from step 6 on its own instead:

```bash
uv run --no-sync dfwb eval runs/scores/ffpp-c23-vit-b16/*/celebdf-v2-*/*.scores.csv --bootstrap 2000
```

See [Cross-dataset evaluation](cross-dataset-eval.md) for `--by`, the coverage policy
(`--missing`/`--min-coverage`) and comparing two detectors with `dfwb eval compare`.

## See also

- [Reproduce a run](reproduce-a-run.md) -- from a run directory to a score file, and what makes
  two score files comparable.
- [Cross-dataset evaluation](cross-dataset-eval.md) -- suites, the coverage policy, and `dfwb eval
  compare`.
- [Adding a dataset](add-a-dataset.md) -- if your dataset has no inventory builder yet.
